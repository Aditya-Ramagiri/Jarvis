"""Lower-priority extras (spec 7.8): package tracking, maths, transcription,
finance tracker.

These are marked as lower priority in the spec, so each one takes the simplest
approach that genuinely works rather than the most complete one:

* Package tracking has no free universal API. Rather than pretend, Adrien
  identifies the carrier from the tracking number's own format and hands back
  a tracking URL plus what it can infer - which is what actually helps.
* Meeting transcription reuses the existing Whisper path rather than adding a
  second STT stack.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from adrien.config import env_str
from adrien.logging_setup import get_logger
from adrien.tools.registry import ToolResult, tool

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Package tracking
# --------------------------------------------------------------------------
# Carrier fingerprints, in the order they should be tested (most specific
# first). Universal tracking APIs are all paid, so identifying the carrier and
# handing back its tracking URL is the honest maximum here.
_CARRIERS: list[tuple[str, re.Pattern[str], str]] = [
    ("UPS", re.compile(r"^1Z[0-9A-Z]{16}$", re.I),
     "https://www.ups.com/track?tracknum={}"),
    ("FedEx", re.compile(r"^\d{12}$|^\d{15}$|^\d{20}$"),
     "https://www.fedex.com/fedextrack/?trknbr={}"),
    ("USPS", re.compile(r"^(94|93|92|94|95)\d{20}$|^[A-Z]{2}\d{9}[A-Z]{2}$"),
     "https://tools.usps.com/go/TrackConfirmAction?tLabels={}"),
    ("DHL", re.compile(r"^\d{10}$|^\d{11}$"),
     "https://www.dhl.com/en/express/tracking.html?AWB={}"),
    ("An Post / international", re.compile(r"^[A-Z]{2}\d{9}[A-Z]{2}$", re.I),
     "https://track.aftership.com/{}"),
]


@tool(category="extras")
def track_package(order_info: str) -> ToolResult:
    """Identify the carrier for a tracking number and give a tracking link.

    Args:
        order_info: The tracking number, or a sentence containing one.
    """
    candidates = re.findall(r"\b[0-9A-Z]{9,26}\b", order_info.upper())
    if not candidates:
        return ToolResult.failure("no tracking number in that - read out the number itself")

    number = max(candidates, key=len)
    for carrier, pattern, url in _CARRIERS:
        if pattern.match(number):
            return ToolResult.success(
                {"carrier": carrier, "tracking_number": number, "url": url.format(number)},
                speak=f"that looks like a {carrier} parcel; the tracking page is open to check",
            )

    return ToolResult.success(
        {"tracking_number": number, "carrier": "unknown",
         "url": f"https://track.aftership.com/{number}"},
        speak="that number does not match a carrier Adrien recognises, "
              "but a universal tracking link will work",
    )


# --------------------------------------------------------------------------
# Money maths
# --------------------------------------------------------------------------
@tool(category="extras")
def split_bill(amount: float, people: int, tip_percent: float = 0.0) -> ToolResult:
    """Split a bill between people, optionally adding a tip.

    Args:
        amount: The total before tip.
        people: How many people are splitting it.
        tip_percent: Tip to add, as a percentage.
    """
    if people < 1:
        return ToolResult.failure("that needs at least one person")
    if amount < 0:
        return ToolResult.failure("the amount cannot be negative")

    total = amount * (1 + tip_percent / 100)
    each = round(total / people, 2)
    # Rounding each share up to the cent can leave the total a cent or two
    # short; the remainder is called out rather than silently lost.
    remainder = round(total - each * people, 2)

    speak = f"{each:.2f} each"
    if tip_percent:
        speak = f"{each:.2f} each, including the {tip_percent:.0f} percent tip"
    return ToolResult.success(
        {"total": round(total, 2), "per_person": each, "people": people,
         "tip_percent": tip_percent, "rounding_remainder": remainder},
        speak=speak,
    )


@tool(category="extras")
def calculate_tip(amount: float, percent: float = 15.0) -> ToolResult:
    """Work out a tip and the resulting total.

    Args:
        amount: The bill before tip.
        percent: The tip percentage.
    """
    tip = round(amount * percent / 100, 2)
    return ToolResult.success(
        {"tip": tip, "total": round(amount + tip, 2), "percent": percent},
        speak=f"{tip:.2f} tip, {amount + tip:.2f} in total",
    )


# --------------------------------------------------------------------------
# Meeting transcription
# --------------------------------------------------------------------------
@tool(category="extras", timeout=300.0)
async def transcribe_meeting(audio_source: str, summarize: bool = True) -> ToolResult:
    """Transcribe a recorded meeting or call from an audio file, and summarise it.

    Args:
        audio_source: Path to the audio file.
        summarize: Also produce a short summary with any action items.
    """
    path = Path(audio_source).expanduser()
    if not path.exists():
        return ToolResult.failure(f"there is no audio file at {path}")

    from adrien.core.stt import Transcriber

    # Reuse the same Whisper path as live speech - one STT stack, one set of
    # keys, one place where accuracy is tuned.
    try:
        import soundfile
    except ImportError:
        return ToolResult.failure("soundfile is not installed, so Adrien cannot read that file")

    try:
        data, sample_rate = soundfile.read(str(path), dtype="int16", always_2d=True)
        mono = data[:, 0].tobytes()
    except Exception as exc:
        return ToolResult.failure(f"could not read the audio: {exc}")

    transcription = await Transcriber().transcribe(mono, sample_rate=sample_rate)
    if transcription.is_empty:
        return ToolResult.failure("the transcription came back empty")

    payload: dict[str, Any] = {
        "transcript": transcription.text,
        "duration_seconds": round(transcription.duration_s, 1),
        "words": len(transcription.text.split()),
    }
    if not summarize:
        return ToolResult.success(payload, speak=f"transcribed {payload['words']} words")

    from adrien.core.llm_router import LLMRouter

    summary = await LLMRouter().complete(
        "Summarise this meeting in three or four sentences, then list any action "
        "items as plain sentences. No markdown.\n\n" + transcription.text[:12000],
        tier="smart",
        max_tokens=500,
    )
    payload["summary"] = summary
    return ToolResult.success(payload, speak=summary)


# --------------------------------------------------------------------------
# Finance tracker (the user's own app at ledger.raidnxt.com)
# --------------------------------------------------------------------------
@tool(category="extras", timeout=25.0)
async def read_finance_tracker_summary(period: str = "this month") -> ToolResult:
    """Check the budget from the user's own finance tracker app.

    Args:
        period: Which period to summarise, e.g. "this month" or "this week".
    """
    from adrien.core.http import get_client

    base = env_str("FINANCE_TRACKER_URL")
    token = env_str("FINANCE_TRACKER_TOKEN")
    if not base:
        return ToolResult.failure("FINANCE_TRACKER_URL is not set in .env")

    headers = {"authorization": f"Bearer {token}"} if token else {}
    # The tracker is the user's own app and its API is not public, so Adrien
    # tries the conventional endpoints and reports plainly when none answer,
    # rather than guessing at a response shape it has never seen.
    for endpoint in ("/api/summary", "/api/v1/summary", "/summary.json", "/api/budget"):
        try:
            response = await get_client().get(
                f"{base.rstrip('/')}{endpoint}",
                params={"period": period},
                headers=headers,
                timeout=12,
            )
        except Exception:
            continue
        if response.status_code == 401:
            return ToolResult.failure("the finance tracker rejected the token in .env")
        if response.status_code != 200:
            continue
        try:
            body = response.json()
        except ValueError:
            continue

        spent = body.get("spent") or body.get("total_spent") or body.get("expenses")
        budget = body.get("budget") or body.get("limit")
        speak = "budget summary is in"
        if spent is not None and budget:
            remaining = round(float(budget) - float(spent), 2)
            speak = (f"{spent} of {budget} spent {period}, {remaining} left"
                     if remaining >= 0 else
                     f"{spent} spent {period}, {abs(remaining)} over budget")
        elif spent is not None:
            speak = f"{spent} spent {period}"
        return ToolResult.success({"period": period, "endpoint": endpoint, "data": body},
                                  speak=speak)

    return ToolResult.failure(
        f"the finance tracker at {base} did not answer on any of the endpoints Adrien knows. "
        "Point FINANCE_TRACKER_URL at its API, or add the endpoint to extra_tools.py."
    )
