"""Post-session summarisation and fact extraction (spec 6.2).

When a conversation ends, one LLM call does two jobs at once: write a summary
of what happened, and pull out any durable facts worth keeping. One call rather
than two because they need the same context and the latency is invisible here -
this runs after the user has walked away.

Facts are extracted as `(subject, predicate, value, category)` triples rather
than free text, so the structured store can supersede an old value when it
changes ("the server moved") instead of accumulating contradictions.

The extraction prompt is deliberately strict about what counts as durable. A
memory full of "the user asked about the weather" is worse than no memory: it
crowds out the real facts at retrieval time and makes every future prompt
longer and slower.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from adrien.logging_setup import get_logger
from adrien.memory.structured_store import Fact

log = get_logger(__name__)

# Categories seed the model's choices without constraining it - a new one is
# just a new string, which is what keeps the schema open (spec 6.2).
SUGGESTED_CATEGORIES = [
    "preference", "project", "person", "device", "schedule", "credential-free-config",
    "gaming", "development", "finance", "general",
]

EXTRACTION_PROMPT = """You are reviewing a conversation between a user and their voice assistant.

Return ONE JSON object, nothing else, with exactly these keys:

  "summary": 1-3 sentences on what actually happened and what was decided.
  "facts": a list of durable facts learned about the user or their projects.

Each fact is an object: {{"subject": ..., "predicate": ..., "value": ..., "category": ...}}
For example: {{"subject": "the Minecraft server", "predicate": "is hosted at",
"value": "rhs.raidnxt.com", "category": "gaming"}}

Only record something as a fact if it will STILL BE TRUE AND USEFUL NEXT MONTH.

  Record: stable preferences, project names and addresses, people's names and
  how they relate to the user, hardware, recurring commitments, how the user
  likes things done.

  Do NOT record: anything the assistant did in this conversation, one-off
  questions, the weather, the time, transient state ("the server is up right
  now"), or anything already obvious from the assistant's own tools.

An empty facts list is the correct answer for most conversations. Never invent
a fact that was not stated. Never record API keys, tokens or passwords.

Suggested categories (use another if none fit): {categories}

Conversation:
{transcript}"""


@dataclass
class SessionDigest:
    summary: str
    facts: list[Fact]
    raw: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.summary and not self.facts


class Summarizer:
    """Turns a finished transcript into a summary and durable facts."""

    def __init__(self, router: Any = None) -> None:
        # Injected in tests; built on demand in production so importing the
        # memory package does not require any keys.
        self._router = router

    @property
    def router(self) -> Any:
        if self._router is None:
            from adrien.core.llm_router import LLMRouter

            self._router = LLMRouter()
        return self._router

    async def digest(self, transcript: list[dict[str, Any]], session_id: str = "") -> SessionDigest:
        """Summarise a session and extract its durable facts."""
        rendered = render_transcript(transcript)
        if len(rendered.split()) < 12:
            # Nothing happened worth a round trip - "what time is it" does not
            # need a summary, and storing one would pollute recall.
            log.debug("session %s too short to summarise", session_id or "?")
            return SessionDigest(summary="", facts=[])

        prompt = EXTRACTION_PROMPT.format(
            categories=", ".join(SUGGESTED_CATEGORIES),
            transcript=rendered[:12000],
        )
        try:
            # Smart tier: fact extraction is exactly the judgement call the
            # small model is worst at, and nobody is waiting on this.
            raw = await self.router.complete(
                prompt,
                system="You extract structured memory. You reply with JSON and nothing else.",
                tier="smart",
                max_tokens=800,
                temperature=0.1,
            )
        except Exception as exc:
            log.error("session summarisation failed: %s", exc)
            return SessionDigest(summary="", facts=[])

        summary, facts = parse_digest(raw, session_id)
        log.info("session %s: %d chars of summary, %d fact(s)",
                 session_id or "?", len(summary), len(facts))
        return SessionDigest(summary=summary, facts=facts, raw=raw)


def render_transcript(transcript: list[dict[str, Any]]) -> str:
    """Turn stored messages into a readable script for the model."""
    lines: list[str] = []
    for message in transcript:
        role = message.get("role", "")
        content = (message.get("content") or "").strip()
        if not content or role == "system":
            continue
        speaker = {"user": "User", "assistant": "Adrien", "tool": "Tool result"}.get(role, role)
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_digest(raw: str, session_id: str = "") -> tuple[str, list[Fact]]:
    """Parse the model's JSON, tolerating the ways models wrap it.

    Models fence JSON in markdown, prepend "Here is the JSON:", and
    occasionally trail a sentence after it. Rather than fight that with
    prompting alone, the outermost braces are extracted and parsed.
    """
    if not raw or not raw.strip():
        return "", []

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()

    payload: dict[str, Any] | None = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        # Not JSON at all. The prose is still a usable summary; better to keep
        # it than to throw the whole session away.
        log.warning("digest was not JSON; keeping it as a plain summary")
        return text[:600], []

    summary = str(payload.get("summary") or "").strip()
    facts: list[Fact] = []
    for item in payload.get("facts") or []:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        value = str(item.get("value") or "").strip()
        if not (subject and predicate and value):
            continue
        if _looks_like_a_secret(value):
            log.info("dropped an extracted fact that looked like a credential")
            continue
        facts.append(Fact(
            subject=subject,
            predicate=predicate,
            value=value,
            category=str(item.get("category") or "general").strip() or "general",
            confidence=float(item.get("confidence", 0.8)),
            source_session=session_id or None,
        ))
    return summary, facts


_SECRET_SHAPES = re.compile(
    r"(gsk_|sk-|ghp_|AIza|GOCSPX-|Bearer\s)|(^[A-Za-z0-9+/]{40,}={0,2}$)"
)


def _looks_like_a_secret(value: str) -> bool:
    """Last line of defence for spec 9: no credentials in the memory store."""
    return bool(_SECRET_SHAPES.search(value.strip()))
