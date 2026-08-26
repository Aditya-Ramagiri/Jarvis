"""Productivity tools: reminders, timers, notes, todos, Calendar, Gmail (7.4).

Reminders and todos live in Adrien's own SQLite store rather than macOS
Reminders, because they have to survive a reboot, be queryable by the memory
system, and fire through Adrien's own voice - none of which the system app
gives us from a background service.

Email is **draft-only by default**: `draft_email_reply` writes a Gmail draft and
stops. Actually sending is a separate tool behind the confirmation layer
(spec 7.4 and section 9), so no plausible chain of tool calls ends with Adrien
sending mail on its own.
"""

from __future__ import annotations

import datetime as dt
import re
import time

from adrien.config import data_dir, env_str
from adrien.logging_setup import get_logger
from adrien.memory.structured_store import StructuredStore
from adrien.tools.registry import ToolResult, tool

log = get_logger(__name__)

_store: StructuredStore | None = None


def store() -> StructuredStore:
    """Lazily opened shared store (also injected by the orchestrator)."""
    global _store
    if _store is None:
        _store = StructuredStore()
    return _store


def set_store(instance: StructuredStore) -> None:
    global _store
    _store = instance


# --------------------------------------------------------------------------
# Time parsing
# --------------------------------------------------------------------------
_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.I,
)
_UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}
_CLOCK = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)


def parse_duration(text: str) -> float | None:
    """Seconds from '10 minutes', 'an hour and a half', '90s'."""
    total = 0.0
    for amount, unit in _DURATION.findall(text or ""):
        total += float(amount) * _UNIT_SECONDS.get(unit.lower().rstrip("."), 0)
    if "half an hour" in (text or "").lower():
        total += 1800
    if re.search(r"\ban hour\b", text or "", re.I) and total < 3600:
        total += 3600
    return total or None


def parse_when(text: str, *, now: dt.datetime | None = None) -> float | None:
    """Absolute epoch time from a spoken time expression.

    Handles the shapes people actually say to a voice assistant - "in ten
    minutes", "at 6pm", "tomorrow at 9" - and gives up cleanly on anything
    else rather than guessing, so the model can ask.
    """
    if not text:
        return None
    now = now or dt.datetime.now()
    lowered = text.strip().lower()

    if lowered.startswith("in ") or _DURATION.search(lowered) and "at " not in lowered:
        seconds = parse_duration(lowered)
        if seconds:
            return (now + dt.timedelta(seconds=seconds)).timestamp()

    base = now
    if "tomorrow" in lowered:
        base = now + dt.timedelta(days=1)
    elif "tonight" in lowered:
        base = now

    match = _CLOCK.search(lowered)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = (match.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif not meridiem and "tonight" in lowered and hour < 12:
            hour += 12
        target = base.replace(hour=min(hour, 23), minute=minute, second=0, microsecond=0)
        # "at 8" said at 9pm means tomorrow morning, not twelve hours ago.
        if target <= now and "tomorrow" not in lowered:
            target += dt.timedelta(days=1)
        return target.timestamp()

    seconds = parse_duration(lowered)
    return (now + dt.timedelta(seconds=seconds)).timestamp() if seconds else None


def describe_delay(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)} seconds"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    if seconds < 86400:
        return f"{round(seconds / 3600, 1)} hours"
    return f"{round(seconds / 86400, 1)} days"


# --------------------------------------------------------------------------
# Reminders and timers
# --------------------------------------------------------------------------
@tool(category="productivity")
def set_reminder(text: str, time: str) -> ToolResult:
    """Set a reminder for a specific time.

    Args:
        text: What to be reminded about.
        time: When, e.g. "in 20 minutes", "at 6pm", "tomorrow at 9".
    """
    due = parse_when(time)
    if due is None:
        return ToolResult.failure(f"could not work out when '{time}' is - ask for a clearer time")

    reminder_id = store().add_reminder(text, due, kind="reminder")
    when = dt.datetime.fromtimestamp(due)
    return ToolResult.success(
        {"id": reminder_id, "text": text, "due_at": due, "due_human": when.strftime("%H:%M on %d %b")},
        speak=f"reminder set for {when.strftime('%H:%M')}",
    )


@tool(category="productivity")
def set_timer(duration: str, label: str = "") -> ToolResult:
    """Set a countdown timer.

    Args:
        duration: How long, e.g. "10 minutes", "90 seconds".
        label: What the timer is for.
    """
    seconds = parse_duration(duration)
    if seconds is None:
        return ToolResult.failure(f"could not work out how long '{duration}' is")

    timer_id = store().add_reminder(label or "timer", time.time() + seconds, kind="timer")
    return ToolResult.success(
        {"id": timer_id, "seconds": seconds, "label": label},
        speak=f"timer set for {describe_delay(seconds)}",
    )


@tool(category="productivity")
def list_reminders() -> ToolResult:
    """List reminders and timers that have not gone off yet."""
    pending = store().pending_reminders()
    if not pending:
        return ToolResult.success({"reminders": []}, speak="nothing pending")
    items = [
        {
            "text": row["text"],
            "kind": row["kind"],
            "due_in": describe_delay(max(0, row["due_at"] - time.time())),
        }
        for row in pending
    ]
    return ToolResult.success(
        {"count": len(items), "reminders": items},
        speak=f"{len(items)} pending, next is {items[0]['text']} in {items[0]['due_in']}",
    )


@tool(category="productivity", destructive=True, confirm="Cancel the reminder about {text}?")
def cancel_reminder(text: str) -> ToolResult:
    """Cancel a pending reminder or timer.

    Args:
        text: Enough of the reminder's wording to identify it.
    """
    for row in store().pending_reminders():
        if text.lower() in row["text"].lower():
            store().cancel_reminder(row["id"])
            return ToolResult.success({"cancelled": row["text"]}, speak="cancelled")
    return ToolResult.failure(f"no pending reminder matching '{text}'")


# --------------------------------------------------------------------------
# Notes and todos
# --------------------------------------------------------------------------
@tool(category="productivity")
def create_note(text: str, tag: str = "") -> ToolResult:
    """Write down a note.

    Args:
        text: The note's content.
        tag: Optional label to group it under.
    """
    note_id = store().add_note(text, tag)
    return ToolResult.success({"id": note_id}, speak="noted")


@tool(category="productivity")
def read_notes(tag: str = "", limit: int = 10) -> ToolResult:
    """Read back recent notes.

    Args:
        tag: Only notes with this label.
        limit: How many to read back.
    """
    notes = store().get_notes(tag, limit)
    if not notes:
        return ToolResult.success({"notes": []}, speak="no notes saved")
    return ToolResult.success(
        {"count": len(notes),
         "notes": [{"text": note["text"], "tag": note["tag"],
                    "when": dt.datetime.fromtimestamp(note["created_at"]).strftime("%d %b %H:%M")}
                   for note in notes]},
        speak=f"{len(notes)} note{'s' if len(notes) != 1 else ''}, most recent: {notes[0]['text'][:160]}",
    )


@tool(category="productivity")
def manage_todo(action: str, item: str = "") -> ToolResult:
    """Add, complete or list todo items.

    Args:
        action: One of add, complete, or list.
        item: The todo's text. Required for add and complete.
    """
    verb = action.strip().lower()

    if verb in ("add", "create", "new"):
        if not item:
            return ToolResult.failure("no todo text was given")
        store().add_todo(item)
        return ToolResult.success({"added": item}, speak="added")

    if verb in ("complete", "done", "finish", "tick", "check"):
        if not item:
            return ToolResult.failure("say which todo to complete")
        completed = store().complete_todo(item)
        if completed is None:
            return ToolResult.failure(f"no open todo matching '{item}'")
        return ToolResult.success({"completed": completed["text"]},
                                  speak=f"ticked off {completed['text']}")

    if verb in ("list", "show", "read"):
        todos = store().get_todos()
        if not todos:
            return ToolResult.success({"todos": []}, speak="the todo list is empty")
        return ToolResult.success(
            {"count": len(todos), "todos": [todo["text"] for todo in todos]},
            speak=f"{len(todos)} open: " + ", ".join(todo["text"] for todo in todos[:5]),
        )

    return ToolResult.failure(f"unknown todo action '{action}' - use add, complete or list")


# --------------------------------------------------------------------------
# Google: Calendar and Gmail
# --------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]
TOKEN_PATH = data_dir() / "google_token.json"


def google_service(api: str, version: str):
    """Authorised Google API client, or (None, error).

    The OAuth *consent* step needs a browser and cannot happen inside a
    background service, so it is a one-off setup command
    (`python -m adrien auth-google`). At runtime this only ever refreshes an
    existing token, and says so plainly if there isn't one.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        return None, "the Google API client libraries are not installed"

    if not TOKEN_PATH.exists():
        return None, (
            "Google is not connected yet - run 'python -m adrien auth-google' "
            "once from a terminal to authorise it"
        )

    try:
        credentials = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if not credentials.valid:
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")
            else:
                return None, "the Google authorisation expired - run 'python -m adrien auth-google'"
        return build(api, version, credentials=credentials, cache_discovery=False), ""
    except Exception as exc:
        return None, f"could not authorise with Google: {exc}"


def run_google_oauth_flow() -> tuple[bool, str]:
    """Interactive consent, run once from a terminal by `adrien auth-google`."""
    client_id = env_str("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = env_str("GOOGLE_OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return False, "GOOGLE_OAUTH_CLIENT_ID / _SECRET are missing from .env"

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return False, "google-auth-oauthlib is not installed"

    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        SCOPES,
    )
    credentials = flow.run_local_server(port=0)
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")
    TOKEN_PATH.chmod(0o600)
    return True, f"Google connected. Token stored at {TOKEN_PATH}"


@tool(category="productivity", timeout=25.0)
def check_calendar(date_range: str = "today") -> ToolResult:
    """Check what is on the calendar.

    Args:
        date_range: today, tomorrow, this week, or next week.
    """
    service, error = google_service("calendar", "v3")
    if service is None:
        return ToolResult.failure(error)

    now = dt.datetime.now(dt.UTC).astimezone()
    window = (date_range or "today").strip().lower()
    if "tomorrow" in window:
        start = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + dt.timedelta(days=1)
    elif "next week" in window:
        start = (now + dt.timedelta(days=7 - now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0)
        end = start + dt.timedelta(days=7)
    elif "week" in window:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + dt.timedelta(days=7)
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + dt.timedelta(days=1)

    try:
        response = service.events().list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=20,
        ).execute()
    except Exception as exc:
        return ToolResult.failure(f"could not read the calendar: {exc}")

    events = []
    for event in response.get("items", []):
        when = event.get("start", {})
        starts = when.get("dateTime") or when.get("date") or ""
        events.append({
            "summary": event.get("summary", "(no title)"),
            "start": starts,
            "location": event.get("location", ""),
            "all_day": "date" in when,
        })

    if not events:
        return ToolResult.success({"events": []}, speak=f"nothing on {window}")

    first = events[0]
    label = first["start"][11:16] if len(first["start"]) > 15 else "all day"
    return ToolResult.success(
        {"range": window, "count": len(events), "events": events},
        speak=f"{len(events)} thing{'s' if len(events) != 1 else ''} {window}, "
              f"first is {first['summary']} at {label}",
    )


@tool(category="productivity", destructive=True,
      confirm="Add '{details}' to your calendar?", timeout=25.0)
def add_calendar_event(details: str, start: str = "", duration_minutes: int = 60) -> ToolResult:
    """Add an event to the calendar.

    Args:
        details: The event's title.
        start: When it starts, e.g. "tomorrow at 3pm".
        duration_minutes: How long it runs for.
    """
    service, error = google_service("calendar", "v3")
    if service is None:
        return ToolResult.failure(error)

    starts_at = parse_when(start) if start else None
    if starts_at is None:
        return ToolResult.failure("could not work out when the event starts")

    begin = dt.datetime.fromtimestamp(starts_at).astimezone()
    finish = begin + dt.timedelta(minutes=max(5, duration_minutes))
    try:
        created = service.events().insert(
            calendarId="primary",
            body={
                "summary": details,
                "start": {"dateTime": begin.isoformat()},
                "end": {"dateTime": finish.isoformat()},
            },
        ).execute()
    except Exception as exc:
        return ToolResult.failure(f"could not create the event: {exc}")

    return ToolResult.success(
        {"id": created.get("id"), "summary": details, "start": begin.isoformat()},
        speak=f"added {details} at {begin.strftime('%H:%M on %A')}",
    )


@tool(category="productivity", timeout=25.0)
def read_gmail(filter: str = "is:unread", limit: int = 5) -> ToolResult:
    """Read recent email headers and snippets.

    Args:
        filter: A Gmail search query, e.g. "is:unread" or "from:alice".
        limit: How many messages to read.
    """
    service, error = google_service("gmail", "v1")
    if service is None:
        return ToolResult.failure(error)

    try:
        listing = service.users().messages().list(
            userId="me", q=filter or "is:unread", maxResults=max(1, min(limit, 10))
        ).execute()
        messages = []
        for stub in listing.get("messages", []):
            detail = service.users().messages().get(
                userId="me", id=stub["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
            messages.append({
                "id": stub["id"],
                "from": headers.get("From", ""),
                "subject": headers.get("Subject", "(no subject)"),
                "snippet": detail.get("snippet", "")[:200],
            })
    except Exception as exc:
        return ToolResult.failure(f"could not read Gmail: {exc}")

    if not messages:
        return ToolResult.success({"messages": []}, speak="no matching email")
    top = messages[0]
    sender = top["from"].split("<")[0].strip().strip('"') or top["from"]
    return ToolResult.success(
        {"count": len(messages), "messages": messages},
        speak=f"{len(messages)} message{'s' if len(messages) != 1 else ''}, "
              f"top one from {sender}: {top['subject']}",
    )


@tool(category="productivity", timeout=45.0)
async def draft_email_reply(context: str, message_id: str = "", tone: str = "friendly") -> ToolResult:
    """Draft a reply to an email and save it to Gmail drafts without sending it.

    Args:
        context: What the reply should say.
        message_id: The Gmail message id being replied to, if there is one.
        tone: The tone to write in, e.g. friendly, formal or brief.
    """
    import base64
    from email.message import EmailMessage

    from adrien.core.llm_router import LLMRouter

    service, error = google_service("gmail", "v1")
    if service is None:
        return ToolResult.failure(error)

    original_subject, recipient, quoted = "", "", ""
    if message_id:
        try:
            detail = service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
            headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
            original_subject = headers.get("Subject", "")
            recipient = headers.get("From", "")
            quoted = detail.get("snippet", "")[:800]
        except Exception as exc:
            return ToolResult.failure(f"could not load that message: {exc}")

    router = LLMRouter()
    body = await router.complete(
        "Write the body of an email reply. No subject line, no placeholders "
        f"like [Name], no markdown. Tone: {tone}.\n\n"
        + (f"The email being replied to said: {quoted}\n\n" if quoted else "")
        + f"What the reply should get across: {context}",
        tier="smart",
        max_tokens=400,
    )

    email = EmailMessage()
    email.set_content(body)
    if recipient:
        email["To"] = recipient
    if original_subject:
        email["Subject"] = original_subject if original_subject.lower().startswith("re:") \
            else f"Re: {original_subject}"

    try:
        draft = service.users().drafts().create(
            userId="me",
            body={"message": {"raw": base64.urlsafe_b64encode(email.as_bytes()).decode()}},
        ).execute()
    except Exception as exc:
        return ToolResult.failure(f"could not save the draft: {exc}")

    return ToolResult.success(
        {"draft_id": draft.get("id"), "body": body, "to": recipient},
        # Deliberately explicit that nothing was sent, so neither the model nor
        # the user can mistake a draft for a sent message.
        speak=f"drafted, not sent. It reads: {body[:300]}",
    )


@tool(category="messaging", irreversible=True,
      confirm="Send the draft email to {to}? This actually sends it.", timeout=25.0)
def send_email(draft_id: str, to: str = "") -> ToolResult:
    """Send an email draft that was prepared earlier. Always confirms first.

    Args:
        draft_id: The id of the draft to send.
        to: The recipient, used only for the confirmation prompt.
    """
    service, error = google_service("gmail", "v1")
    if service is None:
        return ToolResult.failure(error)
    try:
        sent = service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
    except Exception as exc:
        return ToolResult.failure(f"could not send the draft: {exc}")
    return ToolResult.success({"message_id": sent.get("id")}, speak="sent")
