"""Conversation state: the follow-up window and interruption memory (spec 5.2, 5.3).

Two behaviours the spec is precise about, and this module keeps precise:

**The follow-up window is not a chat mode.** After Adrien finishes speaking it
keeps listening for a few seconds so the user can refine the task in flight -
"wait, actually say I'll be there by 8" - without saying the wake word again.
It does not prompt, it does not fill silence, and if nothing is said the window
closes and Adrien goes back to passive wake-word listening. `WindowState` makes
that lifecycle explicit rather than leaving it implied by a sleep somewhere in
the main loop.

**An interrupted reply is remembered, not lost.** When the user barges in,
Adrien records what it *intended* to say and how much of it was actually
spoken. If the next thing said is "keep going" or "what were you saying", it
resumes from where it was cut off instead of starting over or drawing a blank.
This lives in short-term state only - it is context for the next few seconds,
not a durable fact about the user.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from adrien.core.llm_types import Message
from adrien.logging_setup import get_logger

log = get_logger(__name__)


class WindowState(str, Enum):
    """Where the conversation is in its lifecycle."""

    IDLE = "idle"                 # passive: only the wake word gets through
    LISTENING = "listening"       # actively recording a request
    THINKING = "thinking"         # LLM and tools running
    SPEAKING = "speaking"         # TTS playing, barge-in armed
    FOLLOW_UP = "follow_up"       # brief window for a refinement, no wake word
    CONFIRMING = "confirming"     # waiting on a yes/no for a destructive tool


# Phrases that mean "carry on with what you were saying". Matched as whole
# phrases rather than keywords so "finish the deploy script" is a new request,
# not a resume.
_CONTINUATION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"^\s*(please\s+)?(keep|carry)\s+(going|on)\s*[.!?]?\s*$",
        r"^\s*(please\s+)?(go|carry)\s+on\s*[.!?]?\s*$",
        r"^\s*(finish|complete)\s+(that|it|what you were saying)\s*[.!?]?\s*$",
        r"^\s*what\s+(were|was)\s+you\s+saying\s*[.!?]?\s*$",
        r"^\s*(sorry|sorry,)?\s*(say|repeat)\s+(that|it)\s+again\s*[.!?]?\s*$",
        r"^\s*continue\s*[.!?]?\s*$",
        r"^\s*you\s+were\s+saying\s*[.!?]?\s*$",
    )
]


def is_continuation_request(text: str) -> bool:
    """True when the user is asking Adrien to resume an interrupted reply."""
    cleaned = (text or "").strip()
    return any(pattern.match(cleaned) for pattern in _CONTINUATION_PATTERNS)


@dataclass
class InterruptedReply:
    """What Adrien was in the middle of saying when it got cut off."""

    full_text: str
    spoken_ratio: float
    at: float = field(default_factory=time.time)

    @property
    def spoken(self) -> str:
        """The part the user actually heard, rounded to a word boundary."""
        if self.spoken_ratio >= 1.0:
            return self.full_text
        words = self.full_text.split()
        if not words:
            return ""
        # Round down: better to repeat a word than to skip one.
        count = int(len(words) * max(0.0, self.spoken_ratio))
        return " ".join(words[:count])

    @property
    def remaining(self) -> str:
        """The part that never came out."""
        spoken_words = len(self.spoken.split())
        return " ".join(self.full_text.split()[spoken_words:])

    @property
    def is_meaningful(self) -> bool:
        """Worth offering to resume? A few cut-off words are not."""
        return len(self.remaining.split()) >= 3

    def resume_text(self) -> str:
        """What to say when asked to continue.

        A word or two of overlap makes the resumption sound like a person
        picking up a sentence rather than a recording restarting.
        """
        remaining = self.remaining
        if not remaining:
            return self.full_text
        spoken_words = self.spoken.split()
        overlap = " ".join(spoken_words[-2:]) if len(spoken_words) >= 2 else ""
        return f"{overlap} {remaining}".strip()


@dataclass
class Conversation:
    """Short-term state for one continuous interaction."""

    history_turns: int = 12
    follow_up_seconds: float = 6.0
    state: WindowState = WindowState.IDLE
    messages: list[Message] = field(default_factory=list)
    interrupted: InterruptedReply | None = None
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0

    # -- history ----------------------------------------------------------
    def add_user(self, text: str) -> None:
        self.messages.append(Message.user(text))
        self.turn_count += 1
        self.touch()

    def add_assistant(self, message: Message) -> None:
        self.messages.append(message)
        self.touch()

    def add_tool_result(self, message: Message) -> None:
        self.messages.append(message)

    def touch(self) -> None:
        self.last_activity = time.time()

    def trim(self) -> None:
        """Keep the history bounded, without orphaning tool results.

        A `tool` message whose originating `assistant` turn has been trimmed
        away is rejected by both providers, so trimming walks back to a clean
        boundary rather than slicing at a fixed index.
        """
        limit = max(2, self.history_turns)
        if len(self.messages) <= limit:
            return
        start = len(self.messages) - limit
        while start < len(self.messages) and self.messages[start].role in ("tool", "assistant"):
            start += 1
        self.messages = self.messages[start:]

    def build_messages(self, system_prompt: str, recalled: str = "") -> list[Message]:
        """The full prompt for this turn: persona, memory, then history."""
        self.trim()
        prompt: list[Message] = [Message.system(system_prompt)]
        if recalled:
            # Recalled memory goes in its own system message so it can be
            # swapped per turn without rebuilding the persona.
            prompt.append(Message.system(recalled))
        prompt.extend(self.messages)
        return prompt

    # -- window lifecycle -------------------------------------------------
    def open_follow_up(self) -> None:
        """Start the short refinement window (spec 5.2)."""
        self.state = WindowState.FOLLOW_UP
        self.touch()

    def follow_up_remaining(self) -> float:
        if self.state is not WindowState.FOLLOW_UP:
            return 0.0
        return max(0.0, self.follow_up_seconds - (time.time() - self.last_activity))

    @property
    def follow_up_open(self) -> bool:
        return self.follow_up_remaining() > 0

    def close_follow_up(self) -> None:
        """Nothing was said - back to passive wake-word listening."""
        self.state = WindowState.IDLE

    # -- interruption -----------------------------------------------------
    def note_interruption(self, full_text: str, spoken_ratio: float) -> InterruptedReply | None:
        """Record what was cut off, so "keep going" can work."""
        if spoken_ratio >= 0.995 or not full_text.strip():
            self.interrupted = None
            return None
        self.interrupted = InterruptedReply(full_text=full_text, spoken_ratio=spoken_ratio)
        log.info("interrupted after %.0f%% of a %d-word reply",
                 spoken_ratio * 100, len(full_text.split()))
        # The history records what was *heard*, not what was intended: the
        # model should not later assume the user knows the rest.
        return self.interrupted

    def take_resume_text(self) -> str | None:
        """Consume the interrupted reply, if there is one worth resuming."""
        pending, self.interrupted = self.interrupted, None
        if pending is None:
            return None
        if not pending.is_meaningful:
            return None
        # Stale interruptions are not resumable - "keep going" ten minutes
        # later means something else entirely.
        if time.time() - pending.at > 120:
            return None
        return pending.resume_text()

    def clear_interruption(self) -> None:
        self.interrupted = None

    # -- session ----------------------------------------------------------
    def reset(self) -> None:
        """Start a fresh interaction, keeping nothing but the settings."""
        self.messages.clear()
        self.interrupted = None
        self.turn_count = 0
        self.state = WindowState.IDLE
        self.touch()

    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    @classmethod
    def from_settings(cls, settings) -> "Conversation":
        return cls(
            history_turns=int(settings.get("conversation.history_turns", 12)),
            follow_up_seconds=float(settings.get("conversation.follow_up_window_seconds", 6.0)),
        )


def summarise_history(messages: Iterable[Message], limit: int = 400) -> str:
    """Compact recent turns into a Whisper decoding hint.

    Feeding Whisper the words that were just used markedly improves how it
    hears proper nouns - "Modrinth", "raidnxt", the names of the user's
    friends - which are exactly the words it otherwise mangles.
    """
    parts: list[str] = []
    for message in list(messages)[-6:]:
        if message.role in ("user", "assistant") and message.content:
            parts.append(message.content)
    hint = " ".join(parts)
    return hint[-limit:] if len(hint) > limit else hint
