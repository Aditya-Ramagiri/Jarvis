"""Spoken confirmation for actions that are hard to undo (spec section 9).

Resolution order for a tool, most specific first:

1. `permissions.tools.<tool_name>` in `settings.json`
2. `permissions.categories.<category>`
3. `permissions.default`

with one rule layered on top: **a tool that is not marked `destructive` is
always auto**. Confirming "what's the weather" would make Adrien exhausting to
use, and the read-only tools have nothing to undo. Only the tools that send a
message, spend money, write to someone else's inbox or power down the machine
ever reach the confirmation path.

Modes:

* `auto`    - run it
* `confirm` - say what is about to happen, wait for a yes
* `deny`    - never run it, and tell the model why so it can say so

The confirmation itself is injected as a callable, so the same policy works for
a spoken yes/no on the Mac, a tap on the phone client, or an auto-yes in tests.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from adrien.config import Settings
from adrien.config import settings as global_settings
from adrien.logging_setup import get_logger
from adrien.tools.registry import ToolResult, ToolSpec

log = get_logger(__name__)

Mode = str  # "auto" | "confirm" | "deny"

# `confirm_fn(prompt) -> bool`. Returning False means "the user said no or said
# nothing", and either way the tool does not run.
ConfirmFn = Callable[[str], Awaitable[bool]]

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go", "go ahead", "do it",
    "send it", "send", "confirm", "confirmed", "affirmative", "please", "please do",
    "that's right", "correct", "right", "fine", "alright", "aye",
}
_NEGATIVE = {
    "no", "nope", "nah", "don't", "dont", "stop", "cancel", "wait", "never mind",
    "nevermind", "abort", "forget it", "hold on", "negative", "no thanks",
}


# Phrases containing a negative word that mean the opposite. Without these,
# "no problem, go ahead" would be read as a refusal.
_FALSE_NEGATIVES = re.compile(r"\bno (problem|worries|probs|bother)\b")

# Strong negatives: if one of these appears anywhere in the answer, it is a no.
_STRONG_NEGATIVE = {
    "no", "nope", "nah", "don't", "dont", "stop", "cancel", "abort", "wait",
    "negative", "never",
}


def interpret_confirmation(text: str) -> bool | None:
    """Read a spoken yes/no. Returns None when the answer is neither.

    Two rules, both biased towards *not* acting:

    * Ambiguity is never consent. The caller turns None into "did not confirm".
    * A negative anywhere beats an affirmative anywhere. "Yeah, actually no,
      don't" is a refusal, and someone who has to say no twice should not have
      to say it a third time. The cost of being wrong is asymmetric: failing to
      send a message is a small annoyance, sending the wrong one is not.
    """
    cleaned = re.sub(r"[^\w\s']", " ", (text or "").strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    without_false_negatives = _FALSE_NEGATIVES.sub(" ", cleaned).strip()
    words = without_false_negatives.split()

    if any(word in _STRONG_NEGATIVE for word in words):
        return False
    if without_false_negatives in _NEGATIVE:
        return False

    if cleaned in _AFFIRMATIVE or without_false_negatives in _AFFIRMATIVE:
        return True
    # Leading phrases: "sure go ahead", "okay do it".
    for size in (3, 2, 1):
        if " ".join(words[:size]) in _AFFIRMATIVE:
            return True
    if any(word in _AFFIRMATIVE for word in words):
        return True
    return None


@dataclass
class Decision:
    allowed: bool
    mode: Mode
    reason: str = ""
    prompt: str = ""


class PermissionManager:
    """Applies the settings policy and, when needed, asks the user."""

    def __init__(self, settings: Settings | None = None, confirm_fn: ConfirmFn | None = None) -> None:
        self.settings = settings or global_settings()
        self.confirm_fn = confirm_fn

    # -- policy -----------------------------------------------------------
    def mode_for(self, spec: ToolSpec) -> Mode:
        tools = self.settings.get("permissions.tools", {}) or {}
        explicit = tools.get(spec.name)
        if explicit in ("auto", "confirm", "deny"):
            # A per-tool setting is a deliberate statement about *this* tool,
            # so it wins outright - including for irreversible ones.
            return explicit
        if not spec.destructive:
            return "auto"

        categories = self.settings.get("permissions.categories", {}) or {}
        category = categories.get(spec.category)
        default = self.settings.get("permissions.default", "confirm")
        inherited = category if category in ("auto", "confirm", "deny") else (
            default if default in ("auto", "confirm", "deny") else "confirm"
        )

        # A category-wide or global "auto" must not silently authorise
        # shutting the machine down or sending someone a message. Turning
        # "system" to auto is a reasonable thing to want for volume and app
        # control; it is not consent to skip the question on a shutdown.
        # Opting out of *that* takes naming the tool explicitly above.
        if spec.irreversible and inherited == "auto":
            log.debug("%s is irreversible; ignoring inherited 'auto'", spec.name)
            return "confirm"
        return inherited

    def set_mode(self, *, tool: str | None = None, category: str | None = None,
                 mode: Mode = "confirm", persist: bool = True) -> None:
        """Retune the policy at runtime (menu bar, or "stop asking me about X")."""
        if mode not in ("auto", "confirm", "deny"):
            raise ValueError(f"unknown permission mode: {mode}")
        if tool:
            self.settings.set(f"permissions.tools.{tool}", mode)
        elif category:
            self.settings.set(f"permissions.categories.{category}", mode)
        else:
            self.settings.set("permissions.default", mode)
        if persist:
            self.settings.save()

    # -- the gate ---------------------------------------------------------
    async def check(self, spec: ToolSpec, arguments: dict[str, object]) -> Decision:
        mode = self.mode_for(spec)

        if mode == "deny":
            return Decision(
                allowed=False, mode=mode,
                reason=f"{spec.name} is switched off in Adrien's settings.",
            )
        if mode == "auto":
            return Decision(allowed=True, mode=mode)

        prompt = spec.confirmation_prompt(dict(arguments))
        if self.confirm_fn is None:
            # No way to ask - e.g. a headless client with no audio path. Not
            # asking and doing it anyway would be the worst of both.
            log.warning("%s needs confirmation but no confirmation channel exists", spec.name)
            return Decision(
                allowed=False, mode=mode, prompt=prompt,
                reason=f"{spec.name} needs a spoken confirmation and there is no way to ask right now.",
            )

        log.info("asking for confirmation: %s", spec.name)
        confirmed = await self.confirm_fn(prompt)
        if confirmed:
            return Decision(allowed=True, mode=mode, prompt=prompt)
        return Decision(
            allowed=False, mode=mode, prompt=prompt,
            reason=f"The user did not confirm, so {spec.name} was not run.",
        )

    @staticmethod
    def denial_result(decision: Decision) -> ToolResult:
        """Turn a refusal into a tool result the model can speak about."""
        return ToolResult.failure(decision.reason or "not permitted")
