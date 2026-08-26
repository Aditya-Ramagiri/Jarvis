"""The confirmation layer (spec section 9)."""

from __future__ import annotations

import pytest

from adrien.config import DEFAULT_SETTINGS, Settings
from adrien.tools.permissions import PermissionManager, interpret_confirmation
from adrien.tools.registry import ToolRegistry, ToolResult

pytestmark = pytest.mark.asyncio


@pytest.fixture
def registry_with_two_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(category="info")
    def harmless(place: str = "here") -> ToolResult:
        """Look something up.

        Args:
            place: Where.
        """
        return ToolResult.success()

    @reg.tool(category="messaging", destructive=True,
              confirm="Send {recipient}: {message}. Send it?")
    def risky(recipient: str, message: str) -> ToolResult:
        """Send a message to someone.

        Args:
            recipient: Who.
            message: What to say.
        """
        return ToolResult.success()

    return reg


def make_manager(confirm_fn=None, **overrides) -> PermissionManager:
    import copy

    data = copy.deepcopy(DEFAULT_SETTINGS)
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return PermissionManager(Settings(data), confirm_fn)


# -- spoken yes / no --------------------------------------------------------
@pytest.mark.parametrize("phrase", [
    "yes", "Yeah", "yep", "sure", "do it", "send it", "confirm", "go ahead",
    "yeah go ahead", "okay", "alright", "please do",
])
def test_affirmatives_are_understood(phrase):
    assert interpret_confirmation(phrase) is True


@pytest.mark.parametrize("phrase", [
    "no", "nope", "don't", "cancel", "wait", "never mind", "stop",
    "no don't do that", "actually no", "hold on",
])
def test_negatives_are_understood(phrase):
    assert interpret_confirmation(phrase) is False


@pytest.mark.parametrize("phrase", ["", "   ", "what time is it", "the weather in Dublin"])
def test_anything_else_is_not_treated_as_consent(phrase):
    # Ambiguity must never mean yes - that is the point of the whole layer.
    assert interpret_confirmation(phrase) is None


def test_a_negation_anywhere_beats_an_affirmative():
    assert interpret_confirmation("yeah actually no don't") is False


# -- policy resolution ------------------------------------------------------
async def test_a_harmless_tool_never_asks(registry_with_two_tools):
    manager = make_manager()
    decision = await manager.check(registry_with_two_tools.get("harmless"), {})
    assert decision.allowed and decision.mode == "auto"


async def test_a_destructive_tool_asks_first(registry_with_two_tools):
    asked: list[str] = []

    async def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return True

    manager = make_manager(confirm)
    decision = await manager.check(
        registry_with_two_tools.get("risky"),
        {"recipient": "John", "message": "running late"},
    )
    assert decision.allowed
    assert asked == ["Send John: running late. Send it?"]


async def test_a_refusal_blocks_the_tool(registry_with_two_tools):
    async def refuse(prompt: str) -> bool:
        return False

    manager = make_manager(refuse)
    decision = await manager.check(
        registry_with_two_tools.get("risky"), {"recipient": "John", "message": "hi"}
    )
    assert not decision.allowed
    assert "did not confirm" in decision.reason


async def test_a_per_tool_auto_override_skips_the_question(registry_with_two_tools):
    async def fail_if_called(prompt: str) -> bool:
        raise AssertionError("should not have asked")

    manager = make_manager(fail_if_called, **{"permissions.tools": {"risky": "auto"}})
    decision = await manager.check(
        registry_with_two_tools.get("risky"), {"recipient": "J", "message": "hi"}
    )
    assert decision.allowed and decision.mode == "auto"


async def test_a_category_set_to_auto_applies_to_its_destructive_tools(registry_with_two_tools):
    manager = make_manager(**{"permissions.categories": {"messaging": "auto"}})
    decision = await manager.check(
        registry_with_two_tools.get("risky"), {"recipient": "J", "message": "hi"}
    )
    assert decision.allowed


async def test_a_per_tool_setting_beats_its_category(registry_with_two_tools):
    manager = make_manager(
        None,
        **{"permissions.categories": {"messaging": "auto"},
           "permissions.tools": {"risky": "deny"}},
    )
    decision = await manager.check(
        registry_with_two_tools.get("risky"), {"recipient": "J", "message": "hi"}
    )
    assert not decision.allowed and decision.mode == "deny"


async def test_a_denied_tool_is_never_run_and_explains_itself(registry_with_two_tools):
    manager = make_manager(**{"permissions.tools": {"risky": "deny"}})
    decision = await manager.check(
        registry_with_two_tools.get("risky"), {"recipient": "J", "message": "hi"}
    )
    assert not decision.allowed
    assert "settings" in PermissionManager.denial_result(decision).error


async def test_with_no_way_to_ask_the_tool_does_not_run(registry_with_two_tools):
    """A headless client must not become a way to skip confirmation."""
    manager = make_manager(confirm_fn=None)
    decision = await manager.check(
        registry_with_two_tools.get("risky"), {"recipient": "J", "message": "hi"}
    )
    assert not decision.allowed
    assert "no way to ask" in decision.reason


def test_the_shipped_settings_never_auto_send_a_message():
    """Regression guard on the defaults themselves."""
    from adrien.tools.registry import load_all_tools

    manager = PermissionManager(Settings(DEFAULT_SETTINGS))
    for name in ("send_discord_message", "send_email", "shutdown_mac", "restart_mac"):
        spec = load_all_tools().get(name)
        assert spec is not None, f"{name} is not registered"
        assert spec.destructive, f"{name} must be marked destructive"
        assert manager.mode_for(spec) != "auto", f"{name} must not default to auto"
