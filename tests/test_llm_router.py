"""Provider fallback, tiering and the all-keys-down path (spec section 4)."""

from __future__ import annotations

import pytest

from adrien.config import DEFAULT_SETTINGS, Settings
from adrien.core.keypool import KeyPool
from adrien.core.llm_router import LLMRouter, ProviderSlot
from adrien.core.llm_types import AllProvidersFailed, ChatResult, Message, ProviderError

pytestmark = pytest.mark.asyncio


class ScriptedProvider:
    """Replays a scripted list of outcomes, recording the keys it was given."""

    def __init__(self, name: str, outcomes: list[object]) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    def model_for(self, tier: str) -> str:
        return f"{self.name}-{tier}"

    async def chat(self, *, api_key, model, messages, tools=None, **kwargs) -> ChatResult:
        self.calls.append((api_key, model))
        outcome = self.outcomes.pop(0) if self.outcomes else ChatResult(text="ok")
        if isinstance(outcome, Exception):
            raise outcome
        outcome.provider = self.name
        outcome.model = model
        return outcome


def build_router(*providers_and_keys, clock=None) -> LLMRouter:
    settings = Settings(dict(DEFAULT_SETTINGS))
    slots = [
        ProviderSlot(
            provider=provider,
            pool=KeyPool(provider.name, keys, clock=clock) if clock
            else KeyPool(provider.name, keys),
        )
        for provider, keys in providers_and_keys
    ]
    return LLMRouter(settings=settings, slots=slots)


def rate_limited(provider="groq"):
    return ProviderError("429", provider=provider, rate_limited=True)


def transient(provider="groq"):
    return ProviderError("503", provider=provider, retryable=True)


def bad_request(provider="groq"):
    return ProviderError("400 bad schema", provider=provider, retryable=False)


async def test_first_key_answers():
    provider = ScriptedProvider("groq", [ChatResult(text="hello")])
    router = build_router((provider, ["a", "b", "c"]))
    result = await router.chat([Message.user("hi")])
    assert result.text == "hello"
    assert result.key_label == "groq#1"
    assert len(provider.calls) == 1


async def test_rate_limit_rotates_to_the_next_key_transparently():
    provider = ScriptedProvider("groq", [rate_limited(), rate_limited(), ChatResult(text="third")])
    router = build_router((provider, ["a", "b", "c"]))
    result = await router.chat([Message.user("hi")])
    assert result.text == "third"
    assert result.key_label == "groq#3"
    # Each attempt used a *different* key, never the cooling one again.
    assert [key for key, _ in provider.calls] == ["a", "b", "c"]


async def test_falls_back_to_gemini_when_every_groq_key_is_limited():
    groq = ScriptedProvider("groq", [rate_limited(), rate_limited()])
    gemini = ScriptedProvider("gemini", [ChatResult(text="from gemini")])
    router = build_router((groq, ["a", "b"]), (gemini, ["g1", "g2"]))

    result = await router.chat([Message.user("hi")])
    assert result.text == "from gemini"
    assert result.provider == "gemini"
    assert len(groq.calls) == 2


async def test_transient_failures_also_rotate():
    provider = ScriptedProvider("groq", [transient(), ChatResult(text="recovered")])
    router = build_router((provider, ["a", "b"]))
    assert (await router.chat([Message.user("hi")])).text == "recovered"


async def test_bad_request_is_not_retried_across_keys():
    provider = ScriptedProvider("groq", [bad_request(), ChatResult(text="never reached")])
    router = build_router((provider, ["a", "b", "c"]))
    with pytest.raises(ProviderError):
        await router.chat([Message.user("hi")])
    assert len(provider.calls) == 1, "a malformed request must not burn the pool"


async def test_all_providers_failing_raises_the_one_user_visible_error():
    groq = ScriptedProvider("groq", [rate_limited(), rate_limited()])
    gemini = ScriptedProvider("gemini", [rate_limited("gemini")])
    router = build_router((groq, ["a", "b"]), (gemini, ["g1"]))

    with pytest.raises(AllProvidersFailed) as excinfo:
        await router.chat([Message.user("hi")])
    assert len(excinfo.value.attempts) == 3


async def test_attempt_budget_is_capped():
    provider = ScriptedProvider("groq", [rate_limited()] * 10)
    router = build_router((provider, list("abcdefghij")))
    router.settings.set("keys.max_attempts_per_call", 3)
    with pytest.raises(AllProvidersFailed):
        await router.chat([Message.user("hi")])
    assert len(provider.calls) == 3


async def test_short_command_uses_the_fast_model():
    provider = ScriptedProvider("groq", [ChatResult(text="ok")])
    router = build_router((provider, ["a"]))
    await router.chat([Message.user("set a timer for ten minutes")])
    assert provider.calls[0][1] == "groq-fast"


async def test_reasoning_keyword_forces_the_smart_model():
    provider = ScriptedProvider("groq", [ChatResult(text="ok")])
    router = build_router((provider, ["a"]))
    await router.chat([Message.user("explain that")])
    assert provider.calls[0][1] == "groq-smart"


async def test_mid_tool_chain_stays_on_the_smart_model():
    provider = ScriptedProvider("groq", [ChatResult(text="ok")])
    router = build_router((provider, ["a"]))
    await router.chat([
        Message.user("weather"),
        Message(role="tool", content="18C", tool_call_id="1", name="get_weather"),
    ])
    assert provider.calls[0][1] == "groq-smart"


async def test_explicit_tier_overrides_the_heuristic():
    provider = ScriptedProvider("groq", [ChatResult(text="ok")])
    router = build_router((provider, ["a"]))
    await router.chat([Message.user("explain everything about this")], tier="fast")
    assert provider.calls[0][1] == "groq-fast"


async def test_status_reports_health_without_leaking_keys():
    provider = ScriptedProvider("groq", [])
    router = build_router((provider, ["gsk_secret_value"]))
    status = router.status()
    assert status["healthy"] is True
    assert "gsk_secret_value" not in repr(status)


# -- diagnostics ------------------------------------------------------------
# Regression: with every key in cooldown the router made zero attempts, and
# the empty attempt list rendered as "no keys configured" - sending whoever is
# debugging to look for a missing .env when the keys were fine all along.
async def test_all_keys_cooling_is_not_reported_as_missing_keys(clock):
    provider = ScriptedProvider("groq", [rate_limited(), rate_limited()])
    router = build_router((provider, ["a", "b"]), clock=clock)

    with pytest.raises(AllProvidersFailed):
        await router.chat([Message.user("hi")])

    # Second call: nothing is available, so no attempt is made at all.
    with pytest.raises(AllProvidersFailed) as excinfo:
        await router.chat([Message.user("hi again")])

    message = str(excinfo.value)
    assert "cooling down" in message
    assert "frees up in" in message
    assert "no API keys are configured" not in message


async def test_genuinely_missing_keys_say_so():
    router = LLMRouter(settings=Settings(dict(DEFAULT_SETTINGS)), slots=[])
    with pytest.raises(AllProvidersFailed) as excinfo:
        await router.chat([Message.user("hi")])
    assert "no providers configured" in str(excinfo.value)


async def test_a_pool_with_no_keys_reports_missing_configuration(clock):
    provider = ScriptedProvider("groq", [])
    router = build_router((provider, []), clock=clock)
    with pytest.raises(AllProvidersFailed) as excinfo:
        await router.chat([Message.user("hi")])
    assert "no API keys are configured" in str(excinfo.value)
