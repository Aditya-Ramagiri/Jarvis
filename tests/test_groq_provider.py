"""The Groq request and response path, against a mocked transport.

Groq is the primary provider - it serves both chat and Whisper - and until now
it had no tests at all. The Gemini adapter was covered because its dialect
translation is obviously risky; Groq's was not, on the grounds that it is
"just OpenAI-shaped". That reasoning is wrong: the status classification in
`_raise_for_status` is what decides whether a failure rotates a key, burns the
whole pool, or aborts the turn, and it is exercised on every single request.

These tests drive the real `GroqProvider` and `Transcriber` through an
`httpx.MockTransport`, so the actual request that would go on the wire is
asserted - URL, auth header, body shape, multipart file - along with how each
response class is interpreted.
"""

from __future__ import annotations

import json

import httpx
import pytest

from adrien.core import http as http_module
from adrien.core.llm_types import Message, ProviderError, ToolCall
from adrien.core.providers.groq import GroqProvider

pytestmark = pytest.mark.asyncio


# The modules under test do `from adrien.core.http import get_client` at import
# time, so the name they call is bound in *their* namespace. Patching
# `adrien.core.http.get_client` would miss it entirely - and the request would
# go out to the real network, which is exactly the failure this fixture exists
# to prevent.
_PATCH_TARGETS = ("adrien.core.providers.groq", "adrien.core.stt")


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Fail loudly rather than reaching the network if a test forgets to
    install a transport."""
    def refuse(timeout=None):
        raise AssertionError("this test made a real network call")

    for target in _PATCH_TARGETS:
        monkeypatch.setattr(f"{target}.get_client", refuse, raising=False)
    yield
    http_module._client = None
    http_module._client_loop = None


def install_transport(handler, monkeypatch=None) -> list[httpx.Request]:
    """Point the modules under test at `handler`, recording every request."""
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    for target in _PATCH_TARGETS:
        module = __import__(target, fromlist=["get_client"])
        module.get_client = lambda timeout=None, _c=client: _c
    return seen


def models_response(ids: list[str]) -> httpx.Response:
    """What GET /openai/v1/models returns."""
    return httpx.Response(200, json={"data": [{"id": name} for name in ids]})


def json_response(payload: dict, status: int = 200, headers: dict | None = None):
    return lambda request: httpx.Response(status, json=payload, headers=headers or {})


CHAT_OK = {
    "choices": [{
        "message": {"role": "assistant", "content": "It's 18 degrees in Dublin."},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 100, "completion_tokens": 8},
}


# -- the outgoing request ---------------------------------------------------
async def test_the_request_is_shaped_the_way_groq_expects():
    seen = install_transport(json_response(CHAT_OK))

    await GroqProvider().chat(
        api_key="gsk_testkey",
        model="llama-3.1-8b-instant",
        messages=[Message.system("You are Adrien."), Message.user("weather?")],
        temperature=0.4,
        max_tokens=200,
    )

    request = seen[0]
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer gsk_testkey"

    body = json.loads(request.content)
    assert body["model"] == "llama-3.1-8b-instant"
    assert body["temperature"] == 0.4
    assert body["max_tokens"] == 200
    assert body["messages"] == [
        {"role": "system", "content": "You are Adrien."},
        {"role": "user", "content": "weather?"},
    ]
    # No tools were passed, so neither key should appear at all.
    assert "tools" not in body and "tool_choice" not in body


async def test_tools_are_sent_with_auto_choice():
    seen = install_transport(json_response(CHAT_OK))
    schema = {"type": "function", "function": {"name": "get_weather",
                                               "description": "Weather.",
                                               "parameters": {"type": "object", "properties": {}}}}

    await GroqProvider().chat(api_key="k", model="m",
                              messages=[Message.user("weather?")], tools=[schema])

    body = json.loads(seen[0].content)
    assert body["tools"] == [schema]
    assert body["tool_choice"] == "auto"


async def test_an_assistant_tool_call_serialises_with_null_content():
    """OpenAI-shaped APIs reject "" alongside tool_calls; it must be null."""
    seen = install_transport(json_response(CHAT_OK))
    call = ToolCall(name="mute", arguments={}, id="call_abc")

    await GroqProvider().chat(
        api_key="k", model="m",
        messages=[
            Message.user("mute it"),
            Message.assistant(tool_calls=[call]),
            Message.tool_result(call, '{"ok": true}'),
        ],
    )

    messages = json.loads(seen[0].content)["messages"]
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["function"]["name"] == "mute"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert messages[2] == {"role": "tool", "content": '{"ok": true}',
                           "tool_call_id": "call_abc", "name": "mute"}


# -- the incoming response --------------------------------------------------
async def test_a_plain_reply_is_parsed():
    install_transport(json_response(CHAT_OK))
    result = await GroqProvider().chat(api_key="k", model="llama-3.1-8b-instant",
                                       messages=[Message.user("weather?")])

    assert result.text == "It's 18 degrees in Dublin."
    assert result.tool_calls == []
    assert result.provider == "groq"
    assert result.finish_reason == "stop"
    assert result.usage["completion_tokens"] == 8
    assert result.latency_ms >= 0


async def test_tool_calls_are_parsed_with_their_arguments():
    install_transport(json_response({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": '{"location": "Dublin"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }))

    result = await GroqProvider().chat(api_key="k", model="m",
                                       messages=[Message.user("weather?")])
    assert result.wants_tools
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"location": "Dublin"}
    assert result.tool_calls[0].id == "call_xyz"


async def test_unparseable_tool_arguments_do_not_drop_the_call():
    """Models occasionally emit not-quite-JSON; the tool layer should get a
    chance to report a clean validation error instead of the call vanishing."""
    install_transport(json_response({
        "choices": [{"message": {"tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "set_timer", "arguments": "{not json"},
        }]}, "finish_reason": "tool_calls"}],
    }))

    result = await GroqProvider().chat(api_key="k", model="m",
                                       messages=[Message.user("timer")])
    assert result.tool_calls[0].name == "set_timer"
    assert result.tool_calls[0].arguments == {"_raw": "{not json"}


async def test_a_nameless_tool_call_is_discarded():
    install_transport(json_response({
        "choices": [{"message": {"tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "", "arguments": "{}"}}
        ]}, "finish_reason": "tool_calls"}],
    }))
    result = await GroqProvider().chat(api_key="k", model="m",
                                       messages=[Message.user("hi")])
    assert result.tool_calls == []


async def test_a_response_with_no_choices_is_retryable():
    install_transport(json_response({"choices": []}))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.retryable


async def test_non_json_body_is_retryable():
    install_transport(lambda request: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.retryable


# -- status classification --------------------------------------------------
# This is the part that decides whether a failure rotates a key or aborts the
# turn, so each class is pinned explicitly.
async def test_429_is_a_rate_limit_and_carries_retry_after():
    install_transport(json_response(
        {"error": {"message": "rate limit reached"}}, 429, {"retry-after": "7.5"}
    ))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])

    error = excinfo.value
    assert error.rate_limited and error.retryable
    assert error.retry_after == 7.5


async def test_a_rate_limit_without_a_header_still_rotates():
    install_transport(json_response({"error": {"message": "slow down"}}, 429))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.rate_limited
    assert excinfo.value.retry_after is None  # the pool's default applies


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_key_rotates_rather_than_aborting(status):
    """One dead key in the pool must not take the whole turn down."""
    install_transport(json_response({"error": {"message": "invalid api key"}}, status))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.retryable and not excinfo.value.rate_limited


@pytest.mark.parametrize("status", [500, 502, 503, 408])
async def test_server_errors_are_retryable(status):
    install_transport(json_response({"error": {"message": "upstream"}}, status))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.retryable


@pytest.mark.parametrize("status", [400, 404, 422])
async def test_request_errors_are_not_retried_across_keys(status):
    """A malformed request fails identically on every key, so trying the rest
    of the pool only wastes the user's time."""
    install_transport(json_response({"error": {"message": "unknown model"}}, status))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert not excinfo.value.retryable
    assert "unknown model" in str(excinfo.value)


async def test_an_error_body_that_is_not_json_still_classifies():
    install_transport(lambda request: httpx.Response(503, text="upstream timeout"))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.retryable


async def test_a_timeout_is_retryable():
    def timeout(request):
        raise httpx.ReadTimeout("too slow", request=request)

    install_transport(timeout)
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.retryable
    assert "timed out" in str(excinfo.value)


async def test_a_transport_error_is_retryable():
    def boom(request):
        raise httpx.ConnectError("connection refused", request=request)

    install_transport(boom)
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="k", model="m", messages=[Message.user("hi")])
    assert excinfo.value.retryable


async def test_the_key_never_appears_in_an_error_message():
    install_transport(json_response({"error": {"message": "nope"}}, 401))
    with pytest.raises(ProviderError) as excinfo:
        await GroqProvider().chat(api_key="gsk_verysecretvalue", model="m",
                                  messages=[Message.user("hi")])
    assert "gsk_verysecretvalue" not in str(excinfo.value)


# -- model tiering ----------------------------------------------------------
def test_the_tier_selects_the_configured_model():
    provider = GroqProvider(fast_model="fast-model", smart_model="smart-model")
    assert provider.model_for("fast") == "fast-model"
    assert provider.model_for("smart") == "smart-model"
    # Anything unrecognised falls back to fast rather than raising mid-turn.
    assert provider.model_for("nonsense") == "fast-model"


# --------------------------------------------------------------------------
# Whisper / STT
# --------------------------------------------------------------------------
# The other half of the Groq surface. Its failure handling is deliberately
# different from chat's: a voice loop that raises on a failed transcription
# would speak an error at the user for what is usually just silence, so this
# path returns an empty result and lets the orchestrator go back to listening.
def pcm(seconds: float = 1.0, rate: int = 16_000) -> bytes:
    return b"\x10\x27" * int(rate * seconds)


async def test_the_upload_is_multipart_with_the_model_and_a_wav():
    from adrien.core.stt import Transcriber

    seen = install_transport(json_response({"text": "what's the weather"}))
    transcriber = Transcriber()
    transcriber.pool = _pool_with(["gsk_stt_key"])

    result = await transcriber.transcribe(pcm(), prompt="Modrinth raidnxt")

    request = seen[0]
    assert str(request.url) == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer gsk_stt_key"
    body = request.content
    assert b'name="model"' in body and b"whisper-large-v3" in body
    assert b'filename="utterance.wav"' in body
    assert b"RIFF" in body and b"WAVE" in body   # a real container, not raw PCM
    # The decoding hint markedly improves proper nouns, so it must be sent.
    assert b"Modrinth raidnxt" in body
    assert result.text == "what's the weather"
    assert result.key_label == "groq-stt#1"


async def test_audio_below_the_floor_is_never_uploaded():
    """Whisper hallucinates confidently on near-silence; skipping saves both
    the round trip and a confident reply to nothing."""
    from adrien.core.stt import Transcriber

    seen = install_transport(json_response({"text": "Thank you."}))
    transcriber = Transcriber()
    transcriber.pool = _pool_with(["k"])

    result = await transcriber.transcribe(pcm(0.1))
    assert result.is_empty
    assert seen == []


async def test_a_known_silence_artefact_is_discarded():
    from adrien.core.stt import Transcriber

    install_transport(json_response({"text": "Thanks for watching!"}))
    transcriber = Transcriber()
    transcriber.pool = _pool_with(["k"])

    assert (await transcriber.transcribe(pcm())).is_empty


async def test_a_rate_limited_stt_key_rotates_to_the_next():
    from adrien.core.stt import Transcriber

    responses = [
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json={"text": "second key worked"}),
    ]
    seen = install_transport(lambda request: responses.pop(0))
    transcriber = Transcriber()
    transcriber.pool = _pool_with(["k1", "k2"])

    result = await transcriber.transcribe(pcm())
    assert result.text == "second key worked"
    assert len(seen) == 2
    assert transcriber.pool.available_count() == 1   # the limited key is cooling


async def test_stt_returns_empty_rather_than_raising_when_every_key_fails():
    from adrien.core.stt import Transcriber

    install_transport(lambda request: httpx.Response(500, text="upstream"))
    transcriber = Transcriber()
    transcriber.pool = _pool_with(["k1", "k2"])

    result = await transcriber.transcribe(pcm())
    assert result.is_empty and result.duration_s == pytest.approx(1.0)


async def test_a_bad_stt_request_does_not_burn_the_pool():
    """400 means the request was wrong, not the key - the key stays usable."""
    from adrien.core.stt import Transcriber

    seen = install_transport(json_response({"error": {"message": "bad audio"}}, 400))
    transcriber = Transcriber()
    transcriber.pool = _pool_with(["k1", "k2"])

    assert (await transcriber.transcribe(pcm())).is_empty
    assert len(seen) == 1, "a malformed request must not be retried on every key"
    assert transcriber.pool.available_count() == 2


async def test_stt_survives_an_unreadable_body():
    from adrien.core.stt import Transcriber

    install_transport(lambda request: httpx.Response(200, text="not json"))
    transcriber = Transcriber()
    transcriber.pool = _pool_with(["k"])

    assert (await transcriber.transcribe(pcm())).is_empty


async def test_stt_with_no_keys_configured_is_quiet():
    from adrien.core.stt import Transcriber

    seen = install_transport(json_response({"text": "unreachable"}))
    transcriber = Transcriber()
    transcriber.pool = _pool_with([])

    assert (await transcriber.transcribe(pcm())).is_empty
    assert seen == []


def _pool_with(keys: list[str]):
    from adrien.core.keypool import KeyPool

    return KeyPool("groq-stt", keys)


# -- model selection --------------------------------------------------------
# Regression: with every preferred model retired, the fallback sorted the
# remaining ids alphabetically and picked `allam-2-7b` - an Arabic-specialised
# model that cannot do tool calling at all. Every turn then died on
# "`tool calling` is not supported with this model". Alphabetical order is not
# a capability ranking.
def test_ranking_prefers_capable_families_over_alphabetical():
    from adrien.core.providers.groq import rank_candidate

    offered = ["allam-2-7b", "gemma2-9b-it", "llama-3.3-70b-versatile", "qwen-2.5-32b"]
    assert sorted(offered, key=rank_candidate)[0] == "llama-3.3-70b-versatile"
    # The model that caused the outage must never sort first again.
    assert sorted(offered, key=rank_candidate)[-1] == "allam-2-7b"


def test_ranking_prefers_bigger_within_a_family():
    from adrien.core.providers.groq import rank_candidate

    offered = ["llama-3.3-8b-versatile", "llama-3.3-70b-versatile"]
    assert sorted(offered, key=rank_candidate)[0] == "llama-3.3-70b-versatile"


async def test_a_model_without_tool_support_is_struck_off_and_replaced():
    from adrien.core.providers.groq import GroqProvider

    provider = GroqProvider()
    attempts: list[str] = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return models_response(["allam-2-7b", "llama-3.3-70b-versatile"])
        model = json.loads(request.content)["model"]
        attempts.append(model)
        if model == "allam-2-7b":
            return httpx.Response(400, json={
                "error": {"message": "`tool calling` is not supported with this model"}
            })
        return httpx.Response(200, json=CHAT_OK)

    install_transport(handler)
    result = await provider.chat(
        api_key="k", model="allam-2-7b", messages=[Message.user("hi")],
        tools=[{"type": "function", "function": {"name": "x", "description": "y",
                                                 "parameters": {"type": "object",
                                                                "properties": {}}}}],
    )

    assert result.text == "It's 18 degrees in Dublin."
    assert attempts == ["allam-2-7b", "llama-3.3-70b-versatile"]
    assert "allam-2-7b" in provider._no_tools


async def test_a_retired_model_is_replaced():
    from adrien.core.providers.groq import GroqProvider

    provider = GroqProvider()
    attempts: list[str] = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return models_response(["llama-3.3-70b-versatile"])
        model = json.loads(request.content)["model"]
        attempts.append(model)
        if model == "llama-3.1-8b-instant":
            return httpx.Response(404, json={
                "error": {"message": "The model `llama-3.1-8b-instant` does not exist"}
            })
        return httpx.Response(200, json=CHAT_OK)

    install_transport(handler)
    result = await provider.chat(
        api_key="k", model="llama-3.1-8b-instant", messages=[Message.user("hi")]
    )
    assert result.text == "It's 18 degrees in Dublin."
    assert attempts == ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]


async def test_non_chat_models_are_never_selected():
    from adrien.core.providers.groq import GroqProvider

    provider = GroqProvider()
    provider._available = {"whisper-large-v3", "llama-guard-4-12b", "llama-3.3-70b-versatile"}
    assert await provider.resolve_model("fast", "k") == "llama-3.3-70b-versatile"
