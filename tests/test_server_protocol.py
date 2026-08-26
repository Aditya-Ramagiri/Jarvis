"""The client protocol and the server's guards (spec section 8)."""

from __future__ import annotations

import copy

import pytest

from adrien.config import DEFAULT_SETTINGS, Settings
from adrien.server.protocol import (
    CLIENT_SAMPLE_RATE,
    MAX_UTTERANCE_BYTES,
    SERVER_SAMPLE_RATE,
    MessageType,
    decode,
    encode,
    error,
    welcome,
)


# -- framing ----------------------------------------------------------------
def test_a_frame_round_trips():
    frame = decode(encode(MessageType.REPLY, text="hello", tools=["get_weather"]))
    assert frame is not None
    assert frame.type == MessageType.REPLY
    assert frame.get("text") == "hello"
    assert frame.get("tools") == ["get_weather"]


def test_malformed_frames_are_rejected_without_raising():
    """A client sending nonsense must not take down its connection handler."""
    for raw in ("not json", "", "[]", '"a string"', "null", '{"no": "type"}',
                '{"type": ""}', '{"type": 42}'):
        assert decode(raw) is None


def test_missing_fields_come_back_as_defaults():
    frame = decode(encode(MessageType.TEXT))
    assert frame.get("text", "fallback") == "fallback"


def test_unicode_survives_the_round_trip():
    frame = decode(encode(MessageType.REPLY, text="it's 18° and raining 🌧"))
    assert frame.get("text") == "it's 18° and raining 🌧"


def test_the_welcome_frame_announces_both_sample_rates():
    """Clients read the rates from here rather than hardcoding them."""
    frame = decode(welcome())
    assert frame.get("client_sample_rate") == CLIENT_SAMPLE_RATE == 16_000
    assert frame.get("server_sample_rate") == SERVER_SAMPLE_RATE == 24_000
    assert frame.get("protocol") == 1


def test_a_fatal_error_says_so():
    frame = decode(error("bad token", fatal=True))
    assert frame.get("reason") == "bad token"
    assert frame.get("fatal") is True


def test_the_utterance_cap_is_a_minute_of_audio():
    assert MAX_UTTERANCE_BYTES == CLIENT_SAMPLE_RATE * 2 * 60


# -- server guards ----------------------------------------------------------
def build_server(monkeypatch, host: str = "0.0.0.0"):
    from adrien.server.ws_server import AdrienServer

    monkeypatch.setenv("ADRIEN_WS_HOST", host)
    monkeypatch.setenv("ADRIEN_WS_PORT", "8765")
    monkeypatch.setenv("ADRIEN_WS_TOKEN", "test-token")

    class StubOrchestrator:
        def status(self):
            return {"running": True}

    return AdrienServer(StubOrchestrator(), Settings(copy.deepcopy(DEFAULT_SETTINGS)))


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1", "192.168.1.10", "10.0.0.5", "::"])
def test_local_addresses_are_allowed(monkeypatch, host):
    build_server(monkeypatch, host)._assert_local_only()


# 203.0.113.0/24 and friends are documentation ranges, which Python correctly
# classes as private - a genuinely routable address is what must be refused.
@pytest.mark.parametrize("host", ["8.8.8.8", "93.184.216.34"])
def test_binding_to_a_public_address_is_refused(monkeypatch, host):
    """Spec 8: local network only, and loudly so rather than by accident."""
    server = build_server(monkeypatch, host)
    with pytest.raises(RuntimeError, match="local-network only"):
        server._assert_local_only()


def test_a_nonsense_bind_address_is_refused(monkeypatch):
    server = build_server(monkeypatch, "not-an-address")
    with pytest.raises(RuntimeError, match="not an IP address"):
        server._assert_local_only()


def test_the_server_reads_its_token_from_the_environment(monkeypatch):
    assert build_server(monkeypatch).token == "test-token"


# -- session state ----------------------------------------------------------
def test_a_session_starts_unauthenticated():
    from adrien.server.ws_server import ClientSession

    session = ClientSession()
    assert session.authenticated is False
    assert session.receiving_audio is False
    assert session.label == "unknown"


def test_a_session_label_includes_the_platform():
    from adrien.server.ws_server import ClientSession

    session = ClientSession(device="Pixel 8", platform="android")
    assert session.label == "Pixel 8/android"
