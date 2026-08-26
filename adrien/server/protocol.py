"""The client<->brain wire protocol.

Deliberately generic (spec section 2): a future watch or TV client should be
able to speak this without the protocol changing. Nothing in it assumes
Android, iPadOS, or even a screen.

Frames come in two flavours over one WebSocket:

* **Text frames** are JSON control messages: `{"type": ..., ...}`.
* **Binary frames** are raw 16 kHz mono 16-bit PCM audio, in either
  direction. Audio has no JSON envelope because base64 would inflate every
  utterance by a third for no benefit, and a binary frame is unambiguous.

A turn looks like:

    client -> {"type": "hello", "token": ..., "device": ...}
    server -> {"type": "welcome", "assistant": "Adrien", "sample_rate": 16000}

    client -> {"type": "audio_start"}
    client -> <binary PCM frames...>
    client -> {"type": "audio_end"}
    server -> {"type": "transcript", "text": "what's the weather"}
    server -> {"type": "reply", "text": "It's 18 degrees."}
    server -> {"type": "audio_start", "sample_rate": 24000}
    server -> <binary PCM frames...>
    server -> {"type": "audio_end"}

Clients that only want text (a watch complication, a shortcut) send
`{"type": "text", "text": ...}` and set `"want_audio": false`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1

# The mDNS service type clients browse for (spec 8: discovery, not typed IPs).
MDNS_SERVICE_TYPE = "_adrien._tcp.local."

# Audio formats, fixed on both sides so no negotiation is needed.
CLIENT_SAMPLE_RATE = 16_000   # what clients must send: matches Whisper
SERVER_SAMPLE_RATE = 24_000   # what the server sends back: matches Fish Audio


class MessageType:
    # client -> server
    HELLO = "hello"
    AUDIO_START = "audio_start"
    AUDIO_END = "audio_end"
    TEXT = "text"
    CANCEL = "cancel"
    PING = "ping"
    STATUS = "status"

    # server -> client
    WELCOME = "welcome"
    TRANSCRIPT = "transcript"
    REPLY = "reply"
    STATE = "state"
    CONFIRM = "confirm"        # server asks; client answers with `text`
    ERROR = "error"
    PONG = "pong"
    STATUS_REPORT = "status_report"


@dataclass
class Frame:
    """One decoded control frame."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def encode(message_type: str, **fields: Any) -> str:
    """Build a control frame."""
    return json.dumps({"type": message_type, **fields}, ensure_ascii=False)


def decode(raw: str) -> Frame | None:
    """Parse a control frame. Returns None for anything malformed.

    Never raises: a client sending nonsense should get an error frame back,
    not take a connection handler down with it.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    message_type = payload.get("type")
    if not isinstance(message_type, str) or not message_type:
        return None
    return Frame(type=message_type, data={k: v for k, v in payload.items() if k != "type"})


def welcome(assistant: str = "Adrien", **extra: Any) -> str:
    return encode(
        MessageType.WELCOME,
        assistant=assistant,
        protocol=PROTOCOL_VERSION,
        client_sample_rate=CLIENT_SAMPLE_RATE,
        server_sample_rate=SERVER_SAMPLE_RATE,
        **extra,
    )


def error(reason: str, *, fatal: bool = False) -> str:
    return encode(MessageType.ERROR, reason=reason, fatal=fatal)


# Hard cap on one utterance: 16 kHz * 2 bytes * 60 s. A client that keeps
# streaming past this is broken or hostile, and either way the server should
# not accumulate it in memory.
MAX_UTTERANCE_BYTES = CLIENT_SAMPLE_RATE * 2 * 60
