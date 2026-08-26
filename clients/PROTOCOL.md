# Adrien client protocol

One WebSocket, two frame kinds. Deliberately generic: nothing below assumes a
phone, a tablet, or a screen, so a watch or TV client can speak it later
without the protocol changing (spec section 2).

- **Text frames** are JSON control messages, always `{"type": "...", ...}`.
- **Binary frames** are raw PCM audio. No envelope, no base64 — a base64 layer
  would inflate every utterance by a third and buy nothing.

| Direction | Format | Sample rate | Notes |
|---|---|---|---|
| client → server | 16-bit signed PCM, mono, little-endian | **16000 Hz** | matches Whisper's native rate |
| server → client | 16-bit signed PCM, mono, little-endian | **24000 Hz** | matches Fish Audio output |

Both rates are announced in the `welcome` frame; read them from there rather
than hardcoding, so a future change does not break older clients.

## Connecting

Adrien advertises itself over mDNS as `_adrien._tcp.local.`. **Browse for it —
do not ask the user to type an IP.** The TXT record carries `assistant`,
`protocol` and `path`; refuse to connect if `protocol` is higher than the one
your client implements.

Connect to `ws://<host>:<port>/`, then immediately send:

```json
{"type": "hello", "token": "<ADRIEN_WS_TOKEN>", "device": "Pixel 8", "platform": "android"}
```

The token is the `ADRIEN_WS_TOKEN` value from the Mac's `.env`. Everything
except `hello` is rejected until it is accepted. The server replies:

```json
{"type": "welcome", "assistant": "Adrien", "protocol": 1,
 "client_sample_rate": 16000, "server_sample_rate": 24000}
```

A bad token gets `{"type": "error", "reason": "bad token", "fatal": true}` and
the socket closes. Do not retry in a tight loop — surface it to the user, since
it means the token is actually wrong.

## A voice turn

```
client → {"type": "audio_start", "want_audio": true}
client → <binary PCM, 16 kHz, any chunk size up to 1 MiB>
client → {"type": "audio_end"}

server → {"type": "state", "state": "thinking"}
server → {"type": "reply", "text": "It's 18 degrees.", "tools": ["get_weather"]}
server → {"type": "audio_start", "sample_rate": 24000, "bytes": 96000}
server → <binary PCM chunks>
server → {"type": "audio_end"}
server → {"type": "state", "state": "idle"}
```

Start playing on the first binary frame rather than waiting for `audio_end` —
that is the whole reason audio is chunked.

Set `"want_audio": false` for a text-only client (a watch complication, a
Shortcut); the server then skips synthesis entirely, which is faster.

## A text turn

```
client → {"type": "text", "text": "what's on my calendar", "want_audio": false}
server → {"type": "reply", "text": "Two things today ...", "tools": ["check_calendar"]}
```

## Confirmations

When a turn hits a tool that needs confirmation (sending a message, shutting
the Mac down), the server asks **the client that started the turn**:

```
server → {"type": "confirm", "prompt": "Send John: running late. Should I send it?"}
client → {"type": "text", "text": "yes"}
```

Your next `text` frame is consumed as the answer, not as a new request. Show
the prompt and offer an explicit yes/no — do not auto-answer, and do not answer
on the user's behalf if they walk away. No answer within 45 seconds counts as
"no" and the tool does not run.

## Other frames

| Frame | Direction | Purpose |
|---|---|---|
| `{"type": "ping"}` | client → | Liveness. Answered with `pong`. |
| `{"type": "cancel"}` | client → | Stop playback and discard the in-flight utterance. |
| `{"type": "status"}` | client → | Answered with `status_report`: provider health, tool count, memory stats. |
| `{"type": "transcript", "text": ...}` | server → | What Adrien heard, when the server chooses to echo it. |
| `{"type": "state", "state": ...}` | server → | `idle`, `listening`, `thinking`, `speaking`, `confirming`. Drive your UI from this. |
| `{"type": "error", "reason": ..., "fatal": bool}` | server → | `fatal` means the socket is closing. |

## Limits

- Frames are capped at 1 MiB.
- One utterance is capped at 60 seconds of audio (about 1.9 MB). Past that the
  server discards the buffer and sends an error.
- At most 8 clients at once, configurable in `config/settings.json`.
- The server pings every 20 seconds and drops a client that misses two.

## Availability

There is no remote access, by design (spec section 8). No tunnel, no port
forwarding, no relay. When the Mac is off, asleep, or on another network,
**say "Adrien unavailable"** and keep browsing for the service — do not attempt
a workaround, and do not fall back to a cloud endpoint.

## Reference implementation

`clients/python_client.py` is a complete working client in about 200 lines —
discovery, handshake, audio streaming, playback. Read it first; it is the
shortest description of the protocol that actually runs.

```bash
python clients/python_client.py --text "what's the weather"
python clients/python_client.py --listen        # push-to-talk from a terminal
```
