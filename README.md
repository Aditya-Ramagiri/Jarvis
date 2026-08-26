# Adrien

A persistent, wake-word-triggered personal voice assistant. The brain runs as a
background service on a MacBook; an Android phone and an iPad connect to it over
the home WiFi as thin clients.

Not a voice command tool — a personal agent with long-term memory, 55 tools it
can chain together, and a conversational flow that survives being interrupted.

```
  ┌──────────────────────── MACBOOK (the brain) ──────────────────────────┐
  │                                                                        │
  │   wake word ──▶ Whisper STT ──▶ LLM router ──┬──▶ tool executor        │
  │  (local, free)   (Groq)        (Groq→Gemini) │    (55 integrations)    │
  │                                               └──▶ long-term memory    │
  │                                                    (Chroma + SQLite)   │
  │                              Fish Audio TTS ◀──────────┘               │
  │                                    │                                   │
  │                          local WebSocket ◀──── Android / iPad          │
  │                                                (same WiFi only)        │
  └────────────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
git clone <this repo> && cd Jarvis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env && $EDITOR .env      # paste your API keys
cp config/settings.example.json config/settings.json

python -m adrien doctor                   # check everything before running
python -m adrien chat                     # try the brain without a microphone
python -m adrien run                      # start the full voice loop
```

Once it works, install it as a login service:

```bash
./service/install_service.sh
```

## What it needs

| | Why | Required? |
|---|---|---|
| **Groq** keys | LLM inference and `whisper-large-v3` speech recognition | yes |
| **Fish Audio** keys | Adrien's voice | yes |
| **Gemini** keys | Fallback when every Groq key is rate limited | recommended |
| GitHub token | The GitHub tools | optional |
| Google OAuth | Calendar and Gmail (`python -m adrien auth-google`) | optional |

Add as many accounts per provider as you have: `GROQ_API_KEY_1`,
`GROQ_API_KEY_2`, … Adrien discovers them by numeric suffix and rotates
through them, so adding a fourth account is one line in `.env`.

Weather, news and web search work with **no keys at all** (Open-Meteo, Google
News RSS, DuckDuckGo). Adding `OPENWEATHER_API_KEY`, `NEWSAPI_KEY` or
`BRAVE_SEARCH_API_KEY` upgrades them.

## Using it

Say **"Adrien"**, wait for the tone, then talk.

> **"Adrien, message John on Discord saying I'm running late"**
> *"Send John: I'm running late. Should I send it?"*
> **"wait, actually say I'll be there by eight"** ← no wake word needed
> *"Send John: I'll be there by eight. Should I send it?"*
> **"yeah, send it"**

Three behaviours worth knowing:

- **The follow-up window.** For a few seconds after Adrien finishes, you can
  keep talking without the wake word — for refining what it is doing. It does
  not chat: if you say nothing, it goes quiet.
- **Interrupting works.** Talk over Adrien and it stops. Say "keep going" and it
  picks up where it was cut off.
- **It asks before anything irreversible.** Sending a message, sending mail,
  shutting the Mac down. Tune this per tool or per category in
  `config/settings.json`; a category-wide "auto" still will not auto-approve a
  shutdown.

## Commands

```bash
python -m adrien run          # start the assistant
python -m adrien chat         # type instead of talking - no mic needed
python -m adrien doctor       # check keys, packages, devices, permissions
python -m adrien status       # provider health and memory stats
python -m adrien tools -v     # every registered tool and its permission mode
python -m adrien memory       # what Adrien remembers
python -m adrien devices      # audio devices, for pinning one in settings
python -m adrien discover     # find Adrien on the LAN
```

## What it can do

55 tools, chained by the model in a single request:

| Category | Examples |
|---|---|
| **dev** | git status/commit/push/pull, run a script, GitHub notifications, open PRs, CI status, explain an error log |
| **gaming** | start/stop/restart the Minecraft server, who's online, search Modrinth, check a mod's latest release, open a launcher |
| **productivity** | reminders, timers, notes, todos, Google Calendar, Gmail (draft-only by default) |
| **system** | volume, open/quit apps, Spotlight search, screenshot + describe what's on screen, clipboard and clipboard history, lock/sleep/restart/shutdown |
| **info** | weather, news, web search with a synthesised answer, time anywhere |
| **messaging** | Discord messages, sending a prepared email draft |
| **extras** | package tracking, bill splitting, meeting transcription, finance tracker |

*"Check my GitHub notifications, summarise the open PRs, then message me the
summary on Discord"* is one request; the model calls three tools in sequence.

## Adding a tool

One function. Nothing else changes — the schema the LLM sees is derived from
the signature and docstring.

```python
# adrien/tools/my_tools.py
from adrien.tools.registry import ToolResult, tool

@tool(category="info")
def check_bin_day(area: str = "home") -> ToolResult:
    """Check which bin goes out this week.

    Args:
        area: Which address to check.
    """
    return ToolResult.success({"bin": "green"}, speak="green bin this week")
```

Register the module in `load_all_tools()` and it is live. Mark anything
irreversible with `irreversible=True` and a `confirm=` prompt, and the
permission layer handles the rest.

## Documentation

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pieces fit, and why they are shaped this way |
| [docs/WAKE_WORD.md](docs/WAKE_WORD.md) | **Read this** — "Adrien" needs a custom model, and there is a fallback until you train one |
| [docs/SETUP.md](docs/SETUP.md) | Full install, macOS permissions, troubleshooting |
| [clients/PROTOCOL.md](clients/PROTOCOL.md) | The client wire protocol |
| [clients/android/README.md](clients/android/README.md) | Android client |
| [clients/ipad/README.md](clients/ipad/README.md) | iPad client, and its platform limits |

## Two things to know up front

**The wake word is not "Adrien" until you train it.** openWakeWord has no
pretrained model for that name. Adrien falls back to "hey Jarvis" and tells you
so; [docs/WAKE_WORD.md](docs/WAKE_WORD.md) is a 30-minute fix on a free Colab GPU.

**Clients are local-network only, by design.** No tunnel, no port forwarding, no
relay. Off the WiFi, the clients say "Adrien unavailable" — that is the intended
behaviour, not a missing feature.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest              # 216 tests, no audio hardware or API keys needed
ruff check adrien tests
```

The whole test suite runs without the audio/ML stack installed: every heavy
dependency is imported lazily by the module that needs it, so rotation, tools,
permissions, memory and the wire protocol are all testable on a bare machine.

## Licence

AGPL-3.0. See [LICENSE](LICENSE).
