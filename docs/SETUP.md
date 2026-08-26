# Setup

Full install, the macOS permissions Adrien needs, and what to do when
something does not work.

## 1. Install

```bash
git clone <this repo> && cd Jarvis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python **3.11 or newer**. On Apple Silicon everything installs from wheels; if
`chromadb` fails to build, see [Troubleshooting](#troubleshooting) — Adrien
still runs without it.

## 2. Configure

```bash
cp .env.example .env
cp config/settings.example.json config/settings.json
```

Fill in `.env`. Only three things are actually required:

```bash
GROQ_API_KEY_1=...          # LLM + Whisper.  console.groq.com
FISH_AUDIO_API_KEY_1=...    # voice.          fish.audio
FISH_AUDIO_VOICE_ID=...     # which voice
```

Add every account you have — `GROQ_API_KEY_2`, `_3`, … Adrien finds them by
numeric suffix and rotates through them. Gemini keys
(`GEMINI_API_KEY_1`, …) are the fallback for when every Groq key is rate
limited; strongly recommended, since without them a Groq outage is silence.

Generate the client token while you are in there:

```bash
echo "ADRIEN_WS_TOKEN=$(openssl rand -hex 24)" >> .env
```

## 3. Check before running

```bash
python -m adrien doctor
```

Every dependency, key, audio device and macOS permission, with `✓`, `!`
(optional, degraded) or `✗` (broken). Fix the `✗` lines first.

## 4. Try it without a microphone

```bash
python -m adrien chat
```

The full brain — memory, tools, permissions — typed instead of spoken. The
fastest way to tell whether a problem is in the assistant or in the audio path.

## 5. Run it

```bash
python -m adrien run
```

Say the wake word, wait for the tone, talk. **Note that the wake word is not
"Adrien" until you train it** — see [WAKE_WORD.md](WAKE_WORD.md). Until then it
is "hey Jarvis", and Adrien says so at startup.

## 6. Install as a service

```bash
./service/install_service.sh
```

Installs a **LaunchAgent** (`~/Library/LaunchAgents/com.raidnxt.adrien.plist`),
not a LaunchDaemon: Adrien needs the user's audio session and GUI session, which
a system daemon does not have. It starts at login, has no dock icon, and
restarts if it crashes.

```bash
launchctl kickstart -k gui/$(id -u)/com.raidnxt.adrien   # restart
tail -f logs/adrien.err.log                              # watch it
./service/uninstall_service.sh                           # remove
```

For a menu bar status item instead of a bare service, run `python -m
adrien.menubar` (needs `rumps`, macOS only).

## macOS permissions

macOS will prompt on first use. **Grant them to the Python binary the service
runs, not to Terminal** — otherwise the service is denied when launchd starts
it, even though it worked when you ran it by hand. The installer prints the
exact path.

| Permission | Needed for | Without it |
|---|---|---|
| **Microphone** | everything | Adrien hears nothing |
| **Accessibility** | Discord, app control | keystrokes silently do nothing |
| **Automation** → System Events | volume, apps, power | those tools fail |
| **Automation** → Discord | messaging | Discord tool fails |
| **Full Disk Access** *(optional)* | Spotlight search across all folders | search misses some locations |

All under **System Settings → Privacy & Security**.

## Google Calendar and Gmail

OAuth consent needs a browser, so it is a one-off command rather than something
the background service attempts:

```bash
python -m adrien auth-google
```

Put `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET` in `.env` first
(Google Cloud Console → APIs & Services → Credentials → OAuth client ID →
*Desktop app*). Enable the Calendar and Gmail APIs for the project. The token is
stored at `~/Library/Application Support/Adrien/google_token.json` and refreshed
automatically after that.

## Minecraft server control

Adrien does not guess how your server starts. Tell it:

```bash
MINECRAFT_HOST=rhs.raidnxt.com
MINECRAFT_START_CMD=/Users/you/minecraft/start.sh
MINECRAFT_STOP_CMD=screen -S mc -X stuff "stop\n"
MINECRAFT_RESTART_CMD=/Users/you/minecraft/restart.sh
```

`check_minecraft_players_online` works with just `MINECRAFT_HOST`.

## Connecting the phone and iPad

1. Make sure `ADRIEN_WS_TOKEN` is set and the service is running.
2. Build the client (`clients/android` or `clients/ipad` — each has a README).
3. Enter the token once on the pairing screen.
4. The client finds the Mac over mDNS. No IP address to type.

Check the server is reachable from the Mac itself first:

```bash
python clients/python_client.py --discover
python clients/python_client.py --text "what's the weather"
```

That reference client is a complete implementation of the protocol, so if it
works and the phone does not, the problem is on the phone.

## Tuning

`config/settings.json`, reloadable from the menu bar:

```json
"conversation": {
  "follow_up_window_seconds": 6.0,      // longer = more time to refine
  "endpoint_silence_seconds": 1.0,      // longer = fewer cut-off sentences
  "barge_in_speech_frames": 6           // higher = harder to interrupt
},
"permissions": {
  "categories": { "messaging": "confirm" },
  "tools": { "send_discord_message": "auto" }   // stop asking about one tool
}
```

Permission modes are `auto`, `confirm`, `deny`. A category-wide `auto` still
will not auto-approve an irreversible tool — that takes naming the tool.

## Troubleshooting

**Adrien does not respond to the wake word**
Check `python -m adrien doctor` for the microphone. Then watch detections live:
`python -m adrien --log-level DEBUG run` logs a score for every near-miss. Lower
`wake_word.threshold` if they are close but under.

**It triggers on the TV**
Raise `wake_word.threshold` toward 0.7, or train "hey Adrien" instead of
"Adrien" — the extra syllable is a large accuracy win.

**It hears itself and interrupts itself**
Raise `conversation.barge_in_speech_frames`, or use headphones. There is no
echo cancellation on the laptop's mic; the guards are described in
[ARCHITECTURE.md](ARCHITECTURE.md#conversation-flow-spec-52-53).

**"I'm having trouble connecting right now"**
Every key of every provider is unusable. `python -m adrien status` shows which
are cooling down and why. Usually rate limits — add another account, or wait.

**Discord messages do nothing**
Accessibility permission, on the *service's* Python binary. `python -m adrien
doctor` reports it.

**`chromadb` will not install**
Adrien falls back to keyword-based memory and logs a warning. Recall gets
noticeably worse but nothing breaks. On Apple Silicon, `pip install --upgrade
pip setuptools wheel` first usually fixes the build.

**The service will not start**
`tail -50 logs/adrien.err.log`. Most often: the plist points at a Python
without the dependencies (rerun the installer from inside the venv), or `.env`
is missing.

**Clients cannot find it**
Same WiFi? Some routers block mDNS between wireless and wired clients, and
guest networks almost always do. `python -m adrien discover` on the Mac proves
whether the advertisement is up. On iPad, missing
`NSLocalNetworkUsageDescription` fails the browse *silently* — check that
first.
