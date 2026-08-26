"""macOS system control (spec 7.5).

Implemented through the AppleScript bridge (`osascript`) and a few native CLIs
(`mdfind`, `pbcopy`, `screencapture`) rather than a UI automation library,
because these are all things macOS exposes properly - reaching for simulated
keystrokes when there is a real API is how automation becomes fragile. UI
automation is reserved for Discord, where there is no acceptable API path
(see `discord_automation.py`).

Power tools (shutdown, restart, sleep) are marked destructive and go through
the confirmation layer.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from adrien.config import data_dir, env_key_pool
from adrien.logging_setup import get_logger
from adrien.tools._shell import applescript_quote, osascript, require_macos, run
from adrien.tools.registry import ToolResult, tool

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------
@tool(category="system")
def set_volume(level: int) -> ToolResult:
    """Set the Mac's output volume.

    Args:
        level: Volume from 0 to 100.
    """
    if (error := require_macos("set_volume")):
        return ToolResult.failure(error)
    level = max(0, min(100, int(level)))
    result = osascript(f"set volume output volume {level}")
    if not result.ok:
        return ToolResult.failure(f"could not set the volume: {result.output}")
    return ToolResult.success({"level": level}, speak=f"volume {level}")


@tool(category="system")
def get_volume() -> ToolResult:
    """Check the Mac's current output volume and whether it is muted."""
    if (error := require_macos("get_volume")):
        return ToolResult.failure(error)
    result = osascript(
        'set v to output volume of (get volume settings)\n'
        'set m to output muted of (get volume settings)\n'
        'return (v as text) & "," & (m as text)'
    )
    if not result.ok:
        return ToolResult.failure(f"could not read the volume: {result.output}")
    level, _, muted = result.stdout.strip().partition(",")
    return ToolResult.success({"level": int(level or 0), "muted": muted.strip() == "true"})


@tool(category="system")
def mute() -> ToolResult:
    """Mute the Mac's audio output."""
    if (error := require_macos("mute")):
        return ToolResult.failure(error)
    result = osascript("set volume with output muted")
    return (ToolResult.success({"muted": True}, speak="muted") if result.ok
            else ToolResult.failure(result.output))


@tool(category="system")
def unmute() -> ToolResult:
    """Unmute the Mac's audio output."""
    if (error := require_macos("unmute")):
        return ToolResult.failure(error)
    result = osascript("set volume without output muted")
    return (ToolResult.success({"muted": False}, speak="unmuted") if result.ok
            else ToolResult.failure(result.output))


# --------------------------------------------------------------------------
# Applications
# --------------------------------------------------------------------------
@tool(category="system")
def open_app(name: str) -> ToolResult:
    """Open or focus an application.

    Args:
        name: The application's name, e.g. Safari or Visual Studio Code.
    """
    if (error := require_macos("open_app")):
        return ToolResult.failure(error)
    result = run(["open", "-a", name])
    if not result.ok:
        return ToolResult.failure(f"could not open {name}: {result.output.strip()[:160]}")
    return ToolResult.success({"app": name}, speak=f"opened {name}")


@tool(category="system", destructive=True, confirm="Quit {name}?")
def close_app(name: str, force: bool = False) -> ToolResult:
    """Quit an application.

    Args:
        name: The application's name.
        force: Force quit without letting it save. Use only if asked.
    """
    if (error := require_macos("close_app")):
        return ToolResult.failure(error)
    safe = applescript_quote(name)
    script = (
        f'tell application "System Events" to set pids to unix id of '
        f'(every process whose name is "{safe}")'
    ) if force else f'tell application "{safe}" to quit'

    result = osascript(script)
    if force and result.ok:
        for pid in result.stdout.replace(",", " ").split():
            run(["kill", "-9", pid.strip()])
    if not result.ok:
        return ToolResult.failure(f"could not quit {name}: {result.output[:160]}")
    return ToolResult.success({"app": name}, speak=f"quit {name}")


@tool(category="system")
def list_running_apps() -> ToolResult:
    """List the applications currently open, with the frontmost one first."""
    if (error := require_macos("list_running_apps")):
        return ToolResult.failure(error)
    result = osascript(
        'tell application "System Events" to get name of every process whose background only is false'
    )
    if not result.ok:
        return ToolResult.failure(result.output)
    apps = [item.strip() for item in result.stdout.split(",") if item.strip()]
    front = osascript(
        'tell application "System Events" to get name of first process whose frontmost is true'
    )
    return ToolResult.success({"apps": apps, "frontmost": front.stdout.strip()})


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
@tool(category="system", timeout=20.0)
def search_files(query: str, limit: int = 10, folder: str = "") -> ToolResult:
    """Find files on the Mac by name or content, using Spotlight.

    Args:
        query: What to look for.
        limit: How many results to return at most.
        folder: Restrict the search to this folder.
    """
    if (error := require_macos("search_files")):
        return ToolResult.failure(error)

    command = ["mdfind"]
    if folder:
        command += ["-onlyin", str(Path(folder).expanduser())]
    command.append(query)

    result = run(command, timeout=18)
    if not result.ok:
        return ToolResult.failure(f"Spotlight search failed: {result.output[:160]}")

    paths = [line for line in result.stdout.splitlines() if line.strip()][:limit]
    if not paths:
        return ToolResult.success({"results": []}, speak=f"nothing matching {query}")
    return ToolResult.success(
        {"count": len(paths), "results": [{"name": Path(p).name, "path": p} for p in paths]},
        speak=f"{len(paths)} match{'es' if len(paths) != 1 else ''}, top one is {Path(paths[0]).name}",
    )


# --------------------------------------------------------------------------
# Screenshot (+ optional vision description)
# --------------------------------------------------------------------------
@tool(category="system", timeout=45.0)
async def take_screenshot(describe: bool = True, save_to: str = "") -> ToolResult:
    """Take a screenshot of the screen, and optionally describe what is on it.

    Args:
        describe: Also have Adrien look at the image and describe it out loud.
        save_to: Where to save the file. Defaults to a temporary file.
    """
    if (error := require_macos("take_screenshot")):
        return ToolResult.failure(error)

    target = Path(save_to).expanduser() if save_to else Path(
        tempfile.gettempdir()) / f"adrien-screen-{int(time.time())}.png"
    # -x suppresses the shutter sound, which would otherwise be picked up by
    # the mic and read as barge-in.
    result = run(["screencapture", "-x", str(target)], timeout=15)
    if not result.ok or not target.exists():
        return ToolResult.failure(f"could not take the screenshot: {result.output[:160]}")

    payload: dict[str, Any] = {"path": str(target), "bytes": target.stat().st_size}
    if not describe:
        return ToolResult.success(payload, speak="screenshot taken")

    description = await _describe_image(target)
    if description.startswith("__error__"):
        return ToolResult.success(
            payload, speak=f"screenshot saved, but I could not look at it: {description[9:]}"
        )
    payload["description"] = description
    return ToolResult.success(payload, speak=description)


# Groq's vision-capable model. Kept local to this tool rather than pushed into
# the core message types: one tool needing images is not a reason to make every
# message in the system carry an optional image list.
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


async def _describe_image(path: Path, prompt: str = "") -> str:
    from adrien.core.http import get_client
    from adrien.core.keypool import KeyPool

    keys = env_key_pool("GROQ_API_KEY")
    if not keys:
        return "__error__no Groq key is configured"

    encoded = base64.b64encode(path.read_bytes()).decode()
    body = {
        "model": VISION_MODEL,
        "max_tokens": 220,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt or
                 "Describe what is on this screen in two sentences, spoken aloud. "
                 "Mention the app in focus and anything that looks like an error."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }],
    }

    pool = KeyPool("groq-vision", keys)
    client = get_client()
    for lease in pool.leases():
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=body,
                headers={"authorization": f"Bearer {lease.key}"},
                timeout=40,
            )
        except Exception as exc:
            lease.failed()
            log.warning("vision call failed on %s: %s", lease.label, type(exc).__name__)
            continue
        if response.status_code == 429:
            lease.rate_limited()
            continue
        if response.status_code >= 400:
            lease.success()
            return f"__error__the vision model returned {response.status_code}"
        lease.success()
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError):
            return "__error__the vision model returned something unreadable"
    return "__error__every Groq key is rate limited"


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------
@tool(category="system", destructive=True, confirm="Lock the screen?")
def lock_screen() -> ToolResult:
    """Lock the Mac's screen."""
    if (error := require_macos("lock_screen")):
        return ToolResult.failure(error)
    result = run(["pmset", "displaysleepnow"])
    return (ToolResult.success({"locked": True}, speak="locking up") if result.ok
            else ToolResult.failure(result.output))


@tool(category="system", irreversible=True, confirm="Put the Mac to sleep?")
def sleep_mac() -> ToolResult:
    """Put the Mac to sleep."""
    if (error := require_macos("sleep_mac")):
        return ToolResult.failure(error)
    result = osascript('tell application "System Events" to sleep')
    return (ToolResult.success({"sleeping": True}, speak="going to sleep") if result.ok
            else ToolResult.failure(result.output))


@tool(category="system", irreversible=True,
      confirm="Restart the Mac? Anything unsaved will be lost, and Adrien will go down with it.")
def restart_mac() -> ToolResult:
    """Restart the Mac. Requires confirmation."""
    if (error := require_macos("restart_mac")):
        return ToolResult.failure(error)
    result = osascript('tell application "System Events" to restart')
    return (ToolResult.success({"restarting": True}, speak="restarting now") if result.ok
            else ToolResult.failure(result.output))


@tool(category="system", irreversible=True,
      confirm="Shut the Mac down? Anything unsaved will be lost, and Adrien will go down with it.")
def shutdown_mac() -> ToolResult:
    """Shut the Mac down. Requires confirmation."""
    if (error := require_macos("shutdown_mac")):
        return ToolResult.failure(error)
    result = osascript('tell application "System Events" to shut down')
    return (ToolResult.success({"shutting_down": True}, speak="shutting down") if result.ok
            else ToolResult.failure(result.output))


# --------------------------------------------------------------------------
# Clipboard
# --------------------------------------------------------------------------
class ClipboardHistory:
    """A polling clipboard recorder.

    macOS has no clipboard-change notification available to a plain CLI
    process, so history means polling `pbpaste`. 1.5 s is frequent enough to
    catch a copy the user is about to ask about and infrequent enough to be
    invisible in Activity Monitor.

    Entries live in the data directory. Nothing here is sent anywhere, and the
    same redaction that guards the logs is applied before writing, so a copied
    API key does not end up on disk in plain text.
    """

    def __init__(self, maxlen: int = 50, interval: float = 1.5) -> None:
        self.entries: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.interval = interval
        self.path = data_dir() / "clipboard_history.json"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                for entry in json.loads(self.path.read_text(encoding="utf-8")):
                    self.entries.append(entry)
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(list(self.entries)), encoding="utf-8")
        except OSError:  # pragma: no cover
            pass

    def record(self, text: str) -> None:
        from adrien.logging_setup import redact

        if not text.strip():
            return
        if self.entries and self.entries[-1]["text"] == text:
            return
        self.entries.append({"text": redact(text)[:4000], "at": time.time()})
        self._save()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            current = read_clipboard()
            if current is not None:
                self.record(current)

    def start(self) -> ClipboardHistory:
        if shutil.which("pbpaste") is None:
            log.info("clipboard history disabled: pbpaste not available")
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True, name="clipboard")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


_history: ClipboardHistory | None = None


def clipboard_history() -> ClipboardHistory:
    global _history
    if _history is None:
        _history = ClipboardHistory()
    return _history


def read_clipboard() -> str | None:
    try:
        completed = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


@tool(category="system")
def clipboard_get() -> ToolResult:
    """Read what is currently on the clipboard."""
    if (error := require_macos("clipboard_get")):
        return ToolResult.failure(error)
    text = read_clipboard()
    if text is None:
        return ToolResult.failure("could not read the clipboard")
    if not text.strip():
        return ToolResult.success({"text": ""}, speak="the clipboard is empty")
    clipboard_history().record(text)
    return ToolResult.success({"text": text[:4000], "length": len(text)})


@tool(category="system")
def clipboard_set(text: str) -> ToolResult:
    """Put text on the clipboard.

    Args:
        text: The text to copy.
    """
    if (error := require_macos("clipboard_set")):
        return ToolResult.failure(error)
    try:
        process = subprocess.run(["pbcopy"], input=text, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolResult.failure(f"could not write to the clipboard: {exc}")
    if process.returncode != 0:
        return ToolResult.failure("pbcopy refused the text")
    return ToolResult.success({"length": len(text)}, speak="copied")


@tool(category="system")
def clipboard_history_recall(n: int = 1) -> ToolResult:
    """Recall something copied earlier.

    Args:
        n: How far back to go. 1 is the most recent entry before the current one.
    """
    entries = list(clipboard_history().entries)
    if not entries:
        return ToolResult.failure(
            "no clipboard history yet - Adrien starts recording when the service starts"
        )
    index = max(1, int(n))
    if index > len(entries):
        return ToolResult.failure(f"only {len(entries)} entries are remembered")
    entry = entries[-index]
    return ToolResult.success(
        {"text": entry["text"], "copied_at": entry["at"], "position": index},
        speak=entry["text"][:200],
    )
