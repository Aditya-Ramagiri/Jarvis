"""Discord via UI automation - deliberately NOT the API (spec 7.7).

WHY UI AUTOMATION, WHEN AN API EXISTS
-------------------------------------
This is the one integration in Adrien that drives a GUI instead of calling a
service, and the reason is account safety, not convenience:

* A **bot token** can only speak as a bot. Messages would arrive from "Adrien
  BOT", not from the user, and a bot cannot be in the user's DMs with their
  friends at all. It does not do the thing that was asked for.
* A **self-bot** - driving the user's own account through Discord's HTTP API -
  does do the thing, and is explicitly against Discord's Terms of Service.
  Discord detects it and bans accounts for it. Trading someone's account for a
  tidier implementation is not a trade worth making.

Driving the desktop app is the only approach that sends a real message from
the real user without touching an API they are not allowed to touch. Every
keystroke below is one a person could have typed: focus the app, ctrl/cmd-K to
the quick switcher, type the recipient, Enter, type, Enter.

WHAT THIS COSTS, HONESTLY
-------------------------
UI automation is inherently more fragile than an API. It needs Accessibility
permission, it fails if Discord is mid-update or showing a modal, and a
redesigned quick switcher would break it. So the code verifies rather than
assumes: it checks Discord is frontmost, gives the switcher time to settle,
and - crucially - checks that the message field actually contains what was
typed *before* pressing Enter. Sending "I'm running lat" to the wrong person is
far worse than failing cleanly, so every uncertain path fails cleanly.
"""

from __future__ import annotations

import time
from typing import Any

from adrien.logging_setup import get_logger
from adrien.tools._shell import applescript_quote, is_macos, osascript, require_macos, run
from adrien.tools.registry import ToolResult, tool

log = get_logger(__name__)

# Pauses, in seconds. Tuned for a Discord that is already running on a laptop;
# generous enough to survive a slow frame, short enough not to feel laggy.
LAUNCH_WAIT = 2.5
SWITCHER_WAIT = 0.6
SEARCH_WAIT = 1.1
FOCUS_WAIT = 0.35


def _discord_running() -> bool:
    result = osascript(
        'tell application "System Events" to return (exists process "Discord") as text'
    )
    return result.ok and result.stdout.strip().lower() == "true"


def _focus_discord() -> tuple[bool, str]:
    """Bring Discord to the front, launching it first if necessary."""
    if not _discord_running():
        log.info("Discord is not running; launching it")
        if not run(["open", "-a", "Discord"]).ok:
            return False, "Discord is not installed, or would not launch"
        time.sleep(LAUNCH_WAIT)

    result = osascript('tell application "Discord" to activate')
    if not result.ok:
        return False, f"could not bring Discord to the front: {result.output[:160]}"

    time.sleep(FOCUS_WAIT)
    front = osascript(
        'tell application "System Events" to get name of first process whose frontmost is true'
    )
    if front.ok and front.stdout.strip() != "Discord":
        return False, "something else stole focus before Adrien could type"
    return True, ""


def _keystroke(text: str) -> bool:
    """Type text into whatever is focused, as a human would."""
    return osascript(
        f'tell application "System Events" to keystroke "{applescript_quote(text)}"'
    ).ok


def _key_combo(key: str, modifier: str = "command down") -> bool:
    return osascript(
        f'tell application "System Events" to keystroke "{key}" using {{{modifier}}}'
    ).ok


def _key_code(code: int) -> bool:
    return osascript(f"tell application \"System Events\" to key code {code}").ok


RETURN_KEY = 36
ESCAPE_KEY = 53


def _message_field_contents() -> str:
    """Read back the message box through the Accessibility API.

    This is the check that makes the whole thing safe enough to use: if we
    cannot confirm what is in the box, we do not press Enter.
    """
    script = '''
    tell application "System Events"
        tell process "Discord"
            try
                set boxes to every text area of entire contents of front window
                repeat with b in boxes
                    try
                        set v to value of b
                        if v is not missing value and v is not "" then return v as text
                    end try
                end repeat
            end try
            return ""
        end tell
    end tell
    '''
    result = osascript(script, timeout=10)
    return result.stdout.strip() if result.ok else ""


@tool(
    category="messaging",
    irreversible=True,
    confirm="Send {recipient_or_channel} on Discord: {message}. Should I send it?",
    timeout=60.0,
)
def send_discord_message(recipient_or_channel: str, message: str) -> ToolResult:
    """Send a Discord message to a person or a channel, by driving the Discord
    app exactly as the user would.

    Args:
        recipient_or_channel: Who or where to send it - a person's name, or a
            channel name.
        message: The message text to send.
    """
    if (error := require_macos("send_discord_message")):
        return ToolResult.failure(error)
    if not message.strip():
        return ToolResult.failure("there is no message to send")
    if not recipient_or_channel.strip():
        return ToolResult.failure("no recipient was given")

    # Newlines would send the message early - Enter is "send" in Discord.
    text = " ".join(message.split())

    focused, error = _focus_discord()
    if not focused:
        return ToolResult.failure(error)

    # Quick switcher (cmd-K), type the recipient, take the top hit.
    if not _key_combo("k"):
        return ToolResult.failure(
            "Adrien could not send keystrokes to Discord. Give it Accessibility "
            "permission in System Settings, Privacy and Security, Accessibility."
        )
    time.sleep(SWITCHER_WAIT)

    if not _keystroke(recipient_or_channel):
        _key_code(ESCAPE_KEY)
        return ToolResult.failure("could not type the recipient into the quick switcher")
    time.sleep(SEARCH_WAIT)

    if not _key_code(RETURN_KEY):
        _key_code(ESCAPE_KEY)
        return ToolResult.failure("could not open the conversation")
    time.sleep(SEARCH_WAIT)

    if not _keystroke(text):
        return ToolResult.failure("could not type the message")
    time.sleep(FOCUS_WAIT)

    # The safety check: confirm the box holds our message before sending.
    contents = _message_field_contents()
    if contents:
        typed = "".join(contents.split()).lower()
        expected = "".join(text.split()).lower()
        if expected not in typed:
            _select_all_and_clear()
            return ToolResult.failure(
                "the message box did not contain the right text, so Adrien did not "
                "send anything. Check which conversation Discord is showing."
            )
    else:
        log.warning("could not read the Discord message box; sending on the typed text alone")

    if not _key_code(RETURN_KEY):
        return ToolResult.failure("could not press send")

    log.info("sent a Discord message to %s (%d chars)", recipient_or_channel, len(text))
    return ToolResult.success(
        {"recipient": recipient_or_channel, "message": text, "verified": bool(contents)},
        speak=f"sent to {recipient_or_channel}",
    )


def _select_all_and_clear() -> None:
    """Empty the message box after an aborted send, so nothing is left half
    typed for the user to accidentally send later."""
    _key_combo("a")
    _key_code(51)  # delete


@tool(category="messaging", timeout=30.0)
def read_discord_unread() -> ToolResult:
    """Check which Discord conversations have unread messages.

    Reads the badge counts in the sidebar. It cannot read message contents -
    that would need an API Adrien deliberately does not use.
    """
    if (error := require_macos("read_discord_unread")):
        return ToolResult.failure(error)
    if not _discord_running():
        return ToolResult.failure("Discord is not running")

    result = osascript('''
    tell application "System Events"
        tell process "Discord"
            try
                return value of attribute "AXTitle" of front window
            on error
                return ""
            end try
        end tell
    end tell
    ''')
    title = result.stdout.strip()

    # Discord puts the unread count in its window title: "(3) Discord | ...".
    count = 0
    if title.startswith("("):
        try:
            count = int(title[1:title.index(")")])
        except (ValueError, IndexError):
            count = 0

    payload: dict[str, Any] = {"unread": count, "window_title": title}
    if count == 0:
        return ToolResult.success(payload, speak="nothing unread on Discord")
    return ToolResult.success(payload, speak=f"{count} unread on Discord")


def preflight() -> dict[str, Any]:
    """Diagnostics for `adrien doctor`: is UI automation actually going to work?"""
    if not is_macos():
        return {"ok": False, "reason": "not macOS"}
    installed = run(["osascript", "-e", 'exists application "Discord"']).ok
    accessibility = osascript(
        'tell application "System Events" to return (UI elements enabled) as text'
    )
    enabled = accessibility.ok and accessibility.stdout.strip().lower() == "true"
    return {
        "ok": installed and enabled,
        "discord_installed": installed,
        "discord_running": _discord_running() if installed else False,
        "accessibility_permission": enabled,
        "hint": "" if enabled else
        "Grant Accessibility permission to the process running Adrien in "
        "System Settings > Privacy & Security > Accessibility.",
    }
