"""Shared subprocess and AppleScript helpers for the tool modules."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from adrien.logging_setup import get_logger, redact

log = get_logger(__name__)


@dataclass
class CommandResult:
    ok: bool
    stdout: str
    stderr: str
    code: int

    @property
    def output(self) -> str:
        return (self.stdout or self.stderr).strip()


def run(
    command: list[str] | str,
    *,
    cwd: str | Path | None = None,
    timeout: float = 30.0,
    shell: bool = False,
) -> CommandResult:
    """Run a command and capture its output.

    `shell=False` by default: arguments arrive from an LLM, and a shell would
    turn a stray semicolon in a commit message into a second command. The few
    places that genuinely need a shell (user-configured server start scripts)
    pass `shell=True` explicitly and are documented where they do.
    """
    if isinstance(command, str) and not shell:
        command = shlex.split(command)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
            check=False,
        )
    except FileNotFoundError:
        name = command[0] if isinstance(command, list) else str(command).split()[0]
        return CommandResult(False, "", f"{name} is not installed", 127)
    except subprocess.TimeoutExpired:
        return CommandResult(False, "", f"timed out after {timeout:.0f}s", 124)
    except OSError as exc:
        return CommandResult(False, "", str(exc), 1)

    return CommandResult(
        ok=completed.returncode == 0,
        stdout=redact(completed.stdout or ""),
        stderr=redact(completed.stderr or ""),
        code=completed.returncode,
    )


def osascript(script: str, *, timeout: float = 15.0) -> CommandResult:
    """Run AppleScript. The macOS bridge for app control (spec section 3)."""
    return run(["osascript", "-e", script], timeout=timeout)


def applescript_quote(value: str) -> str:
    """Escape a string for safe interpolation into an AppleScript literal.

    AppleScript has no parameter binding, so text from the LLM has to be
    escaped by hand before it lands inside quotes - otherwise a message
    containing a double quote ends the literal and the rest is executed.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def is_macos() -> bool:
    import platform

    return platform.system() == "Darwin"


def require_macos(tool_name: str) -> str | None:
    """Error string when a macOS-only tool is called elsewhere, else None."""
    if is_macos():
        return None
    return f"{tool_name} only works on macOS, and Adrien is running on another platform"
