"""Configuration: environment secrets, user settings and on-disk paths.

Two distinct sources, deliberately kept apart:

* **`.env`** - secrets only (API keys, tokens). Loaded via `python-dotenv`,
  never written back to disk, never logged, never stored in memory records.
* **`config/settings.json`** - user-tunable behaviour (timings, permission
  policy, model names). Safe to read, safe to show the user, safe to commit a
  redacted example of.

Key pools are discovered by numeric suffix rather than hardcoded, because the
number of accounts the user holds changes over time: `GROQ_API_KEY_1`,
`GROQ_API_KEY_2`, ... are read until a gap is found. Adding a fourth Groq
account means adding one line to `.env`, not editing code.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Highest suffix we will probe when discovering a key pool. Generous, because
# probing is a dict lookup and the user may add accounts at any time.
_MAX_POOL_PROBE = 32


# --------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------
def load_env(dotenv_path: Path | None = None, *, override: bool = False) -> None:
    """Load `.env` into `os.environ`. Safe to call more than once."""
    path = dotenv_path or (PROJECT_ROOT / ".env")
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard runtime dep
        # Minimal fallback so the process still starts if python-dotenv is
        # missing; the real parser handles quoting and export statements.
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                name, value = name.strip(), value.strip().strip("'\"")
                if override or name not in os.environ:
                    os.environ[name] = value
        return
    load_dotenv(path, override=override)


def env_key_pool(prefix: str, environ: dict[str, str] | None = None) -> list[str]:
    """Collect `PREFIX_1`, `PREFIX_2`, ... from the environment.

    Stops at the first missing index so a commented-out key never silently
    hides the ones after it. Blank values and obvious placeholders are skipped
    but do *not* stop the scan, so a half-filled `.env` still yields the keys
    that are present.
    """
    src = environ if environ is not None else os.environ
    keys: list[str] = []
    for index in range(1, _MAX_POOL_PROBE + 1):
        name = f"{prefix}_{index}"
        if name not in src:
            break
        value = (src.get(name) or "").strip()
        if value and not value.startswith("<"):
            keys.append(value)
    # Also accept an unsuffixed single key (`GROQ_API_KEY=...`), which is what
    # people naturally write when they only have one account.
    solo = (src.get(prefix) or "").strip()
    if solo and solo not in keys:
        keys.append(solo)
    return keys


def env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name) or default)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def data_dir() -> Path:
    """Writable directory for the vector store, SQLite DB, notes and logs."""
    override = env_str("ADRIEN_DATA_DIR")
    if override:
        path = Path(override).expanduser()
    elif platform.system() == "Darwin":
        path = Path.home() / "Library" / "Application Support" / "Adrien"
    else:
        path = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        ) / "adrien"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chroma_dir() -> Path:
    path = data_dir() / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sqlite_path() -> Path:
    return data_dir() / "adrien.sqlite3"


# --------------------------------------------------------------------------
# settings.json
# --------------------------------------------------------------------------
DEFAULT_SETTINGS: dict[str, Any] = {
    "assistant": {
        "name": "Adrien",
        "persona": (
            "You are Adrien, the user's personal voice assistant. You speak out "
            "loud, so keep answers short, natural and conversational - usually "
            "one or two sentences. No markdown, no bullet lists, no emoji: "
            "everything you write is read aloud. Never mention which model, API "
            "key or provider you are using. When a tool fails, say plainly what "
            "went wrong."
        ),
        "timezone": "local",
    },
    "wake_word": {
        "model_path": "models/adrien.onnx",
        "fallback_model": "hey_jarvis",
        "threshold": 0.55,
        "refractory_seconds": 1.5,
        "acknowledgement": "tone",
    },
    "audio": {
        "input_device": None,
        "output_device": None,
        "sample_rate": 16000,
        "frame_ms": 30,
        "vad_aggressiveness": 2,
    },
    "conversation": {
        "follow_up_window_seconds": 6.0,
        "endpoint_silence_seconds": 1.0,
        "max_utterance_seconds": 30.0,
        "min_utterance_seconds": 0.35,
        "history_turns": 12,
        "barge_in_enabled": True,
        "barge_in_speech_frames": 6,
    },
    "llm": {
        "temperature": 0.6,
        "max_tokens": 700,
        "max_tool_iterations": 5,
        "force_smart_keywords": [
            "why", "explain", "compare", "plan", "summarize", "summarise",
            "debug", "refactor", "and then", "after that",
        ],
    },
    "keys": {
        "cooldown_seconds": 60.0,
        "failure_cooldown_seconds": 15.0,
        "max_attempts_per_call": 8,
        "request_timeout_seconds": 25.0,
    },
    "tts": {"format": "mp3", "latency": "balanced", "stream": True},
    "memory": {
        "enabled": True,
        "top_k": 6,
        "max_distance": 1.1,
        "summarize_on_session_end": True,
    },
    "server": {"enabled": True, "advertise_mdns": True, "max_clients": 8},
    "permissions": {
        "default": "confirm",
        "categories": {
            "dev": "confirm",
            "gaming": "auto",
            "productivity": "confirm",
            "system": "auto",
            "info": "auto",
            "messaging": "confirm",
            "extras": "auto",
        },
        "tools": {},
    },
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay `overlay` onto a copy of `base`."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Settings:
    """Dotted-path read access over the merged settings dictionary."""

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        self._data = data
        self.path = path

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def save(self) -> None:
        """Persist back to `settings.json` (used by the menu bar toggles)."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Settings(path={self.path})"


def load_settings(path: Path | None = None) -> Settings:
    """Load `config/settings.json`, layered over `DEFAULT_SETTINGS`.

    A missing or malformed file is not fatal: Adrien is a background service
    and must still come up. A malformed file is reported on stderr and the
    defaults are used.
    """
    settings_path = path or (PROJECT_ROOT / "config" / "settings.json")
    overlay: dict[str, Any] = {}
    if settings_path.exists():
        try:
            overlay = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[adrien] ignoring bad settings file {settings_path}: {exc}",
                  file=sys.stderr)
            overlay = {}
    return Settings(_deep_merge(DEFAULT_SETTINGS, overlay), settings_path)


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings singleton."""
    load_env()
    return load_settings()


def reload_settings() -> Settings:
    settings.cache_clear()
    return settings()
