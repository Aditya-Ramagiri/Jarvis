"""Logging with mandatory secret redaction.

Spec section 9: "Never log or store raw API keys in the memory system, logs,
or any persisted conversation data." Rather than trusting every call site to
remember that, redaction is enforced centrally by a `logging.Filter` installed
on the root logger, plus a `redact()` helper used by anything that persists
text (memory records, transcripts, tool results).

The filter rewrites the *formatted* message, so it catches secrets that arrive
through `%s` arguments and exception tracebacks too.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path

from adrien.config import env_str, log_dir

# Patterns for secrets that may appear in third-party error strings even if we
# never log them ourselves (e.g. an HTTP client echoing a request header).
_SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),                 # Groq
    re.compile(r"sk-fish-[A-Za-z0-9_\-]{10,}"),          # Fish Audio
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),              # Google API keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),           # GitHub tokens
    re.compile(r"GOCSPX-[A-Za-z0-9_\-]{10,}"),           # Google OAuth secret
    re.compile(r"ya29\.[A-Za-z0-9_\-]{20,}"),            # Google access tokens
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.I),  # generic auth headers
]

_REDACTED = "[redacted]"

_SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _known_secret_values() -> list[str]:
    """Exact secret values currently in the environment.

    Belt and braces: catches keys whose shape does not match any pattern above
    (a provider changing its prefix, a user-supplied token).
    """
    values: list[str] = []
    for name, value in os.environ.items():
        if not value or len(value) < 12:
            continue
        upper = name.upper()
        if any(marker in upper for marker in _SECRET_NAME_MARKERS):
            values.append(value)
    # Longest first so a key that contains another as a prefix redacts fully.
    return sorted(set(values), key=len, reverse=True)


def redact(text: str) -> str:
    """Strip anything that looks like a credential out of `text`."""
    if not text:
        return text
    for value in _known_secret_values():
        if value in text:
            text = text.replace(value, _REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


class RedactionFilter(logging.Filter):
    """Rewrites every record so no credential ever reaches a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - broken %-format in a caller
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


_CONFIGURED = False


def setup_logging(level: str | None = None, *, to_file: bool = True) -> logging.Logger:
    """Configure root logging once. Returns the `adrien` logger."""
    global _CONFIGURED
    logger = logging.getLogger("adrien")
    if _CONFIGURED:
        return logger

    resolved = (level or env_str("ADRIEN_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(resolved)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    redaction = RedactionFilter()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(redaction)
    root.addHandler(console)

    if to_file:
        try:
            path: Path = log_dir() / "adrien.log"
            rotating = logging.handlers.RotatingFileHandler(
                path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            rotating.setFormatter(fmt)
            rotating.addFilter(redaction)
            root.addHandler(rotating)
        except OSError as exc:  # pragma: no cover - unwritable data dir
            logger.warning("file logging disabled: %s", exc)

    # Third-party libraries are chatty at DEBUG and echo request URLs.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """`get_logger(__name__)` -> a child of the `adrien` logger."""
    if name.startswith("adrien"):
        return logging.getLogger(name)
    return logging.getLogger(f"adrien.{name}")
