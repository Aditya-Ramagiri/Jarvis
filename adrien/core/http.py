"""One shared HTTP client for every outbound API call.

Latency matters (spec 4.6) and the single biggest avoidable cost on a small
request is a fresh TLS handshake. A process-wide `httpx.AsyncClient` keeps
connections to Groq, Gemini and Fish Audio warm between utterances, so key
rotation costs a request, not a reconnect.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

_client: Any = None
_client_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()

# Connect fast or move on: a provider that cannot be reached in two seconds is
# not the provider we should be waiting on when three others are configured.
CONNECT_TIMEOUT = 2.0
DEFAULT_TIMEOUT = 25.0


def get_client(timeout: float = DEFAULT_TIMEOUT) -> Any:
    """Return the shared `httpx.AsyncClient`, creating it on first use.

    Recreated if the event loop changed (test runs, service restart), because
    an httpx client binds to the loop that owns its connection pool.
    """
    global _client, _client_loop
    import httpx

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    with _lock:
        if _client is not None and _client_loop is loop and not _client.is_closed:
            return _client
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
            follow_redirects=True,
            headers={"user-agent": "Adrien/0.1 (personal assistant)"},
        )
        _client_loop = loop
        return _client


async def close_client() -> None:
    """Close the shared client on shutdown."""
    global _client, _client_loop
    client = _client
    _client, _client_loop = None, None
    if client is not None and not client.is_closed:
        await client.aclose()


def parse_retry_after(headers: Any) -> float | None:
    """Seconds from a `Retry-After` header, if the provider sent a usable one.

    Groq sends fractional seconds ("7.2"); the HTTP spec also allows a date,
    which we ignore rather than mis-parse - the pool's default cooldown is a
    fine fallback.
    """
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        # Groq also exposes a reset hint on its own headers.
        raw = headers.get("x-ratelimit-reset-requests") or headers.get(
            "x-ratelimit-reset-tokens"
        )
    if not raw:
        return None
    text = str(raw).strip().rstrip("s")
    try:
        value = float(text)
    except ValueError:
        return None
    # Guard against a provider asking us to wait out the afternoon.
    return max(0.0, min(value, 300.0))
