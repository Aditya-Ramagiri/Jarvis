"""API key rotation with a circuit breaker.

Spec section 4. The user holds several accounts per provider purely to work
around per-account rate limits, so rotation is not an optimisation here - it is
the thing that keeps Adrien answering.

Design decisions, and why:

* **Least-recently-used selection**, not round-robin. LRU spreads load the same
  way round-robin does, but it also naturally prefers a key that has been
  resting when some keys have just been burned by a retry storm.
* **Circuit breaker instead of retry-in-place.** A key that returns 429 is
  marked "cooling" for N seconds and skipped entirely. Retrying a key that is
  still rate limited costs a full round trip and buys nothing, and spec 4.6
  makes latency a top priority. `Retry-After`, when the provider sends one, is
  honoured over the configured default.
* **No sleeping inside the pool.** Acquiring a key is a lock plus a scan over a
  handful of entries - microseconds. When every key is cooling, the caller is
  told immediately (`None`) so it can fall through to the next provider rather
  than blocking.
* **Key material never leaves the pool.** Callers log `lease.label`
  ("groq#2"), never the key itself.

The pool is thread-safe: the orchestrator drives it from an asyncio loop while
the WebSocket server and menu bar may read stats from other threads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from adrien.logging_setup import get_logger

log = get_logger(__name__)

Clock = Callable[[], float]


@dataclass
class KeyState:
    """Bookkeeping for one API key. `key` is never logged or serialised."""

    key: str
    label: str
    cooling_until: float = 0.0
    last_used: float = 0.0
    successes: int = 0
    rate_limits: int = 0
    failures: int = 0
    consecutive_failures: int = 0

    def is_available(self, now: float) -> bool:
        return now >= self.cooling_until


@dataclass
class KeyLease:
    """A key handed to a caller for exactly one request."""

    key: str
    label: str
    pool: "KeyPool"
    _state: KeyState = field(repr=False)

    def success(self) -> None:
        self.pool.report_success(self)

    def rate_limited(self, retry_after: float | None = None) -> None:
        self.pool.report_rate_limited(self, retry_after)

    def failed(self, cooldown: float | None = None) -> None:
        self.pool.report_failure(self, cooldown)

    def __repr__(self) -> str:  # pragma: no cover - key material stays hidden
        return f"KeyLease({self.label})"


class KeyPool:
    """A rotating pool of interchangeable API keys for one provider."""

    def __init__(
        self,
        name: str,
        keys: list[str],
        *,
        cooldown_seconds: float = 60.0,
        failure_cooldown_seconds: float = 15.0,
        clock: Clock = time.monotonic,
    ) -> None:
        self.name = name
        self.cooldown_seconds = cooldown_seconds
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self._clock = clock
        self._lock = threading.Lock()
        # De-duplicate: the same key pasted into two slots would otherwise look
        # like two independent quotas and defeat the whole mechanism.
        seen: set[str] = set()
        self._states: list[KeyState] = []
        for index, key in enumerate(keys, start=1):
            key = (key or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            self._states.append(KeyState(key=key, label=f"{name}#{len(self._states) + 1}"))
        if len(self._states) != len(keys):
            log.debug("%s pool: %d usable keys from %d entries",
                      name, len(self._states), len(keys))

    # -- introspection ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._states)

    @property
    def configured(self) -> bool:
        return bool(self._states)

    def available_count(self) -> int:
        now = self._clock()
        with self._lock:
            return sum(1 for state in self._states if state.is_available(now))

    def seconds_until_available(self) -> float | None:
        """How long until *some* key frees up, or None if one is free now."""
        now = self._clock()
        with self._lock:
            if not self._states:
                return None
            if any(state.is_available(now) for state in self._states):
                return 0.0
            return max(0.0, min(state.cooling_until for state in self._states) - now)

    def stats(self) -> list[dict[str, object]]:
        """Per-key counters for the menu bar / `adrien status`. No key material."""
        now = self._clock()
        with self._lock:
            return [
                {
                    "label": state.label,
                    "available": state.is_available(now),
                    "cooling_for": round(max(0.0, state.cooling_until - now), 1),
                    "successes": state.successes,
                    "rate_limits": state.rate_limits,
                    "failures": state.failures,
                }
                for state in self._states
            ]

    # -- acquisition ------------------------------------------------------
    def acquire(self) -> KeyLease | None:
        """Least-recently-used available key, or None if all are cooling."""
        now = self._clock()
        with self._lock:
            candidates = [s for s in self._states if s.is_available(now)]
            if not candidates:
                return None
            state = min(candidates, key=lambda s: s.last_used)
            state.last_used = now
            return KeyLease(key=state.key, label=state.label, pool=self, _state=state)

    def leases(self, max_attempts: int | None = None) -> Iterator[KeyLease]:
        """Yield leases until the pool is exhausted or `max_attempts` is hit.

        The caller reports the outcome on each lease; a lease that reports a
        rate limit or failure puts its key into cooldown, so the next
        iteration naturally moves on instead of looping on a dead key.
        """
        attempts = 0
        limit = max_attempts if max_attempts is not None else len(self._states)
        while attempts < limit:
            lease = self.acquire()
            if lease is None:
                return
            attempts += 1
            yield lease

    # -- outcome reporting ------------------------------------------------
    def report_success(self, lease: KeyLease) -> None:
        with self._lock:
            state = lease._state
            state.successes += 1
            state.consecutive_failures = 0
            state.cooling_until = 0.0

    def report_rate_limited(self, lease: KeyLease, retry_after: float | None = None) -> None:
        now = self._clock()
        with self._lock:
            state = lease._state
            state.rate_limits += 1
            # Trust the provider's own Retry-After when it sends one; it knows
            # when the window resets and we do not.
            cooldown = retry_after if retry_after and retry_after > 0 else self.cooldown_seconds
            state.cooling_until = now + cooldown
        log.info("%s cooling for %.0fs after rate limit", lease.label, cooldown)

    def report_failure(self, lease: KeyLease, cooldown: float | None = None) -> None:
        """A non-rate-limit failure (network error, 5xx, auth rejection).

        Backed off more gently than a 429, and exponentially if the same key
        keeps failing - a revoked key should stop being tried quickly, but a
        single blip should not sideline a good key for a minute.
        """
        now = self._clock()
        with self._lock:
            state = lease._state
            state.failures += 1
            state.consecutive_failures += 1
            base = cooldown if cooldown is not None else self.failure_cooldown_seconds
            backoff = base * min(2 ** (state.consecutive_failures - 1), 8)
            state.cooling_until = now + backoff
        log.info("%s cooling for %.0fs after failure (%d in a row)",
                 lease.label, backoff, state.consecutive_failures)

    def reset(self) -> None:
        """Clear all cooldowns - used by the menu bar's 'retry now' action."""
        with self._lock:
            for state in self._states:
                state.cooling_until = 0.0
                state.consecutive_failures = 0
