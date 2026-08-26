"""Key rotation and circuit-breaker behaviour (spec section 4)."""

from __future__ import annotations

from adrien.core.keypool import KeyPool


def make_pool(clock, keys=("k1", "k2", "k3"), **kwargs):
    return KeyPool("groq", list(keys), clock=clock, **kwargs)


def test_duplicate_keys_are_collapsed(clock):
    pool = make_pool(clock, keys=("same", "same", "other"))
    assert len(pool) == 2
    assert [entry["label"] for entry in pool.stats()] == ["groq#1", "groq#2"]


def test_blank_keys_are_ignored(clock):
    pool = make_pool(clock, keys=("k1", "", "   ", "k2"))
    assert len(pool) == 2


def test_least_recently_used_rotation(clock):
    pool = make_pool(clock)
    seen = []
    for _ in range(6):
        lease = pool.acquire()
        assert lease is not None
        lease.success()
        seen.append(lease.label)
        clock.advance(0.1)
    # Every key gets used before any key is reused.
    assert set(seen[:3]) == {"groq#1", "groq#2", "groq#3"}
    assert seen[:3] == seen[3:]


def test_rate_limited_key_is_skipped_until_cooldown_expires(clock):
    pool = make_pool(clock, cooldown_seconds=60.0)
    first = pool.acquire()
    assert first is not None
    first.rate_limited()

    # The cooling key is not handed out again...
    labels = set()
    for _ in range(4):
        lease = pool.acquire()
        assert lease is not None
        labels.add(lease.label)
        lease.success()
        clock.advance(0.1)
    assert first.label not in labels
    assert pool.available_count() == 2

    # ...until its window passes.
    clock.advance(61)
    assert pool.available_count() == 3


def test_retry_after_header_overrides_default_cooldown(clock):
    pool = make_pool(clock, cooldown_seconds=60.0)
    lease = pool.acquire()
    assert lease is not None
    lease.rate_limited(retry_after=5.0)

    clock.advance(6)
    assert pool.available_count() == 3


def test_acquire_returns_none_when_every_key_is_cooling(clock):
    pool = make_pool(clock)
    for _ in range(3):
        lease = pool.acquire()
        assert lease is not None
        lease.rate_limited()
    # The router relies on this to fall through to the next provider without
    # ever sleeping.
    assert pool.acquire() is None
    assert pool.available_count() == 0
    assert pool.seconds_until_available() == 60.0


def test_leases_iterator_stops_when_pool_is_exhausted(clock):
    pool = make_pool(clock)
    used = []
    for lease in pool.leases():
        used.append(lease.label)
        lease.rate_limited()
    assert len(used) == 3


def test_repeated_failures_back_off_exponentially(clock):
    pool = make_pool(clock, keys=("only",), failure_cooldown_seconds=10.0)
    for expected in (10, 20, 40):
        lease = pool.acquire()
        assert lease is not None, f"expected a lease before {expected}s backoff"
        lease.failed()
        clock.advance(expected - 1)
        assert pool.available_count() == 0, "key freed too early"
        clock.advance(1.5)
        assert pool.available_count() == 1


def test_success_clears_the_failure_streak(clock):
    pool = make_pool(clock, keys=("only",), failure_cooldown_seconds=10.0)
    lease = pool.acquire()
    assert lease is not None
    lease.failed()
    clock.advance(11)

    lease = pool.acquire()
    assert lease is not None
    lease.success()

    lease = pool.acquire()
    assert lease is not None
    lease.failed()
    clock.advance(11)
    assert pool.available_count() == 1, "backoff should have reset to the base"


def test_stats_never_expose_key_material(clock):
    pool = make_pool(clock, keys=("gsk_supersecret",))
    rendered = repr(pool.stats()) + repr(pool.acquire())
    assert "gsk_supersecret" not in rendered


def test_reset_clears_cooldowns(clock):
    pool = make_pool(clock)
    for lease in pool.leases():
        lease.rate_limited()
    assert pool.available_count() == 0
    pool.reset()
    assert pool.available_count() == 3
