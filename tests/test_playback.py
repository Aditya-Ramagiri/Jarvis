"""Playback termination and barge-in (spec 5.1, 5.3).

Regression tests for the deadlock that stalled the whole assistant right after
the wake word fired.

`Speaker.play_stream` used to finish by polling `while stream.active`. For a
*blocking* PortAudio stream - one with no callback, which is what Adrien uses -
`active` stays True until `stop()` or `abort()` is called explicitly. It never
flips on its own when the buffer drains. So the loop never exited: the wake
acknowledgement tone passes no `should_stop` and never sets `_stop`, meaning
`await speaker.play_tone()` never returned and recording never began. From the
outside it looked exactly like the wake word being ignored.

The fake stream below reproduces that behaviour faithfully - `active` is True
from `start()` until `stop()`/`abort()` - so any return to polling it will hang
these tests rather than shipping.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from adrien.core.audio import AudioConfig, Speaker

pytestmark = pytest.mark.asyncio

# If the bug returns these hang forever; fail fast instead of blocking CI.
TIMEOUT = 5.0


class FakeStream:
    """Mimics a blocking sounddevice.OutputStream."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.written = bytearray()
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def write(self, data) -> None:
        self.written.extend(bytes(data))

    @property
    def active(self) -> bool:
        # The crux: a blocking stream stays active until it is told to stop.
        return self.started and not (self.stopped or self.aborted)

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_audio(monkeypatch):
    """Install fake `sounddevice` and `numpy` modules for the duration."""
    created: list[FakeStream] = []

    def make_stream(**kwargs):
        stream = FakeStream(**kwargs)
        created.append(stream)
        return stream

    sounddevice = types.ModuleType("sounddevice")
    sounddevice.OutputStream = make_stream
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)

    # Real numpy when it is installed (the tone needs actual maths); a
    # byte-passthrough stub otherwise, so the streaming tests still run on a
    # machine with no audio stack at all.
    try:
        import numpy  # noqa: F401
    except ImportError:
        stub = types.ModuleType("numpy")
        stub.int16 = "int16"
        stub.frombuffer = lambda buffer, dtype=None: buffer
        stub.ndarray = bytes
        monkeypatch.setitem(sys.modules, "numpy", stub)

    return created


async def chunks_of(*payloads):
    for payload in payloads:
        yield payload


def make_speaker() -> Speaker:
    return Speaker(AudioConfig(), sample_rate=24_000)


# -- termination ------------------------------------------------------------
async def test_playback_terminates(fake_audio):
    """The deadlock itself: this hung forever before the fix."""
    speaker = make_speaker()
    played = await asyncio.wait_for(
        speaker.play_stream(chunks_of(b"\x01\x00" * 2400)), timeout=TIMEOUT
    )
    assert played == 1.0
    assert fake_audio[0].closed


async def test_a_stream_that_stays_active_still_terminates(fake_audio):
    """Explicitly pins the property that caused the hang."""
    speaker = make_speaker()
    await asyncio.wait_for(
        speaker.play_stream(chunks_of(b"\x02\x00" * 5000)), timeout=TIMEOUT
    )
    stream = fake_audio[0]
    # It was never allowed to go inactive on its own - the code must not wait
    # for that - and it was stopped rather than aborted on a clean finish.
    assert stream.stopped and not stream.aborted


async def test_an_empty_stream_terminates(fake_audio):
    speaker = make_speaker()
    assert await asyncio.wait_for(speaker.play_stream(chunks_of()), timeout=TIMEOUT) == 1.0


async def test_empty_chunks_are_skipped(fake_audio):
    speaker = make_speaker()
    await asyncio.wait_for(
        speaker.play_stream(chunks_of(b"", b"\x01\x00" * 500, b"")), timeout=TIMEOUT
    )
    assert len(fake_audio[0].written) == 1000


async def test_everything_handed_over_gets_written(fake_audio):
    speaker = make_speaker()
    payload = b"\x03\x00" * 3000
    await asyncio.wait_for(speaker.play_stream(chunks_of(payload)), timeout=TIMEOUT)
    assert bytes(fake_audio[0].written) == payload
    assert speaker.bytes_played == len(payload)


# -- the wake acknowledgement ----------------------------------------------
async def test_the_wake_tone_returns(fake_audio):
    """The exact call that stalled after the wake word fired."""
    pytest.importorskip("numpy", reason="the tone needs real numpy maths")
    speaker = make_speaker()
    await asyncio.wait_for(speaker.play_tone(), timeout=TIMEOUT)
    assert fake_audio[0].written, "the tone produced no audio"


# -- barge-in ---------------------------------------------------------------
async def test_barge_in_stops_playback_early(fake_audio):
    speaker = make_speaker()
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    played = await asyncio.wait_for(
        speaker.play_stream(chunks_of(b"\x04\x00" * 24_000), should_stop=should_stop),
        timeout=TIMEOUT,
    )
    assert played < 1.0, "interrupted playback should report a partial fraction"
    # Interrupted playback aborts, dropping what is queued, rather than
    # draining it - otherwise the user keeps hearing the rest of the sentence.
    assert fake_audio[0].aborted and not fake_audio[0].stopped


async def test_stop_halts_playback(fake_audio):
    speaker = make_speaker()

    async def slow_chunks():
        for _ in range(50):
            yield b"\x05\x00" * 2400
            await asyncio.sleep(0)
            speaker.stop()

    played = await asyncio.wait_for(speaker.play_stream(slow_chunks()), timeout=TIMEOUT)
    assert played < 1.0
    assert fake_audio[0].aborted


async def test_the_played_fraction_reflects_what_was_heard(fake_audio):
    """`Conversation.note_interruption` relies on this to resume correctly."""
    speaker = make_speaker()
    stop_after = {"n": 0}

    def should_stop() -> bool:
        stop_after["n"] += 1
        return stop_after["n"] > 5

    played = await asyncio.wait_for(
        speaker.play_stream(chunks_of(b"\x06\x00" * 24_000), should_stop=should_stop),
        timeout=TIMEOUT,
    )
    assert 0.0 < played < 1.0
