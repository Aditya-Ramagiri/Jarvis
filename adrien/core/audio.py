"""Microphone capture, endpointing and speaker playback.

Everything here works in one canonical format - **16 kHz, mono, signed 16-bit
little-endian PCM** - because that is what openWakeWord, webrtcvad and Whisper
all want. Converting once at the edges is cheaper and less bug-prone than
having each consumer resample.

Three responsibilities:

* `MicrophoneStream` - a always-on capture thread feeding a bounded queue. It
  stays open across wake word, recording and playback, because opening a
  CoreAudio input device costs 100-300 ms and doing that after the wake word
  would clip the first syllable of the user's request.
* `record_utterance` - VAD-based endpointing: start on speech, stop after a
  configured run of silence.
* `Speaker` - plays PCM chunks as they arrive and can be stopped mid-word,
  which is what makes barge-in (spec 5.3) possible.

Hardware libraries (`sounddevice`, `webrtcvad`, `numpy`) are imported lazily so
the rest of Adrien - and the whole test suite - runs on a machine without an
audio stack.
"""

from __future__ import annotations

import array
import asyncio
import math
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass

from adrien.logging_setup import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes, int16
CHANNELS = 1


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    frame_ms: int = 30
    input_device: int | str | None = None
    output_device: int | str | None = None
    vad_aggressiveness: int = 2

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * SAMPLE_WIDTH

    @classmethod
    def from_settings(cls, settings) -> AudioConfig:
        audio = settings.get("audio", {}) or {}
        return cls(
            sample_rate=int(audio.get("sample_rate", SAMPLE_RATE)),
            frame_ms=int(audio.get("frame_ms", 30)),
            input_device=audio.get("input_device"),
            output_device=audio.get("output_device"),
            vad_aggressiveness=int(audio.get("vad_aggressiveness", 2)),
        )


def frame_rms(frame: bytes) -> float:
    """Root-mean-square level of one PCM frame, 0.0-1.0.

    Written against `array` rather than `audioop`: audioop is deprecated in
    3.11 and gone in 3.13, and this runs on every 30 ms frame forever.
    """
    if len(frame) < SAMPLE_WIDTH:
        return 0.0
    samples = array.array("h")
    # An odd trailing byte would raise; drop it rather than lose the frame.
    samples.frombytes(frame[: len(frame) - (len(frame) % SAMPLE_WIDTH)])
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples)) / 32768.0


def rms_to_db(rms: float) -> float:
    return 20 * math.log10(rms) if rms > 1e-9 else -120.0


class MicrophoneStream:
    """Continuous 16 kHz mono capture into a bounded queue of frames.

    The queue is bounded and drops the *oldest* frame when full: if a consumer
    stalls we would rather lose 30 ms of history than accumulate a growing
    delay between what the user said and what Adrien is processing.
    """

    def __init__(self, config: AudioConfig | None = None, max_frames: int = 200) -> None:
        self.config = config or AudioConfig()
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_frames)
        self._stream = None
        self._muted = threading.Event()
        self.dropped_frames = 0

    # -- lifecycle --------------------------------------------------------
    def start(self) -> MicrophoneStream:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                log.debug("input stream status: %s", status)
            if self._muted.is_set():
                return
            try:
                self._queue.put_nowait(bytes(indata))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(bytes(indata))
                    self.dropped_frames += 1
                except queue.Empty:  # pragma: no cover - race, harmless
                    pass

        self._stream = sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.frame_samples,
            device=self.config.input_device,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        log.info("microphone open at %d Hz, %d ms frames",
                 self.config.sample_rate, self.config.frame_ms)
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> MicrophoneStream:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- reading ----------------------------------------------------------
    def read(self, timeout: float | None = 1.0) -> bytes | None:
        """Next frame, or None if none arrived within `timeout`."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def read_async(self, timeout: float | None = 1.0) -> bytes | None:
        """`read` off the event loop, so the orchestrator never blocks."""
        return await asyncio.to_thread(self.read, timeout)

    def flush(self) -> None:
        """Discard buffered audio.

        Called after Adrien finishes speaking: whatever the mic picked up
        during playback is mostly Adrien's own voice, and replaying it into the
        follow-up window would make it talk to itself.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def frames(self, timeout: float | None = 1.0) -> Iterable[bytes]:
        while True:
            frame = self.read(timeout)
            if frame is None:
                return
            yield frame

    # Muting is used while a client on the WebSocket is driving the
    # conversation, so the Mac's own mic does not race a phone's.
    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()
        self.flush()


class VoiceActivityDetector:
    """webrtcvad wrapper with an energy floor.

    webrtcvad alone is generous - it flags keyboard clatter and fan noise as
    speech. Requiring the frame to also clear an energy floor cuts most of
    that, which matters because a false positive here either ends a recording
    early or falsely triggers barge-in.
    """

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30,
                 energy_floor: float = 0.006) -> None:
        self.frame_ms = frame_ms
        self.energy_floor = energy_floor
        self._vad = None
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
        except ImportError:  # pragma: no cover - optional at import time
            log.warning("webrtcvad unavailable; falling back to energy-only VAD")

    def is_speech(self, frame: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        if frame_rms(frame) < self.energy_floor:
            return False
        if self._vad is None:
            return True  # energy-only fallback
        # webrtcvad only accepts 10/20/30 ms frames at 8/16/32/48 kHz.
        expected = int(sample_rate * self.frame_ms / 1000) * SAMPLE_WIDTH
        if len(frame) != expected:
            return True
        try:
            return self._vad.is_speech(frame, sample_rate)
        except Exception:  # pragma: no cover - malformed frame
            return True


@dataclass
class Utterance:
    """One recorded stretch of speech."""

    pcm: bytes
    duration_s: float
    timed_out: bool = False
    started: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.started or self.duration_s <= 0


def record_utterance(
    mic: MicrophoneStream,
    vad: VoiceActivityDetector,
    *,
    silence_seconds: float = 1.0,
    max_seconds: float = 30.0,
    start_timeout: float = 6.0,
    preroll_frames: int = 8,
) -> Utterance:
    """Record until the user stops talking.

    `preroll_frames` of audio from *before* speech was detected are kept, so
    the recording does not clip the attack of the first word - VAD needs a
    frame or two of evidence before it fires.

    Returns an empty `Utterance` if nobody spoke within `start_timeout`, which
    is how the follow-up window (spec 5.2) closes itself.
    """
    frame_ms = mic.config.frame_ms
    silence_limit = max(1, int(silence_seconds * 1000 / frame_ms))
    max_frames = max(1, int(max_seconds * 1000 / frame_ms))

    preroll: list[bytes] = []
    collected: list[bytes] = []
    silent_run = 0
    started = False
    deadline = time.monotonic() + start_timeout

    while True:
        frame = mic.read(timeout=0.5)
        if frame is None:
            if not started and time.monotonic() >= deadline:
                return Utterance(pcm=b"", duration_s=0.0, started=False)
            continue

        speech = vad.is_speech(frame, mic.config.sample_rate)

        if not started:
            preroll.append(frame)
            if len(preroll) > preroll_frames:
                preroll.pop(0)
            if speech:
                started = True
                collected.extend(preroll)
                collected.append(frame)
            elif time.monotonic() >= deadline:
                return Utterance(pcm=b"", duration_s=0.0, started=False)
            continue

        collected.append(frame)
        silent_run = 0 if speech else silent_run + 1

        if silent_run >= silence_limit:
            break
        if len(collected) >= max_frames:
            log.info("utterance hit the %.0fs cap", max_seconds)
            return Utterance(
                pcm=b"".join(collected),
                duration_s=len(collected) * frame_ms / 1000,
                timed_out=True,
                started=True,
            )

    # Trim the trailing silence we used to detect the endpoint.
    keep = max(1, len(collected) - silence_limit + 2)
    pcm = b"".join(collected[:keep])
    return Utterance(pcm=pcm, duration_s=len(pcm) / (SAMPLE_RATE * SAMPLE_WIDTH), started=True)


class BargeInMonitor:
    """Watches the mic during playback and reports when the user cuts in.

    Acoustic caveat, stated plainly because it shapes the tuning: on a laptop
    with no echo cancellation the mic hears Adrien's own voice. Two guards
    handle that without an AEC dependency - a run of *consecutive* speech
    frames (a syllable, not a click) and an energy threshold calibrated
    against the first moments of playback, when only Adrien is audible. With
    headphones neither guard is doing much work; on open speakers they are
    what stop Adrien interrupting itself.
    """

    def __init__(
        self,
        mic: MicrophoneStream,
        vad: VoiceActivityDetector,
        *,
        speech_frames: int = 6,
        margin_db: float = 6.0,
    ) -> None:
        self.mic = mic
        self.vad = vad
        self.speech_frames = speech_frames
        self.margin_db = margin_db
        self._run = 0
        self._floor_db: float | None = None
        self._calibration: list[float] = []
        self.captured: list[bytes] = []

    def reset(self) -> None:
        self._run = 0
        self._floor_db = None
        self._calibration.clear()
        self.captured.clear()

    def check(self) -> bool:
        """Drain pending frames; True once the user is clearly speaking."""
        while True:
            frame = self.mic.read(timeout=0.0)
            if not frame:
                return False  # nothing buffered right now; caller polls again

            level_db = rms_to_db(frame_rms(frame))

            # First ~300 ms of playback: whatever the mic hears is Adrien.
            if self._floor_db is None:
                self._calibration.append(level_db)
                if len(self._calibration) >= 10:
                    self._floor_db = max(self._calibration) + self.margin_db
                continue

            if level_db > self._floor_db and self.vad.is_speech(frame, self.mic.config.sample_rate):
                self._run += 1
                self.captured.append(frame)
                if self._run >= self.speech_frames:
                    log.info("barge-in detected at %.0f dB (floor %.0f dB)",
                             level_db, self._floor_db)
                    return True
            else:
                self._run = 0
                self.captured.clear()


class Speaker:
    """Plays 16-bit PCM, chunk by chunk, and can be stopped mid-word."""

    def __init__(self, config: AudioConfig | None = None, sample_rate: int = 24_000) -> None:
        self.config = config or AudioConfig()
        self.sample_rate = sample_rate
        self._stop = threading.Event()
        self._playing = threading.Event()
        self.bytes_played = 0

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def stop(self) -> None:
        """Cut playback immediately (barge-in, or a new request arriving)."""
        self._stop.set()

    @property
    def seconds_played(self) -> float:
        return self.bytes_played / (self.sample_rate * SAMPLE_WIDTH)

    async def play_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        should_stop: Callable[[], bool] | None = None,
        poll_interval: float = 0.02,
    ) -> float:
        """Play PCM chunks as they arrive; return the fraction actually played.

        The returned fraction is what lets `ConversationState` know how much of
        a response the user actually heard when they interrupt (spec 5.3).
        """
        import numpy as np
        import sounddevice as sd

        self._stop.clear()
        self._playing.set()
        self.bytes_played = 0
        total_bytes = 0
        pending = bytearray()

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="int16",
            device=self.config.output_device,
            blocksize=0,
        )
        stream.start()
        try:
            async for chunk in chunks:
                if not chunk:
                    continue
                total_bytes += len(chunk)
                pending.extend(chunk)

                # Write in small blocks so a stop request lands within ~20 ms
                # rather than at the end of whatever chunk arrived.
                block = int(self.sample_rate * 0.02) * SAMPLE_WIDTH
                while len(pending) >= block:
                    if self._stop.is_set() or (should_stop and should_stop()):
                        return self._finish(total_bytes)
                    piece = bytes(pending[:block])
                    del pending[:block]
                    await asyncio.to_thread(
                        stream.write, np.frombuffer(piece, dtype=np.int16)
                    )
                    self.bytes_played += len(piece)
                await asyncio.sleep(0)

            if pending and not self._stop.is_set():
                await asyncio.to_thread(
                    stream.write, np.frombuffer(bytes(pending), dtype=np.int16)
                )
                self.bytes_played += len(pending)

            # Everything has been handed to the device. Draining is left to
            # `stream.stop()` in the finally block, which blocks until the
            # queued audio has actually played.
            #
            # Do NOT poll `stream.active` here. This is a *blocking* stream -
            # it has no callback - and for those PortAudio keeps `active` True
            # until stop() or abort() is called explicitly. It never flips on
            # its own when the buffer empties, so polling it is an infinite
            # loop: the wake tone passes no `should_stop` and never sets
            # `_stop`, so `await self.speaker.play_tone()` never returned and
            # the whole conversation stalled immediately after the wake word.
            return self._finish(total_bytes)
        finally:
            self._playing.clear()
            try:
                if self._stop.is_set() or (should_stop and should_stop()):
                    # Interrupted: drop whatever is still queued so the user
                    # hears silence now, not the rest of the sentence.
                    stream.abort()
                else:
                    stream.stop()   # blocks until the queued audio has played
            except Exception as exc:  # pragma: no cover - device disappeared
                log.warning("could not close the output stream cleanly: %s", exc)
            stream.close()

    def _finish(self, total_bytes: int) -> float:
        if total_bytes <= 0:
            return 1.0
        return min(1.0, self.bytes_played / total_bytes)

    async def play_bytes(self, pcm: bytes) -> float:
        async def one() -> AsyncIterator[bytes]:
            yield pcm

        return await self.play_stream(one())

    async def play_tone(self, frequency: float = 880.0, duration: float = 0.12,
                        volume: float = 0.22) -> None:
        """The wake-word acknowledgement (spec 5.1: snappy, not a phrase)."""
        import numpy as np

        samples = int(self.sample_rate * duration)
        t = np.arange(samples, dtype=np.float32) / self.sample_rate
        wave = np.sin(2 * np.pi * frequency * t)
        # Short fades stop the click a hard-edged tone makes on small speakers.
        fade = max(1, int(self.sample_rate * 0.01))
        envelope = np.ones(samples, dtype=np.float32)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        pcm = (wave * envelope * volume * 32767).astype(np.int16).tobytes()
        await self.play_bytes(pcm)


def list_devices() -> list[dict[str, object]]:
    """Input/output devices, for `adrien devices` and settings tuning."""
    import sounddevice as sd

    devices = []
    for index, device in enumerate(sd.query_devices()):
        devices.append({
            "index": index,
            "name": device["name"],
            "inputs": device["max_input_channels"],
            "outputs": device["max_output_channels"],
            "default_samplerate": device["default_samplerate"],
        })
    return devices
