"""Always-on wake word detection with openWakeWord.

Runs entirely locally on CPU (spec 5.1: listening must never touch an API and
must cost nothing). openWakeWord's ONNX models score an 80 ms window in well
under a millisecond, so continuous detection sits around 1-2% of one core.

**"Adrien" is not one of openWakeWord's pretrained models.** Nothing can change
that: the pretrained set is "alexa", "hey jarvis", "hey mycroft", "hey rhasspy"
and a few others. A custom model has to be trained once - it takes a few
minutes in openWakeWord's own synthetic-data notebook and needs no recordings
of the user. `docs/WAKE_WORD.md` walks through it.

Until `models/adrien.onnx` exists, this module falls back to a pretrained model
(default "hey jarvis") and says so loudly at startup, so the system is usable
end-to-end from the first run instead of blocked on a training step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from adrien.config import PROJECT_ROOT
from adrien.core.audio import SAMPLE_WIDTH
from adrien.logging_setup import get_logger

log = get_logger(__name__)

# openWakeWord's models are trained on 80 ms windows at 16 kHz.
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH


@dataclass
class Detection:
    name: str
    score: float
    at: float


class WakeWordDetector:
    """Feed it PCM frames; it tells you when the wake word was spoken."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        fallback_model: str = "hey_jarvis",
        threshold: float = 0.55,
        refractory_seconds: float = 1.5,
    ) -> None:
        self.threshold = threshold
        self.refractory_seconds = refractory_seconds
        self.model_path = Path(model_path) if model_path else None
        self.fallback_model = fallback_model
        self._model = None
        self._buffer = bytearray()
        self._last_detection = 0.0
        self.using_fallback = False
        self.label = "unloaded"

    # -- loading ----------------------------------------------------------
    def load(self) -> WakeWordDetector:
        from openwakeword.model import Model

        path = self.model_path
        if path is not None and not path.is_absolute():
            path = PROJECT_ROOT / path

        if path is not None and path.exists():
            self._model = Model(wakeword_models=[str(path)], inference_framework="onnx")
            self.label = path.stem
            log.info("wake word: custom model %s", path.name)
        else:
            if path is not None:
                log.warning(
                    "wake word model %s not found - falling back to the pretrained "
                    "'%s'. Train a real 'Adrien' model with docs/WAKE_WORD.md.",
                    path, self.fallback_model,
                )
            self._download_pretrained()
            self._model = Model(
                wakeword_models=[self.fallback_model], inference_framework="onnx"
            )
            self.using_fallback = True
            self.label = self.fallback_model
        return self

    @staticmethod
    def _download_pretrained() -> None:
        """Fetch openWakeWord's pretrained models on first run."""
        try:
            import openwakeword

            openwakeword.utils.download_models()
        except Exception as exc:  # pragma: no cover - network / already present
            log.debug("pretrained model download skipped: %s", exc)

    # -- detection --------------------------------------------------------
    def process(self, frame: bytes) -> Detection | None:
        """Feed one PCM frame. Returns a `Detection` on the frame that fires.

        Frames arriving from the mic are 30 ms; openWakeWord wants 80 ms, so
        they are re-chunked here rather than forcing the whole pipeline onto an
        80 ms cadence that webrtcvad cannot use.
        """
        if self._model is None:
            raise RuntimeError("WakeWordDetector.load() was never called")

        import numpy as np

        self._buffer.extend(frame)
        detection: Detection | None = None

        while len(self._buffer) >= CHUNK_BYTES:
            chunk = bytes(self._buffer[:CHUNK_BYTES])
            del self._buffer[:CHUNK_BYTES]
            scores = self._model.predict(np.frombuffer(chunk, dtype=np.int16))
            for name, score in scores.items():
                if score < self.threshold:
                    continue
                now = time.monotonic()
                # One utterance of "Adrien" spans several windows and would
                # otherwise fire repeatedly.
                if now - self._last_detection < self.refractory_seconds:
                    continue
                self._last_detection = now
                detection = Detection(name=name, score=float(score), at=now)
                log.info("wake word '%s' at %.2f", name, score)

        return detection

    def reset(self) -> None:
        """Clear internal state after a conversation, so stale audio in
        openWakeWord's own feature buffer cannot re-trigger it."""
        self._buffer.clear()
        if self._model is not None:
            try:
                self._model.reset()
            except AttributeError:  # pragma: no cover - older openWakeWord
                pass

    @classmethod
    def from_settings(cls, settings) -> WakeWordDetector:
        cfg = settings.get("wake_word", {}) or {}
        return cls(
            model_path=cfg.get("model_path"),
            fallback_model=cfg.get("fallback_model", "hey_jarvis"),
            threshold=float(cfg.get("threshold", 0.55)),
            refractory_seconds=float(cfg.get("refractory_seconds", 1.5)),
        )
