"""Speech to text via Groq-hosted `whisper-large-v3`.

Spec section 3 is explicit that this is a hard requirement: large-v3, not a
distilled or turbo variant, even though those are faster. The reason is
accuracy on disfluent speech - stutters, restarts, filler - and a smaller model
that mangles the request costs far more time than it saves.

**Why a separate key pool from the LLM's.** Groq's rate limits are per model
per account, so a Whisper 429 says nothing about whether that same account can
still serve llama. Sharing one pool would sideline a perfectly good chat key
every time a long dictation hit the audio quota. The pools hold the same keys
and cool independently.
"""

from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass

from adrien.config import Settings, env_key_pool, env_str
from adrien.config import settings as global_settings
from adrien.core.audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from adrien.core.http import get_client, parse_retry_after
from adrien.core.keypool import KeyPool
from adrien.logging_setup import get_logger

log = get_logger(__name__)

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Whisper hallucinates confidently on silence; below this there is nothing to
# transcribe and we save the round trip.
MIN_AUDIO_SECONDS = 0.3


@dataclass
class Transcription:
    text: str
    duration_s: float = 0.0
    latency_ms: float = 0.0
    key_label: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM in a WAV container in memory.

    The API needs a recognisable container; writing a 44-byte header beats
    touching the disk for every utterance.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


class Transcriber:
    """Rotating-key wrapper around Groq's Whisper endpoint."""

    def __init__(self, settings: Settings | None = None, pool: KeyPool | None = None) -> None:
        self.settings = settings or global_settings()
        self.model = env_str("GROQ_STT_MODEL", "whisper-large-v3")
        keys_cfg = self.settings.get("keys", {}) or {}
        self.pool = pool or KeyPool(
            "groq-stt",
            env_key_pool("GROQ_API_KEY"),
            cooldown_seconds=float(keys_cfg.get("cooldown_seconds", 60.0)),
            failure_cooldown_seconds=float(keys_cfg.get("failure_cooldown_seconds", 15.0)),
        )
        self.timeout = float(keys_cfg.get("request_timeout_seconds", 25.0))

    async def transcribe(
        self,
        pcm: bytes,
        *,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcription:
        """Transcribe raw PCM. Returns an empty result rather than raising.

        A failed transcription is a normal event in a voice loop (the user
        coughed, the network blipped) and the orchestrator handles it by
        quietly going back to listening. Raising here would mean an error
        spoken aloud for what is usually silence.
        """
        duration = len(pcm) / (sample_rate * SAMPLE_WIDTH)
        if duration < MIN_AUDIO_SECONDS:
            log.debug("skipping %.2fs of audio, below the floor", duration)
            return Transcription(text="", duration_s=duration)

        if not self.pool.configured:
            log.error("no Groq keys configured for STT")
            return Transcription(text="", duration_s=duration)

        wav = pcm_to_wav(pcm, sample_rate)
        data: dict[str, str] = {
            "model": self.model,
            "response_format": "json",
            "temperature": "0",
        }
        # Whisper auto-detects by default, and on a short or noisy utterance it
        # guesses badly - a two-second English phrase came back as Russian.
        # Pinning the language removes that entire failure mode; set
        # stt.language to "" in settings.json to go back to auto-detection.
        language = language or str(self.settings.get("stt.language", "en") or "")
        if language:
            data["language"] = language
        if prompt:
            # Whisper's `prompt` biases decoding: feeding it Adrien's own name
            # and the user's project nouns markedly improves recognition of
            # them. The orchestrator passes recent context here.
            data["prompt"] = prompt[:800]

        client = get_client()
        started = time.perf_counter()

        for lease in self.pool.leases():
            try:
                response = await client.post(
                    API_URL,
                    headers={"authorization": f"Bearer {lease.key}"},
                    data=data,
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                    timeout=self.timeout,
                )
            except Exception as exc:
                log.warning("stt transport error on %s: %s", lease.label, type(exc).__name__)
                lease.failed()
                continue

            if response.status_code == 429:
                lease.rate_limited(parse_retry_after(response.headers))
                continue
            if response.status_code in (401, 403) or response.status_code >= 500:
                log.warning("stt %s returned %d", lease.label, response.status_code)
                lease.failed()
                continue
            if response.status_code >= 400:
                lease.success()  # the key is fine; the request was not
                log.error("stt rejected the request (%d): %s",
                          response.status_code, (response.text or "")[:200])
                return Transcription(text="", duration_s=duration)

            lease.success()
            try:
                text = (response.json().get("text") or "").strip()
            except ValueError:
                log.error("stt returned unreadable JSON")
                return Transcription(text="", duration_s=duration)

            latency_ms = (time.perf_counter() - started) * 1000
            log.info("stt %.1fs audio -> %d chars in %.0fms (%s)",
                     duration, len(text), latency_ms, lease.label)
            return Transcription(
                text=_strip_hallucinations(text),
                duration_s=duration,
                latency_ms=latency_ms,
                key_label=lease.label,
            )

        log.error("every STT key is cooling down")
        return Transcription(text="", duration_s=duration)


# Whisper emits these for near-silence or background noise. They are not
# things the user said, and passing them to the LLM produces confident replies
# to nothing at all.
_HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "you", ".", "..", "...",
    "thank you for watching.", "bye.", "[blank_audio]", "[silence]",
    "subtitles by the amara.org community", "please subscribe.",
}


def _strip_hallucinations(text: str) -> str:
    if text.strip().lower().strip('"') in _HALLUCINATIONS:
        log.debug("discarded a known Whisper silence artefact")
        return ""
    return text
