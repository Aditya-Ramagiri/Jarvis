"""Text to speech via Fish Audio, with the same key rotation as everything else.

Two decisions worth stating up front:

**PCM, not MP3.** Fish Audio can return mp3/opus/wav, but asking for raw PCM
means the bytes coming off the socket can go straight into the output device.
No decoder dependency, no buffering a whole file before the first word, and -
the one that matters for spec 5.3 - playback can be cut mid-word, because we
are always holding at most 20 ms of audio. An mp3 path would have to decode a
frame at a time or hand the file to `afplay`, which cannot be interrupted
cleanly.

**Rotation only before the first byte.** Once audio is playing, a mid-stream
failure cannot be retried invisibly: the user has already heard half a
sentence. Failures before the first byte rotate keys exactly like the LLM path
(spec 4.7); failures after it are logged and the stream ends, and the
orchestrator treats the reply as interrupted.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator

from adrien.config import Settings, env_key_pool, env_str, settings as global_settings
from adrien.core.http import get_client, parse_retry_after
from adrien.core.keypool import KeyPool
from adrien.logging_setup import get_logger

log = get_logger(__name__)

API_URL = "https://api.fish.audio/v1/tts"

# 24 kHz is the sweet spot for a speaking voice: indistinguishable from 44.1
# through a laptop speaker, and 45% fewer bytes to move per second.
OUTPUT_SAMPLE_RATE = 24_000

# Fish Audio's TTS backend model (distinct from the *voice*, which is
# `reference_id`). s1 is the current low-latency model.
BACKEND_MODEL = "s1"


class TextToSpeech:
    """Streams spoken PCM for a piece of text."""

    def __init__(self, settings: Settings | None = None, pool: KeyPool | None = None) -> None:
        self.settings = settings or global_settings()
        keys_cfg = self.settings.get("keys", {}) or {}
        self.pool = pool or KeyPool(
            "fish",
            env_key_pool("FISH_AUDIO_API_KEY"),
            cooldown_seconds=float(keys_cfg.get("cooldown_seconds", 60.0)),
            failure_cooldown_seconds=float(keys_cfg.get("failure_cooldown_seconds", 15.0)),
        )
        self.voice_id = env_str("FISH_AUDIO_VOICE_ID")
        self.backup_voice_id = env_str("FISH_AUDIO_VOICE_ID_BACKUP")
        self.timeout = float(keys_cfg.get("request_timeout_seconds", 25.0))
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.latency_mode = str(self.settings.get("tts.latency", "balanced"))
        if not self.voice_id:
            log.warning("FISH_AUDIO_VOICE_ID is unset; Fish Audio will use its default voice")

    # -- request ----------------------------------------------------------
    def _payload(self, text: str, voice_id: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "text": text,
            "format": "pcm",
            "sample_rate": self.sample_rate,
            # "balanced" trades a little quality for a much earlier first byte,
            # which is what a conversation actually feels like.
            "latency": self.latency_mode,
            "normalize": True,
            # Smaller chunks -> the first audio arrives sooner.
            "chunk_length": 200,
        }
        if voice_id:
            payload["reference_id"] = voice_id
        return payload

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM chunks for `text` as they arrive.

        Yields nothing (rather than raising) when TTS is unavailable: the
        orchestrator's fallback is to stay quiet and log, not to crash the
        service mid-conversation.
        """
        import httpx

        text = _clean_for_speech(text)
        if not text:
            return
        if not self.pool.configured:
            log.error("no Fish Audio keys configured")
            return

        voices = [voice for voice in (self.voice_id, self.backup_voice_id) if voice] or [""]
        started = time.perf_counter()
        first_byte_logged = False

        for voice_index, voice in enumerate(voices):
            for lease in self.pool.leases():
                try:
                    async with get_client().stream(
                        "POST",
                        API_URL,
                        json=self._payload(text, voice),
                        headers={
                            "authorization": f"Bearer {lease.key}",
                            "model": BACKEND_MODEL,
                        },
                        timeout=self.timeout,
                    ) as response:
                        if response.status_code == 429:
                            await response.aread()
                            lease.rate_limited(parse_retry_after(response.headers))
                            continue
                        if response.status_code in (401, 403) or response.status_code >= 500:
                            await response.aread()
                            log.warning("tts %s returned %d", lease.label, response.status_code)
                            lease.failed()
                            continue
                        if response.status_code >= 400:
                            body = (await response.aread())[:200]
                            lease.success()  # key is fine, request is not
                            log.error("tts rejected the request (%d): %s",
                                      response.status_code, body.decode("utf-8", "replace"))
                            # A bad voice id is worth one retry on the backup.
                            if voice_index + 1 < len(voices):
                                break
                            return

                        lease.success()
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            if not first_byte_logged:
                                log.info("tts first byte in %.0fms (%s)",
                                         (time.perf_counter() - started) * 1000, lease.label)
                                first_byte_logged = True
                            yield chunk
                        return

                except httpx.HTTPError as exc:
                    if first_byte_logged:
                        # Already speaking; nothing to fail over to.
                        log.error("tts stream broke mid-sentence: %s", type(exc).__name__)
                        return
                    log.warning("tts transport error on %s: %s", lease.label, type(exc).__name__)
                    lease.failed()
                    continue

        log.error("every Fish Audio key failed; staying silent")

    async def synthesize(self, text: str) -> bytes:
        """Whole utterance as one PCM buffer - used by the WebSocket server,
        which sends complete audio messages rather than a device stream."""
        chunks = [chunk async for chunk in self.stream(text)]
        return b"".join(chunks)

    async def stream_sentences(self, text: str) -> AsyncIterator[bytes]:
        """Stream sentence by sentence, overlapping synthesis with playback.

        For a long reply this cuts perceived latency substantially: sentence
        two is being synthesised while sentence one is still being spoken.
        Short replies go through in one request, since splitting them would
        only add a round trip.
        """
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            async for chunk in self.stream(text):
                yield chunk
            return

        pending: asyncio.Task | None = None

        async def collect(sentence: str) -> bytes:
            return b"".join([chunk async for chunk in self.stream(sentence)])

        for index, sentence in enumerate(sentences):
            if pending is None:
                pending = asyncio.create_task(collect(sentence))
            audio = await pending
            pending = (
                asyncio.create_task(collect(sentences[index + 1]))
                if index + 1 < len(sentences)
                else None
            )
            if audio:
                yield audio


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, *, max_chars: int = 240) -> list[str]:
    """Split into speakable chunks, merging fragments too short to be worth
    their own request."""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) + len(part) + 1 <= max_chars and len(merged[-1]) < 60:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


# Adrien's replies are spoken, so anything that only makes sense on a screen is
# noise. The persona prompt asks the model to avoid markdown; this is the
# safety net for when it slips.
_MARKDOWN_PATTERNS = [
    (re.compile(r"```[\s\S]*?```"), " "),          # fenced code blocks
    (re.compile(r"`([^`]*)`"), r"\1"),             # inline code
    (re.compile(r"\*\*([^*]*)\*\*"), r"\1"),       # bold
    (re.compile(r"(?<!\w)\*([^*]+)\*(?!\w)"), r"\1"),  # italics
    (re.compile(r"^#{1,6}\s*", re.M), ""),         # headings
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),       # bullets
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # links -> link text
    (re.compile(r"[\U0001F300-\U0001FAFF☀-➿]"), ""),  # emoji
]


def _clean_for_speech(text: str) -> str:
    cleaned = text.strip()
    for pattern, replacement in _MARKDOWN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
