"""Pure-logic pieces of the speech path (no audio hardware needed)."""

from __future__ import annotations

import wave

from adrien.core.audio import SAMPLE_RATE, frame_rms, rms_to_db
from adrien.core.stt import _strip_hallucinations, pcm_to_wav
from adrien.core.tts import _clean_for_speech, split_sentences


def silence(samples: int = 480) -> bytes:
    return b"\x00\x00" * samples


def loud(samples: int = 480, amplitude: int = 12000) -> bytes:
    return amplitude.to_bytes(2, "little", signed=True) * samples


def test_frame_rms_distinguishes_silence_from_speech():
    assert frame_rms(silence()) == 0.0
    assert frame_rms(loud()) > 0.3
    assert rms_to_db(frame_rms(silence())) == -120.0


def test_frame_rms_survives_an_odd_trailing_byte():
    assert frame_rms(loud() + b"\x01") > 0.3
    assert frame_rms(b"") == 0.0
    assert frame_rms(b"\x01") == 0.0


def test_pcm_to_wav_produces_a_readable_container():
    pcm = loud(SAMPLE_RATE)  # one second
    with wave.open(__import__("io").BytesIO(pcm_to_wav(pcm)), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getnframes() == SAMPLE_RATE


def test_whisper_silence_artefacts_are_discarded():
    assert _strip_hallucinations("Thank you.") == ""
    assert _strip_hallucinations("Thanks for watching!") == ""
    assert _strip_hallucinations("thank you for the update") == "thank you for the update"


def test_markdown_is_stripped_before_speaking():
    spoken = _clean_for_speech(
        "## Status\n- **three** PRs are open\n- run `git status`\n"
        "See [the docs](https://example.com) 🎉"
    )
    assert "#" not in spoken and "*" not in spoken and "`" not in spoken
    assert "🎉" not in spoken
    assert "three PRs are open" in spoken
    assert "the docs" in spoken and "example.com" not in spoken


def test_code_blocks_do_not_get_read_aloud():
    spoken = _clean_for_speech("Here it is:\n```python\nprint('hi')\n```\nthat's all")
    assert "print" not in spoken
    assert "that's all" in spoken


def test_sentence_splitting_merges_short_fragments():
    assert split_sentences("Done. Anything else?") == ["Done. Anything else?"]


def test_sentence_splitting_keeps_long_sentences_separate():
    long_one = "This is a reasonably long first sentence that stands on its own well."
    long_two = "And this is a second sentence that also carries plenty of its own weight."
    assert split_sentences(f"{long_one} {long_two}") == [long_one, long_two]


def test_sentence_splitting_of_empty_text():
    assert split_sentences("   ") == []
