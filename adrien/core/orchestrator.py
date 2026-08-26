"""The main loop: wake word -> STT -> LLM + tools -> TTS (spec 5, 11).

    ┌ passive ─────────────────────────────────────────────────────┐
    │ wake word (local, free)                                      │
    └──────┬───────────────────────────────────────────────────────┘
           │ detected -> short tone
    ┌──────▼───────────────────────────────────────────────────────┐
    │ record utterance (VAD endpointing)                           │
    │ transcribe (Whisper large-v3)                                │
    │ recall relevant memory                                       │
    │ LLM turn, with tools; each destructive call gated on a yes   │
    │ speak the reply, listening for barge-in throughout           │
    └──────┬───────────────────────────────────────────────────────┘
           │ follow-up window (a few seconds, no wake word needed)
           └─ nothing said -> back to passive

The same `handle_text` entry point serves the WebSocket clients, so a phone
gets identical behaviour without a second copy of the logic - the only
difference is where the audio comes from and where it goes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from adrien.config import Settings, settings as global_settings
from adrien.core.audio import (
    AudioConfig,
    BargeInMonitor,
    MicrophoneStream,
    Speaker,
    VoiceActivityDetector,
    record_utterance,
)
from adrien.core.conversation import (
    Conversation,
    WindowState,
    is_continuation_request,
    summarise_history,
)
from adrien.core.llm_router import LLMRouter
from adrien.core.llm_types import AllProvidersFailed, Message, ToolCall
from adrien.core.stt import Transcriber
from adrien.core.tts import TextToSpeech
from adrien.core.wake_word import WakeWordDetector
from adrien.logging_setup import get_logger
from adrien.memory.manager import MemoryManager
from adrien.tools.permissions import PermissionManager, interpret_confirmation
from adrien.tools.registry import ToolRegistry, ToolResult, load_all_tools

log = get_logger(__name__)

# Spec 4.5: the one plumbing problem Adrien is allowed to mention out loud.
ALL_PROVIDERS_DOWN = "I'm having trouble connecting right now."

# A conversation with no activity for this long is finished, and gets
# summarised into long-term memory.
SESSION_IDLE_SECONDS = 180.0


@dataclass
class TurnResult:
    """Everything one turn produced - used by the WebSocket server too."""

    reply: str = ""
    tool_calls: list[str] = field(default_factory=list)
    error: str = ""
    interrupted: bool = False
    latency_ms: float = 0.0


class Orchestrator:
    """Owns the conversation. One instance per running service."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        router: LLMRouter | None = None,
        transcriber: Transcriber | None = None,
        tts: TextToSpeech | None = None,
        memory: MemoryManager | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings or global_settings()
        self.router = router or LLMRouter(self.settings)
        self.transcriber = transcriber or Transcriber(self.settings)
        self.tts = tts or TextToSpeech(self.settings)
        self.memory = memory or MemoryManager(self.settings)
        self.registry = registry or load_all_tools()
        self.permissions = PermissionManager(self.settings, confirm_fn=self._confirm_by_voice)
        self.conversation = Conversation.from_settings(self.settings)

        self.audio_config = AudioConfig.from_settings(self.settings)
        self.mic: MicrophoneStream | None = None
        self.speaker = Speaker(self.audio_config, sample_rate=self.tts.sample_rate)
        self.vad = VoiceActivityDetector(
            aggressiveness=self.audio_config.vad_aggressiveness,
            frame_ms=self.audio_config.frame_ms,
        )
        self.wake = WakeWordDetector.from_settings(self.settings)

        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Set while a remote client owns the conversation, so the Mac's own mic
        # does not answer a question asked from the phone.
        self._remote_lock = asyncio.Lock()
        self.on_state_change: Callable[[WindowState], None] | None = None

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        persona = str(self.settings.get("assistant.persona", ""))
        clock = time.strftime("%A %d %B %Y, %H:%M")
        return (
            f"{persona}\n\n"
            f"The current time is {clock}. "
            "You can use tools; prefer doing the thing over describing it. "
            "If a tool fails, say what failed in one short sentence - never claim "
            "something worked when the tool said it did not."
        )

    # ------------------------------------------------------------------
    # One turn
    # ------------------------------------------------------------------
    async def handle_text(self, text: str, *, speak: bool = True,
                          source: str = "mac") -> TurnResult:
        """Process one user utterance end to end.

        Shared by the microphone loop and every WebSocket client.
        """
        started = time.perf_counter()
        text = text.strip()
        if not text:
            return TurnResult()

        if not self.memory.session_id:
            self.memory.start_session(source)

        # "Keep going" is answered from short-term state, with no model call:
        # the reply already exists, and re-deriving it would both cost latency
        # and risk saying something different the second time.
        if is_continuation_request(text):
            resumed = self.conversation.take_resume_text()
            if resumed:
                log.info("resuming an interrupted reply")
                self.conversation.add_user(text)
                self.conversation.add_assistant(Message.assistant(resumed))
                if speak:
                    await self.speak(resumed)
                return TurnResult(reply=resumed,
                                  latency_ms=(time.perf_counter() - started) * 1000)
            # Nothing to resume: fall through and let the model answer.

        self.conversation.add_user(text)
        self.memory.record_turn("user", text)

        recalled = self.memory.recall(text).as_prompt()
        tool_schemas = self.registry.schemas()
        max_iterations = int(self.settings.get("llm.max_tool_iterations", 5))
        called: list[str] = []

        self._set_state(WindowState.THINKING)

        for iteration in range(max_iterations):
            messages = self.conversation.build_messages(self.system_prompt(), recalled)
            try:
                result = await self.router.chat(messages, tools=tool_schemas)
            except AllProvidersFailed as exc:
                log.error("no provider could answer: %s", exc)
                if speak:
                    await self.speak(ALL_PROVIDERS_DOWN)
                return TurnResult(reply=ALL_PROVIDERS_DOWN, error=str(exc))
            except Exception as exc:
                log.exception("the turn failed")
                message = "Something went wrong on my end."
                if speak:
                    await self.speak(message)
                return TurnResult(reply=message, error=str(exc))

            if not result.wants_tools:
                reply = result.text.strip()
                self.conversation.add_assistant(Message.assistant(reply))
                self.memory.record_turn("assistant", reply)
                interrupted = False
                if speak and reply:
                    interrupted = await self.speak(reply)
                return TurnResult(
                    reply=reply,
                    tool_calls=called,
                    interrupted=interrupted,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            # Tool round. The assistant turn carrying the calls must be in the
            # history before the results, or the provider rejects the sequence.
            self.conversation.add_assistant(
                Message.assistant(result.text, tool_calls=result.tool_calls)
            )
            for call in result.tool_calls:
                called.append(call.name)
                tool_result = await self._run_tool(call)
                self.conversation.add_tool_result(
                    Message.tool_result(call, tool_result.to_json())
                )
                self.memory.record_turn(
                    "tool", f"{call.name}: {tool_result.to_json(max_chars=600)}"
                )

            if iteration == max_iterations - 1:
                log.warning("hit the tool-chain limit after %d rounds", max_iterations)

        # Ran out of iterations mid-chain: ask for a plain answer with the
        # results already gathered rather than leaving the user with silence.
        fallback = "That turned into more steps than I could finish. Want me to keep going?"
        try:
            messages = self.conversation.build_messages(self.system_prompt(), recalled)
            # No tools this time: the point is to get words out of the model,
            # and offering tools is what got us into the loop.
            final = await self.router.chat(messages, tier="smart")
            reply = final.text.strip() or fallback
        except Exception as exc:
            log.warning("could not wrap up the tool chain: %s", exc)
            reply = fallback

        self.conversation.add_assistant(Message.assistant(reply))
        self.memory.record_turn("assistant", reply)
        interrupted = await self.speak(reply) if speak and reply else False
        return TurnResult(reply=reply, tool_calls=called, interrupted=interrupted,
                          latency_ms=(time.perf_counter() - started) * 1000)

    async def _run_tool(self, call: ToolCall) -> ToolResult:
        """Permission gate, then execution (spec section 9)."""
        spec = self.registry.get(call.name)
        if spec is None:
            return await self.registry.execute(call)

        decision = await self.permissions.check(spec, call.arguments or {})
        if not decision.allowed:
            log.info("%s blocked: %s", call.name, decision.reason)
            return PermissionManager.denial_result(decision)
        return await self.registry.execute(call)

    # ------------------------------------------------------------------
    # Speaking
    # ------------------------------------------------------------------
    async def speak(self, text: str) -> bool:
        """Say `text`, watching for barge-in. True if the user cut in."""
        if not text.strip():
            return False

        self._set_state(WindowState.SPEAKING)
        barge_in_enabled = bool(self.settings.get("conversation.barge_in_enabled", True))
        monitor: BargeInMonitor | None = None

        if barge_in_enabled and self.mic is not None:
            monitor = BargeInMonitor(
                self.mic,
                self.vad,
                speech_frames=int(self.settings.get("conversation.barge_in_speech_frames", 6)),
            )
            monitor.reset()
            self.mic.flush()

        chunks = (
            self.tts.stream_sentences(text) if self.settings.get("tts.stream", True)
            else self.tts.stream(text)
        )
        try:
            played_ratio = await self.speaker.play_stream(
                chunks, should_stop=(monitor.check if monitor else None)
            )
        except Exception as exc:
            log.error("playback failed: %s", exc)
            return False

        interrupted = played_ratio < 0.995
        if interrupted:
            self.conversation.note_interruption(text, played_ratio)
        else:
            self.conversation.clear_interruption()

        if self.mic is not None:
            # Whatever the mic captured while Adrien was talking is mostly
            # Adrien; keeping it would feed its own voice into the next turn.
            self.mic.flush()
        return interrupted

    # ------------------------------------------------------------------
    # Confirmation (spoken yes/no)
    # ------------------------------------------------------------------
    async def _confirm_by_voice(self, prompt: str) -> bool:
        """Ask the user out loud and wait for a real yes."""
        if self.mic is None:
            return False

        previous_state = self.conversation.state
        self._set_state(WindowState.CONFIRMING)
        try:
            await self.speak(prompt)
            for attempt in range(2):
                utterance = await asyncio.to_thread(
                    record_utterance,
                    self.mic,
                    self.vad,
                    silence_seconds=0.7,
                    max_seconds=8.0,
                    start_timeout=6.0,
                )
                if utterance.is_empty:
                    log.info("no answer to the confirmation; treating it as no")
                    return False

                transcription = await self.transcriber.transcribe(utterance.pcm)
                answer = interpret_confirmation(transcription.text)
                log.info("confirmation heard %r -> %s", transcription.text, answer)
                if answer is not None:
                    return answer
                if attempt == 0:
                    await self.speak("Sorry, was that a yes?")
            return False
        finally:
            self._set_state(previous_state)

    # ------------------------------------------------------------------
    # The microphone loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Run until stopped. This is what the launchd service starts."""
        self._running = True
        log.info("Adrien starting")

        self.wake.load()
        if self.wake.using_fallback:
            log.warning(
                "wake word is '%s', not 'Adrien' - see docs/WAKE_WORD.md to train "
                "the real one", self.wake.label,
            )

        self.mic = MicrophoneStream(self.audio_config).start()
        self._tasks.append(asyncio.create_task(self._reminder_loop()))
        self._tasks.append(asyncio.create_task(self._session_idle_loop()))

        try:
            while self._running:
                await self._wait_for_wake_word()
                if not self._running:
                    break
                await self._converse()
        finally:
            await self.shutdown()

    async def _wait_for_wake_word(self) -> None:
        """Passive listening. Cheap enough to run forever (spec 5.1)."""
        self._set_state(WindowState.IDLE)
        self.wake.reset()
        while self._running:
            if self._remote_lock.locked():
                # A phone is mid-conversation; stay out of the way.
                await asyncio.sleep(0.2)
                continue
            frame = await self.mic.read_async(timeout=0.5)
            if frame is None:
                continue
            if await asyncio.to_thread(self.wake.process, frame):
                return

    async def _converse(self) -> None:
        """One wake-word-initiated interaction, plus its follow-up window."""
        async with self._remote_lock:
            await self._acknowledge()

            first = True
            while self._running:
                self._set_state(WindowState.LISTENING if first else WindowState.FOLLOW_UP)
                utterance = await asyncio.to_thread(
                    record_utterance,
                    self.mic,
                    self.vad,
                    silence_seconds=float(
                        self.settings.get("conversation.endpoint_silence_seconds", 1.0)),
                    max_seconds=float(
                        self.settings.get("conversation.max_utterance_seconds", 30.0)),
                    start_timeout=(
                        6.0 if first
                        else float(self.settings.get(
                            "conversation.follow_up_window_seconds", 6.0))
                    ),
                )
                if utterance.is_empty:
                    # Spec 5.2: nothing said in the window, so stop listening.
                    # Adrien does not prompt or fill the silence.
                    log.debug("follow-up window closed with nothing said")
                    self.conversation.close_follow_up()
                    return

                min_seconds = float(self.settings.get("conversation.min_utterance_seconds", 0.35))
                if utterance.duration_s < min_seconds:
                    log.debug("ignoring %.2fs of noise", utterance.duration_s)
                    if first:
                        return
                    continue

                transcription = await self.transcriber.transcribe(
                    utterance.pcm,
                    prompt=summarise_history(self.conversation.messages),
                )
                if transcription.is_empty:
                    log.debug("nothing transcribable")
                    if first:
                        return
                    continue

                log.info("heard: %s", transcription.text)
                await self.handle_text(transcription.text, speak=True, source="mac")
                self.conversation.open_follow_up()
                first = False

    async def _acknowledge(self) -> None:
        """The brief wake acknowledgement - a tone, not a phrase (spec 5.1)."""
        style = str(self.settings.get("wake_word.acknowledgement", "tone"))
        if style == "none":
            return
        if style == "tone":
            try:
                await self.speaker.play_tone()
            except Exception as exc:  # pragma: no cover - no output device
                log.debug("could not play the wake tone: %s", exc)
            return
        await self.speak(style if len(style) < 24 else "yeah?")

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------
    async def _reminder_loop(self, interval: float = 15.0) -> None:
        """Speak reminders and timers when they come due."""
        while self._running:
            await asyncio.sleep(interval)
            try:
                due = self.memory.store.due_reminders()
            except Exception as exc:  # pragma: no cover
                log.error("could not read reminders: %s", exc)
                continue

            for row in due:
                self.memory.store.mark_reminder_fired(row["id"])
                # Wait for a gap rather than talking over the user.
                while self.conversation.state in (WindowState.LISTENING,
                                                  WindowState.SPEAKING,
                                                  WindowState.CONFIRMING):
                    await asyncio.sleep(1.0)
                text = ("Your timer's up." if row["kind"] == "timer" and row["text"] == "timer"
                        else f"Reminder: {row['text']}")
                log.info("firing %s %s", row["kind"], row["id"][:8])
                await self.speak(text)

    async def _session_idle_loop(self, interval: float = 30.0) -> None:
        """Close and summarise a conversation once it has gone quiet."""
        while self._running:
            await asyncio.sleep(interval)
            if not self.memory.session_id:
                continue
            if self.conversation.idle_seconds() < SESSION_IDLE_SECONDS:
                continue
            if self.conversation.state is not WindowState.IDLE:
                continue
            log.info("session idle; summarising it into long-term memory")
            try:
                await self.memory.end_session()
            except Exception as exc:  # pragma: no cover
                log.error("session summarisation failed: %s", exc)
            self.conversation.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _set_state(self, state: WindowState) -> None:
        if self.conversation.state is state:
            return
        self.conversation.state = state
        log.debug("state -> %s", state.value)
        if self.on_state_change:
            try:
                self.on_state_change(state)
            except Exception:  # pragma: no cover - a menu bar callback
                log.debug("state callback raised", exc_info=True)

    def stop(self) -> None:
        log.info("Adrien stopping")
        self._running = False
        self.speaker.stop()

    async def shutdown(self) -> None:
        """Clean up, and get the last conversation into long-term memory."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        try:
            await asyncio.wait_for(self.memory.end_session(), timeout=30)
        except Exception as exc:
            # A slow or failed summary must never stop the service exiting.
            log.warning("could not finish the closing summary: %s", exc)

        if self.mic is not None:
            self.mic.stop()
            self.mic = None

        from adrien.core.http import close_client

        await close_client()
        self.memory.close()
        log.info("Adrien stopped")

    # ------------------------------------------------------------------
    # Used by the WebSocket server
    # ------------------------------------------------------------------
    async def handle_remote_audio(self, pcm: bytes, sample_rate: int = 16_000,
                                  source: str = "client") -> tuple[TurnResult, bytes]:
        """Full turn for a client: audio in, reply text and reply audio out.

        Takes the same lock the microphone loop uses, so a question asked on
        the phone is not answered simultaneously by the Mac.
        """
        async with self._remote_lock:
            transcription = await self.transcriber.transcribe(
                pcm, sample_rate=sample_rate,
                prompt=summarise_history(self.conversation.messages),
            )
            if transcription.is_empty:
                return TurnResult(error="nothing transcribable"), b""

            log.info("[%s] heard: %s", source, transcription.text)
            result = await self.handle_text(transcription.text, speak=False, source=source)
            audio = await self.tts.synthesize(result.reply) if result.reply else b""
            return result, audio

    async def stream_reply_audio(self, text: str) -> AsyncIterator[bytes]:
        """PCM for a reply, for clients that want it chunk by chunk."""
        async for chunk in self.tts.stream_sentences(text):
            yield chunk

    def status(self) -> dict[str, Any]:
        """Health snapshot for the menu bar, the CLI and the clients."""
        return {
            "running": self._running,
            "state": self.conversation.state.value,
            "wake_word": self.wake.label,
            "wake_word_is_fallback": self.wake.using_fallback,
            "tools": len(self.registry),
            "providers": self.router.status(),
            "memory": self.memory.stats(),
            "session": self.memory.session_id[:8] if self.memory.session_id else None,
        }


async def _amain(confirm_fn: Callable[[str], Awaitable[bool]] | None = None) -> None:
    orchestrator = Orchestrator()
    if confirm_fn is not None:
        orchestrator.permissions.confirm_fn = confirm_fn
    await orchestrator.run()


def main() -> None:  # pragma: no cover - process entry point
    from adrien.logging_setup import setup_logging

    setup_logging()
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        log.info("interrupted")


if __name__ == "__main__":  # pragma: no cover
    main()
