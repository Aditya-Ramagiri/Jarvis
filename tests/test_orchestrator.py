"""End-to-end turn logic, with every external service faked.

These exercise the part of the orchestrator that has no hardware in it: the
tool-calling loop, the permission gate, memory writes and the interruption
path. Audio is covered by the helpers' own tests; what matters here is that a
turn does the right sequence of things.
"""

from __future__ import annotations

import copy

import pytest

from adrien.config import DEFAULT_SETTINGS, Settings
from adrien.core.conversation import WindowState
from adrien.core.llm_types import AllProvidersFailed, ChatResult, Message, ToolCall
from adrien.core.orchestrator import ALL_PROVIDERS_DOWN, Orchestrator
from adrien.memory.manager import MemoryManager
from adrien.memory.structured_store import StructuredStore
from adrien.memory.summarizer import Summarizer
from adrien.memory.vector_store import KeywordVectorStore
from adrien.tools.registry import ToolRegistry, ToolResult

pytestmark = pytest.mark.asyncio


class FakeRouter:
    """Replays scripted ChatResults and records what it was asked."""

    def __init__(self, results: list[ChatResult]) -> None:
        self.results = list(results)
        self.calls: list[list[Message]] = []

    async def chat(self, messages, *, tools=None, tier="auto", **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.results:
            return ChatResult(text="(nothing scripted)")
        return self.results.pop(0)

    async def complete(self, prompt, **kwargs) -> str:
        return "completed"

    def status(self) -> dict:
        return {"healthy": True, "providers": []}


class FakeTTS:
    """Records what was spoken instead of making any sound."""

    sample_rate = 24_000

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def stream(self, text):
        self.spoken.append(text)
        yield b"\x00\x00"

    async def stream_sentences(self, text):
        self.spoken.append(text)
        yield b"\x00\x00"

    async def synthesize(self, text) -> bytes:
        self.spoken.append(text)
        return b"\x00\x00"


class FakeTranscriber:
    def __init__(self, text: str = "") -> None:
        self.text = text

    async def transcribe(self, pcm, **kwargs):
        from adrien.core.stt import Transcription

        return Transcription(text=self.text)


def build_orchestrator(tmp_path, results, registry=None, **setting_overrides):
    settings = Settings(copy.deepcopy(DEFAULT_SETTINGS))
    for dotted, value in setting_overrides.items():
        settings.set(dotted, value)

    memory = MemoryManager(
        settings=settings,
        store=StructuredStore(tmp_path / "orch.sqlite3"),
        vectors=KeywordVectorStore(),
        summarizer=Summarizer(router=None),
    )
    orchestrator = Orchestrator(
        settings,
        router=FakeRouter(results),
        transcriber=FakeTranscriber(),
        tts=FakeTTS(),
        memory=memory,
        registry=registry if registry is not None else ToolRegistry(),
    )
    # No microphone in tests: playback is faked, so barge-in never fires.
    orchestrator.speaker = FakeSpeaker()
    return orchestrator


class FakeSpeaker:
    """Plays nothing; reports how much of the reply "came out"."""

    def __init__(self, played_ratio: float = 1.0) -> None:
        self.played_ratio = played_ratio
        self.chunks_played = 0

    async def play_stream(self, chunks, should_stop=None, **kwargs) -> float:
        async for _ in chunks:
            self.chunks_played += 1
        return self.played_ratio

    async def play_tone(self, **kwargs) -> None:
        return None

    def stop(self) -> None:
        return None


# -- plain turns ------------------------------------------------------------
async def test_a_simple_turn_answers_and_speaks(tmp_path):
    orchestrator = build_orchestrator(tmp_path, [ChatResult(text="It's 18 degrees.")])
    result = await orchestrator.handle_text("what's the weather")

    assert result.reply == "It's 18 degrees."
    assert orchestrator.tts.spoken == ["It's 18 degrees."]
    assert [m.role for m in orchestrator.conversation.messages] == ["user", "assistant"]


async def test_an_empty_utterance_does_nothing(tmp_path):
    orchestrator = build_orchestrator(tmp_path, [ChatResult(text="unused")])
    assert (await orchestrator.handle_text("   ")).reply == ""
    assert orchestrator.tts.spoken == []


async def test_the_turn_is_written_to_memory(tmp_path):
    orchestrator = build_orchestrator(tmp_path, [ChatResult(text="Done.")])
    await orchestrator.handle_text("tidy up")

    transcript = orchestrator.memory.store.get_transcript(orchestrator.memory.session_id)
    assert [row["role"] for row in transcript] == ["user", "assistant"]


async def test_recalled_memory_is_injected_into_the_prompt(tmp_path):
    from adrien.memory.structured_store import Fact

    orchestrator = build_orchestrator(tmp_path, [ChatResult(text="rhs.raidnxt.com")])
    orchestrator.memory.remember_fact(
        Fact("the minecraft server", "is hosted at", "rhs.raidnxt.com", "gaming")
    )

    await orchestrator.handle_text("what's the minecraft server address")
    prompt = orchestrator.router.calls[0]
    assert any("rhs.raidnxt.com" in (m.content or "") for m in prompt if m.role == "system")


# -- tool calling -----------------------------------------------------------
def registry_with_weather() -> tuple[ToolRegistry, list[dict]]:
    registry = ToolRegistry()
    seen: list[dict] = []

    @registry.tool(category="info")
    def get_weather(location: str = "here") -> ToolResult:
        """Weather for a place.

        Args:
            location: Where.
        """
        seen.append({"location": location})
        return ToolResult.success({"temp": 18}, speak="18 degrees")

    return registry, seen


async def test_a_tool_call_runs_and_its_result_goes_back_to_the_model(tmp_path):
    registry, seen = registry_with_weather()
    call = ToolCall(name="get_weather", arguments={"location": "Dublin"})
    orchestrator = build_orchestrator(
        tmp_path,
        [ChatResult(tool_calls=[call]), ChatResult(text="It's 18 in Dublin.")],
        registry=registry,
    )

    result = await orchestrator.handle_text("weather in Dublin")
    assert seen == [{"location": "Dublin"}]
    assert result.tool_calls == ["get_weather"]
    assert result.reply == "It's 18 in Dublin."

    # The second model call must carry the assistant turn *and* the tool result.
    second = orchestrator.router.calls[1]
    assert [m.role for m in second[-2:]] == ["assistant", "tool"]
    assert "18" in second[-1].content


async def test_multi_step_chains_keep_going(tmp_path):
    """Spec 7.1: several tools in one request, in sequence."""
    registry = ToolRegistry()
    order: list[str] = []

    @registry.tool(category="dev")
    def check_prs() -> ToolResult:
        """List open pull requests."""
        order.append("check_prs")
        return ToolResult.success({"count": 2})

    @registry.tool(category="messaging")
    def notify(message: str) -> ToolResult:
        """Send a summary somewhere.

        Args:
            message: What to send.
        """
        order.append("notify")
        return ToolResult.success()

    orchestrator = build_orchestrator(
        tmp_path,
        [
            ChatResult(tool_calls=[ToolCall(name="check_prs", arguments={})]),
            ChatResult(tool_calls=[ToolCall(name="notify", arguments={"message": "2 open"})]),
            ChatResult(text="Sent you the summary."),
        ],
        registry=registry,
    )

    result = await orchestrator.handle_text("check my PRs then message me the summary")
    assert order == ["check_prs", "notify"]
    assert result.reply == "Sent you the summary."


async def test_a_runaway_chain_is_capped_and_still_answers(tmp_path):
    registry, _ = registry_with_weather()
    call = ToolCall(name="get_weather", arguments={})
    orchestrator = build_orchestrator(
        tmp_path,
        [ChatResult(tool_calls=[call]) for _ in range(6)] + [ChatResult(text="Enough.")],
        registry=registry,
        **{"llm.max_tool_iterations": 3},
    )

    result = await orchestrator.handle_text("loop forever")
    assert len(result.tool_calls) == 3
    assert result.reply  # never leaves the user in silence


async def test_a_failing_tool_is_reported_not_papered_over(tmp_path):
    """Spec 7.9: the model must be told the push failed."""
    registry = ToolRegistry()

    @registry.tool(category="dev")
    def git_push() -> ToolResult:
        """Push to the remote."""
        return ToolResult.failure("the remote rejected it - non-fast-forward")

    orchestrator = build_orchestrator(
        tmp_path,
        [ChatResult(tool_calls=[ToolCall(name="git_push", arguments={})]),
         ChatResult(text="The push was rejected, you need to pull first.")],
        registry=registry,
    )

    await orchestrator.handle_text("push it")
    tool_message = orchestrator.router.calls[1][-1]
    assert '"ok": false' in tool_message.content
    assert "non-fast-forward" in tool_message.content


# -- permissions ------------------------------------------------------------
async def test_a_destructive_tool_asks_before_running(tmp_path):
    registry = ToolRegistry()
    sent: list[str] = []

    @registry.tool(category="messaging", irreversible=True,
                   confirm="Send {recipient}: {message}. Send it?")
    def send_discord_message(recipient: str, message: str) -> ToolResult:
        """Send a Discord message.

        Args:
            recipient: Who to.
            message: What to say.
        """
        sent.append(message)
        return ToolResult.success()

    orchestrator = build_orchestrator(
        tmp_path,
        [ChatResult(tool_calls=[ToolCall(
            name="send_discord_message",
            arguments={"recipient": "John", "message": "running late"})]),
         ChatResult(text="Sent.")],
        registry=registry,
    )

    asked: list[str] = []

    async def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return True

    orchestrator.permissions.confirm_fn = confirm
    await orchestrator.handle_text("tell John I'm running late on discord")

    assert asked == ["Send John: running late. Send it?"]
    assert sent == ["running late"]


async def test_a_refused_tool_does_not_run_and_the_model_is_told(tmp_path):
    registry = ToolRegistry()
    sent: list[str] = []

    @registry.tool(category="messaging", irreversible=True, confirm="Send it?")
    def send_discord_message(recipient: str, message: str) -> ToolResult:
        """Send a Discord message.

        Args:
            recipient: Who to.
            message: What to say.
        """
        sent.append(message)
        return ToolResult.success()

    orchestrator = build_orchestrator(
        tmp_path,
        [ChatResult(tool_calls=[ToolCall(
            name="send_discord_message",
            arguments={"recipient": "John", "message": "oops"})]),
         ChatResult(text="Okay, I didn't send it.")],
        registry=registry,
    )

    async def refuse(prompt: str) -> bool:
        return False

    orchestrator.permissions.confirm_fn = refuse
    result = await orchestrator.handle_text("message John")

    assert sent == []
    assert "did not confirm" in orchestrator.router.calls[1][-1].content
    assert result.reply == "Okay, I didn't send it."


# -- failure handling -------------------------------------------------------
async def test_every_provider_failing_is_the_one_thing_adrien_says_aloud(tmp_path):
    """Spec 4.5."""
    class DeadRouter(FakeRouter):
        async def chat(self, messages, **kwargs):
            raise AllProvidersFailed(["groq#1: 429", "gemini#1: 429"])

    orchestrator = build_orchestrator(tmp_path, [])
    orchestrator.router = DeadRouter([])

    result = await orchestrator.handle_text("what's the weather")
    assert result.reply == ALL_PROVIDERS_DOWN
    assert orchestrator.tts.spoken == [ALL_PROVIDERS_DOWN]


async def test_an_unexpected_error_does_not_leave_the_user_in_silence(tmp_path):
    class BrokenRouter(FakeRouter):
        async def chat(self, messages, **kwargs):
            raise RuntimeError("something odd")

    orchestrator = build_orchestrator(tmp_path, [])
    orchestrator.router = BrokenRouter([])

    result = await orchestrator.handle_text("hello")
    assert result.error == "something odd"
    assert orchestrator.tts.spoken


# -- interruption and resumption --------------------------------------------
async def test_an_interrupted_reply_can_be_resumed_without_another_model_call(tmp_path):
    long_reply = ("There are three open pull requests, two of them are failing CI, "
                  "and the third one is waiting on a review from Sam")
    orchestrator = build_orchestrator(tmp_path, [ChatResult(text=long_reply)])
    orchestrator.speaker = FakeSpeaker(played_ratio=0.4)

    first = await orchestrator.handle_text("what's on my PRs")
    assert first.interrupted is True

    calls_before = len(orchestrator.router.calls)
    second = await orchestrator.handle_text("keep going")

    assert "waiting on a review from Sam" in second.reply
    assert len(orchestrator.router.calls) == calls_before, "resuming must not call the model"


async def test_keep_going_with_nothing_pending_falls_through_to_the_model(tmp_path):
    orchestrator = build_orchestrator(tmp_path, [ChatResult(text="Going where?")])
    result = await orchestrator.handle_text("keep going")
    assert result.reply == "Going where?"


async def test_a_completed_reply_leaves_nothing_to_resume(tmp_path):
    orchestrator = build_orchestrator(tmp_path, [ChatResult(text="All good.")])
    await orchestrator.handle_text("status")
    assert orchestrator.conversation.interrupted is None


# -- status -----------------------------------------------------------------
async def test_status_reports_the_running_state(tmp_path):
    orchestrator = build_orchestrator(tmp_path, [])
    status = orchestrator.status()
    assert status["state"] == WindowState.IDLE.value
    assert status["providers"]["healthy"] is True
    assert "memory" in status
