"""Follow-up window and interruption memory (spec 5.2, 5.3)."""

from __future__ import annotations

import time

from adrien.core.conversation import (
    Conversation,
    InterruptedReply,
    WindowState,
    is_continuation_request,
    summarise_history,
)
from adrien.core.llm_types import Message, ToolCall


# -- continuation detection -------------------------------------------------
def test_continuation_phrases_are_recognised():
    for phrase in ("keep going", "carry on", "go on", "continue",
                   "finish that", "what were you saying", "you were saying",
                   "Keep going.", "please keep going"):
        assert is_continuation_request(phrase), phrase


def test_a_new_request_is_not_mistaken_for_a_continuation():
    for phrase in ("finish the deploy script", "keep going to the shop",
                   "what were you saying about the server yesterday",
                   "continue the download when it's done", "what's the weather"):
        assert not is_continuation_request(phrase), phrase


# -- interrupted replies ----------------------------------------------------
def test_an_interrupted_reply_splits_at_a_word_boundary():
    reply = InterruptedReply(full_text="one two three four five six seven eight",
                             spoken_ratio=0.5)
    assert reply.spoken == "one two three four"
    assert reply.remaining == "five six seven eight"


def test_resuming_overlaps_a_couple_of_words():
    reply = InterruptedReply(
        full_text="There are three open pull requests and two of them are failing CI",
        spoken_ratio=0.5,
    )
    resumed = reply.resume_text()
    # Picks up mid-thought rather than restarting the sentence.
    assert resumed.endswith("two of them are failing CI")
    assert not resumed.startswith("There are three")


def test_a_reply_that_finished_has_nothing_to_resume():
    conversation = Conversation()
    assert conversation.note_interruption("all done", 1.0) is None
    assert conversation.take_resume_text() is None


def test_barely_interrupted_replies_are_not_worth_resuming():
    conversation = Conversation()
    conversation.note_interruption("the build passed", 0.9)
    # Two words left is not a thread worth picking up.
    assert conversation.take_resume_text() is None


def test_a_meaningfully_interrupted_reply_can_be_resumed():
    conversation = Conversation()
    conversation.note_interruption(
        "There are three open pull requests and two of them are failing CI right now", 0.4
    )
    resumed = conversation.take_resume_text()
    assert resumed and "failing CI right now" in resumed


def test_resuming_consumes_the_pending_reply():
    conversation = Conversation()
    conversation.note_interruption("a b c d e f g h i j k l", 0.25)
    assert conversation.take_resume_text() is not None
    assert conversation.take_resume_text() is None


def test_a_stale_interruption_is_not_resumed():
    conversation = Conversation()
    conversation.note_interruption("a b c d e f g h i j k l", 0.25)
    conversation.interrupted.at = time.time() - 600
    assert conversation.take_resume_text() is None


# -- the follow-up window ---------------------------------------------------
def test_the_follow_up_window_opens_and_expires():
    conversation = Conversation(follow_up_seconds=0.05)
    conversation.open_follow_up()
    assert conversation.state is WindowState.FOLLOW_UP
    assert conversation.follow_up_open

    conversation.last_activity -= 1.0
    assert not conversation.follow_up_open
    assert conversation.follow_up_remaining() == 0.0


def test_closing_the_window_returns_to_passive_listening():
    conversation = Conversation()
    conversation.open_follow_up()
    conversation.close_follow_up()
    # Spec 5.2: not a chat mode - it goes quiet rather than prompting.
    assert conversation.state is WindowState.IDLE


def test_no_window_is_open_when_idle():
    assert Conversation().follow_up_remaining() == 0.0


# -- history ----------------------------------------------------------------
def test_the_prompt_starts_with_persona_and_memory():
    conversation = Conversation()
    conversation.add_user("hello")
    messages = conversation.build_messages("You are Adrien.", "The user lives in Dublin.")
    assert [message.role for message in messages] == ["system", "system", "user"]
    assert messages[1].content == "The user lives in Dublin."


def test_memory_is_omitted_when_there_is_none():
    conversation = Conversation()
    conversation.add_user("hello")
    assert len(conversation.build_messages("You are Adrien.")) == 2


def test_history_is_trimmed_without_orphaning_tool_results():
    """A tool message whose assistant turn was trimmed is rejected by the API."""
    conversation = Conversation(history_turns=4)
    call = ToolCall(name="get_weather", arguments={})
    for index in range(6):
        conversation.add_user(f"question {index}")
        conversation.add_assistant(Message.assistant(tool_calls=[call]))
        conversation.add_tool_result(Message.tool_result(call, "18C"))

    conversation.trim()
    assert conversation.messages[0].role == "user"
    for position, message in enumerate(conversation.messages):
        if message.role == "tool":
            assert conversation.messages[position - 1].role == "assistant"


def test_resetting_clears_everything_short_term():
    conversation = Conversation()
    conversation.add_user("hello")
    conversation.note_interruption("a b c d e f g h", 0.2)
    conversation.reset()
    assert conversation.messages == []
    assert conversation.interrupted is None
    assert conversation.state is WindowState.IDLE


# -- Whisper biasing --------------------------------------------------------
def test_recent_turns_become_a_transcription_hint():
    conversation = Conversation()
    conversation.add_user("check modrinth for create")
    conversation.add_assistant(Message.assistant("Create is on version 6.0"))
    hint = summarise_history(conversation.messages)
    assert "modrinth" in hint and "Create" in hint


def test_the_hint_is_length_capped():
    conversation = Conversation()
    conversation.add_user("word " * 500)
    assert len(summarise_history(conversation.messages, limit=100)) == 100
