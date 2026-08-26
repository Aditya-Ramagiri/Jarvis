"""Gemini dialect translation - the fallback provider's riskiest surface."""

from __future__ import annotations

from adrien.core.llm_types import Message, ToolCall
from adrien.core.providers.gemini import (
    sanitize_schema,
    to_gemini_contents,
    to_gemini_tools,
)


def test_system_prompt_moves_out_of_contents():
    system, contents = to_gemini_contents([
        Message.system("you are Adrien"),
        Message.user("hi"),
    ])
    assert system == {"parts": [{"text": "you are Adrien"}]}
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_multiple_system_messages_are_merged():
    system, _ = to_gemini_contents([
        Message.system("persona"),
        Message.system("recalled facts"),
        Message.user("hi"),
    ])
    assert system["parts"][0]["text"] == "persona\n\nrecalled facts"


def test_assistant_tool_call_becomes_a_function_call_part():
    call = ToolCall(name="get_weather", arguments={"location": "Dublin"})
    _, contents = to_gemini_contents([
        Message.user("weather?"),
        Message.assistant(tool_calls=[call]),
        Message.tool_result(call, "18C and raining"),
    ])
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "Dublin"}}}],
    }
    # Tool results ride on a user turn, and the payload must be an object.
    assert contents[2]["role"] == "user"
    response = contents[2]["parts"][0]["functionResponse"]
    assert response["name"] == "get_weather"
    assert response["response"] == {"result": "18C and raining"}


def test_assistant_text_and_tool_calls_coexist():
    call = ToolCall(name="set_timer", arguments={"duration": "10m"})
    _, contents = to_gemini_contents([Message.assistant("on it", tool_calls=[call])])
    assert contents[0]["parts"][0] == {"text": "on it"}
    assert "functionCall" in contents[0]["parts"][1]


def test_empty_assistant_turn_is_dropped():
    _, contents = to_gemini_contents([Message.user("hi"), Message.assistant("")])
    assert len(contents) == 1


def test_schema_is_reduced_to_the_accepted_subset():
    cleaned = sanitize_schema({
        "type": "object",
        "additionalProperties": False,
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "properties": {
            "name": {"type": "string", "description": "who", "title": "Name"},
            "tags": {"type": "array", "items": {"type": "string", "default": "x"}},
        },
        "required": ["name"],
    })
    assert cleaned == {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING", "description": "who"},
            "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["name"],
    }


def test_no_argument_tool_gets_an_empty_property_bag():
    # Gemini rejects an OBJECT schema with no `properties` key at all.
    assert sanitize_schema({"type": "object"}) == {"type": "OBJECT", "properties": {}}


def test_openai_tools_become_function_declarations():
    tools = to_gemini_tools([{
        "type": "function",
        "function": {
            "name": "mute",
            "description": "Mute the Mac",
            "parameters": {"type": "object", "properties": {}},
        },
    }])
    assert tools[0]["functionDeclarations"][0]["name"] == "mute"
    assert tools[0]["functionDeclarations"][0]["parameters"]["type"] == "OBJECT"


# -- thought signatures -----------------------------------------------------
# Regression: current Gemini models attach a `thoughtSignature` to every
# function call and reject the follow-up request with a 400 if it is not
# handed back. Dropping it silently broke every multi-step tool chain on the
# fallback provider - and only showed up against the live API.
def test_a_thought_signature_is_carried_back_to_gemini():
    call = ToolCall(
        name="get_weather",
        arguments={"location": "Dublin"},
        provider_state={"thoughtSignature": "Cs0BAdHtim8abc"},
    )
    _, contents = to_gemini_contents([
        Message.user("weather?"),
        Message.assistant(tool_calls=[call]),
        Message.tool_result(call, "18C"),
    ])
    model_turn = contents[1]["parts"][0]
    assert model_turn["functionCall"]["name"] == "get_weather"
    assert model_turn["thoughtSignature"] == "Cs0BAdHtim8abc"


def test_a_call_without_a_signature_carries_no_empty_key():
    """Groq-originated calls have no signature; sending an empty one is worse
    than sending none."""
    call = ToolCall(name="mute", arguments={})
    _, contents = to_gemini_contents([Message.assistant(tool_calls=[call])])
    assert "thoughtSignature" not in contents[0]["parts"][0]


def test_signatures_are_captured_from_a_response():
    from adrien.core.llm_types import ToolCall as TC

    part = {
        "functionCall": {"name": "set_timer", "args": {"duration": "10m"}},
        "thoughtSignature": "sig-123",
    }
    # Mirrors the parsing branch in GeminiProvider.chat.
    state = {"thoughtSignature": part["thoughtSignature"]} if part.get("thoughtSignature") else {}
    call = TC(name=part["functionCall"]["name"],
              arguments=part["functionCall"]["args"], provider_state=state)
    assert call.provider_state["thoughtSignature"] == "sig-123"
    # And it survives a round trip back into a request.
    _, contents = to_gemini_contents([Message.assistant(tool_calls=[call])])
    assert contents[0]["parts"][0]["thoughtSignature"] == "sig-123"
