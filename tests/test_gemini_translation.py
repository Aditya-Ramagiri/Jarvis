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
