"""Tool registration, schema derivation and execution (spec 7.1, 7.9)."""

from __future__ import annotations

import asyncio

import pytest

from adrien.core.llm_types import ToolCall
from adrien.tools.registry import (
    ToolRegistry,
    ToolResult,
    build_parameters_schema,
    parse_docstring,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def reg() -> ToolRegistry:
    return ToolRegistry()


# -- docstring and schema ---------------------------------------------------
def test_docstring_splits_into_summary_and_arguments():
    summary, args = parse_docstring(
        """Send a message to someone.

        Args:
            recipient: Who to send it to.
            message: The text to send,
                which may wrap onto another line.

        Returns:
            Whether it worked.
        """
    )
    assert summary == "Send a message to someone."
    assert args["recipient"] == "Who to send it to."
    assert args["message"] == "The text to send, which may wrap onto another line."
    assert "Returns" not in args


def test_schema_is_derived_from_hints_defaults_and_docstring():
    def sample(name: str, count: int = 3, ratio: float = 1.0,
               loud: bool = False, tags: list[str] | None = None) -> None:
        """Do a thing.

        Args:
            name: Who to do it to.
            count: How many times.
        """

    schema = build_parameters_schema(sample)
    assert schema["required"] == ["name"]
    assert schema["properties"]["name"] == {"type": "string", "description": "Who to do it to."}
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["properties"]["count"]["default"] == 3
    assert schema["properties"]["ratio"]["type"] == "number"
    assert schema["properties"]["loud"]["type"] == "boolean"
    assert schema["properties"]["tags"]["type"] == "array"


def test_a_tool_without_a_docstring_is_rejected(reg):
    with pytest.raises(ValueError, match="docstring"):
        @reg.tool()
        def undocumented(x: str) -> None:
            pass


# -- execution --------------------------------------------------------------
async def test_a_tool_runs_and_returns_its_result(reg):
    @reg.tool(category="info")
    def add(a: int, b: int) -> ToolResult:
        """Add two numbers.

        Args:
            a: First.
            b: Second.
        """
        return ToolResult.success({"sum": a + b}, speak=f"{a + b}")

    result = await reg.execute(ToolCall(name="add", arguments={"a": 2, "b": 3}))
    assert result.ok and result.data == {"sum": 5}


async def test_an_async_tool_is_awaited(reg):
    @reg.tool()
    async def slow() -> ToolResult:
        """Wait a moment then succeed."""
        await asyncio.sleep(0)
        return ToolResult.success("done")

    assert (await reg.execute(ToolCall(name="slow", arguments={}))).data == "done"


async def test_a_plain_return_value_counts_as_success(reg):
    @reg.tool()
    def plain() -> str:
        """Return a bare string."""
        return "hello"

    result = await reg.execute(ToolCall(name="plain", arguments={}))
    assert result.ok and result.data == "hello"


async def test_an_unknown_tool_fails_helpfully(reg):
    @reg.tool()
    def known() -> ToolResult:
        """A tool that exists."""
        return ToolResult.success()

    result = await reg.execute(ToolCall(name="nonexistent", arguments={}))
    assert not result.ok
    assert "known" in result.error  # tells the model what it could call instead


async def test_a_raising_tool_becomes_a_failure_not_an_exception(reg):
    @reg.tool()
    def explode() -> ToolResult:
        """Raise on purpose."""
        raise RuntimeError("the disk is on fire")

    result = await reg.execute(ToolCall(name="explode", arguments={}))
    assert not result.ok
    assert "the disk is on fire" in result.error


async def test_a_hanging_tool_is_cut_off(reg):
    @reg.tool(timeout=0.05)
    async def hang() -> ToolResult:
        """Never return."""
        await asyncio.sleep(30)

    result = await reg.execute(ToolCall(name="hang", arguments={}))
    assert not result.ok and "timed out" in result.error


async def test_missing_required_arguments_are_reported(reg):
    @reg.tool()
    def needs(value: str) -> ToolResult:
        """Need one argument.

        Args:
            value: The thing.
        """
        return ToolResult.success(value)

    result = await reg.execute(ToolCall(name="needs", arguments={}))
    assert not result.ok and "missing required" in result.error


async def test_unknown_arguments_are_dropped_rather_than_crashing(reg):
    @reg.tool()
    def strict(value: str) -> ToolResult:
        """Take exactly one argument.

        Args:
            value: The thing.
        """
        return ToolResult.success(value)

    result = await reg.execute(
        ToolCall(name="strict", arguments={"value": "x", "hallucinated": "y"})
    )
    assert result.ok and result.data == "x"


async def test_string_arguments_are_coerced_to_their_declared_types(reg):
    @reg.tool()
    def typed(count: int, ratio: float, loud: bool) -> ToolResult:
        """Take typed arguments.

        Args:
            count: An integer.
            ratio: A float.
            loud: A boolean.
        """
        return ToolResult.success({"count": count, "ratio": ratio, "loud": loud})

    result = await reg.execute(
        ToolCall(name="typed", arguments={"count": "7", "ratio": "0.5", "loud": "yes"})
    )
    assert result.data == {"count": 7, "ratio": 0.5, "loud": True}


async def test_uncoercible_arguments_report_the_problem(reg):
    @reg.tool()
    def typed(count: int) -> ToolResult:
        """Take an integer.

        Args:
            count: An integer.
        """
        return ToolResult.success(count)

    result = await reg.execute(ToolCall(name="typed", arguments={"count": "many"}))
    assert not result.ok and "integer" in result.error


async def test_a_tool_missing_its_env_var_says_so(reg):
    @reg.tool(requires_env=["DEFINITELY_NOT_SET_XYZ"])
    def needs_key() -> ToolResult:
        """Need a key that is not there."""
        return ToolResult.success()

    result = await reg.execute(ToolCall(name="needs_key", arguments={}))
    assert not result.ok and "DEFINITELY_NOT_SET_XYZ" in result.error


# -- serialisation ----------------------------------------------------------
def test_results_are_redacted_before_reaching_the_model(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "hunter2-hunter2-hunter2")
    payload = ToolResult.success({"config": "key is hunter2-hunter2-hunter2"}).to_json()
    assert "hunter2" not in payload


def test_oversized_results_are_truncated():
    payload = ToolResult.success({"log": "x" * 50_000}).to_json(max_chars=500)
    assert len(payload) < 600
    assert "truncated" in payload


def test_unserialisable_data_does_not_break_the_turn():
    payload = ToolResult.success({"handle": object()}).to_json()
    assert '"ok": true' in payload


# -- registry surface -------------------------------------------------------
def test_schemas_can_be_filtered_by_category(reg):
    @reg.tool(category="info")
    def a() -> ToolResult:
        """Tool A."""
        return ToolResult.success()

    @reg.tool(category="system")
    def b() -> ToolResult:
        """Tool B."""
        return ToolResult.success()

    names = [schema["function"]["name"] for schema in reg.schemas(categories=["info"])]
    assert names == ["a"]
    assert len(reg.schemas()) == 2


def test_every_shipped_tool_has_a_usable_schema():
    """The whole registry, as the LLM will actually receive it."""
    from adrien.tools.registry import load_all_tools

    loaded = load_all_tools()
    assert len(loaded) >= 40, "tool modules failed to import"

    for schema in loaded.schemas():
        function = schema["function"]
        assert function["name"], "every tool needs a name"
        assert function["description"], f"{function['name']} has no description"
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        for argument, spec in parameters["properties"].items():
            assert "type" in spec, f"{function['name']}.{argument} has no type"


def test_destructive_tools_have_confirmation_wording():
    """Anything irreversible must be able to explain itself before running."""
    from adrien.tools.registry import load_all_tools

    for spec in load_all_tools().tools.values():
        if not spec.destructive:
            continue
        prompt = spec.confirmation_prompt({key: "something" for key in
                                           spec.parameters.get("properties", {})})
        assert "?" in prompt, f"{spec.name} does not ask a question"
        assert len(prompt) > 10
