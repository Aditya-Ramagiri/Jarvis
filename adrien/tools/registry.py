"""The tool-calling framework (spec 7.1).

A tool is an ordinary Python function wearing a decorator:

    @tool(category="info")
    def get_weather(location: str = "here", units: str = "celsius") -> ToolResult:
        '''Current weather for a place.

        Args:
            location: City or place name. Defaults to the user's location.
            units: "celsius" or "fahrenheit".
        '''

Everything the LLM needs - name, description, JSON-Schema parameters - is
derived from the signature, type hints and docstring, so a tool cannot drift
out of sync with its own schema. Adding an integration is one function in one
module; nothing in the orchestrator changes.

There is no if/elif dispatch anywhere: `registry.schemas()` feeds the provider's
native function-calling API, and `registry.execute()` runs whatever comes back.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from adrien.core.llm_types import ToolCall
from adrien.logging_setup import get_logger, redact

log = get_logger(__name__)

# A tool that hangs would hang the whole conversation, so every call is capped.
DEFAULT_TIMEOUT = 20.0

CATEGORIES = ("dev", "gaming", "productivity", "system", "info", "messaging", "extras")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class ToolResult:
    """Structured outcome (spec 7.9).

    The LLM must be able to tell success from failure without parsing prose,
    so a failed git push comes back as `ok=False` with the reason attached -
    which is what stops Adrien cheerfully saying "done" over an error.
    """

    ok: bool = True
    data: Any = None
    error: str = ""
    # Optional phrasing hint for the spoken reply. The model is free to ignore
    # it, but for terse confirmations it usually shouldn't.
    speak: str = ""

    @classmethod
    def success(cls, data: Any = None, speak: str = "") -> "ToolResult":
        return cls(ok=True, data=data, speak=speak)

    @classmethod
    def failure(cls, error: str, data: Any = None) -> "ToolResult":
        return cls(ok=False, error=error, data=data)

    def to_json(self, max_chars: int = 4000) -> str:
        """Serialise for the model, redacted and length-capped.

        Tool output goes straight into the prompt, so an unbounded `git log` or
        a screenful of stack trace would blow the context window and cost real
        latency. Truncation is explicit so the model knows it happened.
        """
        payload: dict[str, Any] = {"ok": self.ok}
        if self.error:
            payload["error"] = self.error
        if self.speak:
            payload["speak"] = self.speak
        if self.data is not None:
            payload["data"] = self.data
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = json.dumps({"ok": self.ok, "data": str(self.data)[:max_chars]})
        text = redact(text)
        if len(text) > max_chars:
            text = text[:max_chars] + '... [truncated]"}'
        return text


# --------------------------------------------------------------------------
# Schema derivation
# --------------------------------------------------------------------------
_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.M)
_ARG_LINE = re.compile(r"^\s{0,8}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, per-argument help)."""
    if not doc:
        return "", {}
    text = inspect.cleandoc(doc)
    match = _ARGS_HEADER.search(text)
    if not match:
        # Everything up to the first blank-line-separated trailing section.
        return text.split("\n\n")[0].strip(), {}

    summary = text[: match.start()].strip().split("\n\n")[0].strip()
    params: dict[str, str] = {}
    current: str | None = None
    for line in text[match.end():].splitlines():
        if not line.strip():
            continue
        if re.match(r"^\s*(Returns|Raises|Example|Examples|Note)\s*:\s*$", line):
            break
        arg_match = _ARG_LINE.match(line)
        if arg_match:
            current = arg_match.group(1)
            params[current] = arg_match.group(2).strip()
        elif current:  # continuation line
            params[current] = f"{params[current]} {line.strip()}"
    return summary, params


_JSON_TYPES: dict[Any, str] = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object",
}


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    """Map a type hint onto a JSON-Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Both spellings of a union: `Optional[X]` (origin `typing.Union`) and the
    # PEP 604 `X | None` (origin `types.UnionType`). They are distinct objects,
    # so both have to be checked.
    if origin is typing.Union or origin is types.UnionType:
        # Optional[X] -> X's schema; the parameter's default makes it optional.
        inner = [arg for arg in args if arg is not type(None)]
        return _schema_for_annotation(inner[0]) if inner else {"type": "string"}

    if origin is typing.Literal:
        return {"type": "string", "enum": [str(arg) for arg in args]}

    if origin in (list, tuple, set):
        item = args[0] if args else str
        return {"type": "array", "items": _schema_for_annotation(item)}

    if origin is dict:
        return {"type": "object"}

    if isinstance(annotation, type) and issubclass(annotation, str) and hasattr(annotation, "__members__"):
        return {"type": "string", "enum": list(annotation.__members__)}  # str Enum

    return {"type": _JSON_TYPES.get(annotation, "string")}


def build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """JSON Schema for `func`'s arguments, from hints + docstring."""
    signature = inspect.signature(func)
    _, arg_docs = parse_docstring(func.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = typing.get_type_hints(func) if func.__annotations__ else {}

    for name, parameter in signature.parameters.items():
        if name in ("self", "cls") or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        ):
            continue
        schema = _schema_for_annotation(hints.get(name, parameter.annotation))
        if name in arg_docs:
            schema["description"] = arg_docs[name]
        if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
            schema["default"] = parameter.default
        properties[name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# --------------------------------------------------------------------------
# Tool specs and the registry
# --------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    category: str = "extras"
    destructive: bool = False
    # Destructive *and* unrecoverable: powering the machine down, sending a
    # message someone else will read, running arbitrary code. A category-wide
    # "auto" must not silently cover these - see PermissionManager.mode_for.
    irreversible: bool = False
    confirm_template: str = ""
    requires_env: tuple[str, ...] = ()
    timeout: float = DEFAULT_TIMEOUT
    is_async: bool = False

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def confirmation_prompt(self, arguments: dict[str, Any]) -> str:
        """What Adrien says out loud before doing something irreversible."""
        if self.confirm_template:
            try:
                return self.confirm_template.format(**arguments)
            except (KeyError, IndexError):
                pass
        rendered = ", ".join(f"{k} {v}" for k, v in arguments.items() if v not in (None, ""))
        return f"Do you want me to run {self.name.replace('_', ' ')}{' with ' + rendered if rendered else ''}?"


@dataclass
class ToolRegistry:
    """Central registry passed to the LLM's function-calling API."""

    tools: dict[str, ToolSpec] = field(default_factory=dict)

    # -- registration -----------------------------------------------------
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self.tools:
            log.warning("tool %s registered twice; the later one wins", spec.name)
        self.tools[spec.name] = spec
        return spec

    def tool(
        self,
        *,
        name: str | None = None,
        category: str = "extras",
        destructive: bool = False,
        irreversible: bool = False,
        confirm: str = "",
        requires_env: Iterable[str] = (),
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that turns a function into a registered tool."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            summary, _ = parse_docstring(func.__doc__)
            if not summary:
                raise ValueError(
                    f"tool {func.__name__} needs a docstring - it is what the "
                    "model reads to decide when to call it"
                )
            self.register(ToolSpec(
                name=name or func.__name__,
                description=summary,
                parameters=build_parameters_schema(func),
                func=func,
                category=category,
                destructive=destructive or irreversible,
                irreversible=irreversible,
                confirm_template=confirm,
                requires_env=tuple(requires_env),
                timeout=timeout,
                is_async=inspect.iscoroutinefunction(func),
            ))
            return func

        return decorator

    # -- lookup -----------------------------------------------------------
    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def __len__(self) -> int:
        return len(self.tools)

    def by_category(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for spec in self.tools.values():
            grouped.setdefault(spec.category, []).append(spec.name)
        return {key: sorted(value) for key, value in sorted(grouped.items())}

    def schemas(self, *, categories: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Tool schemas for the provider's function-calling API."""
        allowed = set(categories) if categories else None
        return [
            spec.to_openai()
            for spec in self.tools.values()
            if allowed is None or spec.category in allowed
        ]

    # -- execution --------------------------------------------------------
    def validate_arguments(self, spec: ToolSpec, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Drop unknown keys, check required ones, coerce obvious types.

        Models improvise argument names and send "5" where an int was asked
        for. Coercing here means one clean error message back to the model
        instead of a TypeError traceback.
        """
        properties: dict[str, Any] = spec.parameters.get("properties", {})
        required: list[str] = spec.parameters.get("required", [])

        cleaned: dict[str, Any] = {}
        for key, value in arguments.items():
            if key not in properties:
                log.debug("%s: ignoring unknown argument %r", spec.name, key)
                continue
            expected = properties[key].get("type")
            try:
                if expected == "integer" and not isinstance(value, bool):
                    cleaned[key] = int(float(value)) if value not in (None, "") else None
                elif expected == "number":
                    cleaned[key] = float(value)
                elif expected == "boolean":
                    cleaned[key] = (
                        value if isinstance(value, bool)
                        else str(value).strip().lower() in ("true", "yes", "1", "on")
                    )
                elif expected == "array" and isinstance(value, str):
                    cleaned[key] = [item.strip() for item in value.split(",") if item.strip()]
                elif expected == "string" and value is not None and not isinstance(value, str):
                    cleaned[key] = str(value)
                else:
                    cleaned[key] = value
            except (TypeError, ValueError):
                return {}, f"{key} should be a {expected}, got {value!r}"

        missing = [key for key in required if cleaned.get(key) in (None, "")]
        if missing:
            return {}, f"missing required argument(s): {', '.join(missing)}"
        return {key: value for key, value in cleaned.items() if value is not None}, ""

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Never raises - failures come back as results."""
        spec = self.get(call.name)
        if spec is None:
            available = ", ".join(sorted(self.tools)[:20])
            return ToolResult.failure(
                f"no tool named {call.name}. Available tools include: {available}"
            )

        arguments, error = self.validate_arguments(spec, call.arguments or {})
        if error:
            return ToolResult.failure(error)

        missing_env = [name for name in spec.requires_env if not _env_present(name)]
        if missing_env:
            return ToolResult.failure(
                f"{spec.name} is not configured: {', '.join(missing_env)} missing from .env"
            )

        started = time.perf_counter()
        try:
            if spec.is_async:
                result = await asyncio.wait_for(spec.func(**arguments), timeout=spec.timeout)
            else:
                # Tools are mostly blocking I/O (subprocess, HTTP, UI
                # automation); a thread keeps the event loop responsive so
                # barge-in still works while a tool runs.
                result = await asyncio.wait_for(
                    asyncio.to_thread(spec.func, **arguments), timeout=spec.timeout
                )
        except asyncio.TimeoutError:
            log.warning("%s timed out after %.0fs", spec.name, spec.timeout)
            return ToolResult.failure(f"{spec.name} timed out after {spec.timeout:.0f} seconds")
        except Exception as exc:
            log.exception("%s raised", spec.name)
            return ToolResult.failure(f"{spec.name} failed: {type(exc).__name__}: {exc}")

        elapsed = (time.perf_counter() - started) * 1000
        log.info("tool %s -> %s in %.0fms", spec.name,
                 "ok" if getattr(result, "ok", True) else "error", elapsed)

        if isinstance(result, ToolResult):
            return result
        # A tool that just returns a value is treated as a success.
        return ToolResult.success(result)


def _env_present(name: str) -> bool:
    import os

    return bool((os.environ.get(name) or "").strip())


# The process-wide registry every tool module decorates into.
registry = ToolRegistry()
tool = registry.tool


def load_all_tools() -> ToolRegistry:
    """Import every tool module so its decorators run.

    Import errors are survivable on purpose: a missing optional dependency
    (say `pyautogui` on a Linux box) should cost that one category, not the
    whole assistant.
    """
    modules = [
        "adrien.tools.dev_tools",
        "adrien.tools.gaming_tools",
        "adrien.tools.productivity_tools",
        "adrien.tools.system_tools",
        "adrien.tools.info_tools",
        "adrien.tools.discord_automation",
        "adrien.tools.extra_tools",
    ]
    import importlib

    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            log.error("could not load %s: %s", module, exc)
    log.info("loaded %d tools: %s", len(registry), registry.by_category())
    return registry
