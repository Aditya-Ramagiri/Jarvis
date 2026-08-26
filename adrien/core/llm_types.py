"""Provider-neutral chat types.

Adrien talks to Groq (OpenAI-shaped) and Gemini (its own shape). Rather than
letting either dialect leak into the orchestrator, everything upstream speaks
these types and each provider adapts at its own boundary. Adding a third
provider later means writing one adapter, not touching the orchestrator.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A model's request to run one tool."""

    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")

    @classmethod
    def from_openai(cls, payload: dict[str, Any]) -> "ToolCall":
        function = payload.get("function") or {}
        raw = function.get("arguments") or "{}"
        if isinstance(raw, str):
            try:
                arguments = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                # Models occasionally emit not-quite-JSON. Surface it as a
                # string argument rather than dropping the call: the tool layer
                # reports a clean validation error the model can recover from.
                arguments = {"_raw": raw}
        else:
            arguments = dict(raw)
        return cls(
            name=function.get("name", ""),
            arguments=arguments if isinstance(arguments, dict) else {"_raw": arguments},
            id=payload.get("id") or f"call_{uuid.uuid4().hex[:12]}",
        )

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class Message:
    """One turn in the conversation, in provider-neutral form."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.role == "tool":
            payload["content"] = self.content
            payload["tool_call_id"] = self.tool_call_id or ""
            if self.name:
                payload["name"] = self.name
            return payload
        payload["content"] = self.content or ""
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai() for call in self.tool_calls]
            # OpenAI-shaped APIs expect null, not "", alongside tool_calls.
            if not self.content:
                payload["content"] = None
        return payload

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=call.id, name=call.name)


@dataclass
class ChatResult:
    """A completed model response plus the routing metadata we log locally."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    key_label: str = ""
    latency_ms: float = 0.0
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ProviderError(Exception):
    """A failed provider call, classified so the router knows what to do next.

    * `rate_limited` - rotate to the next key immediately (spec 4.3).
    * `retryable` - transient (network blip, 5xx); try another key.
    * neither - a request-shaped problem (bad schema, unknown model). Trying
      another key would fail identically, so the router stops and surfaces it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status: int | None = None,
        rate_limited: bool = False,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.rate_limited = rate_limited
        self.retryable = retryable or rate_limited
        self.retry_after = retry_after


class AllProvidersFailed(Exception):
    """Every key of every provider is unusable.

    Spec 4.5: the one case where Adrien speaks up about its own plumbing.
    """

    def __init__(self, attempts: list[str], last_error: Exception | None = None) -> None:
        super().__init__(
            "all LLM providers failed: " + ("; ".join(attempts) or "no keys configured")
        )
        self.attempts = attempts
        self.last_error = last_error
