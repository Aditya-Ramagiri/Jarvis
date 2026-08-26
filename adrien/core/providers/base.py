"""The contract every LLM provider adapter implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from adrien.core.llm_types import ChatResult, Message


@runtime_checkable
class ChatProvider(Protocol):
    """Stateless adapter: the key arrives per call, from the rotating pool."""

    name: str

    def model_for(self, tier: str) -> str:
        """Concrete model id for the `"fast"` or `"smart"` tier."""

    async def chat(
        self,
        *,
        api_key: str,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 700,
        timeout: float = 25.0,
    ) -> ChatResult:
        """One completion. Raises `ProviderError` on any failure."""
