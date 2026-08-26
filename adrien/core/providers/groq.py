"""Groq adapter - primary LLM provider.

Groq exposes an OpenAI-compatible endpoint, so the neutral message and tool
types serialise almost directly. We speak that endpoint over `httpx` rather
than through the `groq` SDK on purpose: the SDK wants to own retries and the
client lifecycle, and both of those belong to `KeyPool` and `LLMRouter` here
(spec section 4). One less layer between a 429 and the next key.
"""

from __future__ import annotations

import time
from typing import Any

from adrien.config import env_str
from adrien.core.http import get_client, parse_retry_after
from adrien.core.llm_types import ChatResult, Message, ProviderError, ToolCall
from adrien.logging_setup import get_logger

log = get_logger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider:
    name = "groq"

    def __init__(self, fast_model: str | None = None, smart_model: str | None = None) -> None:
        self.fast_model = fast_model or env_str("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
        self.smart_model = smart_model or env_str("GROQ_SMART_MODEL", "llama-3.3-70b-versatile")

    def model_for(self, tier: str) -> str:
        return self.smart_model if tier == "smart" else self.fast_model

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
        import httpx

        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.to_openai() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        client = get_client()
        started = time.perf_counter()
        try:
            response = await client.post(
                API_URL,
                json=payload,
                headers={"authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"groq timed out after {timeout:.0f}s", provider=self.name, retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"groq transport error: {type(exc).__name__}",
                provider=self.name,
                retryable=True,
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        _raise_for_status(response, self.name)

        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError) as exc:
            raise ProviderError(
                "groq returned an unreadable response", provider=self.name, retryable=True
            ) from exc

        tool_calls = [ToolCall.from_openai(item) for item in (message.get("tool_calls") or [])]
        return ChatResult(
            text=(message.get("content") or "").strip(),
            tool_calls=[call for call in tool_calls if call.name],
            provider=self.name,
            model=model,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason") or "",
            usage=body.get("usage") or {},
        )


def _raise_for_status(response: Any, provider: str) -> None:
    """Classify an HTTP status into the retry semantics the router needs."""
    status = response.status_code
    if status < 400:
        return

    detail = ""
    try:
        body = response.json()
        error = body.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or "")
        elif error:
            detail = str(error)
    except ValueError:
        detail = (response.text or "")[:200]

    if status == 429:
        raise ProviderError(
            f"{provider} rate limited: {detail}",
            provider=provider,
            status=status,
            rate_limited=True,
            retry_after=parse_retry_after(response.headers),
        )
    if status in (401, 403):
        # A dead key, not a dead provider: cool this key and rotate on.
        raise ProviderError(
            f"{provider} rejected the key ({status})",
            provider=provider,
            status=status,
            retryable=True,
        )
    if status >= 500 or status == 408:
        raise ProviderError(
            f"{provider} server error {status}: {detail}",
            provider=provider,
            status=status,
            retryable=True,
        )
    # 400/404/422: the request itself is wrong. Another key changes nothing.
    raise ProviderError(
        f"{provider} rejected the request ({status}): {detail}",
        provider=provider,
        status=status,
        retryable=False,
    )
