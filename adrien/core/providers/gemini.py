"""Gemini adapter - fallback LLM provider (spec 4.4).

Gemini's REST API is *not* OpenAI-shaped, so this module owns the whole
translation:

* messages       -> `contents` with `user`/`model` roles (no system role;
                    the system prompt goes in `system_instruction`)
* tool calls     -> `functionCall` parts
* tool results   -> `functionResponse` parts, matched back to their call by
                    tool *name* (Gemini has no call ids, so ids are minted on
                    the way back and correlated by name + order)
* tool schemas   -> `functionDeclarations` with a reduced JSON-Schema subset

Using REST directly rather than `google-generativeai` keeps key rotation in our
hands: the SDK configures a key globally at import time, which is exactly the
wrong shape for a pool of five keys.
"""

from __future__ import annotations

import time
from typing import Any

from adrien.config import env_str
from adrien.core.http import get_client, parse_retry_after
from adrien.core.llm_types import ChatResult, Message, ProviderError, ToolCall
from adrien.logging_setup import get_logger

log = get_logger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Gemini accepts a subset of JSON Schema. Anything else (additionalProperties,
# $schema, oneOf, ...) is rejected outright, so tool schemas are filtered
# rather than passed through.
_ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum", "properties",
    "required", "items", "minimum", "maximum",
}


def sanitize_schema(schema: Any) -> Any:
    """Recursively reduce a JSON Schema to the subset Gemini accepts."""
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {name: sanitize_schema(sub) for name, sub in value.items()}
        elif key == "items":
            cleaned[key] = sanitize_schema(value)
        elif key == "type" and isinstance(value, str):
            cleaned[key] = value.upper()  # Gemini wants STRING/OBJECT/ARRAY
        else:
            cleaned[key] = value
    if cleaned.get("type") == "OBJECT" and "properties" not in cleaned:
        # Gemini rejects an object schema with no properties; a tool that takes
        # no arguments is expressed as an empty property bag.
        cleaned["properties"] = {}
    return cleaned


def to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI `tools` array -> Gemini `functionDeclarations`."""
    declarations = []
    for tool in tools:
        function = tool.get("function") or tool
        declaration: dict[str, Any] = {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
        }
        parameters = function.get("parameters")
        if parameters:
            declaration["parameters"] = sanitize_schema(parameters)
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}]


def to_gemini_contents(messages: list[Message]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Neutral messages -> (`system_instruction`, `contents`)."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
            continue

        if message.role == "tool":
            contents.append({
                "role": "user",  # Gemini carries tool results on the user turn
                "parts": [{
                    "functionResponse": {
                        "name": message.name or "tool",
                        # `response` must be an object, never a bare string.
                        "response": {"result": message.content},
                    }
                }],
            })
            continue

        if message.role == "assistant":
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"text": message.content})
            for call in message.tool_calls:
                part: dict[str, Any] = {
                    "functionCall": {"name": call.name, "args": call.arguments}
                }
                # Gemini rejects the follow-up request outright if the
                # signature it issued with the call is not handed back
                # (400: "Function call is missing a thought_signature").
                # Dropping it breaks every multi-step chain on this provider.
                signature = call.provider_state.get("thoughtSignature")
                if signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        contents.append({"role": "user", "parts": [{"text": message.content or ""}]})

    system_instruction = (
        {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    )
    return system_instruction, contents


class GeminiProvider:
    name = "gemini"

    # Moving aliases, not pinned versions. This is the *fallback* provider:
    # it runs only when every Groq key is down, which means it is the least
    # exercised path in the system and the most likely to have rotted
    # unnoticed. A pinned model that Google retires turns "Groq is rate
    # limited" into "Adrien is broken" - which is exactly what happened with
    # gemini-2.0-flash. An alias that tracks the current model is the safer
    # trade here, even though it means the exact model can change underneath.
    def __init__(self, fast_model: str | None = None, smart_model: str | None = None) -> None:
        self.fast_model = fast_model or env_str("GEMINI_FAST_MODEL", "gemini-flash-lite-latest")
        self.smart_model = smart_model or env_str("GEMINI_SMART_MODEL", "gemini-flash-latest")

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

        system_instruction, contents = to_gemini_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction
        if tools:
            payload["tools"] = to_gemini_tools(tools)

        client = get_client()
        started = time.perf_counter()
        try:
            response = await client.post(
                f"{API_BASE}/{model}:generateContent",
                json=payload,
                # Key goes in the header, not the query string: query strings
                # end up in proxy and server logs.
                headers={"x-goog-api-key": api_key},
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"gemini timed out after {timeout:.0f}s", provider=self.name, retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"gemini transport error: {type(exc).__name__}",
                provider=self.name,
                retryable=True,
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        _raise_for_status(response, self.name)

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(
                "gemini returned an unreadable response", provider=self.name, retryable=True
            ) from exc

        candidates = body.get("candidates") or []
        if not candidates:
            # Usually a safety block; there is nothing to retry with a
            # different key, so surface it as non-retryable.
            reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(
                f"gemini returned nothing ({reason})", provider=self.name, retryable=False
            )

        candidate = candidates[0]
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in (candidate.get("content") or {}).get("parts") or []:
            if "text" in part and part["text"]:
                texts.append(part["text"])
            function_call = part.get("functionCall")
            if function_call and function_call.get("name"):
                state: dict[str, Any] = {}
                # Carried opaquely and handed straight back next turn.
                if part.get("thoughtSignature"):
                    state["thoughtSignature"] = part["thoughtSignature"]
                tool_calls.append(
                    ToolCall(
                        name=function_call["name"],
                        arguments=dict(function_call.get("args") or {}),
                        provider_state=state,
                    )
                )

        usage = body.get("usageMetadata") or {}
        return ChatResult(
            text="".join(texts).strip(),
            tool_calls=tool_calls,
            provider=self.name,
            model=model,
            latency_ms=latency_ms,
            finish_reason=candidate.get("finishReason") or "",
            usage={
                "prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount"),
            },
        )


def _raise_for_status(response: Any, provider: str) -> None:
    status = response.status_code
    if status < 400:
        return

    detail = ""
    try:
        error = (response.json() or {}).get("error") or {}
        detail = str(error.get("message") or "")
    except ValueError:
        detail = (response.text or "")[:200]

    if status == 429 or "quota" in detail.lower():
        raise ProviderError(
            f"{provider} rate limited: {detail}",
            provider=provider,
            status=status,
            rate_limited=True,
            retry_after=parse_retry_after(response.headers),
        )
    if status in (401, 403):
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
    raise ProviderError(
        f"{provider} rejected the request ({status}): {detail}",
        provider=provider,
        status=status,
        retryable=False,
    )
