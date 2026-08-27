"""Groq adapter - primary LLM provider.

Groq exposes an OpenAI-compatible endpoint, so the neutral message and tool
types serialise almost directly. We speak that endpoint over `httpx` rather
than through the `groq` SDK on purpose: the SDK wants to own retries and the
client lifecycle, and both of those belong to `KeyPool` and `LLMRouter` here
(spec section 4). One less layer between a 429 and the next key.
"""

from __future__ import annotations

import re
import time
from typing import Any

from adrien.config import env_str
from adrien.core.http import get_client, parse_retry_after
from adrien.core.llm_types import ChatResult, Message, ProviderError, ToolCall
from adrien.logging_setup import get_logger

log = get_logger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS_URL = "https://api.groq.com/openai/v1/models"

# Preference order per tier, best first. Groq retires model ids without
# offering a moving alias (there is no "llama-latest"), so any pinned name is a
# future 404 - `llama-3.1-8b-instant` was the default here and stopped existing.
# These lists are only a *preference*: the live list from /models decides what
# is actually reachable, so a retirement costs a fallback, not an outage.
FAST_PREFERENCES = (
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
)
SMART_PREFERENCES = (
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
)

# Never auto-select these for chat: they are not general chat models, or are
# specialised in a way that rules out tool calling.
_NOT_CHAT = (
    "whisper", "tts", "embed", "guard",      # not chat models at all
    "allam",                                  # Arabic-specialised, no tool calling
    "distil",                                 # distilled, tool support is patchy
)

# Family ranking for the "nothing preferred survived" case. Sorting
# alphabetically instead of by family is how a fallback once picked
# `allam-2-7b`, which cannot do tool calling at all - the worst option in the
# list, chosen because "a" sorts first. Lower index is better.
_FAMILY_RANK = (
    "llama-3.3", "llama-3.1", "llama-4", "llama-3", "llama3", "llama",
    "qwen", "gpt-oss", "kimi", "deepseek", "gemma", "mixtral", "mistral",
)

# How many different models to try before giving up on this provider.
_MAX_MODEL_ATTEMPTS = 3


def rank_candidate(name: str) -> tuple[int, int, str]:
    """Sort key for an unknown model id: known family first, then bigger.

    Returns (family rank, size rank, name) so `sorted()` puts the most
    capable recognisable chat model first.
    """
    lowered = name.lower()
    family = next(
        (index for index, marker in enumerate(_FAMILY_RANK) if marker in lowered),
        len(_FAMILY_RANK),
    )
    # Prefer larger parameter counts within a family: 70b beats 8b.
    size = 0
    for match in re.finditer(r"(\d+)\s*b\b", lowered):
        size = max(size, int(match.group(1)))
    return (family, -size, name)


class GroqProvider:
    """Groq chat, with the model id resolved against what the key can see.

    Model ids are discovered rather than trusted. The first time a tier is
    needed, `/openai/v1/models` says what this account can actually reach and
    the best available preference wins; a 404 for a retired model refreshes
    that list and retries once. Without this, Groq retiring a name is an
    outage that looks like a bug in Adrien.
    """

    name = "groq"

    def __init__(self, fast_model: str | None = None, smart_model: str | None = None) -> None:
        # An explicit setting is a deliberate choice and is tried first, but it
        # is still verified against the live list rather than assumed.
        self.fast_model = fast_model or env_str("GROQ_FAST_MODEL")
        self.smart_model = smart_model or env_str("GROQ_SMART_MODEL")
        self._available: set[str] | None = None
        self._resolved: dict[str, str] = {}
        # Models this key can see but that rejected tool calling. Adrien always
        # sends tools, so one of these is useless to us however good it is.
        self._no_tools: set[str] = set()

    def model_for(self, tier: str) -> str:
        """Best known id for a tier, before any discovery has happened.

        The router calls this to label the request; `chat()` does the real
        resolution once it has a key to ask with.
        """
        if tier in self._resolved:
            return self._resolved[tier]
        configured = self.smart_model if tier == "smart" else self.fast_model
        if configured:
            return configured
        preferences = SMART_PREFERENCES if tier == "smart" else FAST_PREFERENCES
        return preferences[0]

    async def _discover(self, api_key: str) -> set[str]:
        """Ask Groq which models this key can actually use."""
        try:
            response = await get_client().get(
                MODELS_URL, headers={"authorization": f"Bearer {api_key}"}, timeout=15
            )
            if response.status_code != 200:
                log.warning("could not list Groq models (%d)", response.status_code)
                return set()
            ids = {
                str(item.get("id"))
                for item in (response.json() or {}).get("data") or []
                if item.get("id")
            }
            log.info("Groq offers %d models to this key", len(ids))
            return ids
        except Exception as exc:
            log.warning("could not list Groq models: %s", type(exc).__name__)
            return set()

    async def resolve_model(self, tier: str, api_key: str, *, refresh: bool = False) -> str:
        """Pick a model for `tier` that this key can really reach."""
        if not refresh and tier in self._resolved:
            return self._resolved[tier]

        if refresh or self._available is None:
            self._available = await self._discover(api_key)

        available = self._available or set()
        configured = self.smart_model if tier == "smart" else self.fast_model
        preferences = SMART_PREFERENCES if tier == "smart" else FAST_PREFERENCES

        if not available:
            # Discovery failed; go with what we were told and let the call
            # report the truth rather than guessing further.
            chosen = configured or preferences[0]
            self._resolved[tier] = chosen
            return chosen

        usable = available - self._no_tools

        if configured and configured in usable:
            chosen = configured
        else:
            if configured and configured in available:
                log.warning("Groq model %r cannot do tool calling - picking another", configured)
            elif configured:
                log.warning("Groq model %r is not available to this key - picking another",
                            configured)
            chosen = next((name for name in preferences if name in usable), "")

        if not chosen:
            # Nothing from the preference list survived. Rank what is left by
            # family and size rather than alphabetically.
            candidates = sorted(
                (name for name in usable
                 if not any(marker in name.lower() for marker in _NOT_CHAT)),
                key=rank_candidate,
            )
            chosen = candidates[0] if candidates else ""
            if chosen:
                log.warning("no preferred Groq model available; using %r (best of %d)",
                            chosen, len(candidates))
            else:
                chosen = configured or preferences[0]
                log.error("no usable Groq chat model found among %d offered", len(available))

        log.info("Groq %s tier resolved to %s", tier, chosen)
        self._resolved[tier] = chosen
        return chosen

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
        """One completion, moving on if the chosen model turns out unusable.

        Two failures are worth healing rather than surfacing, because both mean
        "this id is wrong" rather than "the request is wrong":

        * the model was retired (404)
        * the model exists but cannot do tool calling (400), which is fatal for
          Adrien since every request carries tools

        Either way the model is struck off and the next best is tried.
        """
        tier = "smart" if model == self._resolved.get("smart") else "fast"
        tried: list[str] = []

        for _ in range(_MAX_MODEL_ATTEMPTS):
            tried.append(model)
            try:
                return await self._chat_once(
                    api_key=api_key, model=model, messages=messages, tools=tools,
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                )
            except ProviderError as exc:
                if _is_no_tool_support(exc):
                    log.warning("Groq model %r cannot do tool calling; striking it off", model)
                    self._no_tools.add(model)
                elif _is_unknown_model(exc):
                    log.warning("Groq model %r is gone; refreshing the list", model)
                else:
                    raise

                replacement = await self.resolve_model(tier, api_key, refresh=True)
                if replacement in tried:
                    raise
                model = replacement

        raise ProviderError(
            f"groq: no usable model found (tried {', '.join(tried)})",
            provider=self.name, retryable=False,
        )

    async def _chat_once(
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


def _is_no_tool_support(error: ProviderError) -> bool:
    """True when the model exists but refuses tool calling."""
    if error.status != 400:
        return False
    text = str(error).lower()
    return "tool calling" in text or "does not support tools" in text


def _is_unknown_model(error: ProviderError) -> bool:
    """True when Groq rejected the request because the model id is retired."""
    if error.status not in (400, 404):
        return False
    text = str(error).lower()
    return "does not exist" in text or "model_not_found" in text or "decommissioned" in text
