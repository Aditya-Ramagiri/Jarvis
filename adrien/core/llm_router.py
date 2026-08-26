"""LLM routing: model tiering, key rotation, cross-provider fallback.

This is the module spec section 4 is about. The contract it offers upstream is
deliberately small - `await router.chat(messages, tools=...)` - and everything
noisy happens behind it:

    for provider in (groq, gemini):          # 4.4 provider fallback
        for lease in provider.pool.leases(): # 4.2 LRU rotation
            try: return await provider.chat(key=lease.key, ...)
            except rate limited: lease.rate_limited(); continue   # 4.3, no sleep
            except transient:     lease.failed();       continue
            except bad request:   raise                 # another key won't help
    raise AllProvidersFailed                            # 4.5, the only case
                                                        # the user hears about

Rotation adds one dictionary scan per attempt and never sleeps, so a 429 costs
the round trip that already failed and nothing more (4.6).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from adrien.config import Settings, env_key_pool
from adrien.config import settings as global_settings
from adrien.core.keypool import KeyPool
from adrien.core.llm_types import (
    AllProvidersFailed,
    ChatResult,
    Message,
    ProviderError,
)
from adrien.core.providers.base import ChatProvider
from adrien.core.providers.gemini import GeminiProvider
from adrien.core.providers.groq import GroqProvider
from adrien.logging_setup import get_logger

log = get_logger(__name__)

Tier = str  # "fast" | "smart" | "auto"

# Below this many words, with no reasoning cue, a request is a command
# ("set a timer for ten minutes") and the 8B model answers it faster.
_FAST_TIER_MAX_WORDS = 12


@dataclass
class ProviderSlot:
    """A provider plus the key pool that feeds it."""

    provider: ChatProvider
    pool: KeyPool

    @property
    def name(self) -> str:
        return self.provider.name


class LLMRouter:
    """Chooses a model tier, then burns through keys until one answers."""

    def __init__(
        self,
        settings: Settings | None = None,
        slots: Sequence[ProviderSlot] | None = None,
    ) -> None:
        self.settings = settings or global_settings()
        self.slots: list[ProviderSlot] = list(slots) if slots is not None else self._build_slots()
        configured = [f"{slot.name}({len(slot.pool)})" for slot in self.slots]
        log.info("LLM router ready: %s", ", ".join(configured) or "NO KEYS CONFIGURED")

    # -- construction -----------------------------------------------------
    def _build_slots(self) -> list[ProviderSlot]:
        keys_cfg = self.settings.get("keys", {}) or {}
        cooldown = float(keys_cfg.get("cooldown_seconds", 60.0))
        failure_cooldown = float(keys_cfg.get("failure_cooldown_seconds", 15.0))

        slots: list[ProviderSlot] = []
        # Order is the fallback order: Groq first (fast + primary), Gemini after.
        for name, prefix, provider in (
            ("groq", "GROQ_API_KEY", GroqProvider()),
            ("gemini", "GEMINI_API_KEY", GeminiProvider()),
        ):
            keys = env_key_pool(prefix)
            if not keys:
                log.warning("no %s keys found in .env (%s_1, ...)", name, prefix)
                continue
            slots.append(
                ProviderSlot(
                    provider=provider,
                    pool=KeyPool(
                        name,
                        keys,
                        cooldown_seconds=cooldown,
                        failure_cooldown_seconds=failure_cooldown,
                    ),
                )
            )
        return slots

    # -- tiering ----------------------------------------------------------
    def choose_tier(self, messages: Sequence[Message]) -> str:
        """Pick the fast or smart model for this turn (spec section 3).

        Default is the smart model: conversation and multi-tool chains are
        explicitly its job. The fast model is used for the short, literal
        commands that make up most utterances, where 70B buys nothing but
        latency.
        """
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user" and m.content), ""
        )
        if not last_user:
            return "smart"

        # A turn that already carries tool results is mid-chain: stay smart so
        # the model can reason over what came back.
        if any(m.role == "tool" for m in messages):
            return "smart"

        lowered = last_user.lower()
        keywords: Iterable[str] = self.settings.get("llm.force_smart_keywords", []) or []
        if any(keyword in lowered for keyword in keywords):
            return "smart"

        return "fast" if len(last_user.split()) <= _FAST_TIER_MAX_WORDS else "smart"

    # -- the call ---------------------------------------------------------
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tier: Tier = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Complete `messages`, rotating keys and providers until one answers.

        Raises `AllProvidersFailed` only when every key of every provider is
        unusable - the single case spec 4.5 lets Adrien complain about.
        """
        if not self.slots:
            raise AllProvidersFailed(["no providers configured"])

        resolved_tier = self.choose_tier(messages) if tier == "auto" else tier
        temperature = (
            temperature if temperature is not None
            else float(self.settings.get("llm.temperature", 0.6))
        )
        max_tokens = (
            max_tokens if max_tokens is not None
            else int(self.settings.get("llm.max_tokens", 700))
        )
        timeout = float(self.settings.get("keys.request_timeout_seconds", 25.0))
        max_attempts = int(self.settings.get("keys.max_attempts_per_call", 8))

        message_list = list(messages)
        attempts: list[str] = []
        last_error: Exception | None = None
        budget = max_attempts
        started = time.perf_counter()

        for slot in self.slots:
            if budget <= 0:
                break
            for lease in slot.pool.leases(max_attempts=budget):
                budget -= 1
                model = slot.provider.model_for(resolved_tier)
                try:
                    result = await slot.provider.chat(
                        api_key=lease.key,
                        model=model,
                        messages=message_list,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                except ProviderError as exc:
                    last_error = exc
                    attempts.append(f"{lease.label}: {exc}")
                    if exc.rate_limited:
                        lease.rate_limited(exc.retry_after)
                        continue
                    if exc.retryable:
                        lease.failed()
                        continue
                    # Malformed request: every key would fail identically.
                    log.error("%s rejected the request, not retrying: %s", slot.name, exc)
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    last_error = exc
                    attempts.append(f"{lease.label}: {type(exc).__name__}")
                    lease.failed()
                    continue

                lease.success()
                result.key_label = lease.label
                # Local debug logging only - never surfaced to the user (4.8).
                log.info(
                    "llm ok tier=%s provider=%s model=%s key=%s %.0fms tools=%d total=%.0fms",
                    resolved_tier, slot.name, model, lease.label, result.latency_ms,
                    len(result.tool_calls), (time.perf_counter() - started) * 1000,
                )
                return result

            if slot.pool.configured and slot.pool.available_count() == 0:
                log.warning("all %s keys are cooling down, falling through", slot.name)

        # Zero attempts means no key was ever handed out, which is a different
        # failure from "every key was tried and rejected" - and the difference
        # is the whole diagnosis. Say which one it was.
        reason = ""
        if not attempts:
            configured = [slot for slot in self.slots if slot.pool.configured]
            if not configured:
                reason = "no API keys are configured in .env"
            else:
                waits = [
                    slot.pool.seconds_until_available() or 0.0 for slot in configured
                ]
                reason = (
                    f"every key across {len(configured)} provider(s) is still cooling "
                    f"down; the next frees up in {min(waits):.0f}s"
                )

        log.error("every provider failed after %d attempts%s",
                  len(attempts), f" ({reason})" if reason else "")
        raise AllProvidersFailed(attempts, last_error, reason=reason)

    async def complete(self, prompt: str, *, system: str = "", tier: Tier = "fast",
                       max_tokens: int = 400, temperature: float = 0.3) -> str:
        """One-shot text completion - used by the summarizer and text tools."""
        messages = [Message.system(system)] if system else []
        messages.append(Message.user(prompt))
        result = await self.chat(
            messages, tier=tier, max_tokens=max_tokens, temperature=temperature
        )
        return result.text

    # -- introspection ----------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Health snapshot for the menu bar and the `status` tool."""
        return {
            "providers": [
                {
                    "name": slot.name,
                    "keys": len(slot.pool),
                    "available": slot.pool.available_count(),
                    "detail": slot.pool.stats(),
                }
                for slot in self.slots
            ],
            "healthy": any(slot.pool.available_count() > 0 for slot in self.slots),
        }

    def reset_cooldowns(self) -> None:
        for slot in self.slots:
            slot.pool.reset()
