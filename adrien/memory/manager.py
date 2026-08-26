"""The memory system's front door (spec section 6).

Ties the three pieces together:

    ChromaDB   - semantic search over facts, summaries and transcript chunks
    SQLite     - exact lookups, timestamps, supersession, productivity data
    Summarizer - post-session digest and fact extraction

The two calls the orchestrator makes:

* `recall(query)` before generating a response - retrieves what is relevant
  and formats it for the prompt (spec 6.2, second to last bullet).
* `end_session()` when a conversation finishes - summarises, extracts facts,
  and writes everything into both stores.

Every fact is written to *both* stores, on purpose. SQLite owns truth and
supersession; Chroma owns "what is this about". Writing to one and deriving the
other would mean either exact recall without semantics or semantics without a
way to say "that changed".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from adrien.config import Settings
from adrien.config import settings as global_settings
from adrien.logging_setup import get_logger
from adrien.memory.structured_store import Fact, StructuredStore
from adrien.memory.summarizer import SessionDigest, Summarizer
from adrien.memory.vector_store import MemoryRecord, SearchHit, VectorStore, open_vector_store

log = get_logger(__name__)

# Transcripts are chunked before embedding: one 40-minute conversation
# embedded as a single vector retrieves for everything and means nothing.
TRANSCRIPT_CHUNK_TURNS = 6


@dataclass
class Recollection:
    """What memory found for the current turn."""

    facts: list[SearchHit]
    history: list[SearchHit]

    @property
    def is_empty(self) -> bool:
        return not self.facts and not self.history

    def as_prompt(self) -> str:
        """Render for injection into the system prompt.

        Kept terse and explicitly labelled as background: a wall of recalled
        text makes the model narrate its own memory ("I remember you said...")
        instead of just using it.
        """
        if self.is_empty:
            return ""
        lines: list[str] = []
        if self.facts:
            lines.append("What you know about the user:")
            lines.extend(f"- {hit.record.text}" for hit in self.facts)
        if self.history:
            if lines:
                lines.append("")
            lines.append("From earlier conversations:")
            for hit in self.history:
                when = _relative_day(hit.record.created_at)
                lines.append(f"- ({when}) {hit.record.text}")
        lines.append(
            "\nUse this only if it is relevant. Do not mention that you looked it up."
        )
        return "\n".join(lines)


class MemoryManager:
    """Long-term memory, as the orchestrator sees it."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: StructuredStore | None = None,
        vectors: VectorStore | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.settings = settings or global_settings()
        self.enabled = bool(self.settings.get("memory.enabled", True))
        self.store = store or StructuredStore()
        self.vectors = vectors if vectors is not None else open_vector_store()
        self.summarizer = summarizer or Summarizer()
        self.session_id: str = ""

    # -- session lifecycle ------------------------------------------------
    def start_session(self, source: str = "mac") -> str:
        self.session_id = self.store.start_session(source)
        log.info("memory session %s started (%s)", self.session_id[:8], source)
        return self.session_id

    def record_turn(self, role: str, content: str,
                    tool_calls: list[dict[str, Any]] | None = None) -> None:
        """Append one turn to the raw transcript (spec 6.1, item 2)."""
        if not self.enabled or not self.session_id or not content.strip():
            return
        self.store.add_message(self.session_id, role, content, tool_calls)

    async def end_session(self) -> SessionDigest | None:
        """Summarise, extract facts and store everything (spec 6.2).

        Failures here are logged and swallowed: losing a summary is a much
        smaller problem than a background service that dies on shutdown.
        """
        if not self.enabled or not self.session_id:
            return None

        session_id, self.session_id = self.session_id, ""
        transcript = self.store.get_transcript(session_id)
        if not transcript:
            return None

        self.remember_transcript(session_id, transcript)

        if not self.settings.get("memory.summarize_on_session_end", True):
            self.store.end_session(session_id)
            return None

        try:
            digest = await self.summarizer.digest(transcript, session_id)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("could not digest session %s: %s", session_id[:8], exc)
            return None

        self.store.end_session(session_id, digest.summary)
        if digest.summary:
            self.vectors.add([MemoryRecord(
                text=digest.summary, kind="summary", session_id=session_id, category="session"
            )])
        for fact in digest.facts:
            self.remember_fact(fact)

        log.info("session %s digested: %d fact(s) stored", session_id[:8], len(digest.facts))
        return digest

    # -- writing ----------------------------------------------------------
    def remember_fact(self, fact: Fact) -> Fact:
        """Store a fact in both stores, superseding any older value."""
        stored = self.store.upsert_fact(fact)
        self.vectors.add([MemoryRecord(
            id=stored.id,
            text=stored.as_sentence(),
            kind="fact",
            session_id=stored.source_session or "",
            category=stored.category,
            created_at=stored.created_at,
        )])
        return stored

    def remember_transcript(self, session_id: str, transcript: list[dict[str, Any]]) -> int:
        """Chunk and embed the raw transcript (spec 6.1: keep both)."""
        turns = [
            f"{'User' if row['role'] == 'user' else 'Adrien'}: {row['content']}"
            for row in transcript
            if row.get("role") in ("user", "assistant") and (row.get("content") or "").strip()
        ]
        if not turns:
            return 0

        records: list[MemoryRecord] = []
        for start in range(0, len(turns), TRANSCRIPT_CHUNK_TURNS):
            chunk = turns[start:start + TRANSCRIPT_CHUNK_TURNS]
            records.append(MemoryRecord(
                text="\n".join(chunk),
                kind="transcript",
                session_id=session_id,
                category="conversation",
                created_at=transcript[0].get("created_at", time.time()),
            ))
        return self.vectors.add(records)

    def forget(self, fact_id: str) -> bool:
        """Retract a fact from both stores."""
        self.vectors.delete([fact_id])
        return self.store.forget_fact(fact_id)

    # -- reading ----------------------------------------------------------
    def recall(self, query: str, *, top_k: int | None = None) -> Recollection:
        """Find what is relevant to the current turn.

        Facts and history are retrieved separately rather than as one ranked
        list, because they play different roles in the prompt: facts are
        standing truth, history is context that may be stale. Merging them
        would let a chatty old transcript crowd out the one fact that matters.
        """
        if not self.enabled or not query.strip():
            return Recollection(facts=[], history=[])

        k = top_k or int(self.settings.get("memory.top_k", 6))
        max_distance = self.settings.get("memory.max_distance")
        max_distance = float(max_distance) if max_distance is not None else None

        try:
            facts = self.vectors.search(
                query, top_k=k, kinds=["fact"], max_distance=max_distance
            )
            history = self.vectors.search(
                query, top_k=max(2, k // 2), kinds=["summary", "transcript"],
                max_distance=max_distance,
            )
        except Exception as exc:  # pragma: no cover - a broken index
            log.error("recall failed: %s", exc)
            return Recollection(facts=[], history=[])

        log.debug("recall for %r: %d fact(s), %d history", query[:60], len(facts), len(history))
        return Recollection(facts=facts, history=history)

    def known_facts(self, category: str = "") -> list[Fact]:
        """Every current fact - used by the menu bar's memory inspector."""
        return self.store.get_facts(category=category)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "facts": len(self.store.get_facts(limit=10_000)),
            "vectors": self.vectors.count(),
            "sessions": len(self.store.recent_sessions(limit=1000)),
        }

    def close(self) -> None:
        self.store.close()


def _relative_day(timestamp: float) -> str:
    if not timestamp:
        return "earlier"
    days = (time.time() - timestamp) / 86400
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 8:
        return f"{int(days)} days ago"
    if days < 60:
        return f"{int(days / 7)} weeks ago"
    return f"{int(days / 30)} months ago"
