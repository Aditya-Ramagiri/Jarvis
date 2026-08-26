"""ChromaDB vector store for semantic recall (spec 6.2).

One collection holds three kinds of record, distinguished by metadata:

* `fact`       - a durable statement about the user or their projects
* `summary`    - what happened in one past session
* `transcript` - a chunk of raw conversation

They share a collection on purpose. The question "what was I frustrated about
last week regarding the build system" (spec 6.1) should be answerable from
whichever of those actually holds the answer, and a single similarity search
across all three is both simpler and better than querying three stores and
trying to merge scores that were never comparable.

Embeddings come from Chroma's bundled MiniLM, which runs locally on CPU - no
embedding API calls, consistent with the local-first principle in spec 2.3.

If ChromaDB is not installed, `open_vector_store()` degrades to a keyword-
overlap implementation rather than failing. Recall is markedly worse and it
says so in the logs, but Adrien still remembers things, which matters more than
purity when a dependency fails to build.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from adrien.config import chroma_dir
from adrien.logging_setup import get_logger, redact

log = get_logger(__name__)

COLLECTION = "adrien_memory"


@dataclass
class MemoryRecord:
    """One retrievable piece of memory."""

    text: str
    kind: str = "fact"           # fact | summary | transcript
    session_id: str = ""
    category: str = "general"
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    extra: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        # Chroma metadata values must be scalars.
        data: dict[str, Any] = {
            "kind": self.kind,
            "session_id": self.session_id,
            "category": self.category,
            "created_at": self.created_at,
        }
        for key, value in self.extra.items():
            if isinstance(value, (str, int, float, bool)):
                data[key] = value
        return data


@dataclass
class SearchHit:
    record: MemoryRecord
    distance: float

    @property
    def relevance(self) -> float:
        """0-1, higher is better. Chroma returns squared L2 distance."""
        return 1.0 / (1.0 + max(0.0, self.distance))


class VectorStore(Protocol):
    def add(self, records: list[MemoryRecord]) -> int: ...
    def search(self, query: str, *, top_k: int = 6, kinds: list[str] | None = None,
               max_distance: float | None = None) -> list[SearchHit]: ...
    def count(self) -> int: ...
    def delete(self, ids: list[str]) -> None: ...


class ChromaVectorStore:
    """The real implementation: persistent, local, semantic."""

    def __init__(self, path: Path | None = None, collection: str = COLLECTION) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path = path or chroma_dir()
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            # Cosine beats Chroma's default L2 for sentence embeddings: it
            # ignores magnitude, so a one-line fact and a long summary compare
            # on meaning rather than length.
            metadata={"hnsw:space": "cosine"},
        )
        log.info("vector store open at %s (%d records)", self.path, self.count())

    def add(self, records: list[MemoryRecord]) -> int:
        if not records:
            return 0
        self._collection.upsert(
            ids=[record.id for record in records],
            documents=[redact(record.text) for record in records],
            metadatas=[record.metadata() for record in records],
        )
        return len(records)

    def search(self, query: str, *, top_k: int = 6, kinds: list[str] | None = None,
               max_distance: float | None = None) -> list[SearchHit]:
        if not query.strip() or self.count() == 0:
            return []
        where = {"kind": {"$in": kinds}} if kinds else None
        try:
            response = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, self.count())),
                where=where,
            )
        except Exception as exc:
            log.error("vector search failed: %s", exc)
            return []

        hits: list[SearchHit] = []
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        ids = (response.get("ids") or [[]])[0]

        for index, text in enumerate(documents):
            distance = float(distances[index]) if index < len(distances) else 1.0
            if max_distance is not None and distance > max_distance:
                continue
            metadata = metadatas[index] if index < len(metadatas) else {}
            hits.append(SearchHit(
                record=MemoryRecord(
                    id=ids[index] if index < len(ids) else uuid.uuid4().hex,
                    text=text,
                    kind=str(metadata.get("kind", "fact")),
                    session_id=str(metadata.get("session_id", "")),
                    category=str(metadata.get("category", "general")),
                    created_at=float(metadata.get("created_at", 0.0)),
                ),
                distance=distance,
            ))
        return hits

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:  # pragma: no cover
            return 0

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)


_WORD = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "about", "that", "this", "it",
    "i", "my", "me", "you", "your", "what", "when", "how", "did", "do", "does",
}


class KeywordVectorStore:
    """Fallback when ChromaDB is unavailable.

    TF-IDF-ish cosine over bag-of-words. It cannot do what the spec asks for -
    "frustrated about the build system" will not match "the CI keeps breaking"
    without shared words - so it warns on every construction. It exists so a
    failed `pip install chromadb` costs recall quality, not the whole memory
    subsystem.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.records: dict[str, MemoryRecord] = {}
        log.warning(
            "ChromaDB unavailable - memory is running on keyword matching. "
            "Semantic recall will be poor until 'pip install chromadb' succeeds."
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]

    def add(self, records: list[MemoryRecord]) -> int:
        for record in records:
            record.text = redact(record.text)
            self.records[record.id] = record
        return len(records)

    def _idf(self, term: str) -> float:
        containing = sum(1 for record in self.records.values()
                         if term in self._tokens(record.text))
        return math.log((1 + len(self.records)) / (1 + containing)) + 1.0

    def search(self, query: str, *, top_k: int = 6, kinds: list[str] | None = None,
               max_distance: float | None = None) -> list[SearchHit]:
        query_terms = set(self._tokens(query))
        if not query_terms:
            return []

        scored: list[SearchHit] = []
        for record in self.records.values():
            if kinds and record.kind not in kinds:
                continue
            terms = set(self._tokens(record.text))
            if not terms:
                continue
            overlap = query_terms & terms
            if not overlap:
                continue
            score = sum(self._idf(term) for term in overlap) / math.sqrt(len(terms))
            # Present as a distance so both stores speak the same language.
            distance = 1.0 / (1.0 + score)
            if max_distance is not None and distance > max_distance:
                continue
            scored.append(SearchHit(record=record, distance=distance))

        scored.sort(key=lambda hit: hit.distance)
        return scored[:top_k]

    def count(self) -> int:
        return len(self.records)

    def delete(self, ids: list[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


def open_vector_store(path: Path | None = None) -> VectorStore:
    """The real store when ChromaDB is importable, the fallback otherwise."""
    try:
        return ChromaVectorStore(path)
    except ImportError:
        return KeywordVectorStore(path)
    except Exception as exc:  # pragma: no cover - corrupt index, locked dir
        log.error("could not open ChromaDB (%s); falling back to keyword memory", exc)
        return KeywordVectorStore(path)
