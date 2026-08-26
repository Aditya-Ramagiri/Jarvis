"""SQLite: structured facts, transcripts and the productivity tools' data.

Two jobs, deliberately in one place:

1. **Memory metadata** (spec 6.2) - facts, sessions, transcripts. The vector
   store answers "what is this about"; SQLite answers "when, in which session,
   how confident, is it still true" and gives exact lookups without an
   embedding round trip.
2. **Productivity state** - reminders, notes, todos. Small, structured,
   queried by time and status. A vector store would be the wrong shape.

The fact schema is intentionally open (spec 6.2, last bullet): a fact is
`(subject, predicate, value)` with a free-form `category`, so facts about the
finance tracker or the Minecraft server need no migration - just a new
category string.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from adrien.config import sqlite_path
from adrien.logging_setup import get_logger, redact

log = get_logger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Durable facts about the user and their projects. Open by design: a new
-- subject area is a new `category` string, never a schema change.
CREATE TABLE IF NOT EXISTS facts (
    id          TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    value       TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    confidence  REAL NOT NULL DEFAULT 0.8,
    source_session TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    superseded  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_facts_subject  ON facts(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category, superseded);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    source      TEXT NOT NULL DEFAULT 'mac',
    summary     TEXT,
    turn_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    tool_calls  TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS reminders (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    due_at      REAL NOT NULL,
    created_at  REAL NOT NULL,
    fired       INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL DEFAULT 'reminder'
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(fired, due_at);

CREATE TABLE IF NOT EXISTS notes (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    tag         TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS todos (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    completed_at REAL
);
"""


@dataclass
class Fact:
    subject: str
    predicate: str
    value: str
    category: str = "general"
    confidence: float = 0.8
    source_session: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_sentence(self) -> str:
        """Natural-language form - what gets embedded and shown to the LLM."""
        return f"{self.subject} {self.predicate} {self.value}".strip()


class StructuredStore:
    """Thin, thread-safe SQLite wrapper."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else sqlite_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # The orchestrator, the reminder scheduler and the WebSocket server all
        # touch this; one connection guarded by a lock is simpler to reason
        # about than a pool, and the write volume here is trivial.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._db.execute(sql, tuple(params))
            self._db.commit()
            return cursor

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(params)).fetchall()

    # -- facts ------------------------------------------------------------
    def upsert_fact(self, fact: Fact) -> Fact:
        """Store a fact, superseding any earlier value for the same subject
        and predicate.

        Superseding rather than deleting: "the server moved to a new address"
        should not erase the fact that it used to be somewhere else, because
        the user may well ask what it was before.
        """
        now = time.time()
        with self._lock:
            existing = self._db.execute(
                "SELECT id, value FROM facts WHERE subject=? AND predicate=? AND superseded=0",
                (fact.subject, fact.predicate),
            ).fetchone()
            if existing and existing["value"] == fact.value:
                self._db.execute(
                    "UPDATE facts SET updated_at=?, confidence=MAX(confidence, ?) WHERE id=?",
                    (now, fact.confidence, existing["id"]),
                )
                self._db.commit()
                fact.id = existing["id"]
                return fact
            if existing:
                self._db.execute(
                    "UPDATE facts SET superseded=1, updated_at=? WHERE id=?", (now, existing["id"])
                )
            self._db.execute(
                "INSERT INTO facts (id, subject, predicate, value, category, confidence,"
                " source_session, created_at, updated_at, superseded)"
                " VALUES (?,?,?,?,?,?,?,?,?,0)",
                (fact.id, fact.subject, fact.predicate, redact(fact.value), fact.category,
                 fact.confidence, fact.source_session, now, now),
            )
            self._db.commit()
        return fact

    def get_facts(self, *, subject: str = "", category: str = "",
                  include_superseded: bool = False, limit: int = 200) -> list[Fact]:
        clauses, params = [], []
        if not include_superseded:
            clauses.append("superseded=0")
        if subject:
            clauses.append("LOWER(subject)=LOWER(?)")
            params.append(subject)
        if category:
            clauses.append("category=?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(
            f"SELECT * FROM facts {where} ORDER BY updated_at DESC LIMIT ?", [*params, limit]
        )
        return [_row_to_fact(row) for row in rows]

    def get_fact_by_id(self, fact_id: str) -> Fact | None:
        rows = self._query("SELECT * FROM facts WHERE id=?", (fact_id,))
        return _row_to_fact(rows[0]) if rows else None

    def forget_fact(self, fact_id: str) -> bool:
        cursor = self._execute("UPDATE facts SET superseded=1 WHERE id=?", (fact_id,))
        return cursor.rowcount > 0

    # -- sessions and transcripts -----------------------------------------
    def start_session(self, source: str = "mac") -> str:
        session_id = uuid.uuid4().hex
        self._execute(
            "INSERT INTO sessions (id, started_at, source) VALUES (?,?,?)",
            (session_id, time.time(), source),
        )
        return session_id

    def add_message(self, session_id: str, role: str, content: str,
                    tool_calls: list[dict[str, Any]] | None = None) -> str:
        message_id = uuid.uuid4().hex
        self._execute(
            "INSERT INTO messages (id, session_id, role, content, created_at, tool_calls)"
            " VALUES (?,?,?,?,?,?)",
            (message_id, session_id, role, redact(content), time.time(),
             json.dumps(tool_calls) if tool_calls else None),
        )
        self._execute(
            "UPDATE sessions SET turn_count = turn_count + 1 WHERE id=?", (session_id,)
        )
        return message_id

    def get_transcript(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,),
        )
        return [dict(row) for row in rows]

    def end_session(self, session_id: str, summary: str = "") -> None:
        self._execute(
            "UPDATE sessions SET ended_at=?, summary=? WHERE id=?",
            (time.time(), redact(summary), session_id),
        )

    def recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    # -- reminders and timers ---------------------------------------------
    def add_reminder(self, text: str, due_at: float, kind: str = "reminder") -> str:
        reminder_id = uuid.uuid4().hex
        self._execute(
            "INSERT INTO reminders (id, text, due_at, created_at, kind) VALUES (?,?,?,?,?)",
            (reminder_id, text, due_at, time.time(), kind),
        )
        return reminder_id

    def due_reminders(self, now: float | None = None) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM reminders WHERE fired=0 AND due_at<=? ORDER BY due_at",
            (now if now is not None else time.time(),),
        )
        return [dict(row) for row in rows]

    def pending_reminders(self) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM reminders WHERE fired=0 ORDER BY due_at")
        return [dict(row) for row in rows]

    def mark_reminder_fired(self, reminder_id: str) -> None:
        self._execute("UPDATE reminders SET fired=1 WHERE id=?", (reminder_id,))

    def cancel_reminder(self, reminder_id: str) -> bool:
        return self._execute("DELETE FROM reminders WHERE id=?", (reminder_id,)).rowcount > 0

    # -- notes ------------------------------------------------------------
    def add_note(self, text: str, tag: str = "") -> str:
        note_id = uuid.uuid4().hex
        self._execute(
            "INSERT INTO notes (id, text, tag, created_at) VALUES (?,?,?,?)",
            (note_id, redact(text), tag, time.time()),
        )
        return note_id

    def get_notes(self, tag: str = "", limit: int = 20) -> list[dict[str, Any]]:
        if tag:
            rows = self._query(
                "SELECT * FROM notes WHERE tag=? ORDER BY created_at DESC LIMIT ?", (tag, limit)
            )
        else:
            rows = self._query("SELECT * FROM notes ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    # -- todos ------------------------------------------------------------
    def add_todo(self, text: str) -> str:
        todo_id = uuid.uuid4().hex
        self._execute(
            "INSERT INTO todos (id, text, created_at) VALUES (?,?,?)",
            (todo_id, redact(text), time.time()),
        )
        return todo_id

    def get_todos(self, include_done: bool = False, limit: int = 50) -> list[dict[str, Any]]:
        where = "" if include_done else "WHERE done=0"
        rows = self._query(
            f"SELECT * FROM todos {where} ORDER BY done, created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    def complete_todo(self, text_fragment: str) -> dict[str, Any] | None:
        """Complete the best match for a spoken fragment.

        The user says "tick off the milk one", not a UUID, so matching is by
        substring against open items, most recent first.
        """
        rows = self._query(
            "SELECT * FROM todos WHERE done=0 AND LOWER(text) LIKE ? ORDER BY created_at DESC",
            (f"%{text_fragment.lower()}%",),
        )
        if not rows:
            return None
        row = dict(rows[0])
        self._execute(
            "UPDATE todos SET done=1, completed_at=? WHERE id=?", (time.time(), row["id"])
        )
        return row


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        subject=row["subject"],
        predicate=row["predicate"],
        value=row["value"],
        category=row["category"],
        confidence=row["confidence"],
        source_session=row["source_session"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
