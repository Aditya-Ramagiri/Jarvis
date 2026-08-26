"""The memory pipeline: storage, supersession, recall and digestion (spec 6)."""

from __future__ import annotations

import pytest

from adrien.config import DEFAULT_SETTINGS, Settings
from adrien.memory.manager import MemoryManager
from adrien.memory.structured_store import Fact, StructuredStore
from adrien.memory.summarizer import Summarizer, parse_digest, render_transcript
from adrien.memory.vector_store import KeywordVectorStore, MemoryRecord

pytestmark = pytest.mark.asyncio


@pytest.fixture
def store(tmp_path) -> StructuredStore:
    instance = StructuredStore(tmp_path / "test.sqlite3")
    yield instance
    instance.close()


@pytest.fixture
def memory(store) -> MemoryManager:
    """A manager on the keyword vector store, so tests need no ChromaDB."""
    import copy

    return MemoryManager(
        settings=Settings(copy.deepcopy(DEFAULT_SETTINGS)),
        store=store,
        vectors=KeywordVectorStore(),
        summarizer=Summarizer(router=None),
    )


# -- structured store -------------------------------------------------------
def test_a_fact_round_trips(store):
    store.upsert_fact(Fact("the Minecraft server", "is hosted at", "rhs.raidnxt.com", "gaming"))
    facts = store.get_facts(subject="the Minecraft server")
    assert len(facts) == 1
    assert facts[0].value == "rhs.raidnxt.com"
    assert facts[0].as_sentence() == "the Minecraft server is hosted at rhs.raidnxt.com"


def test_a_changed_value_supersedes_rather_than_deletes(store):
    store.upsert_fact(Fact("the server", "is hosted at", "old.example.com"))
    store.upsert_fact(Fact("the server", "is hosted at", "new.example.com"))

    current = store.get_facts(subject="the server")
    assert [fact.value for fact in current] == ["new.example.com"]
    # The old value is still there to answer "what was it before".
    history = store.get_facts(subject="the server", include_superseded=True)
    assert {fact.value for fact in history} == {"old.example.com", "new.example.com"}


def test_restating_the_same_fact_does_not_duplicate_it(store):
    first = store.upsert_fact(Fact("the user", "prefers", "a casual tone"))
    again = store.upsert_fact(Fact("the user", "prefers", "a casual tone"))
    assert first.id == again.id
    assert len(store.get_facts(subject="the user")) == 1


def test_facts_can_be_filtered_by_category(store):
    store.upsert_fact(Fact("the server", "runs", "Paper", "gaming"))
    store.upsert_fact(Fact("the ledger app", "lives at", "ledger.raidnxt.com", "finance"))
    assert len(store.get_facts(category="finance")) == 1


def test_a_new_subject_area_needs_no_schema_change(store):
    """Spec 6.2: facts about other projects must just work."""
    store.upsert_fact(Fact("the finance tracker", "has an API at", "/api/summary",
                           category="a-category-that-did-not-exist-before"))
    assert store.get_facts(category="a-category-that-did-not-exist-before")


def test_credentials_are_redacted_on_the_way_into_the_store(store, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN_FOR_TEST", "abcdef0123456789abcdef")
    store.upsert_fact(Fact("the api", "uses", "abcdef0123456789abcdef"))
    assert "abcdef0123456789abcdef" not in store.get_facts(subject="the api")[0].value


def test_transcripts_are_stored_per_session(store):
    session = store.start_session("mac")
    store.add_message(session, "user", "what's the server address")
    store.add_message(session, "assistant", "rhs.raidnxt.com")
    transcript = store.get_transcript(session)
    assert [row["role"] for row in transcript] == ["user", "assistant"]
    assert store.recent_sessions()[0]["turn_count"] == 2


# -- productivity data ------------------------------------------------------
def test_reminders_become_due(store):
    store.add_reminder("take the pizza out", due_at=1000.0)
    assert store.due_reminders(now=999.0) == []
    assert len(store.due_reminders(now=1001.0)) == 1


def test_a_fired_reminder_stops_coming_back(store):
    reminder_id = store.add_reminder("stand up", due_at=1000.0)
    store.mark_reminder_fired(reminder_id)
    assert store.due_reminders(now=2000.0) == []


def test_todos_are_completed_by_spoken_fragment(store):
    store.add_todo("buy oat milk")
    store.add_todo("renew the domain")
    completed = store.complete_todo("milk")
    assert completed["text"] == "buy oat milk"
    assert [todo["text"] for todo in store.get_todos()] == ["renew the domain"]


def test_completing_a_todo_that_does_not_exist_returns_nothing(store):
    assert store.complete_todo("something never written down") is None


# -- vector recall ----------------------------------------------------------
def test_recall_finds_a_relevant_fact(memory):
    memory.remember_fact(Fact("the Minecraft server", "is hosted at", "rhs.raidnxt.com", "gaming"))
    memory.remember_fact(Fact("the user", "prefers", "a casual tone", "preference"))

    recalled = memory.recall("what's the minecraft server address")
    assert recalled.facts
    assert "rhs.raidnxt.com" in recalled.facts[0].record.text


def test_recall_of_something_never_mentioned_comes_back_empty(memory):
    memory.remember_fact(Fact("the user", "prefers", "a casual tone"))
    assert memory.recall("quantum chromodynamics").is_empty


def test_facts_and_history_are_kept_apart(memory):
    memory.remember_fact(Fact("the build system", "is", "gradle", "development"))
    memory.vectors.add([MemoryRecord(
        text="User was frustrated that the gradle build kept failing",
        kind="summary", session_id="s1",
    )])

    recalled = memory.recall("gradle build")
    assert any("gradle" in hit.record.text for hit in recalled.facts)
    assert any("frustrated" in hit.record.text for hit in recalled.history)


def test_the_prompt_rendering_labels_both_kinds(memory):
    memory.remember_fact(Fact("the user", "lives in", "Dublin", "general"))
    memory.vectors.add([
        MemoryRecord(text="The user talked about moving away from Dublin", kind="summary")
    ])

    prompt = memory.recall("where does the user live").as_prompt()
    assert "What you know about the user:" in prompt
    assert "From earlier conversations:" in prompt
    assert "Dublin" in prompt
    # The model must use recalled context, not narrate it.
    assert "Do not mention that you looked it up." in prompt


def test_an_empty_recollection_renders_nothing(memory):
    assert memory.recall("anything at all").as_prompt() == ""


def test_disabled_memory_recalls_nothing(memory):
    memory.remember_fact(Fact("the user", "lives in", "Dublin"))
    memory.enabled = False
    assert memory.recall("where does the user live").is_empty


def test_a_forgotten_fact_leaves_both_stores(memory):
    fact = memory.remember_fact(Fact("the user", "drives", "a blue car"))
    assert memory.recall("what car").facts
    assert memory.forget(fact.id) is True
    assert memory.recall("what car").is_empty


def test_transcripts_are_chunked_not_stored_as_one_blob(memory):
    transcript = [
        {"role": "user" if index % 2 == 0 else "assistant",
         "content": f"turn number {index}", "created_at": 1000.0}
        for index in range(14)
    ]
    written = memory.remember_transcript("session-1", transcript)
    assert written == 3  # 14 turns at 6 per chunk


# -- digest parsing ---------------------------------------------------------
def test_a_clean_digest_parses():
    summary, facts = parse_digest(
        '{"summary": "Set up the server.", "facts": [{"subject": "the server",'
        ' "predicate": "runs", "value": "Paper 1.21", "category": "gaming"}]}',
        session_id="s1",
    )
    assert summary == "Set up the server."
    assert facts[0].value == "Paper 1.21"
    assert facts[0].source_session == "s1"


def test_a_fenced_digest_parses():
    summary, facts = parse_digest(
        'Here you go:\n```json\n{"summary": "Talked about CI.", "facts": []}\n```\n'
    )
    assert summary == "Talked about CI."
    assert facts == []


def test_non_json_output_is_still_kept_as_a_summary():
    summary, facts = parse_digest("The user asked about the weather and then left.")
    assert "weather" in summary
    assert facts == []


def test_incomplete_facts_are_dropped():
    _, facts = parse_digest(
        '{"summary": "x", "facts": [{"subject": "a", "predicate": "b"},'
        ' {"subject": "c", "predicate": "d", "value": "e"}]}'
    )
    assert len(facts) == 1


def test_a_fact_that_looks_like_a_credential_is_never_stored():
    _, facts = parse_digest(
        '{"summary": "x", "facts": [{"subject": "groq", "predicate": "key is",'
        ' "value": "gsk_abcdefghijklmnopqrstuvwxyz"}]}'
    )
    assert facts == []


def test_empty_output_is_handled():
    assert parse_digest("") == ("", [])


def test_transcript_rendering_names_the_speakers():
    rendered = render_transcript([
        {"role": "system", "content": "ignore me"},
        {"role": "user", "content": "what's up"},
        {"role": "assistant", "content": "not much"},
    ])
    assert "ignore me" not in rendered
    assert rendered == "User: what's up\nAdrien: not much"


# -- session lifecycle ------------------------------------------------------
class ScriptedSummarizer(Summarizer):
    def __init__(self, payload: str) -> None:
        super().__init__(router=None)
        self.payload = payload
        self.seen: list[str] = []

    async def digest(self, transcript, session_id=""):
        self.seen.append(session_id)
        summary, facts = parse_digest(self.payload, session_id)
        from adrien.memory.summarizer import SessionDigest

        return SessionDigest(summary=summary, facts=facts, raw=self.payload)


async def test_ending_a_session_stores_the_summary_and_its_facts(store):
    import copy

    summarizer = ScriptedSummarizer(
        '{"summary": "Fixed the flaky build.", "facts": [{"subject": "the CI",'
        ' "predicate": "breaks on", "value": "the integration suite",'
        ' "category": "development"}]}'
    )
    memory = MemoryManager(
        settings=Settings(copy.deepcopy(DEFAULT_SETTINGS)),
        store=store, vectors=KeywordVectorStore(), summarizer=summarizer,
    )

    memory.start_session()
    memory.record_turn("user", "the integration suite keeps failing in CI")
    memory.record_turn("assistant", "I had a look, it's a timeout")

    digest = await memory.end_session()
    assert digest is not None and "flaky" in digest.summary
    assert store.get_facts(subject="the CI")[0].value == "the integration suite"
    # And it is retrievable next session.
    assert memory.recall("what breaks in CI").facts


async def test_ending_an_empty_session_is_a_no_op(memory):
    memory.start_session()
    assert await memory.end_session() is None


async def test_ending_without_a_session_is_safe(memory):
    assert await memory.end_session() is None
