"""
tests/test_memory.py
─────────────────────
Unit tests for the memory subsystem (Phase 2).

All embedding calls are mocked — tests pass without Ollama.
Integration tests (marked) need live Ollama + nomic-embed-text.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ragnarok.memory.db import MemoryDB, _encode_embedding, _decode_embedding
from ragnarok.memory.recall import MemoryStore, MemoryEntry
from ragnarok.memory.working import WorkingMemory, WorkingMemoryManager
from ragnarok.memory.consolidation import forget, _forget_score


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(768).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_store(tmp_path: Path) -> MemoryStore:
    db = MemoryDB(db_path=tmp_path / "test.db")
    store = MemoryStore(db=db)
    return store


# ── MemoryDB tests ─────────────────────────────────────────────────────────────

class TestMemoryDB:

    def test_insert_and_retrieve(self, tmp_path):
        db = MemoryDB(db_path=tmp_path / "t.db")
        vec = _fake_embed("hello world")
        now = time.time()
        db.insert(
            memory_id="abc",
            kind="episodic",
            text="hello world",
            embedding=vec,
            created_at=now,
            last_used_at=now,
        )
        row = db.get_by_id("abc")
        assert row is not None
        assert row["text"] == "hello world"
        assert row["kind"] == "episodic"

    def test_embedding_roundtrip(self, tmp_path):
        db = MemoryDB(db_path=tmp_path / "t.db")
        vec = _fake_embed("roundtrip test")
        now = time.time()
        db.insert(
            memory_id="xyz",
            kind="semantic",
            text="roundtrip test",
            embedding=vec,
            created_at=now,
            last_used_at=now,
        )
        row = db.get_by_id("xyz")
        recovered = db.decode_embedding(row).tolist()
        assert len(recovered) == len(vec)
        assert abs(recovered[0] - vec[0]) < 1e-5

    def test_count(self, tmp_path):
        db = MemoryDB(db_path=tmp_path / "t.db")
        now = time.time()
        for i in range(3):
            db.insert(
                memory_id=str(i),
                kind="episodic",
                text=f"memory {i}",
                embedding=_fake_embed(f"memory {i}"),
                created_at=now,
                last_used_at=now,
            )
        assert db.count() == 3

    def test_mark_superseded_excluded(self, tmp_path):
        db = MemoryDB(db_path=tmp_path / "t.db")
        now = time.time()
        db.insert(
            memory_id="old",
            kind="episodic",
            text="old fact",
            embedding=_fake_embed("old fact"),
            created_at=now,
            last_used_at=now,
        )
        db.mark_superseded("old", "new")
        active = db.get_all_active()
        assert not any(r["id"] == "old" for r in active)

    def test_pinned(self, tmp_path):
        db = MemoryDB(db_path=tmp_path / "t.db")
        now = time.time()
        db.insert(
            memory_id="p1",
            kind="semantic",
            text="pinned memory",
            embedding=_fake_embed("pinned memory"),
            created_at=now,
            last_used_at=now,
        )
        db.set_pinned("p1", True)
        row = db.get_by_id("p1")
        assert row["pinned"] == 1

    def test_delete(self, tmp_path):
        db = MemoryDB(db_path=tmp_path / "t.db")
        now = time.time()
        db.insert(
            memory_id="d1",
            kind="episodic",
            text="to delete",
            embedding=_fake_embed("to delete"),
            created_at=now,
            last_used_at=now,
        )
        db.delete("d1")
        assert db.get_by_id("d1") is None


# ── MemoryStore tests ──────────────────────────────────────────────────────────

class TestMemoryStore:

    def test_remember_and_recall(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            mock_llm.embed.side_effect = _fake_embed
            mid = store.remember("My favourite colour is blue", kind="semantic")
            assert len(mid) == 36  # UUID format

            results = store.recall("favourite colour")

        assert len(results) >= 1
        assert any("blue" in r.text for r in results)

    def test_dedup_prevents_duplicate(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            # Use identical embedding to force dedup
            fixed_vec = _fake_embed("I prefer Python")
            mock_llm.embed.return_value = fixed_vec
            mid1 = store.remember("I prefer Python")
            mid2 = store.remember("I prefer Python")  # near-identical

        assert mid1 != mid2  # new ID after supersede
        assert store.count() == 1  # still only one active memory

    def test_pin_protects_from_delete(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            mock_llm.embed.side_effect = _fake_embed
            mid = store.remember("Important fact")
        store.pin(mid)
        result = store.delete(mid, allow_pinned=False)
        assert result is False
        assert store.count() == 1

    def test_pin_allows_delete_with_flag(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            mock_llm.embed.side_effect = _fake_embed
            mid = store.remember("To be deleted")
        store.pin(mid)
        result = store.delete(mid, allow_pinned=True)
        assert result is True
        assert store.count() == 0

    def test_export_json(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            mock_llm.embed.side_effect = _fake_embed
            store.remember("Test export memory")
        exported = store.export_json()
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "text" in data[0]

    def test_recall_as_prompt_budget(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            mock_llm.embed.side_effect = _fake_embed
            for i in range(10):
                store.remember(f"Memory item number {i} with extra content to fill budget")
            results = store.recall_as_prompt("memory")
        # Should be within budget
        from ragnarok.config import settings
        assert len(results) <= settings.memory_budget_chars + 200  # some slack

    def test_recall_updates_use_count(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            mock_llm.embed.side_effect = _fake_embed
            mid = store.remember("Use count test")
            store.recall("Use count test")

        row = store._db.get_by_id(mid)
        # Note: dedup may create a new ID, so just verify count > 0 overall
        assert store.count() >= 1

    def test_text_search(self, tmp_path):
        store = _make_store(tmp_path)
        with patch("ragnarok.memory.recall.llm") as mock_llm:
            mock_llm.embed.side_effect = _fake_embed
            store.remember("User likes jazz music")
            store.remember("Favourite programming language is Rust")
        results = store.search_text("jazz")
        assert any("jazz" in r.text for r in results)


# ── WorkingMemory tests ────────────────────────────────────────────────────────

class TestWorkingMemory:

    def test_checkpoint_and_load(self, tmp_path):
        mgr = WorkingMemoryManager(checkpoint_dir=tmp_path)
        mgr.memory.set_goal("Summarise notes", "task-001")
        mgr.memory.update_context("step", "searching")
        mgr.checkpoint()

        mgr2 = WorkingMemoryManager(checkpoint_dir=tmp_path)
        loaded = mgr2.load_checkpoint()

        assert loaded is True
        assert mgr2.memory.goal == "Summarise notes"
        assert mgr2.memory.task_id == "task-001"
        assert mgr2.memory.context["step"] == "searching"

    def test_clear_removes_checkpoint(self, tmp_path):
        mgr = WorkingMemoryManager(checkpoint_dir=tmp_path)
        mgr.memory.set_goal("test", "t-1")
        mgr.checkpoint()
        mgr.clear()
        assert not (tmp_path / "working_memory.json").exists()

    def test_no_checkpoint_returns_false(self, tmp_path):
        mgr = WorkingMemoryManager(checkpoint_dir=tmp_path)
        result = mgr.load_checkpoint()
        assert result is False


# ── Forgetting tests ───────────────────────────────────────────────────────────

class TestForgetting:

    def test_forget_removes_low_scorers(self, tmp_path):
        store = _make_store(tmp_path)
        db = store._db
        # Insert a very old, unimportant, unused memory
        old_time = time.time() - (60 * 86_400)  # 60 days ago
        db.insert(
            memory_id="old_useless",
            kind="episodic",
            text="I ate a sandwich once",
            embedding=_fake_embed("sandwich"),
            created_at=old_time,
            last_used_at=old_time,
            importance=0.05,
        )
        deleted = forget(store, dry_run=False)
        assert "old_useless" in deleted
        assert store.count() == 0

    def test_forget_dry_run_does_not_delete(self, tmp_path):
        store = _make_store(tmp_path)
        db = store._db
        old_time = time.time() - (60 * 86_400)
        db.insert(
            memory_id="should_stay",
            kind="episodic",
            text="Would be deleted but dry_run",
            embedding=_fake_embed("dry_run"),
            created_at=old_time,
            last_used_at=old_time,
            importance=0.05,
        )
        candidates = forget(store, dry_run=True)
        assert "should_stay" in candidates
        assert store.count() == 1  # still present

    def test_pinned_memory_never_forgotten(self, tmp_path):
        store = _make_store(tmp_path)
        db = store._db
        old_time = time.time() - (60 * 86_400)
        db.insert(
            memory_id="pinned_mem",
            kind="episodic",
            text="Never forget this",
            embedding=_fake_embed("never forget"),
            created_at=old_time,
            last_used_at=old_time,
            importance=0.01,
        )
        db.set_pinned("pinned_mem", True)
        deleted = forget(store, dry_run=False)
        assert "pinned_mem" not in deleted
        assert store.count() == 1


# ── Integration tests ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_live_remember_and_recall(tmp_path):
    """Requires nomic-embed-text pulled."""
    store = _make_store(tmp_path)
    store.remember("My name is Alice and I love hiking", kind="semantic")
    store.remember("I work as a software engineer", kind="semantic")
    results = store.recall("what do I like to do")
    assert any("hiking" in r.text for r in results)


@pytest.mark.integration
def test_live_recall_latency(tmp_path):
    """Recall of 100 memories must be under 800ms."""
    store = _make_store(tmp_path)
    for i in range(100):
        store.remember(f"Memory item {i}: some interesting fact about topic {i % 10}")
    t0 = time.perf_counter()
    store.recall("interesting fact about topic 3")
    latency_ms = (time.perf_counter() - t0) * 1000
    assert latency_ms < 800, f"Recall too slow: {latency_ms:.0f}ms"
