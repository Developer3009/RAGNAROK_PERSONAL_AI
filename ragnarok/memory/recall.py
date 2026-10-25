"""
ragnarok/memory/recall.py
──────────────────────────
High-level memory store: remember(), recall(), and duplicate detection.

remember(text, kind, source) → memory_id
    Embeds text, checks for near-duplicates, saves to DB.

recall(query, top_k) → list[MemoryEntry]
    Embeds query, returns top_k memories by cosine similarity
    with a recency boost, updates use_count on hits.

Also exposes list_memories(), pin(), delete(), export_json().
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from ragnarok import llm
from ragnarok.config import settings
from ragnarok.memory.db import MemoryDB
from ragnarok.safety.encryption import SecretStore, get_secret_store

logger = logging.getLogger(__name__)

# Cosine similarity threshold above which we consider two memories duplicates
_DEDUP_THRESHOLD = 0.95
# How many days old until recency boost is zero
_RECENCY_HALF_LIFE_DAYS = 7.0


@dataclass
class MemoryEntry:
    id: str
    kind: str
    text: str
    importance: float
    pinned: bool
    source: str
    created_at: float
    last_used_at: float
    use_count: int
    score: float = 0.0       # combined similarity + recency score

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86_400


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _recency_boost(created_at: float, weight: float) -> float:
    """
    Exponential decay recency score in [0, weight].
    Decays to weight/2 after _RECENCY_HALF_LIFE_DAYS days.
    """
    age_days = (time.time() - created_at) / 86_400
    decay = np.exp(-age_days / _RECENCY_HALF_LIFE_DAYS * np.log(2))
    return weight * float(decay)


class MemoryStore:
    """
    The main memory interface for Ragnarok.

    Args:
        db: MemoryDB instance (defaults to the configured DB file).
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        secret_store: SecretStore | None = None,
    ) -> None:
        self._db = db or MemoryDB()
        self._secrets = secret_store

    # ── remember ───────────────────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        kind: str = "episodic",
        source: str = "",
        importance: float = 0.5,
        pinned: bool = False,
    ) -> str:
        """
        Save a memory, performing duplicate detection first.

        - If a near-identical memory exists (cosine ≥ 0.95), update it
          instead of inserting a duplicate.
        - If the new fact contradicts an old one with the same embedding
          region, mark the old one superseded.

        Returns:
            The memory_id of the saved (or updated) memory.
        """
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty")

        # Embedding must always be calculated on plaintext for semantic retrieval
        vector = llm.embed(text)

        # Encrypt text if marked sensitive or secret
        stored_text = text
        if (kind in ("secret", "sensitive", "credential") or "sensitive" in source) and self._secrets is not None:
            stored_text = self._secrets.encrypt(text)

        # ── Duplicate / conflict detection ─────────────────────────────────────
        existing = self._db.get_all_active()
        for row in existing:
            old_vec = self._db.decode_embedding(row)
            sim = _cosine_similarity(np.array(vector, dtype=np.float32), old_vec)
            if sim >= _DEDUP_THRESHOLD:
                # Near-identical — update instead of inserting
                logger.info(
                    "Dedup: updating existing memory %s (sim=%.3f)", row["id"], sim
                )
                # Mark old as superseded, insert updated version
                new_id = str(uuid.uuid4())
                now = time.time()
                self._db.insert(
                    memory_id=new_id,
                    kind=kind,
                    text=stored_text,
                    embedding=vector,
                    created_at=row["created_at"],   # keep original timestamp
                    last_used_at=now,
                    importance=max(float(row["importance"]), importance),
                    pinned=bool(row["pinned"]) or pinned,
                    source=source or row["source"],
                )
                self._db.mark_superseded(row["id"], new_id)
                return new_id

        # ── Fresh insert ───────────────────────────────────────────────────────
        memory_id = str(uuid.uuid4())
        now = time.time()
        self._db.insert(
            memory_id=memory_id,
            kind=kind,
            text=stored_text,
            embedding=vector,
            created_at=now,
            last_used_at=now,
            importance=importance,
            pinned=pinned,
            source=source,
        )
        logger.info("Saved new %s memory: %s (chars=%d)", kind, memory_id, len(text))
        return memory_id

    # ── recall ─────────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int | None = None,
        kind_filter: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryEntry]:
        """
        Find the most relevant memories for `query`.

        Scoring = cosine_similarity + recency_boost
        Results are sorted descending by score.
        Updates use_count on returned memories.

        Args:
            query:           The text to match against memories.
            top_k:           How many to return (defaults to config recall_top_k).
            kind_filter:     If set, only return memories of this kind.
            query_embedding: Optional precomputed vector to bypass embed API call.

        Returns:
            List of MemoryEntry sorted by relevance, best first.
        """
        k = top_k or settings.recall_top_k
        t0 = time.perf_counter()

        if query_embedding is not None:
            query_vec = np.array(query_embedding, dtype=np.float32)
        else:
            query_vec = np.array(llm.embed(query), dtype=np.float32)
        norm_q = np.linalg.norm(query_vec)
        rows = self._db.get_all_active()

        if not rows:
            return []

        # Vectorized similarity computation
        filtered_rows = [r for r in rows if not kind_filter or r["kind"] == kind_filter]
        if not filtered_rows:
            return []

        mem_matrices = [self._db.decode_embedding(r) for r in filtered_rows]
        matrix = np.vstack(mem_matrices)  # Shape (N, D)
        norms_m = np.linalg.norm(matrix, axis=1)

        # Handle zero norms safely
        with np.errstate(divide='ignore', invalid='ignore'):
            sims = np.dot(matrix, query_vec) / (norms_m * norm_q)
            sims = np.nan_to_num(sims, nan=0.0)

        weights = settings.recency_boost_weight
        now = time.time()
        scored: list[tuple[float, Any]] = []
        for i, row in enumerate(filtered_rows):
            boost = _recency_boost(row["created_at"], weights)
            score = float(sims[i]) + boost
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        now = time.time()
        results: list[MemoryEntry] = []
        for score, row in top:
            self._db.update_used(row["id"], now)
            txt = row["text"]
            if self._secrets is not None and self._secrets.is_encrypted(txt):
                try:
                    txt = self._secrets.decrypt(txt)
                except Exception as exc:
                    logger.warning("Failed to decrypt memory %s: %s", row["id"], exc)
            results.append(
                MemoryEntry(
                    id=row["id"],
                    kind=row["kind"],
                    text=txt,
                    importance=float(row["importance"]),
                    pinned=bool(row["pinned"]),
                    source=row["source"],
                    created_at=float(row["created_at"]),
                    last_used_at=float(row["last_used_at"]),
                    use_count=int(row["use_count"]),
                    score=score,
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "recall query_len=%d results=%d latency_ms=%.1f",
            len(query), len(results), latency_ms,
        )
        if latency_ms > settings.max_recall_latency_ms:
            logger.warning(
                "⚠ Recall latency %.1fms exceeds target %.0fms — consider indexed search",
                latency_ms, settings.max_recall_latency_ms,
            )
        return results

    def recall_as_prompt(self, query: str) -> str:
        """
        Return recalled memories formatted for injection into the system prompt.
        Respects memory_budget_chars from config.
        """
        memories = self.recall(query)
        if not memories:
            return ""

        lines: list[str] = []
        total = 0
        budget = settings.memory_budget_chars

        for m in memories:
            line = f"- [{m.kind}] {m.text}"
            if total + len(line) > budget:
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines)

    # ── Management API ─────────────────────────────────────────────────────────

    def list_all(self) -> list[MemoryEntry]:
        """Return all active memories, newest first."""
        rows = self._db.get_all_active()
        return [
            MemoryEntry(
                id=row["id"], kind=row["kind"], text=row["text"],
                importance=float(row["importance"]), pinned=bool(row["pinned"]),
                source=row["source"], created_at=float(row["created_at"]),
                last_used_at=float(row["last_used_at"]), use_count=int(row["use_count"]),
            )
            for row in rows
        ]

    def pin(self, memory_id: str, pinned: bool = True) -> None:
        """Pin or unpin a memory (pinned memories are never forgotten)."""
        self._db.set_pinned(memory_id, pinned)

    def delete(self, memory_id: str, allow_pinned: bool = False) -> bool:
        """
        Permanently delete a memory.
        Returns False if memory is pinned and allow_pinned is False.
        """
        row = self._db.get_by_id(memory_id)
        if row is None:
            return False
        if row["pinned"] and not allow_pinned:
            logger.warning("Refusing to delete pinned memory %s", memory_id)
            return False
        self._db.delete(memory_id)
        return True

    def search_text(self, query: str) -> list[MemoryEntry]:
        """Simple text search (no embedding — fast, less accurate)."""
        rows = self._db.search_text(query)
        return [
            MemoryEntry(
                id=row["id"], kind=row["kind"], text=row["text"],
                importance=float(row["importance"]), pinned=bool(row["pinned"]),
                source=row["source"], created_at=float(row["created_at"]),
                last_used_at=float(row["last_used_at"]), use_count=int(row["use_count"]),
            )
            for row in rows
        ]

    def export_json(self) -> str:
        """Export all active memories as a JSON string."""
        entries = [asdict(m) for m in self.list_all()]
        return json.dumps(entries, indent=2, default=str)

    def count(self) -> int:
        return self._db.count()
