"""
ragnarok/memory/db.py
──────────────────────
SQLite database layer for Ragnarok's persistent memory.

Schema:
    memories(
        id            TEXT PRIMARY KEY,   -- UUID4
        kind          TEXT,               -- 'episodic' | 'semantic'
        text          TEXT,               -- raw text content
        embedding     BLOB,               -- numpy float32 array, little-endian
        created_at    REAL,               -- Unix timestamp
        last_used_at  REAL,               -- Unix timestamp
        use_count     INTEGER DEFAULT 0,
        importance    REAL DEFAULT 0.5,   -- 0..1
        pinned        INTEGER DEFAULT 0,  -- 0 | 1
        source        TEXT DEFAULT '',    -- source message / context
        superseded_by TEXT DEFAULT ''     -- id of the memory that replaced this
    )

All write operations are wrapped in transactions.
All read operations use prepared statements with parameterised queries.
"""

from __future__ import annotations

import sqlite3
import struct
import uuid
from pathlib import Path
from typing import Generator

import numpy as np

from ragnarok.config import settings

_DB_FILE_NAME = "ragnarok_memory.db"
_EMBEDDING_DTYPE = np.float32


def _get_db_path() -> Path:
    db_dir = settings.db_dir
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / _DB_FILE_NAME


def _encode_embedding(vector: list[float]) -> bytes:
    """Pack embedding as little-endian float32 bytes."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _decode_embedding(blob: bytes) -> np.ndarray:
    """Unpack little-endian float32 bytes to numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=_EMBEDDING_DTYPE)


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    kind          TEXT    NOT NULL DEFAULT 'episodic',
    text          TEXT    NOT NULL,
    embedding     BLOB    NOT NULL,
    created_at    REAL    NOT NULL,
    last_used_at  REAL    NOT NULL,
    use_count     INTEGER NOT NULL DEFAULT 0,
    importance    REAL    NOT NULL DEFAULT 0.5,
    pinned        INTEGER NOT NULL DEFAULT 0,
    source        TEXT    NOT NULL DEFAULT '',
    superseded_by TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_kind      ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_pinned    ON memories(pinned);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
"""


class MemoryDB:
    """
    Low-level SQLite interface for memory storage and retrieval.
    Use MemoryStore (recall.py) for the high-level API.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or _get_db_path()
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")   # write-ahead logging
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_CREATE_TABLE)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────────

    def insert(
        self,
        *,
        memory_id: str,
        kind: str,
        text: str,
        embedding: list[float],
        created_at: float,
        last_used_at: float,
        importance: float = 0.5,
        pinned: bool = False,
        source: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO memories
                (id, kind, text, embedding, created_at, last_used_at,
                 use_count, importance, pinned, source)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                memory_id, kind, text,
                _encode_embedding(embedding),
                created_at, last_used_at,
                importance, int(pinned), source,
            ),
        )
        self._conn.commit()

    def update_used(self, memory_id: str, now: float) -> None:
        """Increment use_count and update last_used_at."""
        self._conn.execute(
            "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
            (now, memory_id),
        )
        self._conn.commit()

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        self._conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ?",
            (new_id, old_id),
        )
        self._conn.commit()

    def delete(self, memory_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def delete_many(self, ids: list[str]) -> None:
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()

    def set_pinned(self, memory_id: str, pinned: bool) -> None:
        self._conn.execute(
            "UPDATE memories SET pinned = ? WHERE id = ?",
            (int(pinned), memory_id),
        )
        self._conn.commit()

    def update_importance(self, memory_id: str, importance: float) -> None:
        self._conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (min(1.0, max(0.0, importance)), memory_id),
        )
        self._conn.commit()

    def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        embedding: list[float] | None = None,
        importance: float | None = None,
        pinned: bool | None = None,
    ) -> bool:
        """Update fields of an existing memory. Returns True if row was updated."""
        updates: list[str] = []
        params: list[Any] = []
        if text is not None:
            updates.append("text = ?")
            params.append(text)
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(_encode_embedding(embedding))
        if importance is not None:
            updates.append("importance = ?")
            params.append(min(1.0, max(0.0, importance)))
        if pinned is not None:
            updates.append("pinned = ?")
            params.append(int(pinned))

        if not updates:
            return False

        params.append(memory_id)
        cur = self._conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_all_active(self) -> list[sqlite3.Row]:
        """Return all non-superseded memories (for similarity search)."""
        return self._conn.execute(
            "SELECT * FROM memories WHERE superseded_by = '' ORDER BY created_at DESC"
        ).fetchall()

    def get_by_id(self, memory_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

    def get_candidates_for_forgetting(
        self, threshold: float
    ) -> list[sqlite3.Row]:
        """
        Return non-pinned memories below the importance threshold,
        sorted by lowest importance first.
        """
        return self._conn.execute(
            """
            SELECT * FROM memories
            WHERE pinned = 0
              AND superseded_by = ''
              AND importance < ?
            ORDER BY importance ASC, last_used_at ASC
            """,
            (threshold,),
        ).fetchall()

    def get_episodic_for_consolidation(
        self, min_count: int = 5
    ) -> list[sqlite3.Row]:
        """
        Return episodic memories that haven't been consolidated yet
        (superseded_by == '') for grouping.
        """
        return self._conn.execute(
            """
            SELECT * FROM memories
            WHERE kind = 'episodic' AND superseded_by = ''
            ORDER BY created_at ASC
            """,
        ).fetchall()

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by = ''"
        ).fetchone()[0]

    def search_text(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        """Simple LIKE-based text search (fallback when embedding unavailable)."""
        return self._conn.execute(
            "SELECT * FROM memories WHERE text LIKE ? AND superseded_by = '' LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()

    # ── Embedding helpers ──────────────────────────────────────────────────────

    @staticmethod
    def decode_embedding(row: sqlite3.Row) -> np.ndarray:
        return _decode_embedding(row["embedding"])

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()
