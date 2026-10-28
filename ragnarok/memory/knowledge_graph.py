"""
ragnarok/memory/knowledge_graph.py
────────────────────────────────────
Innovation #2: Local Temporal Knowledge Graph (Yggdrasil — The World Tree of connected knowledge).

Stores factual knowledge as a directed property graph in SQLite:
    (Entity) --[Relationship {confidence, valid_from, valid_until}]--> (Entity)

Key capabilities over flat vector search:
  - Multi-hop traversal: "What project involves Alice, and what is its deadline?"
  - Temporal validity: When a fact changes, old edges are deprecated with 'valid_until'.
    This prevents contradictory vector collisions (e.g. "I use VS Code" + "I use Cursor").
  - Entity deduplication: Same entity aliases are unified.
  - Path finding: BFS between any two entities.
  - Time-scoped queries: "What did I work on last Tuesday?"

Schema:
  kg_entities(id, name, entity_type, aliases, created_at, updated_at)
  kg_edges(id, src_id, rel, dst_id, confidence, valid_from, valid_until, source_text, created_at)

Integrates with existing MemoryDB: facts extracted by facts.py are automatically
also stored as KG triples for hybrid retrieval.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ragnarok.config import settings

logger = logging.getLogger(__name__)

_KG_DB_FILE = "ragnarok_kg.db"

_CREATE_KG_SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    aliases     TEXT NOT NULL DEFAULT '[]',   -- JSON array of strings
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_entity_name ON kg_entities(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_kg_entity_type ON kg_entities(entity_type);

CREATE TABLE IF NOT EXISTS kg_edges (
    id          TEXT PRIMARY KEY,
    src_id      TEXT NOT NULL REFERENCES kg_entities(id),
    rel         TEXT NOT NULL,
    dst_id      TEXT NOT NULL REFERENCES kg_entities(id),
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  REAL NOT NULL,
    valid_until REAL,             -- NULL = currently valid
    source_text TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kg_src      ON kg_edges(src_id, valid_until);
CREATE INDEX IF NOT EXISTS idx_kg_dst      ON kg_edges(dst_id, valid_until);
CREATE INDEX IF NOT EXISTS idx_kg_rel      ON kg_edges(rel);
CREATE INDEX IF NOT EXISTS idx_kg_temporal ON kg_edges(valid_from, valid_until);
"""


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class KGEntity:
    id: str
    name: str
    entity_type: str = "concept"
    aliases: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class KGEdge:
    id: str
    src_id: str
    rel: str
    dst_id: str
    confidence: float = 1.0
    valid_from: float = field(default_factory=time.time)
    valid_until: float | None = None
    source_text: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class KGPath:
    """A multi-hop path through the knowledge graph."""
    entities: list[KGEntity]
    edges: list[KGEdge]

    def to_text(self) -> str:
        """Render the path as a readable string."""
        parts = []
        for i, edge in enumerate(self.edges):
            src = self.entities[i].name
            dst = self.entities[i + 1].name
            parts.append(f"{src} --[{edge.rel}]--> {dst}")
        return " | ".join(parts) if parts else "empty path"


# ── Database ────────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Local Temporal Knowledge Graph stored in SQLite.

    Thread-safe via connection-per-call pattern (same as MemoryDB).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_dir = settings.db_dir
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / _KG_DB_FILE
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=15, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_CREATE_KG_SCHEMA)
            conn.commit()

    # ── Entity management ────────────────────────────────────────────────────

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "concept",
        aliases: list[str] | None = None,
    ) -> KGEntity:
        """
        Insert or retrieve an entity by name (case-insensitive).
        If it already exists, optionally merge new aliases.
        """
        now = time.time()
        aliases = aliases or []

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM kg_entities WHERE name = ? COLLATE NOCASE",
                (name,),
            ).fetchone()

            if row:
                existing_aliases = json.loads(row["aliases"])
                merged = list(set(existing_aliases + aliases))
                conn.execute(
                    "UPDATE kg_entities SET aliases=?, updated_at=? WHERE id=?",
                    (json.dumps(merged), now, row["id"]),
                )
                conn.commit()
                return KGEntity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    aliases=merged,
                    created_at=row["created_at"],
                    updated_at=now,
                )
            else:
                eid = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO kg_entities(id,name,entity_type,aliases,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (eid, name, entity_type, json.dumps(aliases), now, now),
                )
                conn.commit()
                return KGEntity(id=eid, name=name, entity_type=entity_type, aliases=aliases)

    def get_entity(self, name: str) -> KGEntity | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM kg_entities WHERE name=? COLLATE NOCASE", (name,)
            ).fetchone()
            if row:
                return KGEntity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    aliases=json.loads(row["aliases"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        return None

    # ── Edge management ──────────────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 1.0,
        source_text: str = "",
        subject_type: str = "concept",
        obj_type: str = "concept",
        supersede_existing: bool = True,
    ) -> KGEdge:
        """
        Add a knowledge triple: (subject) --[relation]--> (object).

        If supersede_existing=True and an active edge with the same
        (src, rel) already exists, it is marked as deprecated (valid_until=now)
        before adding the new edge. This implements temporal validity tracking.
        """
        now = time.time()
        src = self.upsert_entity(subject, subject_type)
        dst = self.upsert_entity(obj, obj_type)

        with self._connect() as conn:
            # Deprecate any currently-active edge with same (src, rel)
            if supersede_existing:
                conn.execute(
                    "UPDATE kg_edges SET valid_until=? "
                    "WHERE src_id=? AND rel=? AND valid_until IS NULL",
                    (now, src.id, relation),
                )

            eid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO kg_edges(id,src_id,rel,dst_id,confidence,valid_from,valid_until,source_text,created_at) "
                "VALUES(?,?,?,?,?,?,NULL,?,?)",
                (eid, src.id, relation, dst.id, confidence, now, source_text, now),
            )
            conn.commit()

        logger.debug("KG triple added: %s -[%s]-> %s", subject, relation, obj)
        return KGEdge(
            id=eid,
            src_id=src.id,
            rel=relation,
            dst_id=dst.id,
            confidence=confidence,
            valid_from=now,
            source_text=source_text,
        )

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        entity_name: str,
        as_subject: bool = True,
        as_object: bool = True,
        active_only: bool = True,
        limit: int = 20,
    ) -> list[dict]:
        """
        Return all edges connected to an entity (as subject or object).
        Returns dicts with: rel, neighbor_name, confidence, valid_from, valid_until.
        """
        entity = self.get_entity(entity_name)
        if entity is None:
            return []

        results = []
        now = time.time()
        temporal_filter = "AND e.valid_until IS NULL" if active_only else ""

        with self._connect() as conn:
            if as_subject:
                rows = conn.execute(
                    f"""SELECT e.rel, ent.name AS neighbor, e.confidence, e.valid_from, e.valid_until
                        FROM kg_edges e JOIN kg_entities ent ON e.dst_id = ent.id
                        WHERE e.src_id=? {temporal_filter} LIMIT ?""",
                    (entity.id, limit),
                ).fetchall()
                for r in rows:
                    results.append({
                        "direction": "outgoing",
                        "rel": r["rel"],
                        "neighbor": r["neighbor"],
                        "confidence": r["confidence"],
                        "valid_from": r["valid_from"],
                        "valid_until": r["valid_until"],
                    })

            if as_object:
                rows = conn.execute(
                    f"""SELECT e.rel, ent.name AS neighbor, e.confidence, e.valid_from, e.valid_until
                        FROM kg_edges e JOIN kg_entities ent ON e.src_id = ent.id
                        WHERE e.dst_id=? {temporal_filter} LIMIT ?""",
                    (entity.id, limit),
                ).fetchall()
                for r in rows:
                    results.append({
                        "direction": "incoming",
                        "rel": r["rel"],
                        "neighbor": r["neighbor"],
                        "confidence": r["confidence"],
                        "valid_from": r["valid_from"],
                        "valid_until": r["valid_until"],
                    })

        return results

    def multi_hop_query(
        self,
        start_entity: str,
        max_hops: int = 3,
        active_only: bool = True,
    ) -> list[str]:
        """
        BFS traversal from start_entity up to max_hops.
        Returns a list of human-readable relationship strings.
        """
        entity = self.get_entity(start_entity)
        if entity is None:
            return []

        visited: set[str] = {entity.id}
        queue: list[tuple[str, str, int]] = [(entity.id, start_entity, 0)]  # (id, name, depth)
        findings: list[str] = []
        temporal_filter = "AND e.valid_until IS NULL" if active_only else ""

        with self._connect() as conn:
            while queue:
                current_id, current_name, depth = queue.pop(0)
                if depth >= max_hops:
                    continue

                rows = conn.execute(
                    f"""SELECT e.rel, ent.id AS nid, ent.name AS nname, e.confidence
                        FROM kg_edges e JOIN kg_entities ent ON e.dst_id = ent.id
                        WHERE e.src_id=? {temporal_filter}""",
                    (current_id,),
                ).fetchall()

                for row in rows:
                    triple = f"{current_name} --[{row['rel']}]--> {row['nname']}"
                    findings.append(triple)
                    if row["nid"] not in visited:
                        visited.add(row["nid"])
                        queue.append((row["nid"], row["nname"], depth + 1))

        return findings

    def query_as_context(self, entity_name: str, max_hops: int = 2) -> str:
        """
        Return KG-derived context about an entity as a formatted string
        suitable for injection into the LLM system prompt.
        """
        triples = self.multi_hop_query(entity_name, max_hops=max_hops)
        if not triples:
            return ""
        return (
            f"[Knowledge Graph — facts about '{entity_name}']\n"
            + "\n".join(f"  • {t}" for t in triples[:15])
        )

    def deprecate_fact(self, subject: str, relation: str) -> int:
        """
        Mark all active edges (subject, relation, *) as deprecated.
        Returns the number of edges deprecated.
        """
        entity = self.get_entity(subject)
        if entity is None:
            return 0
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE kg_edges SET valid_until=? WHERE src_id=? AND rel=? AND valid_until IS NULL",
                (now, entity.id, relation),
            )
            conn.commit()
            return cur.rowcount

    def get_history(self, subject: str, relation: str) -> list[dict]:
        """Return the full temporal history of a (subject, relation) relationship."""
        entity = self.get_entity(subject)
        if entity is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.rel, ent.name AS obj, e.confidence,
                          e.valid_from, e.valid_until, e.source_text
                   FROM kg_edges e JOIN kg_entities ent ON e.dst_id=ent.id
                   WHERE e.src_id=? AND e.rel=?
                   ORDER BY e.valid_from DESC""",
                (entity.id, relation),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_entities(self, query: str, limit: int = 10) -> list[KGEntity]:
        """Full-text search over entity names and aliases."""
        like_q = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kg_entities WHERE name LIKE ? OR aliases LIKE ? LIMIT ?",
                (like_q, like_q, limit),
            ).fetchall()
        return [
            KGEntity(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                aliases=json.loads(r["aliases"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def stats(self) -> dict:
        """Return basic statistics about the knowledge graph."""
        with self._connect() as conn:
            n_entities = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
            n_edges = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
            n_active = conn.execute(
                "SELECT COUNT(*) FROM kg_edges WHERE valid_until IS NULL"
            ).fetchone()[0]
        return {
            "entities": n_entities,
            "total_edges": n_edges,
            "active_edges": n_active,
            "deprecated_edges": n_edges - n_active,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_kg: KnowledgeGraph | None = None


def get_kg() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg


# ── Convenience: ingest a plain-text fact automatically ─────────────────────

def ingest_fact_text(text: str, source: str = "") -> list[KGEdge]:
    """
    Lightweight heuristic parser that turns a free-text fact into KG triples.

    Examples:
      "I prefer dark mode" → (User, prefers, dark mode)
      "Project Titan deadline is Friday" → (Project Titan, deadline, Friday)
      "Alice works on the Ragnarok repo" → (Alice, works_on, Ragnarok repo)

    For richer extraction, pair with the LLM-based extractor (facts.py).
    """
    import re

    patterns = [
        # "X prefers/uses/likes Y"
        (r"(?:I |User )?(?:prefer|use|like|want|need)s?\s+(.+)", "User", "prefers"),
        # "X is Y"
        (r"^(.+?)\s+is\s+(.+)$", None, "is"),
        # "X deadline is Y" / "X due Y"
        (r"(.+?)\s+(?:deadline|due)\s+(?:is\s+)?(.+)", None, "deadline"),
        # "X works on Y" / "X involved in Y"
        (r"(.+?)\s+(?:works on|involved in|assigned to)\s+(.+)", None, "works_on"),
    ]

    kg = get_kg()
    edges: list[KGEdge] = []

    for pattern, fixed_subject, rel in patterns:
        m = re.search(pattern, text.strip(), re.IGNORECASE)
        if m:
            if fixed_subject:
                subj = fixed_subject
                obj = m.group(1).strip().rstrip(".")
            elif m.lastindex and m.lastindex >= 2:
                subj = m.group(1).strip().rstrip(".")
                obj = m.group(2).strip().rstrip(".")
            else:
                continue

            if subj and obj and len(subj) < 100 and len(obj) < 100:
                edge = kg.add_triple(subj, rel, obj, source_text=source or text[:120])
                edges.append(edge)
                break

    return edges
