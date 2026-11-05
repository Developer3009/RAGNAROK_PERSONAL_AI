"""
ragnarok/memory/agenda.py
─────────────────────────
Persistent SQLite-backed Agenda database for calendar events and to-do tasks.

Provides full CRUD for:
  - CalendarEvent  — time-boxed entries with start/end, location, color, source
  - TodoTask       — task items with priority, status, due date, tags

Thread-safety: uses threading.local() for per-thread SQLite connections,
matching the EpisodicRingBuffer pattern already used in this codebase.

DB path: Path(settings.db_dir) / "ragnarok_agenda.db"
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragnarok.config import settings

logger = logging.getLogger(__name__)

# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class CalendarEvent:
    """A calendar event with start/end time range."""

    id: str
    title: str
    description: str
    start_time: str           # ISO 8601: "2026-11-10T09:00:00"
    end_time: str             # ISO 8601
    location: str = ""
    color: str = "#00dbe9"    # Ragnarok cyan default
    source: str = "manual"   # manual | email | agent
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class TodoTask:
    """A to-do task with priority and lifecycle status."""

    id: str
    title: str
    due_date: str             # ISO 8601
    priority: str             # low | medium | high | urgent
    status: str = "pending"   # pending | in_progress | completed | cancelled
    scheduled_time: str = ""
    description: str = ""
    completed_at: float = 0.0
    tags: list[str] = field(default_factory=list)
    source: str = "manual"   # manual | email | agent
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ── AgendaDB ──────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS calendar_events (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    start_time  TEXT NOT NULL,
    end_time    TEXT NOT NULL,
    location    TEXT NOT NULL DEFAULT '',
    color       TEXT NOT NULL DEFAULT '#00dbe9',
    source      TEXT NOT NULL DEFAULT 'manual',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS todo_tasks (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    due_date       TEXT NOT NULL,
    priority       TEXT NOT NULL DEFAULT 'medium',
    status         TEXT NOT NULL DEFAULT 'pending',
    scheduled_time TEXT NOT NULL DEFAULT '',
    description    TEXT NOT NULL DEFAULT '',
    completed_at   REAL NOT NULL DEFAULT 0.0,
    tags           TEXT NOT NULL DEFAULT '[]',
    source         TEXT NOT NULL DEFAULT 'manual',
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
"""


class AgendaDB:
    """Thread-safe SQLite CRUD store for calendar events and to-do tasks."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_dir = Path(settings.db_dir)
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "ragnarok_agenda.db"
        self._db_path = str(db_path)
        self._local = threading.local()
        # Initialise schema on the calling thread
        self._init_schema()
        logger.info("AgendaDB initialised at %s", self._db_path)

    # ── Connection management ─────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return (or create) a per-thread SQLite connection."""
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_DDL)
        conn.commit()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    def _row_to_event(self, row: sqlite3.Row) -> CalendarEvent:
        d = dict(row)
        return CalendarEvent(**d)

    def _row_to_task(self, row: sqlite3.Row) -> TodoTask:
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        return TodoTask(**d)

    # ── Calendar Events CRUD ──────────────────────────────────────────────────

    def create_event(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        color: str = "#00dbe9",
        source: str = "manual",
    ) -> CalendarEvent:
        """Insert a new calendar event and return it."""
        now = time.time()
        ev = CalendarEvent(
            id=self._new_id(),
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            location=location,
            color=color,
            source=source,
            created_at=now,
            updated_at=now,
        )
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO calendar_events
               (id, title, description, start_time, end_time, location, color, source, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ev.id, ev.title, ev.description, ev.start_time, ev.end_time,
             ev.location, ev.color, ev.source, ev.created_at, ev.updated_at),
        )
        conn.commit()
        logger.debug("Created event %s: %s", ev.id, ev.title)
        return ev

    def get_event(self, event_id: str) -> CalendarEvent | None:
        """Retrieve a single calendar event by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM calendar_events WHERE id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def list_events(
        self,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> list[CalendarEvent]:
        """List events, optionally filtered by ISO-8601 range."""
        conn = self._get_conn()
        if start and end:
            rows = conn.execute(
                "SELECT * FROM calendar_events WHERE start_time >= ? AND start_time <= ? ORDER BY start_time LIMIT ?",
                (start, end, limit),
            ).fetchall()
        elif start:
            rows = conn.execute(
                "SELECT * FROM calendar_events WHERE start_time >= ? ORDER BY start_time LIMIT ?",
                (start, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM calendar_events ORDER BY start_time LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def update_event(self, event_id: str, **kwargs: Any) -> CalendarEvent | None:
        """Update allowed fields of a calendar event. Returns updated event."""
        allowed = {"title", "description", "start_time", "end_time", "location", "color", "source"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_event(event_id)
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [event_id]
        conn = self._get_conn()
        conn.execute(f"UPDATE calendar_events SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return self.get_event(event_id)

    def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event. Returns True if a row was deleted."""
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))
        conn.commit()
        return cur.rowcount > 0

    # ── TodoTask CRUD ─────────────────────────────────────────────────────────

    def create_task(
        self,
        title: str,
        due_date: str,
        priority: str = "medium",
        description: str = "",
        scheduled_time: str = "",
        tags: list[str] | None = None,
        source: str = "manual",
    ) -> TodoTask:
        """Insert a new to-do task and return it."""
        now = time.time()
        task = TodoTask(
            id=self._new_id(),
            title=title,
            due_date=due_date,
            priority=priority,
            status="pending",
            scheduled_time=scheduled_time,
            description=description,
            completed_at=0.0,
            tags=tags or [],
            source=source,
            created_at=now,
            updated_at=now,
        )
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO todo_tasks
               (id, title, due_date, priority, status, scheduled_time, description,
                completed_at, tags, source, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (task.id, task.title, task.due_date, task.priority, task.status,
             task.scheduled_time, task.description, task.completed_at,
             json.dumps(task.tags), task.source, task.created_at, task.updated_at),
        )
        conn.commit()
        logger.debug("Created task %s: %s", task.id, task.title)
        return task

    def get_task(self, task_id: str) -> TodoTask | None:
        """Retrieve a single to-do task by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM todo_tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(
        self,
        status: str | None = None,
        priority: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> list[TodoTask]:
        """List tasks, optionally filtered by status, priority, and/or date range."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if priority:
            clauses.append("priority = ?")
            params.append(priority)
        if start:
            clauses.append("due_date >= ?")
            params.append(start)
        if end:
            clauses.append("due_date <= ?")
            params.append(end)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM todo_tasks {where} ORDER BY due_date, created_at LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_task(self, task_id: str, **kwargs: Any) -> TodoTask | None:
        """Update allowed fields of a to-do task. Returns updated task."""
        allowed = {"title", "due_date", "priority", "status", "scheduled_time",
                   "description", "completed_at", "tags", "source"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_task(task_id)
        # Serialise tags if present
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"])
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]
        conn = self._get_conn()
        conn.execute(f"UPDATE todo_tasks SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return self.get_task(task_id)

    def toggle_task(self, task_id: str) -> TodoTask | None:
        """Toggle a task between pending and completed. Returns updated task."""
        task = self.get_task(task_id)
        if not task:
            return None
        if task.status == "completed":
            return self.update_task(task_id, status="pending", completed_at=0.0)
        else:
            return self.update_task(task_id, status="completed", completed_at=time.time())

    def delete_task(self, task_id: str) -> bool:
        """Delete a to-do task. Returns True if a row was deleted."""
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM todo_tasks WHERE id = ?", (task_id,))
        conn.commit()
        return cur.rowcount > 0

    # ── Utility ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the per-thread connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# Module-level singleton — shared across the app process
_agenda_db: AgendaDB | None = None


def get_agenda_db() -> AgendaDB:
    """Return (or lazily create) the module-level AgendaDB singleton."""
    global _agenda_db
    if _agenda_db is None:
        _agenda_db = AgendaDB()
    return _agenda_db
