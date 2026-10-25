"""
ragnarok/memory/working.py
───────────────────────────
Working memory: a short RAM buffer of the current task state,
checkpointed to disk after every significant change.

Used for crash recovery (Phase 6) — on startup the last checkpoint
is loaded to resume interrupted tasks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ragnarok.config import settings

logger = logging.getLogger(__name__)
_CHECKPOINT_FILE = "working_memory.json"


@dataclass
class WorkingMemory:
    """
    In-RAM task state that is checkpointed to disk after every mutation.

    Fields:
        task_id:        Unique ID for the current task (UUID4).
        goal:           What the agent is trying to accomplish.
        plan:           List of plan steps (each a dict with id, status, etc.).
        step_index:     Which step is currently executing.
        context:        Arbitrary key-value context for the current task.
        history_snap:   Snapshot of MessageHistory for recovery.
        started_at:     Unix timestamp when the task started.
        updated_at:     Unix timestamp of last checkpoint.
    """
    task_id: str = ""
    goal: str = ""
    plan: list[dict[str, Any]] = field(default_factory=list)
    step_index: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    history_snap: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── Mutation helpers ───────────────────────────────────────────────────────

    def set_goal(self, goal: str, task_id: str) -> None:
        self.goal = goal
        self.task_id = task_id
        self.started_at = time.time()
        self.updated_at = time.time()

    def update_context(self, key: str, value: Any) -> None:
        self.context[key] = value
        self.updated_at = time.time()

    def set_history(self, snapshot: list[dict]) -> None:
        self.history_snap = snapshot
        self.updated_at = time.time()

    def clear(self) -> None:
        self.task_id = ""
        self.goal = ""
        self.plan = []
        self.step_index = 0
        self.context = {}
        self.history_snap = []
        self.updated_at = time.time()

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemory":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class WorkingMemoryManager:
    """
    Manages a single WorkingMemory instance with automatic disk checkpointing.
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._dir = checkpoint_dir or settings.db_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _CHECKPOINT_FILE
        self._mem = WorkingMemory()

    @property
    def memory(self) -> WorkingMemory:
        return self._mem

    def checkpoint(self) -> None:
        """Write current working memory to disk atomically."""
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._mem.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self._path)
        logger.debug("Working memory checkpointed to %s", self._path)

    def load_checkpoint(self) -> bool:
        """
        Load the last checkpoint from disk.
        Returns True if a checkpoint was found and loaded, False otherwise.
        """
        if not self._path.exists():
            return False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._mem = WorkingMemory.from_dict(data)
            logger.info(
                "Loaded working memory checkpoint: task_id=%s goal=%r",
                self._mem.task_id, self._mem.goal[:60],
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load working memory checkpoint: %s", exc)
            return False

    def clear(self) -> None:
        """Clear working memory and remove checkpoint."""
        self._mem.clear()
        if self._path.exists():
            self._path.unlink()
        logger.info("Working memory cleared")
