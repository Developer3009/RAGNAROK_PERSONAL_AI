"""
ragnarok/safety/audit.py
─────────────────────────
Append-only security audit log for Ragnarok.

Every tool call, approval decision, permission check, and safety event is
recorded as a JSON line to logs/security_audit.jsonl.

Design:
  - Each record is a single JSON object terminated by a newline (JSONL format).
  - Records are appended atomically (one line per event).
  - File is never rewritten — only appended.
  - Structured fields enable grep/jq queries and dashboard visualisation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ragnarok.config import settings

logger = logging.getLogger(__name__)

_AUDIT_FILE = "security_audit.jsonl"


class AuditEvent(str, Enum):
    TOOL_CALL = "tool_call"
    TOOL_SUCCESS = "tool_success"
    TOOL_ERROR = "tool_error"
    TOOL_DENIED = "tool_denied"
    TOOL_RATE_LIMITED = "tool_rate_limited"
    TOOL_DOOM_LOOP = "tool_doom_loop"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    PERMISSION_DENIED = "permission_denied"
    SANDBOX_VIOLATION = "sandbox_violation"
    ENCRYPTION_WRITE = "encryption_write"
    ENCRYPTION_READ = "encryption_read"
    SESSION_START = "session_start"
    SESSION_RESET = "session_reset"


@dataclass
class AuditRecord:
    event: str
    tool_name: str = ""
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    risk_level: str = ""
    permitted: bool = True
    latency_ms: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, default=str)


class AuditLogger:
    """
    Thread-safe append-only JSONL security audit logger.

    Args:
        log_dir: Directory where security_audit.jsonl is written.
                 Defaults to settings.log_dir.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._dir = log_dir or settings.log_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _AUDIT_FILE
        self._lock = threading.Lock()
        self._session_id = f"session-{int(time.time())}"
        logger.debug("AuditLogger initialised: %s", self._path)

    # ── Public event methods ───────────────────────────────────────────────────

    def log_tool_call(
        self, tool_name: str, args: dict[str, Any], risk_level: str = "low"
    ) -> None:
        self._write(AuditRecord(
            event=AuditEvent.TOOL_CALL,
            tool_name=tool_name,
            risk_level=risk_level,
            details={"args_keys": list(args.keys())},
            session_id=self._session_id,
        ))

    def log_tool_success(
        self, tool_name: str, latency_ms: float, result_len: int
    ) -> None:
        self._write(AuditRecord(
            event=AuditEvent.TOOL_SUCCESS,
            tool_name=tool_name,
            latency_ms=latency_ms,
            details={"result_len": result_len},
            session_id=self._session_id,
        ))

    def log_tool_error(
        self, tool_name: str, error: str, latency_ms: float = 0.0
    ) -> None:
        self._write(AuditRecord(
            event=AuditEvent.TOOL_ERROR,
            tool_name=tool_name,
            permitted=False,
            latency_ms=latency_ms,
            details={"error": error[:200]},
            session_id=self._session_id,
        ))

    def log_denied(self, tool_name: str, reason: str, risk_level: str = "") -> None:
        self._write(AuditRecord(
            event=AuditEvent.TOOL_DENIED,
            tool_name=tool_name,
            permitted=False,
            risk_level=risk_level,
            details={"reason": reason},
            session_id=self._session_id,
        ))

    def log_rate_limited(self, tool_name: str, window_sec: float, limit: int) -> None:
        self._write(AuditRecord(
            event=AuditEvent.TOOL_RATE_LIMITED,
            tool_name=tool_name,
            permitted=False,
            details={"window_sec": window_sec, "limit": limit},
            session_id=self._session_id,
        ))

    def log_doom_loop(self, tool_name: str) -> None:
        self._write(AuditRecord(
            event=AuditEvent.TOOL_DOOM_LOOP,
            tool_name=tool_name,
            permitted=False,
            details={"message": "Consecutive identical failing calls detected"},
            session_id=self._session_id,
        ))

    def log_approval_requested(self, tool_name: str, risk_level: str) -> None:
        self._write(AuditRecord(
            event=AuditEvent.APPROVAL_REQUESTED,
            tool_name=tool_name,
            risk_level=risk_level,
            session_id=self._session_id,
        ))

    def log_approval_granted(self, tool_name: str) -> None:
        self._write(AuditRecord(
            event=AuditEvent.APPROVAL_GRANTED,
            tool_name=tool_name,
            permitted=True,
            session_id=self._session_id,
        ))

    def log_approval_denied(self, tool_name: str) -> None:
        self._write(AuditRecord(
            event=AuditEvent.APPROVAL_DENIED,
            tool_name=tool_name,
            permitted=False,
            session_id=self._session_id,
        ))

    def log_permission_denied(self, tool_name: str, reason: str) -> None:
        self._write(AuditRecord(
            event=AuditEvent.PERMISSION_DENIED,
            tool_name=tool_name,
            permitted=False,
            details={"reason": reason},
            session_id=self._session_id,
        ))

    def log_sandbox_violation(self, tool_name: str, path: str, reason: str) -> None:
        self._write(AuditRecord(
            event=AuditEvent.SANDBOX_VIOLATION,
            tool_name=tool_name,
            permitted=False,
            details={"path": path, "reason": reason[:200]},
            session_id=self._session_id,
        ))

    def log_session_start(self) -> None:
        self._write(AuditRecord(
            event=AuditEvent.SESSION_START,
            session_id=self._session_id,
        ))

    def log_session_reset(self) -> None:
        self._write(AuditRecord(
            event=AuditEvent.SESSION_RESET,
            session_id=self._session_id,
        ))

    def log_encryption_event(self, event_type: str, memory_id: str) -> None:
        audit_event = (
            AuditEvent.ENCRYPTION_WRITE
            if event_type == "write"
            else AuditEvent.ENCRYPTION_READ
        )
        self._write(AuditRecord(
            event=audit_event,
            details={"memory_id": memory_id[:16]},
            session_id=self._session_id,
        ))

    # ── Query helpers ──────────────────────────────────────────────────────────

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last n audit records as dicts."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        tail_lines = lines[-n:]
        records = []
        for line in tail_lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records

    def count_by_event(self) -> dict[str, int]:
        """Return event type frequency counts across the full log."""
        if not self._path.exists():
            return {}
        counts: dict[str, int] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                evt = rec.get("event", "unknown")
                counts[evt] = counts.get(evt, 0) + 1
            except json.JSONDecodeError:
                pass
        return counts

    # ── Internal ───────────────────────────────────────────────────────────────

    def _write(self, record: AuditRecord) -> None:
        try:
            line = record.to_json() + "\n"
            with self._lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as exc:
            logger.error("AuditLogger write failed: %s", exc)


# Module-level singleton (overridable for testing)
audit_logger = AuditLogger()
