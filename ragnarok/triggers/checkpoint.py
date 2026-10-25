"""
ragnarok/triggers/checkpoint.py
────────────────────────────────
Phase 6: Session checkpoint — persist agent context across crashes / restarts.

Saves a lightweight JSON snapshot of the conversation to the RAGNAROK_MEMORY
sandbox folder.  On the next start the agent detects the checkpoint and can
offer to resume from where it left off.

Design:
  - Atomic write via a temp-file rename to prevent partial reads on crash.
  - Checksummed JSON so a tampered or corrupted checkpoint is detected and
    ignored rather than blindly loaded.
  - Fits inside the PathSandbox: always writes to settings.db_dir / checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ragnarok.config import settings

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "session_checkpoint.json"


class CheckpointError(Exception):
    """Raised when a checkpoint is corrupt, tampered, or unreadable."""


class SessionCheckpoint:
    """
    Persist and restore agent session state.

    Args:
        checkpoint_dir: Directory to store checkpoint files in.
                        Defaults to settings.db_dir / "checkpoints".

    Usage::

        cp = SessionCheckpoint()
        cp.save(
            history=[{"role": "user", "content": "Hello"}],
            plan_summary="Step 1: read file",
            metadata={"version": "0.3"},
        )

        data = cp.load()
        if data:
            print("Resuming from checkpoint:", data["saved_at"])
            cp.clear()    # Remove after successful resume
    """

    def __init__(self, checkpoint_dir: Path | str | None = None) -> None:
        if checkpoint_dir is None:
            self._dir = settings.db_dir / "checkpoints"
        else:
            self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _CHECKPOINT_FILENAME

    # ── Public API ─────────────────────────────────────────────────────────────

    def save(
        self,
        history: list[dict[str, str]],
        plan_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Atomically write a checkpoint snapshot.

        Args:
            history:      List of {"role": ..., "content": ...} dicts.
            plan_summary: Optional string summary of the active plan.
            metadata:     Any extra key/value data to store alongside.
        """
        payload: dict[str, Any] = {
            "saved_at": datetime.now().isoformat(),
            "history": history,
            "plan_summary": plan_summary,
            "metadata": metadata or {},
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        checksum = _sha256(serialized)
        envelope = {"checksum": checksum, "data": serialized}
        envelope_bytes = json.dumps(envelope, ensure_ascii=False).encode("utf-8")

        # Atomic write via temp file in same directory
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._dir, prefix=".tmp_", suffix=".json"
        )
        try:
            os.write(tmp_fd, envelope_bytes)
            os.close(tmp_fd)
            # Windows requires the target to not exist before replace
            if self._path.exists():
                self._path.unlink()
            os.replace(tmp_path, self._path)
            logger.info(
                "Session checkpoint saved → %s (%d history entries).",
                self._path,
                len(history),
            )
        except Exception:
            os.close(tmp_fd) if not os.get_inheritable(tmp_fd) else None
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> dict[str, Any] | None:
        """
        Load and verify the checkpoint.

        Returns:
            Parsed checkpoint dict, or None if no checkpoint exists.

        Raises:
            CheckpointError: If the checkpoint file is corrupt or tampered.
        """
        if not self._path.exists():
            return None
        try:
            raw = self._path.read_bytes()
            envelope = json.loads(raw)
            stored_checksum = envelope["checksum"]
            serialized = envelope["data"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            raise CheckpointError(
                f"Checkpoint file is unreadable or malformed: {exc}"
            ) from exc

        computed = _sha256(serialized)
        if computed != stored_checksum:
            raise CheckpointError(
                "Checkpoint checksum mismatch — file may be corrupt or tampered. "
                "The checkpoint will not be loaded."
            )

        payload = json.loads(serialized)
        logger.info(
            "Session checkpoint loaded (saved_at=%s, %d history entries).",
            payload.get("saved_at", "?"),
            len(payload.get("history", [])),
        )
        return payload

    def exists(self) -> bool:
        """Return True if a checkpoint file is present."""
        return self._path.exists()

    def clear(self) -> None:
        """Delete the checkpoint file after a successful resume."""
        if self._path.exists():
            self._path.unlink()
            logger.info("Session checkpoint cleared.")

    @property
    def path(self) -> Path:
        return self._path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
