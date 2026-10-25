"""
ragnarok/memory/encrypted_db.py
───────────────────────────────
Encrypted-at-rest database wrapper for Ragnarok's persistent memory.

Protects memory files on disk using AES-128 Fernet encryption derived from SecretStore.
An external process or attacker attempting to inspect the database file on disk
without the encryption key cannot read any memories or SQLite tables.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ragnarok.config import settings
from ragnarok.memory.db import MemoryDB
from ragnarok.safety.encryption import EncryptionError, SecretStore, get_secret_store

logger = logging.getLogger(__name__)


class EncryptedMemoryDB:
    """
    Provides transparent encryption-at-rest for the Ragnarok memory database.

    On disk, the file is saved with .enc suffix as encrypted ciphertext.
    Opening without the valid SecretStore key raises EncryptionError.
    """

    def __init__(
        self,
        enc_path: Path | None = None,
        secret_store: SecretStore | None = None,
        passphrase: str | None = None,
    ) -> None:
        self._enc_path = enc_path or (settings.db_dir / "ragnarok_memory.db.enc")
        if passphrase:
            self._store = SecretStore.from_passphrase(passphrase)
        else:
            self._store = secret_store or get_secret_store()

        self._temp_dir = tempfile.TemporaryDirectory(prefix="ragnarok_sec_")
        self._temp_db_path = Path(self._temp_dir.name) / "decrypted.db"

        # Load and decrypt existing file if present
        if self._enc_path.exists():
            encrypted_data = self._enc_path.read_text(encoding="ascii").strip()
            if not encrypted_data:
                pass
            else:
                try:
                    decrypted_hex = self._store.decrypt(encrypted_data)
                    raw_bytes = bytes.fromhex(decrypted_hex)
                    self._temp_db_path.write_bytes(raw_bytes)
                except Exception as exc:
                    self._temp_dir.cleanup()
                    raise EncryptionError(
                        f"Cannot open database: invalid encryption key or corrupted file. {exc}"
                    ) from exc

        # Initialize underlying MemoryDB on decrypted instance
        self._db = MemoryDB(db_path=self._temp_db_path)

    # ── Forwarding MemoryDB interface ──────────────────────────────────────────

    def insert(
        self,
        *,
        text: str,
        embedding: list[float],
        kind: str = "semantic",
        memory_id: str | None = None,
        created_at: float | None = None,
        last_used_at: float | None = None,
        importance: float = 0.5,
        pinned: bool = False,
        source: str = "",
    ) -> str:
        now = time.time()
        mid = memory_id or str(uuid.uuid4())
        c_at = created_at if created_at is not None else now
        lu_at = last_used_at if last_used_at is not None else now
        self._db.insert(
            memory_id=mid,
            kind=kind,
            text=text,
            embedding=embedding,
            created_at=c_at,
            last_used_at=lu_at,
            importance=importance,
            pinned=pinned,
            source=source,
        )
        self.save_encrypted()
        return mid

    def get_by_id(self, memory_id: str) -> sqlite3.Row | None:
        return self._db.get_by_id(memory_id)

    def get_all_active(self) -> list[sqlite3.Row]:
        return self._db.get_all_active()

    def update_used(self, memory_id: str, now: float | None = None) -> None:
        n = now if now is not None else time.time()
        self._db.update_used(memory_id, n)
        self.save_encrypted()

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        self._db.mark_superseded(old_id, new_id)
        self.save_encrypted()

    def update_importance(self, memory_id: str, importance: float) -> None:
        self._db.update_importance(memory_id, importance)
        self.save_encrypted()

    def set_pinned(self, memory_id: str, pinned: bool = True) -> None:
        self._db.set_pinned(memory_id, pinned)
        self.save_encrypted()

    def delete(self, memory_id: str) -> None:
        self._db.delete(memory_id)
        self.save_encrypted()

    def delete_many(self, ids: list[str]) -> None:
        self._db.delete_many(ids)
        self.save_encrypted()

    def count(self) -> int:
        return self._db.count()

    def save_encrypted(self) -> None:
        """Checkpoint current database state to the encrypted file on disk."""
        if not self._temp_db_path.exists():
            return
        if hasattr(self._db, "_conn") and self._db._conn:
            self._db._conn.commit()
            self._db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        raw_bytes = self._temp_db_path.read_bytes()
        hex_data = raw_bytes.hex()
        ciphertext = self._store.encrypt(hex_data)

        self._enc_path.parent.mkdir(parents=True, exist_ok=True)
        temp_enc = self._enc_path.with_suffix(".tmp_enc")
        temp_enc.write_text(ciphertext, encoding="ascii")
        temp_enc.replace(self._enc_path)


    def close(self) -> None:
        """Flush changes, encrypt to disk, and remove plaintext temp database."""
        self.save_encrypted()
        if hasattr(self._db, "_conn") and self._db._conn:
            self._db._conn.close()
        try:
            self._temp_dir.cleanup()
        except Exception:
            pass

    def __enter__(self) -> "EncryptedMemoryDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
