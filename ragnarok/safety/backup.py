"""
ragnarok/safety/backup.py
─────────────────────────
Encrypted local backups and restoration for Ragnarok database and state.

Features:
  - Generates encrypted backup archives (.enc) using SecretStore AES-128 Fernet key.
  - Verifies HMAC authenticity and integrity upon restore (detects tampering).
  - Restores database cleanly with atomic replace.
  - Protects against key mismatch or corruption.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from ragnarok.config import settings
from ragnarok.safety.encryption import EncryptionError, SecretStore, get_secret_store

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Handles encrypted backup and restore operations.
    """

    def __init__(self, secret_store: SecretStore | None = None) -> None:
        self._store = secret_store or get_secret_store()

    def create_backup(
        self,
        db_path: Path | str,
        backup_dir: Path | str | None = None,
        passphrase: str | None = None,
    ) -> Path:
        """
        Create an encrypted backup file from the source database.

        Returns:
            Path to the generated encrypted backup file (.ragnarok_backup.enc).
        """
        src = Path(db_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Database file to back up does not exist: {src}")

        b_dir = Path(backup_dir).resolve() if backup_dir else (settings.db_dir / "backups")
        b_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        backup_filename = f"{src.stem}_{timestamp}.ragnarok_backup.enc"
        target_path = b_dir / backup_filename

        store = SecretStore.from_passphrase(passphrase) if passphrase else self._store

        # Read binary data and metadata
        raw_bytes = src.read_bytes()
        archive_payload = {
            "source_name": src.name,
            "created_at": timestamp,
            "byte_size": len(raw_bytes),
            "data_hex": raw_bytes.hex(),
        }
        json_str = json.dumps(archive_payload)

        # Encrypt with authentication
        encrypted_token = store.encrypt(json_str)
        target_path.write_text(encrypted_token, encoding="ascii")
        logger.info("Created encrypted backup at %s (%d bytes)", target_path, len(encrypted_token))
        return target_path

    def restore_backup(
        self,
        backup_file: Path | str,
        target_db_path: Path | str,
        passphrase: str | None = None,
    ) -> Path:
        """
        Decrypt and restore database from encrypted backup.

        Returns:
            Restored Path.

        Raises:
            EncryptionError: if decryption fails, key is incorrect, or archive is tampered.
        """
        b_path = Path(backup_file).resolve()
        if not b_path.exists():
            raise FileNotFoundError(f"Backup file not found: {b_path}")

        store = SecretStore.from_passphrase(passphrase) if passphrase else self._store

        encrypted_content = b_path.read_text(encoding="ascii").strip()
        try:
            decrypted_json = store.decrypt(encrypted_content)
            payload = json.loads(decrypted_json)
        except Exception as exc:
            raise EncryptionError(
                f"Failed to decrypt backup: invalid key or tampered ciphertext. {exc}"
            ) from exc

        raw_hex = payload.get("data_hex", "")
        db_bytes = bytes.fromhex(raw_hex)

        dest = Path(target_db_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file then atomic rename
        temp_dest = dest.with_suffix(".tmp_restore")
        temp_dest.write_bytes(db_bytes)
        temp_dest.replace(dest)

        logger.info("Successfully restored database to %s (%d bytes)", dest, len(db_bytes))
        return dest
