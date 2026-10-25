"""
ragnarok/safety/encryption.py
──────────────────────────────
Fernet-based AES-128 encryption at rest for sensitive memories and secrets.

Features:
  - Generates or reads persistent secret key from data/.ragnarok_key (or environment/config)
  - Encrypts memory texts flagged as sensitive / secret
  - Transparent decryption with tamper detection (Fernet HMAC verification)
  - MemoryVault helper for storing encrypted key-value pairs
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from ragnarok.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_KEY_FILENAME = ".ragnarok_key"


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""


class SecretStore:
    """
    Manages symmetric encryption key and provides encrypt/decrypt helpers.
    """

    def __init__(self, key_path: Path | None = None, key_bytes: bytes | None = None) -> None:
        if key_bytes:
            self._key = key_bytes
        else:
            self._key_path = key_path or (settings.data_dir / _DEFAULT_KEY_FILENAME)
            self._key = self._load_or_create_key(self._key_path)

        self._fernet = Fernet(self._key)

    @staticmethod
    def _load_or_create_key(path: Path) -> bytes:
        if path.exists():
            try:
                key = path.read_bytes().strip()
                # Validate key structure
                Fernet(key)
                return key
            except Exception as exc:
                logger.warning("Could not read encryption key from %s (%s). Generating new key.", path, exc)

        # Generate a new Fernet key
        key = Fernet.generate_key()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(key)
            # Make readable only by current user where supported
            try:
                os.chmod(path, 0o600)
            except Exception:
                pass
            logger.info("Generated new encryption key at %s", path)
        except Exception as exc:
            logger.error("Failed to persist encryption key to %s: %s", path, exc)
        return key

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes = b"ragnarok_salt_v1") -> "SecretStore":
        """Derive a consistent Fernet key from a passphrase."""
        key = base64.urlsafe_b64encode(hashlib.scrypt(passphrase.encode(), salt=salt, n=16384, r=8, p=1, maxmem=0, dklen=32))
        return cls(key_bytes=key)

    def encrypt(self, plain_text: str) -> str:
        """
        Encrypt a UTF-8 string. Returns base64 URL-safe ciphertext token.
        """
        if not plain_text:
            return ""
        try:
            token = self._fernet.encrypt(plain_text.encode("utf-8"))
            return token.decode("ascii")
        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    def decrypt(self, token: str) -> str:
        """
        Decrypt a base64 ciphertext token back to plain text.
        """
        if not token:
            return ""
        try:
            data = self._fernet.decrypt(token.encode("ascii"))
            return data.decode("utf-8")
        except InvalidToken as exc:
            raise EncryptionError("Decryption failed: Invalid token or tampered ciphertext.") from exc
        except Exception as exc:
            raise EncryptionError(f"Decryption failed: {exc}") from exc

    def is_encrypted(self, value: str) -> bool:
        """Check if string appears to be a Fernet token (starts with gAAAAA)."""
        if not isinstance(value, str) or not value.startswith("gAAAAA"):
            return False
        try:
            self._fernet.decrypt(value.encode("ascii"))
            return True
        except Exception:
            return False


# Global default secret store
_secret_store: SecretStore | None = None


def get_secret_store() -> SecretStore:
    global _secret_store
    if _secret_store is None:
        _secret_store = SecretStore()
    return _secret_store
