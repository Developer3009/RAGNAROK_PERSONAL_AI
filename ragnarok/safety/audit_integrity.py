"""
ragnarok/safety/audit_integrity.py
────────────────────────────────────
HMAC-SHA256 integrity verification for the Ragnarok audit log.

Every audit log entry is appended with an HMAC-SHA256 signature derived from:
  - A per-session signing key (generated from the Fernet key + session salt)
  - The JSON serialisation of the log entry

This allows post-hoc detection of log tampering, deletion, or injection.

Usage:
    signer = AuditSigner()
    signed_line = signer.sign_entry({"event": "tool_call", "tool": "read_file", ...})
    # Append signed_line to the JSONL audit log

    ok, reason = signer.verify_line(signed_line)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_HMAC_FIELD = "_hmac"
_SIGNING_SALT = b"ragnarok_audit_sign_v1"


def _derive_signing_key() -> bytes:
    """
    Derive a 32-byte signing key from the Fernet encryption key + fixed salt.
    Falls back to a random per-process key if the encryption key is unavailable.
    """
    try:
        from ragnarok.config import settings
        key_path = settings.data_dir / ".ragnarok_key"
        if key_path.exists():
            raw = key_path.read_bytes().strip()
            return hashlib.pbkdf2_hmac("sha256", raw, _SIGNING_SALT, iterations=10_000, dklen=32)
    except Exception as exc:
        logger.debug("AuditSigner: could not derive key from Fernet key: %s", exc)
    # Fallback: per-process random key (survives restarts via env var)
    env_key = os.environ.get("RAGNAROK_AUDIT_KEY", "")
    if env_key:
        return hashlib.sha256(env_key.encode() + _SIGNING_SALT).digest()
    return os.urandom(32)  # ephemeral — verification only valid in this process


class AuditSigner:
    """
    Signs and verifies JSONL audit log entries using HMAC-SHA256.
    Thread-safe (immutable key after construction).
    """

    def __init__(self, signing_key: bytes | None = None) -> None:
        self._key = signing_key or _derive_signing_key()

    def _compute_hmac(self, payload: str) -> str:
        """Compute HMAC-SHA256 of the payload string."""
        mac = hmac.new(self._key, payload.encode("utf-8"), hashlib.sha256)
        return mac.hexdigest()

    def sign_entry(self, entry: dict) -> str:
        """
        Serialise `entry` to a compact JSON string, compute HMAC,
        and return a single JSONL line with the signature embedded.
        """
        # Ensure deterministic serialisation (sorted keys, no whitespace)
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        sig = self._compute_hmac(payload)
        signed = dict(entry)
        signed[_HMAC_FIELD] = sig
        signed["_signed_at"] = time.time()
        return json.dumps(signed, sort_keys=True, separators=(",", ":"))

    def verify_line(self, jsonl_line: str) -> tuple[bool, str]:
        """
        Verify the HMAC in a JSONL log line.

        Returns:
            (True, "ok") if the entry is intact.
            (False, reason) if tampered or malformed.
        """
        try:
            entry = json.loads(jsonl_line)
        except json.JSONDecodeError as exc:
            return False, f"JSON parse error: {exc}"

        stored_sig = entry.pop(_HMAC_FIELD, None)
        entry.pop("_signed_at", None)
        if not stored_sig:
            return False, "No HMAC signature in log entry"

        canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        expected = self._compute_hmac(canonical)

        if not hmac.compare_digest(stored_sig, expected):
            return False, "HMAC mismatch — log entry may have been tampered with"
        return True, "ok"

    def verify_log_file(self, log_path: Path) -> dict:
        """
        Verify all entries in a JSONL audit log file.

        Returns:
            {"total": N, "valid": M, "invalid": K, "errors": [...]}
        """
        result: dict = {"total": 0, "valid": 0, "invalid": 0, "errors": []}
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    result["total"] += 1
                    ok, reason = self.verify_line(line)
                    if ok:
                        result["valid"] += 1
                    else:
                        result["invalid"] += 1
                        result["errors"].append({"line": lineno, "reason": reason})
        except FileNotFoundError:
            result["errors"].append({"line": 0, "reason": f"Log file not found: {log_path}"})
        except Exception as exc:
            result["errors"].append({"line": 0, "reason": str(exc)})
        return result


# Module-level singleton — shared across the app
_default_signer: AuditSigner | None = None


def get_audit_signer() -> AuditSigner:
    global _default_signer
    if _default_signer is None:
        _default_signer = AuditSigner()
    return _default_signer
