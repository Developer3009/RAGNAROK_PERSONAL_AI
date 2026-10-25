"""
ragnarok/safety/secret_masker.py
────────────────────────────────
Masks credentials, API keys, and sensitive tokens from logs and error strings.
"""

from __future__ import annotations

import logging
import re
from typing import Any

# Regexes for sensitive tokens
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token|bearer|auth|credentials)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"),
    re.compile(r"gAAAAA[a-zA-Z0-9_\-]{30,}"),  # Fernet ciphertext token
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{15,}"),
]

_MASK = "***REDACTED***"


class SecretMasker:
    """Utility to mask secrets in text or dictionary objects before logging."""

    @classmethod
    def mask_text(cls, text: str) -> str:
        if not text:
            return ""
        result = text
        for pat in _SECRET_PATTERNS:
            def repl(m: re.Match) -> str:
                # If pattern has 2 capture groups (key + value), preserve key
                if len(m.groups()) >= 2:
                    prefix = m.group(1)
                    return f"{prefix}={_MASK}"
                return _MASK

            result = pat.sub(repl, result)
        return result

    @classmethod
    def mask_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Deep copy and mask any keys containing sensitive words."""
        sensitive_keys = {"password", "secret", "token", "key", "auth", "credential", "private_key"}
        sanitized: dict[str, Any] = {}
        for k, v in data.items():
            k_lower = k.lower()
            if any(s in k_lower for s in sensitive_keys):
                sanitized[k] = _MASK
            elif isinstance(v, dict):
                sanitized[k] = cls.mask_dict(v)
            elif isinstance(v, str):
                sanitized[k] = cls.mask_text(v)
            else:
                sanitized[k] = v
        return sanitized


class SecretScrubbingFilter(logging.Filter):
    """Logging filter to scrub secrets from any record emitted to logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = SecretMasker.mask_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = SecretMasker.mask_dict(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    SecretMasker.mask_text(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True
