"""
ragnarok/skills/read_file.py
─────────────────────────────
Skill: read_file
Reads a text file from an allowed directory and returns its content.
Risk level: low (read-only, allowlist-enforced).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragnarok.config import settings

logger = logging.getLogger(__name__)

MANIFEST = {
    "name": "read_file",
    "description": (
        "Reads a text file and returns its content. "
        "Only files inside allowed folders can be read. "
        "Use when the user asks to read, show, or summarize a file."
    ),
    "risk_level": "low",
    "required_permissions": [],
    "parameters": {
        "path": {
            "type": "string",
            "description": (
                "Relative or absolute path to the file. "
                "Must be inside an allowed folder (data/, logs/)."
            ),
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to return (default 4000).",
        },
    },
    "required": ["path"],
}

_MAX_CHARS_DEFAULT = 4_000
_MAX_CHARS_CAP = 20_000


def run(args: dict) -> str:
    raw_path = args.get("path", "")
    max_chars = min(int(args.get("max_chars", _MAX_CHARS_DEFAULT)), _MAX_CHARS_CAP)

    target = Path(raw_path).resolve()

    # ── Allowlist check ────────────────────────────────────────────────────────
    allowed = [p.resolve() for p in settings.allowed_folders]
    if not any(str(target).startswith(str(a)) for a in allowed):
        allowed_str = ", ".join(str(a) for a in allowed)
        return (
            f"Error: '{target}' is outside allowed folders ({allowed_str}). "
            "Cannot read this file."
        )

    # ── Path traversal guard ───────────────────────────────────────────────────
    try:
        target.relative_to(target.parent)  # ensures no symlink escape
    except ValueError:
        return "Error: Path traversal detected."

    if not target.exists():
        return f"Error: File not found: {target}"
    if not target.is_file():
        return f"Error: '{target}' is a directory, not a file."

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied reading '{target}'."
    except OSError as exc:
        return f"Error reading file: {exc}"

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"

    logger.info("read_file path=%s chars=%d", target, len(content))
    return content
