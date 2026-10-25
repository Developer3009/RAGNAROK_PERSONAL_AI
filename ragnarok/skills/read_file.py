"""
ragnarok/skills/read_file.py
─────────────────────────────
Skill: read_file
Reads a text file from an allowed directory and returns its content.
Risk level: low (read-only, allowlist-enforced via PathSandbox).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragnarok.config import settings
from ragnarok.safety.sandbox import PathSandbox, ResourceGuard, SandboxViolation

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
            "description": "Maximum characters to return (default 4000, max 20000).",
        },
    },
    "required": ["path"],
}

_MAX_CHARS_DEFAULT = 4_000
_MAX_CHARS_CAP = 20_000


def run(args: dict) -> str:
    raw_path = str(args.get("path") or "").strip().strip("'\"")
    if not raw_path:
        return "Error: Path cannot be empty."

    max_chars = min(int(args.get("max_chars", _MAX_CHARS_DEFAULT)), _MAX_CHARS_CAP)

    sandbox = PathSandbox(allowed_dirs=settings.allowed_folders)
    guard = ResourceGuard(max_read_bytes=max_chars * 4)  # utf-8: up to 4 bytes/char

    # ── Sandbox path validation (resolves symlinks, blocks traversal) ──────────
    try:
        target = sandbox.validate(raw_path, must_exist=True)
    except SandboxViolation as exc:
        return f"Error: {exc}"

    if not target.is_file():
        return f"Error: '{target}' is a directory, not a file."

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied reading '{target}'."
    except OSError as exc:
        return f"Error reading file: {exc}"

    # ── Apply read size guard ──────────────────────────────────────────────────
    content = guard.truncate_read(content)

    # Also apply char cap with a human note
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"

    logger.info("read_file path=%s chars=%d", target, len(content))
    return content


def execute(path: str, max_chars: int = _MAX_CHARS_DEFAULT) -> str:
    return run({"path": path, "max_chars": max_chars})
