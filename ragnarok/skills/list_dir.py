"""
ragnarok/skills/list_dir.py
────────────────────────────
Skill: list_dir
Lists files and subdirectories inside an allowed path.
Risk level: low.

Improvements:
  - Uses PathSandbox for proper path resolution and traversal protection.
  - Caps output at _MAX_ENTRIES to avoid flooding the context on huge dirs.
  - Shows file sizes and truncation notice when capped.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragnarok.config import settings
from ragnarok.safety.sandbox import PathSandbox, SandboxViolation

logger = logging.getLogger(__name__)

MANIFEST = {
    "name": "list_dir",
    "description": "List files and subdirectories inside an allowed directory path.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (defaults to data directory).",
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum number of entries to return (default 50, max 200).",
            },
        },
        "required": [],
    },
    "risk_level": "low",
}

_MAX_ENTRIES_DEFAULT = 50
_MAX_ENTRIES_CAP = 200


def run(args: dict) -> str:
    default_dir = str(settings.allowed_folders[0]) if settings.allowed_folders else "data"
    raw_path = args.get("path")
    if not raw_path or str(raw_path).strip() in ("", "."):
        path = default_dir
    else:
        path = str(raw_path).strip().strip("'\"")

    max_entries = min(int(args.get("max_entries", _MAX_ENTRIES_DEFAULT)), _MAX_ENTRIES_CAP)

    sandbox = PathSandbox(allowed_dirs=settings.allowed_folders)

    try:
        target = sandbox.validate(path)
    except SandboxViolation as exc:
        return f"Error: {exc}"

    if not target.exists():
        return f"Error: Directory '{path}' does not exist."

    if not target.is_dir():
        return f"Error: Path '{path}' is not a directory."

    try:
        all_entries = sorted(target.iterdir())
        capped = all_entries[:max_entries]
        entries = []
        for p in capped:
            kind = "[DIR] " if p.is_dir() else "[FILE]"
            size = f"  ({p.stat().st_size:,} bytes)" if p.is_file() else ""
            entries.append(f"{kind} {p.name}{size}")

        result = "\n".join(entries) if entries else "(Directory is empty)"
        if len(all_entries) > max_entries:
            result += f"\n... [{len(all_entries) - max_entries} more entries not shown, use max_entries to see more]"

        logger.info("list_dir path=%s entries=%d", target, len(capped))
        return result
    except PermissionError:
        return f"Error: Permission denied listing '{target}'."
    except OSError as exc:
        return f"Error listing directory: {exc}"


def execute(path: str | None = None, max_entries: int = _MAX_ENTRIES_DEFAULT) -> str:
    default_dir = str(settings.allowed_folders[0]) if settings.allowed_folders else "data"
    return run({"path": path or default_dir, "max_entries": max_entries})
