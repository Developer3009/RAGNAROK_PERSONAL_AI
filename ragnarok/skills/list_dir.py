"""
ragnarok/skills/list_dir.py
────────────────────────────
Skill: list_dir
Lists files and subdirectories inside an allowed path.
Risk level: low.
"""

from __future__ import annotations

from pathlib import Path
from ragnarok.config import settings

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
        },
        "required": [],
    },
    "risk_level": "low",
}


def run(args: dict) -> str:
    default_dir = str(settings.allowed_folders[0]) if settings.allowed_folders else "data"
    path = args.get("path") or default_dir
    target = Path(path).resolve()

    allowed = [p.resolve() for p in settings.allowed_folders]
    if not any(str(target).startswith(str(a)) for a in allowed):
        allowed_str = ", ".join(str(a) for a in allowed)
        return f"Error: Path '{target}' is not within any allowed directory ({allowed_str})."

    if not target.exists():
        return f"Error: Directory '{path}' does not exist."

    if not target.is_dir():
        return f"Error: Path '{path}' is not a directory."

    try:
        entries = []
        for p in sorted(target.iterdir()):
            kind = "[DIR]" if p.is_dir() else "[FILE]"
            size = f"({p.stat().st_size} bytes)" if p.is_file() else ""
            entries.append(f"{kind} {p.name} {size}".strip())
        return "\n".join(entries) if entries else "(Directory is empty)"
    except Exception as exc:
        return f"Error listing directory: {exc}"


def execute(path: str | None = None) -> str:
    default_dir = str(settings.allowed_folders[0]) if settings.allowed_folders else "data"
    return run({"path": path or default_dir})


