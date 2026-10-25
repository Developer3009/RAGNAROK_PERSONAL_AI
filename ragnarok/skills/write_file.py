"""
ragnarok/skills/write_file.py
──────────────────────────────
Skill: write_file
Safely writes content to an allowed file path.
Risk level: high (modifies filesystem).
"""

from __future__ import annotations

import os
from pathlib import Path
from ragnarok.config import settings

MANIFEST = {
    "name": "write_file",
    "description": "Write or overwrite content to a file in an allowed directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative to allowed directory or absolute).",
            },
            "content": {
                "type": "string",
                "description": "The exact text content to write.",
            },
        },
        "required": ["path", "content"],
    },
    "risk_level": "high",
}


def run(args: dict) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    target = Path(path).resolve()

    allowed = [p.resolve() for p in settings.allowed_folders]
    if not any(str(target).startswith(str(a)) for a in allowed):
        allowed_str = ", ".join(str(a) for a in allowed)
        return f"Error: Path '{target}' is not within any allowed directory ({allowed_str})."

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {target.name}."
    except Exception as exc:
        return f"Error writing file: {exc}"


def execute(path: str, content: str) -> str:
    return run({"path": path, "content": content})

