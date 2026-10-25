"""
ragnarok/skills/add_note.py
────────────────────────────
Skill: add_note
Writes a short note to data/knowledge/ for later recall.
Risk level: low (writes only to allowed folder).

Improvements:
  - Tag is sanitized to be filesystem-safe (no slashes, dots, or special chars).
  - Uses PathSandbox to ensure the write target stays in the allowed sandbox.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from ragnarok.config import settings
from ragnarok.safety.sandbox import PathSandbox, SandboxViolation

logger = logging.getLogger(__name__)

MANIFEST = {
    "name": "add_note",
    "description": (
        "Saves a short note or piece of information to the knowledge store "
        "so it can be recalled later. "
        "Use when the user wants to remember something."
    ),
    "risk_level": "low",
    "required_permissions": [],
    "parameters": {
        "content": {
            "type": "string",
            "description": "The text of the note to save.",
        },
        "tag": {
            "type": "string",
            "description": "Optional tag/category for the note (e.g. 'preference', 'task').",
        },
    },
    "required": ["content"],
}

_MAX_NOTE_CHARS = 5_000
_TAG_SAFE = re.compile(r"[^\w\-]")  # only word chars and hyphens allowed in tag


def run(args: dict) -> str:
    content = args.get("content", "").strip()
    raw_tag = args.get("tag", "general").strip() or "general"

    if not content:
        return "Error: Note content cannot be empty."
    if len(content) > _MAX_NOTE_CHARS:
        return f"Error: Note too long (max {_MAX_NOTE_CHARS} chars)."

    # Sanitize tag: strip dangerous characters, limit length
    tag = _TAG_SAFE.sub("_", raw_tag)[:20] or "general"

    knowledge_dir: Path = settings.knowledge_dir
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # Validate note directory is inside the sandbox
    sandbox = PathSandbox(allowed_dirs=settings.allowed_folders)
    try:
        sandbox.validate(knowledge_dir)
    except SandboxViolation as exc:
        return f"Error: knowledge_dir is outside sandbox: {exc}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"note_{timestamp}_{tag}.txt"
    note_path = knowledge_dir / filename

    try:
        note_path.write_text(
            f"# Tag: {tag}\n# Saved: {datetime.now().isoformat()}\n\n{content}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return f"Error saving note: {exc}"

    logger.info("add_note tag=%s path=%s", tag, note_path)
    return f"Note saved as '{filename}' in knowledge store."


def execute(content: str, tag: str = "general") -> str:
    return run({"content": content, "tag": tag})
