"""
ragnarok/skills/add_note.py
────────────────────────────
Skill: add_note
Writes a short note to data/knowledge/ for later recall.
Risk level: low (writes only to allowed folder).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ragnarok.config import settings

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


def run(args: dict) -> str:
    content = args.get("content", "").strip()
    tag = args.get("tag", "general").strip() or "general"

    if not content:
        return "Error: Note content cannot be empty."
    if len(content) > 5000:
        return "Error: Note too long (max 5000 chars)."

    knowledge_dir: Path = settings.knowledge_dir
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"note_{timestamp}_{tag[:20]}.txt"
    note_path = knowledge_dir / filename

    note_path.write_text(
        f"# Tag: {tag}\n# Saved: {datetime.now().isoformat()}\n\n{content}\n",
        encoding="utf-8",
    )

    logger.info("add_note tag=%s path=%s", tag, note_path)
    return f"Note saved as '{filename}' in knowledge store."
