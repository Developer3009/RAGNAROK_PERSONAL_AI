"""
ragnarok/skills/remember.py
───────────────────────────
Skill: remember
Stores personal data, preferences, or lasting facts into the persistent
RAGNAROK_MEMORY database so it remains available across every session.
Risk level: low.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from ragnarok.config import settings
from ragnarok.memory.db import MemoryDB

logger = logging.getLogger(__name__)

MANIFEST = {
    "name": "remember",
    "description": (
        "Stores personal data, facts, or preferences into persistent long-term memory "
        "inside C:\\Users\\divye\\OneDrive\\Desktop\\RAGNAROK_MEMORY so it remains available "
        "across every conversation session. "
        "Use whenever the user shares a personal preference, contact info, goal, or asks you to remember something."
    ),
    "risk_level": "low",
    "required_permissions": [],
    "parameters": {
        "text": {
            "type": "string",
            "description": "The personal information, fact, or preference to remember permanently.",
        },
        "importance": {
            "type": "number",
            "description": "Importance score between 0.0 and 1.0 (defaults to 0.85).",
        },
        "tag": {
            "type": "string",
            "description": "Optional category tag (e.g. 'preference', 'personal', 'work', 'profile').",
        },
    },
    "required": ["text"],
}


def run(args: dict) -> str:
    text = (args.get("text") or "").strip()
    if not text:
        return "Error: Memory text cannot be empty."

    importance = float(args.get("importance", 0.85))
    tag = (args.get("tag") or "personal").strip()

    # 1. Ensure RAGNAROK_MEMORY directory structure exists
    memory_root = settings.allowed_folders[0] if settings.allowed_folders else settings.db_dir
    knowledge_dir = settings.knowledge_dir
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # 2. Store in persistent SQLite / Encrypted DB
    db = MemoryDB()
    now = time.time()
    dummy_vec = [0.0] * 768
    try:
        from ragnarok import llm
        vector = llm.embed(text)
    except Exception:
        vector = dummy_vec

    import uuid
    mem_id = str(uuid.uuid4())
    db.insert(
        memory_id=mem_id,
        kind="semantic",
        text=text,
        embedding=vector,
        created_at=now,
        last_used_at=now,
        importance=importance,
        pinned=True,  # User explicitly requested to remember -> pin so it's never forgotten
        source=f"chat_remember_tool:{tag}",
    )

    # 3. Also write a human-readable journal note into knowledge folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_file = knowledge_dir / f"personal_memory_{timestamp}.txt"
    try:
        note_file.write_text(
            f"# Personal Memory [Tag: {tag}]\n"
            f"# Saved: {datetime.now().isoformat()}\n"
            f"# Importance: {importance}\n\n"
            f"{text}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not write readable mirror note: %s", exc)

    logger.info("Persisted memory '%s' to %s", mem_id, memory_root)
    return (
        f"Remembered: '{text}' (saved to persistent memory in {memory_root}, "
        f"pinned for future sessions)."
    )


def execute(text: str, importance: float = 0.85, tag: str = "personal") -> str:
    return run({"text": text, "importance": importance, "tag": tag})
