"""
ragnarok/skills/search_files.py
────────────────────────────────
Skill: search_files
Searches for files in allowed directories matching a name pattern or
containing a text snippet.
Risk level: low (read-only, allowlist-enforced).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragnarok.config import settings

logger = logging.getLogger(__name__)

MANIFEST = {
    "name": "search_files",
    "description": (
        "Searches for files in allowed directories that match a name pattern "
        "or contain a text string. "
        "Use when the user asks to find a file or search through files."
    ),
    "risk_level": "low",
    "required_permissions": [],
    "parameters": {
        "pattern": {
            "type": "string",
            "description": (
                "Glob pattern for filename matching, e.g. '*.txt' or 'note_*'. "
                "Leave empty to search by content only."
            ),
        },
        "contains": {
            "type": "string",
            "description": "Optional text to search for inside files (case-insensitive).",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 20).",
        },
    },
    "required": [],
}

_MAX_RESULTS_DEFAULT = 20
_MAX_RESULTS_CAP = 100


def run(args: dict) -> str:
    pattern = args.get("pattern", "*").strip() or "*"
    contains = (args.get("contains") or "").strip().lower()
    max_results = min(int(args.get("max_results", _MAX_RESULTS_DEFAULT)), _MAX_RESULTS_CAP)

    allowed = [p.resolve() for p in settings.allowed_folders]
    matches: list[str] = []

    for base in allowed:
        if not base.exists():
            continue
        for path in base.rglob(pattern):
            if not path.is_file():
                continue
            if contains:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    if contains not in text.lower():
                        continue
                except OSError:
                    continue
            matches.append(str(path))
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No files found matching pattern='{pattern}'" + (
            f" containing='{contains}'" if contains else ""
        )

    lines = [f"Found {len(matches)} file(s):"] + [f"  {m}" for m in matches]
    if len(matches) == max_results:
        lines.append(f"  (results capped at {max_results})")

    logger.info("search_files pattern=%s contains=%s results=%d", pattern, contains, len(matches))
    return "\n".join(lines)
