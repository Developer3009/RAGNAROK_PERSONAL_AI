"""
ragnarok/skills/get_time.py
────────────────────────────
Skill: get_time
Returns the current local date and time.
Risk level: low (read-only, no IO).
"""

from __future__ import annotations

from datetime import datetime

MANIFEST = {
    "name": "get_time",
    "description": (
        "Returns the current local date and time. "
        "Use when the user asks what time or date it is."
    ),
    "risk_level": "low",
    "required_permissions": [],
    "parameters": {
        "format": {
            "type": "string",
            "description": (
                "Output format string. Use 'iso' for ISO-8601, "
                "'human' for a friendly string (default)."
            ),
            "enum": ["iso", "human"],
        }
    },
    "required": [],
}


def run(args: dict) -> str:
    fmt = args.get("format", "human")
    now = datetime.now()
    if fmt == "iso":
        return now.isoformat(timespec="seconds")
    return now.strftime("%A, %B %d %Y at %I:%M %p")


def execute(format: str = "human") -> str:
    return run({"format": format})
