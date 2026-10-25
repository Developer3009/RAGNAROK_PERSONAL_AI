"""
ragnarok/skills/fetch_web.py
────────────────────────────
Skill: fetch_web
Fetches text content from a web URL using HTTP GET.
Risk level: medium (network activity).
"""

from __future__ import annotations

import httpx

MANIFEST = {
    "name": "fetch_web",
    "description": "Fetch text content from a web URL (HTTP GET).",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The HTTP or HTTPS URL to fetch.",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum characters to return (defaults to 4000).",
            },
        },
        "required": ["url"],
    },
    "risk_level": "medium",
}


def execute(url: str, max_length: int = 4000) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: URL must start with http:// or https://"

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "RagnarokLocalAgent/0.2"})
            r.raise_for_status()
            text = r.text
            if len(text) > max_length:
                return text[:max_length] + f"\n... [Truncated at {max_length} characters]"
            return text
    except Exception as exc:
        return f"Error fetching URL: {exc}"
