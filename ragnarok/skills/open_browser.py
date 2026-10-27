"""
ragnarok/skills/open_browser.py
───────────────────────────────
Skill: open_browser
Opens a URL or performs a search on YouTube / the web in the user's default browser.
Supports voice command parsing and direct intent execution.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import webbrowser
from typing import Any

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

MANIFEST = SkillManifest(
    name="open_browser",
    description=(
        "Open a web page, YouTube video search, or website in the user's default web browser. "
        "Supports opening specific URLs or searching directly on YouTube."
    ),
    risk_level="low",
    required_permissions=[],
    parameters={
        "url": {
            "type": "string",
            "description": "The exact URL to open (e.g. 'https://www.youtube.com'). Optional if search_query is provided.",
        },
        "search_query": {
            "type": "string",
            "description": "Query to search on YouTube or the web (e.g. 'working of LLM' or 'space documentary').",
        },
        "target": {
            "type": "string",
            "description": "Target platform for search_query: 'youtube' (default) or 'web'.",
        },
    },
    required=[],
)


def parse_voice_command(text: str) -> dict[str, Any] | None:
    """
    Parse natural spoken or typed commands related to opening websites or searching YouTube.
    Examples:
        - 'open youtube and search "working of LLM"' -> {'search_query': 'working of LLM', 'target': 'youtube'}
        - 'open youtube and search working of LLM'   -> {'search_query': 'working of LLM', 'target': 'youtube'}
        - 'search youtube for working of LLM'        -> {'search_query': 'working of LLM', 'target': 'youtube'}
        - 'search working of LLM on youtube'        -> {'search_query': 'working of LLM', 'target': 'youtube'}
        - 'open youtube'                             -> {'url': 'https://www.youtube.com', 'target': 'youtube'}
    """
    cleaned = (text or "").strip().rstrip(".!?,")
    if not cleaned:
        return None

    # 1. "open youtube and search (for) <query>"
    m1 = re.search(r"^(?:please\s+)?open\s+youtube\s+and\s+search(?:\s+for)?\s+[\"']?(.*?)[\"']?$", cleaned, re.IGNORECASE)
    if m1 and m1.group(1).strip():
        return {"search_query": m1.group(1).strip(), "target": "youtube"}

    # 2. "search youtube for <query>"
    m2 = re.search(r"^(?:please\s+)?search\s+youtube\s+for\s+[\"']?(.*?)[\"']?$", cleaned, re.IGNORECASE)
    if m2 and m2.group(1).strip():
        return {"search_query": m2.group(1).strip(), "target": "youtube"}

    # 3. "search <query> on youtube"
    m3 = re.search(r"^(?:please\s+)?search\s+[\"']?(.*?)[\"']?\s+on\s+youtube$", cleaned, re.IGNORECASE)
    if m3 and m3.group(1).strip():
        return {"search_query": m3.group(1).strip(), "target": "youtube"}

    # 4. "open youtube" (just home page)
    if re.search(r"^(?:please\s+)?open\s+youtube\b", cleaned, re.IGNORECASE):
        return {"url": "https://www.youtube.com", "target": "youtube"}

    # 5. "open <url>"
    m5 = re.search(r"^(?:please\s+)?open\s+(https?://\S+|www\.\S+)", cleaned, re.IGNORECASE)
    if m5:
        return {"url": m5.group(1).strip(), "target": "web"}

    return None


def run(args: dict[str, Any]) -> str:
    url = (args.get("url") or "").strip()
    raw_query = (
        args.get("search_query")
        or args.get("query")
        or args.get("q")
        or args.get("search")
        or ""
    ).strip()
    target = (args.get("target") or "youtube").strip().lower()

    # Clean up leading phrases if the entire sentence was passed into search_query
    if raw_query:
        voice_parsed = parse_voice_command(raw_query)
        if voice_parsed and voice_parsed.get("search_query"):
            search_query = voice_parsed["search_query"]
            target = voice_parsed.get("target", target)
        else:
            search_query = raw_query.strip("'\"")
    else:
        search_query = ""

    target_url: str = ""

    if url:
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"https://{url}"
        target_url = url
    elif search_query:
        encoded_query = urllib.parse.quote_plus(search_query)
        if "youtube" in target:
            target_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        else:
            target_url = f"https://www.google.com/search?q={encoded_query}"
    else:
        # Default open YouTube home
        target_url = "https://www.youtube.com"

    try:
        webbrowser.open(target_url, new=2)
        if search_query:
            if "youtube" in target:
                return f"Opening YouTube and searching for '{search_query}'. Browser navigated to: {target_url}"
            return f"Opening web search for '{search_query}'. Browser navigated to: {target_url}"
        return f"Opening browser to: {target_url}"
    except Exception as exc:
        logger.error("Failed to open browser URL %s: %s", target_url, exc)
        return f"Error opening browser: {exc}"


execute = run
