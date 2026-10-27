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


YT_SYNONYMS = r"(?:youtube|yt|you\s*tube)"


def parse_voice_command(text: str) -> dict[str, Any] | None:
    """
    Parse natural spoken or typed commands related to opening websites or searching YouTube.
    Supports 'youtube', 'yt', 'you tube', 'play on youtube', etc.
    """
    cleaned = (text or "").strip().rstrip(".!?,")
    if not cleaned:
        return None

    # Strip conversational prefixes
    cleaned = re.sub(
        r"^(?:please\s+|can\s+you\s+|could\s+you\s+|hey\s+ragnarok\s+|ragnarok\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # 1. "open/launch/go to yt/youtube and search/play <query>"
    m1 = re.search(
        rf"^(?:open|launch|go\s+to)\s+{YT_SYNONYMS}\s+and\s+(?:search(?:\s+for)?|play|watch|find)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m1 and m1.group(1).strip():
        return {"search_query": m1.group(1).strip(), "target": "youtube"}

    # 2. "search (for) <query> on yt/youtube"
    m2 = re.search(
        rf"^(?:search(?:\s+for)?|play|watch|find|look\s+up)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{YT_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    )
    if m2 and m2.group(1).strip():
        return {"search_query": m2.group(1).strip(), "target": "youtube"}

    # 3. "search yt/youtube for <query>"
    m3 = re.search(
        rf"^(?:search|look\s+up)\s+{YT_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m3 and m3.group(1).strip():
        return {"search_query": m3.group(1).strip(), "target": "youtube"}

    # 4. "play <query> on/in yt/youtube"
    m4 = re.search(
        rf"^play\s+[\"'\s]?(.*?)[\"'\s]?\s+(?:on|in)\s+{YT_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    )
    if m4 and m4.group(1).strip():
        return {"search_query": m4.group(1).strip(), "target": "youtube"}

    # 5. "yt/youtube search (for) <query>"
    m5 = re.search(
        rf"^{YT_SYNONYMS}\s+(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m5 and m5.group(1).strip():
        return {"search_query": m5.group(1).strip(), "target": "youtube"}

    # 6. Just open yt/youtube (homepage)
    if re.search(rf"^(?:open|launch|go\s+to)?\s*{YT_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://www.youtube.com", "target": "youtube"}

    # 7. "open <url>"
    m7 = re.search(
        r"^(?:open|launch|go\s+to)\s+(https?://\S+|www\.\S+|\S+\.(?:com|org|net|io|ai|dev)\S*)",
        cleaned,
        re.IGNORECASE,
    )
    if m7:
        return {"url": m7.group(1).strip(), "target": "web"}

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
        if any(k in target for k in ("youtube", "yt")):
            target_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        else:
            target_url = f"https://www.google.com/search?q={encoded_query}"
    else:
        # Default open YouTube home
        target_url = "https://www.youtube.com"

    try:
        webbrowser.open(target_url, new=2)

        if search_query:
            if any(k in target for k in ("youtube", "yt")):
                return f"Opening YouTube and searching for '{search_query}'. Browser navigated to: {target_url}"
            return f"Opening web search for '{search_query}'. Browser navigated to: {target_url}"
        return f"Opening browser to: {target_url}"
    except Exception as exc:
        logger.error("Failed to open browser URL %s: %s", target_url, exc)
        return f"Error opening browser: {exc}"


execute = run
