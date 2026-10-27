"""
ragnarok/skills/open_browser.py
───────────────────────────────
Skill: open_browser
Opens a URL, application, or performs searches across Google, YouTube,
Gmail, Google Maps, Google Drive, GitHub, Wikipedia, Spotify, and more.
Also launches native desktop applications (Calculator, Notepad, Explorer, etc.).
Supports voice command parsing and direct intent execution.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import urllib.parse
import webbrowser
from typing import Any

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

MANIFEST = SkillManifest(
    name="open_browser",
    description=(
        "Open websites, web applications, or search queries across Google, YouTube, "
        "Gmail, Google Maps, GitHub, Wikipedia, Google Drive, Spotify, etc., or launch local desktop applications."
    ),
    risk_level="low",
    required_permissions=[],
    parameters={
        "url": {
            "type": "string",
            "description": "The exact URL to open (e.g. 'https://mail.google.com'). Optional if search_query or target is provided.",
        },
        "search_query": {
            "type": "string",
            "description": "Query to search on Google, Gmail, YouTube, Maps, etc. (e.g. 'working of LLM', 'invoice 2026').",
        },
        "target": {
            "type": "string",
            "description": "Target platform: 'google', 'gmail', 'youtube', 'maps', 'drive', 'github', 'wikipedia', 'spotify', 'reddit', 'linkedin', 'desktop', or 'web'.",
        },
        "app_name": {
            "type": "string",
            "description": "Optional name of local desktop application to launch (e.g. 'calculator', 'notepad', 'explorer').",
        },
    },
    required=[],
)

YT_SYNONYMS = r"(?:youtube|yt|you\s*tube)"
GMAIL_SYNONYMS = r"(?:gmail|gamil|google\s*mail|email|mail|inbox)"
GOOGLE_SYNONYMS = r"(?:google|google\s*search)"
MAPS_SYNONYMS = r"(?:google\s*maps|maps|map)"
DRIVE_SYNONYMS = r"(?:google\s*drive|drive)"
DOCS_SYNONYMS = r"(?:google\s*docs|docs|doc)"
SHEETS_SYNONYMS = r"(?:google\s*sheets|sheets|sheet)"
CALENDAR_SYNONYMS = r"(?:google\s*calendar|calendar)"
GITHUB_SYNONYMS = r"(?:github|git\s*hub)"
WIKI_SYNONYMS = r"(?:wikipedia|wiki)"
SPOTIFY_SYNONYMS = r"(?:spotify)"
REDDIT_SYNONYMS = r"(?:reddit)"
LINKEDIN_SYNONYMS = r"(?:linkedin|linked\s*in)"
DESKTOP_SYNONYMS = r"(?:calculator|calc|notepad|explorer|file\s+explorer|task\s+manager|taskmgr|terminal|powershell|cmd|command\s+prompt)"


def parse_voice_command(text: str) -> dict[str, Any] | None:
    """
    Parse natural spoken or typed commands related to opening websites,
    searching across platforms (Google, Gmail, YouTube, Maps, etc.), or desktop apps.
    """
    cleaned = (text or "").strip().rstrip(".!?,")
    if not cleaned:
        return None

    # Strip conversational prefixes
    cleaned = re.sub(
        r"^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|hey\s+ragnarok\s+|ragnarok\s+|just\s+|go\s+ahead\s+and\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # 1. Desktop local applications
    m_desk = re.search(rf"^(?:open|launch)\s+({DESKTOP_SYNONYMS})$", cleaned, re.IGNORECASE)
    if m_desk:
        return {"target": "desktop", "app_name": m_desk.group(1).lower()}

    # 2. Gmail compose
    if re.search(
        rf"^(?:compose(?:\s+a)?(?:\s+new)?\s+(?:mail|email|message)|send\s+(?:an?\s+)?(?:email|mail))(?:\s+(?:on|in)\s+{GMAIL_SYNONYMS})?$",
        cleaned,
        re.IGNORECASE,
    ):
        return {
            "target": "gmail",
            "url": "https://mail.google.com/mail/u/0/#inbox?compose=new",
            "action": "compose",
        }

    # 3. Gmail search ("search on gmail for ...", "search mail ...", "open gamil and search ...")
    m_gm_s1 = re.search(
        rf"^(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{GMAIL_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    )
    if m_gm_s1 and m_gm_s1.group(1).strip():
        return {"search_query": m_gm_s1.group(1).strip(), "target": "gmail"}

    m_gm_s2 = re.search(
        rf"^(?:search|find|look\s+up)\s+(?:on\s+)?{GMAIL_SYNONYMS}\s+(?:for\s+)?[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_gm_s2 and m_gm_s2.group(1).strip():
        return {"search_query": m_gm_s2.group(1).strip(), "target": "gmail"}

    m_gm_s3 = re.search(
        rf"^(?:open|launch|go\s+to)\s+{GMAIL_SYNONYMS}\s+and\s+search(?:\s+for)?\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_gm_s3 and m_gm_s3.group(1).strip():
        return {"search_query": m_gm_s3.group(1).strip(), "target": "gmail"}

    m_gm_s4 = re.search(
        rf"^search\s+(?:mail|email|inbox|gmail|gamil)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_gm_s4 and m_gm_s4.group(1).strip():
        return {"search_query": m_gm_s4.group(1).strip(), "target": "gmail"}

    # 4. Gmail open ("open gmail", "open gamil", "open mail", "check my email")
    if re.search(
        rf"^(?:open|launch|go\s+to|check(?:\s+my)?)\s+{GMAIL_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    ):
        return {"url": "https://mail.google.com", "target": "gmail"}

    # 5. Google search
    m_g_s1 = re.search(
        rf"^(?:search(?:\s+for)?|find|look\s+up)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{GOOGLE_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    )
    if m_g_s1 and m_g_s1.group(1).strip():
        return {"search_query": m_g_s1.group(1).strip(), "target": "google"}

    m_g_s2 = re.search(
        rf"^(?:search|find|look\s+up)\s+{GOOGLE_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_g_s2 and m_g_s2.group(1).strip():
        return {"search_query": m_g_s2.group(1).strip(), "target": "google"}

    m_g_s3 = re.search(
        rf"^(?:open|launch|go\s+to)\s+{GOOGLE_SYNONYMS}\s+and\s+search(?:\s+for)?\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_g_s3 and m_g_s3.group(1).strip():
        return {"search_query": m_g_s3.group(1).strip(), "target": "google"}

    m_g_s4 = re.search(
        rf"^(?:search\s+on\s+google(?:\s+for)?|google(?:\s+search(?:\s+for)?)?)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if (
        m_g_s4
        and m_g_s4.group(1).strip()
        and not re.match(rf"^{GOOGLE_SYNONYMS}$", m_g_s4.group(1).strip(), re.IGNORECASE)
    ):
        return {"search_query": m_g_s4.group(1).strip(), "target": "google"}

    # 6. Google open ("open google")
    if re.search(rf"^(?:open|launch|go\s+to)?\s*{GOOGLE_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://www.google.com", "target": "google"}

    # 7. Google Maps
    m_m_s1 = re.search(
        rf"^(?:directions\s+to|navigate\s+to)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_m_s1 and m_m_s1.group(1).strip():
        return {"search_query": m_m_s1.group(1).strip(), "target": "maps"}

    m_m_s2 = re.search(
        rf"^(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{MAPS_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    )
    if m_m_s2 and m_m_s2.group(1).strip():
        return {"search_query": m_m_s2.group(1).strip(), "target": "maps"}

    m_m_s3 = re.search(
        rf"^(?:search|find)\s+{MAPS_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_m_s3 and m_m_s3.group(1).strip():
        return {"search_query": m_m_s3.group(1).strip(), "target": "maps"}

    m_m_s4 = re.search(
        rf"^(?:open|launch|go\s+to)\s+{MAPS_SYNONYMS}\s+and\s+search(?:\s+for)?\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_m_s4 and m_m_s4.group(1).strip():
        return {"search_query": m_m_s4.group(1).strip(), "target": "maps"}

    if re.search(rf"^(?:open|launch|go\s+to)?\s*{MAPS_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://maps.google.com", "target": "maps"}

    # 8. YouTube Search & Open
    m_yt_s1 = re.search(
        rf"^(?:open|launch|go\s+to)\s+{YT_SYNONYMS}\s+and\s+(?:search(?:\s+for)?|play|watch|find)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_yt_s1 and m_yt_s1.group(1).strip():
        return {"search_query": m_yt_s1.group(1).strip(), "target": "youtube"}

    m_yt_s2 = re.search(
        rf"^(?:search(?:\s+for)?|play|watch|find|look\s+up)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{YT_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    )
    if m_yt_s2 and m_yt_s2.group(1).strip():
        return {"search_query": m_yt_s2.group(1).strip(), "target": "youtube"}

    m_yt_s3 = re.search(
        rf"^(?:search|look\s+up)\s+{YT_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_yt_s3 and m_yt_s3.group(1).strip():
        return {"search_query": m_yt_s3.group(1).strip(), "target": "youtube"}

    m_yt_s4 = re.search(
        rf"^play\s+[\"'\s]?(.*?)[\"'\s]?\s+(?:on|in)\s+{YT_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    )
    if m_yt_s4 and m_yt_s4.group(1).strip():
        return {"search_query": m_yt_s4.group(1).strip(), "target": "youtube"}

    m_yt_s5 = re.search(
        rf"^{YT_SYNONYMS}\s+(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_yt_s5 and m_yt_s5.group(1).strip():
        return {"search_query": m_yt_s5.group(1).strip(), "target": "youtube"}

    if re.search(rf"^(?:open|launch|go\s+to)?\s*{YT_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://www.youtube.com", "target": "youtube"}

    # 9. Google Drive & Docs & Sheets & Calendar
    m_drv = re.search(
        rf"^(?:search|find)\s+{DRIVE_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^(?:open\s+{DRIVE_SYNONYMS}\s+and\s+search\s+[\"'\s]?(.*?)[\"'\s]?)$",
        cleaned,
        re.IGNORECASE,
    )
    if m_drv and m_drv.group(1).strip():
        return {"search_query": m_drv.group(1).strip(), "target": "drive"}

    if re.search(rf"^(?:open|launch|go\s+to)\s+{DRIVE_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://drive.google.com", "target": "drive"}
    if re.search(rf"^(?:open|launch|go\s+to)\s+{DOCS_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://docs.google.com", "target": "docs"}
    if re.search(rf"^(?:open|launch|go\s+to)\s+{SHEETS_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://sheets.google.com", "target": "sheets"}
    if re.search(
        rf"^(?:open|launch|go\s+to|check(?:\s+my)?)\s+{CALENDAR_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    ):
        return {"url": "https://calendar.google.com", "target": "calendar"}

    # 10. GitHub
    m_gh = re.search(
        rf"^(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{GITHUB_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^(?:search|find)\s+{GITHUB_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^(?:open\s+{GITHUB_SYNONYMS}\s+and\s+search\s+[\"'\s]?(.*?)[\"'\s]?)$",
        cleaned,
        re.IGNORECASE,
    )
    if m_gh and m_gh.group(1).strip():
        return {"search_query": m_gh.group(1).strip(), "target": "github"}
    if re.search(rf"^(?:open|launch|go\s+to)\s+{GITHUB_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://github.com", "target": "github"}

    # 11. Wikipedia
    m_wk = re.search(
        rf"^(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{WIKI_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^(?:search|find)\s+{WIKI_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^wikipedia\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_wk and m_wk.group(1).strip():
        return {"search_query": m_wk.group(1).strip(), "target": "wikipedia"}
    if re.search(rf"^(?:open|launch|go\s+to)\s+{WIKI_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://www.wikipedia.org", "target": "wikipedia"}

    # 12. Spotify
    m_sp = re.search(
        rf"^(?:search(?:\s+for)?|play)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{SPOTIFY_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^(?:search|play)\s+{SPOTIFY_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_sp and m_sp.group(1).strip():
        return {"search_query": m_sp.group(1).strip(), "target": "spotify"}
    if re.search(rf"^(?:open|launch|go\s+to)\s+{SPOTIFY_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://open.spotify.com", "target": "spotify"}

    # 13. Reddit
    m_rd = re.search(
        rf"^(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{REDDIT_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^(?:search|find)\s+{REDDIT_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_rd and m_rd.group(1).strip():
        return {"search_query": m_rd.group(1).strip(), "target": "reddit"}
    if re.search(rf"^(?:open|launch|go\s+to)\s+{REDDIT_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://www.reddit.com", "target": "reddit"}

    # 14. LinkedIn
    m_li = re.search(
        rf"^(?:search(?:\s+for)?|find)\s+[\"'\s]?(.*?)[\"'\s]?\s+on\s+{LINKEDIN_SYNONYMS}$",
        cleaned,
        re.IGNORECASE,
    ) or re.search(
        rf"^(?:search|find)\s+{LINKEDIN_SYNONYMS}\s+for\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_li and m_li.group(1).strip():
        return {"search_query": m_li.group(1).strip(), "target": "linkedin"}
    if re.search(rf"^(?:open|launch|go\s+to)\s+{LINKEDIN_SYNONYMS}$", cleaned, re.IGNORECASE):
        return {"url": "https://www.linkedin.com", "target": "linkedin"}

    # 15. Other popular applications / AI
    if re.search(r"^(?:open|launch|go\s+to)\s+(?:chatgpt|chat\s*gpt)$", cleaned, re.IGNORECASE):
        return {"url": "https://chatgpt.com", "target": "chatgpt"}
    if re.search(r"^(?:open|launch|go\s+to)\s+claude$", cleaned, re.IGNORECASE):
        return {"url": "https://claude.ai", "target": "claude"}
    if re.search(r"^(?:open|launch|go\s+to)\s+whatsapp(?:\s*web)?$", cleaned, re.IGNORECASE):
        return {"url": "https://web.whatsapp.com", "target": "whatsapp"}
    if re.search(r"^(?:open|launch|go\s+to)\s+(?:twitter|x)$", cleaned, re.IGNORECASE):
        return {"url": "https://x.com", "target": "twitter"}
    if re.search(r"^(?:open|launch|go\s+to)\s+notion$", cleaned, re.IGNORECASE):
        return {"url": "https://www.notion.so", "target": "notion"}

    # 16. General Web URL ("open https://example.com" or "go to github.com")
    m_url = re.search(
        r"^(?:open|launch|go\s+to)\s+(https?://\S+|www\.\S+|\S+\.(?:com|org|net|io|ai|dev)\S*)",
        cleaned,
        re.IGNORECASE,
    )
    if m_url:
        return {"url": m_url.group(1).strip(), "target": "web"}

    # 17. General web search ("search web for ...")
    m_web = re.search(
        r"^(?:search(?:\s+the)?\s+web\s+for|search\s+for)\s+[\"'\s]?(.*?)[\"'\s]?$",
        cleaned,
        re.IGNORECASE,
    )
    if m_web and m_web.group(1).strip():
        return {"search_query": m_web.group(1).strip(), "target": "google"}

    return None


def run(args: dict[str, Any]) -> str:
    target = (args.get("target") or "web").strip().lower()
    app_name = (args.get("app_name") or "").strip().lower()

    # ── 1. Desktop Application Launch ──────────────────────────────────────────
    if target == "desktop" or app_name:
        name = app_name or args.get("search_query", "").strip().lower()
        executables: dict[str, str] = {
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "notepad": "notepad.exe",
            "explorer": "explorer.exe",
            "file explorer": "explorer.exe",
            "task manager": "taskmgr.exe",
            "taskmgr": "taskmgr.exe",
            "terminal": "wt.exe",
            "powershell": "powershell.exe",
            "cmd": "cmd.exe",
            "command prompt": "cmd.exe",
        }
        cmd = executables.get(name, f"{name}.exe")
        try:
            subprocess.Popen(cmd, shell=True)
            return f"Launched {name.title()} application on your desktop."
        except Exception as exc:
            logger.error("Failed to launch desktop application %s: %s", name, exc)
            return f"Error launching {name}: {exc}"

    # ── 2. Web / Browser Navigation & Searches ──────────────────────────────────
    url = (args.get("url") or "").strip()
    raw_query = (
        args.get("search_query")
        or args.get("query")
        or args.get("q")
        or args.get("search")
        or ""
    ).strip()

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
    target_label: str = target.title()

    if url:
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"https://{url}"
        target_url = url
    elif search_query:
        encoded_query = urllib.parse.quote_plus(search_query)
        if any(k in target for k in ("gmail", "gamil", "mail", "email")):
            target_url = f"https://mail.google.com/mail/u/0/#search/{encoded_query}"
            target_label = "Gmail"
        elif any(k in target for k in ("youtube", "yt")):
            target_url = f"https://www.youtube.com/results?search_query={encoded_query}"
            target_label = "YouTube"
        elif any(k in target for k in ("map", "maps")):
            target_url = f"https://www.google.com/maps/search/{encoded_query}"
            target_label = "Google Maps"
        elif any(k in target for k in ("drive",)):
            target_url = f"https://drive.google.com/drive/search?q={encoded_query}"
            target_label = "Google Drive"
        elif any(k in target for k in ("github",)):
            target_url = f"https://github.com/search?q={encoded_query}"
            target_label = "GitHub"
        elif any(k in target for k in ("wikipedia", "wiki")):
            target_url = f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_query}"
            target_label = "Wikipedia"
        elif any(k in target for k in ("spotify",)):
            target_url = f"https://open.spotify.com/search/{encoded_query}"
            target_label = "Spotify"
        elif any(k in target for k in ("reddit",)):
            target_url = f"https://www.reddit.com/search/?q={encoded_query}"
            target_label = "Reddit"
        elif any(k in target for k in ("linkedin",)):
            target_url = f"https://www.linkedin.com/search/results/all/?keywords={encoded_query}"
            target_label = "LinkedIn"
        else:
            target_url = f"https://www.google.com/search?q={encoded_query}"
            target_label = "Google"
    else:
        # Default destination based on target
        destinations = {
            "gmail": "https://mail.google.com",
            "youtube": "https://www.youtube.com",
            "maps": "https://maps.google.com",
            "drive": "https://drive.google.com",
            "docs": "https://docs.google.com",
            "sheets": "https://sheets.google.com",
            "calendar": "https://calendar.google.com",
            "github": "https://github.com",
            "wikipedia": "https://www.wikipedia.org",
            "spotify": "https://open.spotify.com",
            "reddit": "https://www.reddit.com",
            "linkedin": "https://www.linkedin.com",
            "chatgpt": "https://chatgpt.com",
            "claude": "https://claude.ai",
            "whatsapp": "https://web.whatsapp.com",
            "twitter": "https://x.com",
            "notion": "https://www.notion.so",
        }
        target_url = destinations.get(target, "https://www.google.com")

    try:
        webbrowser.open(target_url, new=2)

        if search_query:
            return f"Searching {target_label} for '{search_query}'. Browser navigated to: {target_url}"
        if args.get("action") == "compose":
            return f"Opening Gmail compose window. Browser navigated to: {target_url}"
        return f"Opening {target_label}. Browser navigated to: {target_url}"
    except Exception as exc:
        logger.error("Failed to open browser URL %s: %s", target_url, exc)
        return f"Error opening {target_label}: {exc}"


execute = run
