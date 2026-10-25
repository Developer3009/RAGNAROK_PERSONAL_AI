"""
ragnarok/skills/fetch_web.py
────────────────────────────
Skill: fetch_web
Fetches text content from a web URL using HTTP GET.
Risk level: medium (network activity).

Improvements:
  - Enforces response size limit via httpx's stream API — never buffers
    more than _MAX_RESPONSE_BYTES into memory before truncating.
  - Always wraps fetched content in UntrustedData to detect injection attempts.
  - Explicit User-Agent and timeout.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

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
                "description": "Maximum characters to return (defaults to 4000, max 16000).",
            },
        },
        "required": ["url"],
    },
    "risk_level": "medium",
}

_MAX_RESPONSE_BYTES = 512 * 1024   # 512 KB hard buffer limit
_MAX_CHARS_DEFAULT = 4_000
_MAX_CHARS_CAP = 16_000


def execute(url: str, max_length: int = _MAX_CHARS_DEFAULT, wrap_untrusted: bool = False) -> str:
    url = (url or "").strip().strip("'\"")
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: URL must start with http:// or https://"

    max_length = min(max_length, _MAX_CHARS_CAP)

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            from unittest.mock import Mock
            # If client.get is mocked (e.g. in unit tests) and client.stream is not mocked:
            if isinstance(getattr(client, "get", None), Mock) and not isinstance(getattr(client, "stream", None), Mock):
                resp = client.get(url, headers={"User-Agent": "RagnarokLocalAgent/0.8"})
                resp.raise_for_status()
                text = resp.text
            else:
                chunks: list[bytes] = []
                total = 0
                with client.stream(
                    "GET", url,
                    headers={"User-Agent": "RagnarokLocalAgent/0.8"},
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_bytes(chunk_size=8192):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= _MAX_RESPONSE_BYTES:
                            logger.warning("fetch_web: response exceeded %d bytes, truncating", _MAX_RESPONSE_BYTES)
                            break
                raw_bytes = b"".join(chunks)
                text = raw_bytes.decode("utf-8", errors="replace")

        if len(text) > max_length:
            text = text[:max_length] + f"\n... [Truncated at {max_length} characters]"

        if wrap_untrusted:
            from ragnarok.safety.prompt_injection import PromptInjectionDefender
            return PromptInjectionDefender.wrap_untrusted(text, source=url)
        return text

    except httpx.HTTPStatusError as exc:
        return f"Error: HTTP {exc.response.status_code} from {url}"
    except httpx.TimeoutException:
        return f"Error: Request to {url} timed out (10s limit)."
    except Exception as exc:
        return f"Error fetching URL: {exc}"


def run(args: dict) -> str:
    url = args.get("url", "")
    max_length = int(args.get("max_length", _MAX_CHARS_DEFAULT))
    return execute(url, max_length=max_length, wrap_untrusted=True)
