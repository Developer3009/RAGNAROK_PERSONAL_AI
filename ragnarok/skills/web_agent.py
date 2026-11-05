"""
ragnarok/skills/web_agent.py
────────────────────────────
Innovation #9 — Headless Autonomous Web Browser Agent

Uses Playwright (async Chromium) in headless mode.  An LLM-driven reasoning
loop decides which action to take next based on a pruned accessibility tree,
repeating until the task is solved or max_steps is reached.

Security:
  • Only http:// and https:// URLs are allowed.
  • Optional domain block-list to prevent navigation to disallowed sites.

Skill usage:
    run({"task": "Find the Python version on python.org",
         "url": "https://www.python.org",
         "max_steps": 10})
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

# ── Action schema ─────────────────────────────────────────────────────────────

ALLOWED_ACTIONS = {
    "navigate",
    "click",
    "type",
    "extract_text",
    "scroll",
    "done",
    "fail",
}

BLOCKED_DOMAINS: set[str] = set()   # extend via config if desired


def _is_safe_url(url: str) -> bool:
    """Block non-HTTP schemes and optional domain block-list."""
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    for blocked in BLOCKED_DOMAINS:
        if blocked.lower() in url.lower():
            return False
    return True


# ── Accessibility tree helper ─────────────────────────────────────────────────

async def _extract_a11y_tree(page: Any, max_tokens: int = 500) -> str:
    """
    Extract actionable elements from the page accessibility tree.
    Returns a compact text representation pruned to ~max_tokens characters.
    """
    try:
        snapshot = await page.accessibility.snapshot()
        lines: list[str] = []

        def _walk(node: dict | None, depth: int = 0) -> None:
            if not node or depth > 6:
                return
            role = node.get("role", "")
            name = node.get("name", "")
            value = node.get("value", "")

            # Only include actionable or informative roles
            if role in (
                "link", "button", "textbox", "combobox", "checkbox",
                "radio", "menuitem", "tab", "heading", "StaticText",
                "text", "paragraph",
            ) and name:
                indent = "  " * depth
                label = name[:80]
                if value:
                    label += f" (value={value[:40]})"
                lines.append(f"{indent}[{role}] {label}")

            for child in node.get("children", []):
                _walk(child, depth + 1)

        _walk(snapshot)
        tree_text = "\n".join(lines)
        # Prune to max_tokens characters
        if len(tree_text) > max_tokens * 4:
            tree_text = tree_text[: max_tokens * 4] + "\n... (truncated)"
        return tree_text or "(empty accessibility tree)"
    except Exception as exc:
        logger.warning("a11y tree extraction failed: %s", exc)
        return "(accessibility tree unavailable)"


# ── BrowserSession ────────────────────────────────────────────────────────────

class BrowserSession:
    """Wraps a Playwright browser lifecycle."""

    def __init__(self, headless: bool = True, timeout_ms: int = 30_000) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright: Any = None
        self._browser: Any = None
        self.page: Any = None

    async def start(self) -> None:
        """Launch Chromium browser and open a blank page."""
        from playwright.async_api import async_playwright  # noqa: PLC0415
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self.page = await self._browser.new_page()
        self.page.set_default_timeout(self.timeout_ms)
        logger.debug("BrowserSession started (headless=%s)", self.headless)

    async def stop(self) -> None:
        """Close browser and playwright instance."""
        try:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as exc:
            logger.warning("BrowserSession stop error: %s", exc)

    async def navigate(self, url: str) -> str:
        if not _is_safe_url(url):
            return f"BLOCKED: unsafe URL '{url}'"
        try:
            resp = await self.page.goto(url, wait_until="domcontentloaded")
            status = resp.status if resp else "?"
            return f"Navigated to {url} (HTTP {status})"
        except Exception as exc:
            return f"Navigation failed: {exc}"

    async def click(self, selector_or_text: str) -> str:
        try:
            # Try exact text first, then CSS selector
            try:
                await self.page.get_by_text(selector_or_text, exact=False).first.click(timeout=5000)
            except Exception:
                await self.page.click(selector_or_text, timeout=5000)
            return f"Clicked '{selector_or_text}'"
        except Exception as exc:
            return f"Click failed: {exc}"

    async def type_text(self, selector_or_text: str, text: str) -> str:
        try:
            try:
                await self.page.get_by_text(selector_or_text, exact=False).first.fill(text, timeout=5000)
            except Exception:
                await self.page.fill(selector_or_text, text, timeout=5000)
            return f"Typed '{text}' into '{selector_or_text}'"
        except Exception as exc:
            return f"Type failed: {exc}"

    async def extract_text(self) -> str:
        try:
            text = await self.page.inner_text("body")
            # Return first 2000 chars
            return text.strip()[:2000]
        except Exception as exc:
            return f"Extract failed: {exc}"

    async def scroll(self, direction: str = "down") -> str:
        delta = 600 if direction == "down" else -600
        await self.page.evaluate(f"window.scrollBy(0, {delta})")
        return f"Scrolled {direction}"


# ── WebAgentLoop ──────────────────────────────────────────────────────────────

class WebAgentLoop:
    """
    Multi-step LLM-driven reasoning loop over a Playwright browser session.
    """

    ACTION_SYSTEM = textwrap.dedent("""\
        You are an autonomous web browsing agent. You receive:
          1. The user's task
          2. The current page URL
          3. A compact accessibility tree of the page

        Decide the SINGLE best next action and respond ONLY with valid JSON:
        {
          "action": "<one of: navigate | click | type | extract_text | scroll | done | fail>",
          "target": "<URL, element text, or CSS selector — omit if not needed>",
          "text": "<text to type — only for 'type' action>",
          "reason": "<one sentence why>"
        }

        When the task is complete, use action "done" and put the answer in "reason".
        If the task cannot be completed, use "fail".
    """)

    def __init__(self, model: str | None = None) -> None:
        from ragnarok.config import settings  # late import
        self.model = model or settings.model_main

    def _decide(self, task: str, url: str, tree: str, history: str) -> dict:
        """Ask the LLM to decide the next action."""
        from ragnarok import llm  # late import
        user_msg = (
            f"Task: {task}\n"
            f"Current URL: {url}\n"
            f"Accessibility tree:\n{tree}\n"
        )
        if history:
            user_msg += f"\nAction history:\n{history[-600:]}"

        try:
            raw = llm.chat(
                messages=[
                    {"role": "system", "content": self.ACTION_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                model=self.model,
            ) or "{}"
        except Exception as exc:
            logger.warning("WebAgent LLM call failed: %s", exc)
            return {"action": "fail", "reason": str(exc)}

        # llm.chat returns ChatResult; extract .content
        if hasattr(raw, "content"):
            raw = raw.content or "{}"

        # Parse JSON — handle markdown fences
        json_str = raw
        if "```" in raw:
            for part in raw.split("```"):
                if "{" in part:
                    json_str = part.strip().lstrip("json").strip()
                    break
        try:
            return json.loads(json_str)
        except Exception:
            # Try to extract JSON from text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end])
                except Exception:
                    pass
            return {"action": "fail", "reason": f"LLM returned non-JSON: {raw[:200]}"}

    async def run_async(
        self,
        task: str,
        start_url: str,
        max_steps: int = 10,
    ) -> str:
        """Run the browser agent loop asynchronously."""
        from ragnarok.config import settings  # late import
        headless = getattr(settings, "web_agent_headless", True)
        timeout_ms = int(getattr(settings, "web_agent_timeout_sec", 30)) * 1000

        session = BrowserSession(headless=headless, timeout_ms=timeout_ms)
        await session.start()
        history_lines: list[str] = []

        try:
            # Navigate to starting URL
            nav_result = await session.navigate(start_url)
            history_lines.append(f"[navigate] {nav_result}")

            for step in range(max_steps):
                url = session.page.url if session.page else start_url
                tree = await _extract_a11y_tree(session.page)
                history = "\n".join(history_lines)

                decision = self._decide(task, url, tree, history)
                action = decision.get("action", "fail")
                target = decision.get("target", "")
                text   = decision.get("text", "")
                reason = decision.get("reason", "")

                logger.debug(
                    "WebAgent step %d: action=%s target=%s", step + 1, action, target
                )

                if action == "done":
                    return f"Task completed: {reason}"

                if action == "fail":
                    return f"Task failed: {reason}"

                if action == "navigate":
                    result = await session.navigate(target)
                elif action == "click":
                    result = await session.click(target)
                elif action == "type":
                    result = await session.type_text(target, text)
                elif action == "extract_text":
                    result = await session.extract_text()
                elif action == "scroll":
                    result = await session.scroll(target or "down")
                else:
                    result = f"Unknown action '{action}'"

                history_lines.append(f"[{action}] {result[:200]}")

            return f"Max steps ({max_steps}) reached without completing task."

        finally:
            await session.stop()

    def run_sync(self, task: str, start_url: str, max_steps: int = 10) -> str:
        """Synchronous wrapper — runs the async loop in a new event loop."""
        try:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(
                self.run_async(task, start_url, max_steps)
            )
        except Exception as exc:
            logger.exception("WebAgentLoop.run_sync failed: %s", exc)
            return f"ERROR: {exc}"
        finally:
            loop.close()


# ── Singleton ─────────────────────────────────────────────────────────────────

_web_agent: WebAgentLoop | None = None


def get_web_agent() -> WebAgentLoop:
    """Return the module-level WebAgentLoop singleton."""
    global _web_agent
    if _web_agent is None:
        _web_agent = WebAgentLoop()
    return _web_agent


# ── Skill registration ────────────────────────────────────────────────────────

MANIFEST = SkillManifest(
    name="web_agent",
    description=(
        "Headless autonomous web browser agent powered by Playwright + local LLM. "
        "Navigates to a URL, reads the page accessibility tree, and executes a "
        "multi-step reasoning loop to complete a browsing task. Fully offline except "
        "for the target website."
    ),
    risk_level="high",
    required_permissions=["network", "browser"],
    parameters={
        "task": {
            "type": "string",
            "description": "The browsing task to accomplish (e.g. 'Find the latest Python release')",
        },
        "url": {
            "type": "string",
            "description": "Starting URL (must be http:// or https://)",
        },
        "max_steps": {
            "type": "integer",
            "description": "Maximum browser actions before stopping (default 10)",
        },
    },
    required=["task", "url"],
)


def run(args: dict) -> str:
    """
    Skill entry-point for web_agent.

    Args:
        args: dict with keys:
            task     (str, required): The browsing task description.
            url      (str, required): Starting URL.
            max_steps (int, optional): Max steps, default 10.

    Returns:
        str: Final answer or error message.
    """
    task = str(args.get("task", "")).strip()
    url  = str(args.get("url",  "")).strip()
    max_steps = int(args.get("max_steps", 10))

    if not task:
        return "ERROR: 'task' parameter is required."
    if not url:
        return "ERROR: 'url' parameter is required."
    if not _is_safe_url(url):
        return f"ERROR: URL '{url}' is not allowed (must be http/https)."

    try:
        agent = get_web_agent()
        return agent.run_sync(task=task, start_url=url, max_steps=max_steps)
    except Exception as exc:
        logger.exception("web_agent skill failed: %s", exc)
        return f"ERROR: {exc}"
