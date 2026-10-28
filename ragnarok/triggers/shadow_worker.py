"""
ragnarok/triggers/shadow_worker.py
────────────────────────────────────
Innovation #6: Autonomous "Shadow-Worker" — Predictive Proactivity Engine.

Implements TWO proactive background services:

A) Morning Standup Briefing:
   Fires at user-configured time (default 08:30 AM).
   Analyzes: recent git commits, open terminal tasks, upcoming notes,
   pending tasks from memory, and speaks/displays an executive summary.
   "Good morning Divyansh. Yesterday you committed 3 features to the swarm module.
    You have 1 unmerged branch and a pending note about encryption testing."

B) Silent Error Sentinel:
   Monitors developer log/terminal files via watchdog.
   When a Python traceback, test failure, or build error is detected:
     1. Reads the error context (surrounding code, recent memory).
     2. Asks the LLM for a diagnosis + proposed fix.
     3. Stores the proposed fix in memory (pinned) for 1-click approval.
   The user sees: "🔴 Error detected in test_swarm.py. Ragnarok has a proposed fix ready."

Integrates with the existing ScheduledTrigger and MemoryDB systems.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from ragnarok.config import settings

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class BriefingReport:
    date: str
    git_summary: str
    pending_tasks: list[str]
    upcoming_notes: list[str]
    recent_memory_snippets: list[str]
    full_text: str = ""

    def render(self) -> str:
        if self.full_text:
            return self.full_text
        parts = [f"📅 **Morning Briefing — {self.date}**\n"]
        if self.git_summary:
            parts.append(f"🔀 **Git Status**\n{self.git_summary}\n")
        if self.pending_tasks:
            parts.append("📌 **Pending Tasks**")
            parts.extend(f"  • {t}" for t in self.pending_tasks[:5])
            parts.append("")
        if self.upcoming_notes:
            parts.append("📝 **Recent Notes / Context**")
            parts.extend(f"  • {n}" for n in self.upcoming_notes[:5])
            parts.append("")
        if self.recent_memory_snippets:
            parts.append("🧠 **Memory Highlights**")
            parts.extend(f"  • {m}" for m in self.recent_memory_snippets[:3])
        return "\n".join(parts)


@dataclass
class ErrorAlert:
    file_path: str
    error_type: str
    traceback_snippet: str
    proposed_fix: str = ""
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def render(self) -> str:
        parts = [
            f"🔴 **Error Sentinel Alert**",
            f"  File: `{self.file_path}`",
            f"  Type: {self.error_type}",
            f"  Time: {datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S')}",
        ]
        if self.traceback_snippet:
            parts.append(f"\n```\n{self.traceback_snippet[:500]}\n```")
        if self.proposed_fix:
            parts.append(f"\n💡 **Proposed Fix**:\n{self.proposed_fix}")
        return "\n".join(parts)


# ── Git Intelligence ──────────────────────────────────────────────────────────

class GitIntelligence:
    """Extract meaningful summaries from local git repositories."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self._root = repo_root or Path.cwd()

    def _git(self, *args: str, days: int | None = None) -> str:
        """Run a git command and return stdout."""
        cmd = ["git", "-C", str(self._root)] + list(args)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def recent_commits(self, since_hours: int = 24) -> list[str]:
        """Return commit messages from the last N hours."""
        since = (datetime.now() - timedelta(hours=since_hours)).strftime("%Y-%m-%d %H:%M")
        raw = self._git("log", "--oneline", f"--since={since}", "--no-merges")
        if not raw:
            return []
        return [line.strip() for line in raw.splitlines() if line.strip()][:10]

    def unmerged_branches(self) -> list[str]:
        """Return list of local branches not merged into main/master."""
        raw = self._git("branch", "--no-merged", "HEAD")
        return [b.strip().lstrip("* ") for b in raw.splitlines() if b.strip()]

    def uncommitted_changes(self) -> list[str]:
        """Return list of modified files not yet staged/committed."""
        raw = self._git("status", "--short")
        return [line.strip() for line in raw.splitlines() if line.strip()][:10]

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"

    def summarize(self) -> str:
        """Generate a concise git status summary."""
        branch = self.current_branch()
        commits = self.recent_commits(24)
        unmerged = self.unmerged_branches()
        uncommitted = self.uncommitted_changes()

        parts = [f"Branch: `{branch}`"]
        if commits:
            parts.append(f"Last 24h commits ({len(commits)}): " + ", ".join(f'"{c.split(" ", 1)[-1]}"' for c in commits[:3]))
        if unmerged:
            parts.append(f"Unmerged branches: {', '.join(unmerged[:3])}")
        if uncommitted:
            parts.append(f"Uncommitted changes: {', '.join(uncommitted[:5])}")
        if not commits and not unmerged and not uncommitted:
            parts.append("Repository is clean — no recent activity.")
        return "\n".join(parts)


# ── Memory Summarizer ──────────────────────────────────────────────────────────

def _get_recent_memory_snippets(limit: int = 5) -> list[str]:
    """Pull recent memory entries for the briefing."""
    try:
        from ragnarok.memory.db import MemoryDB
        db = MemoryDB()
        recent = db.get_recent(limit=limit * 2, kind="episodic")
        snippets = []
        for m in recent[:limit]:
            text = m.text[:100] + ("..." if len(m.text) > 100 else "")
            snippets.append(text)
        return snippets
    except Exception as exc:
        logger.debug("Memory snippet fetch failed: %s", exc)
        return []


def _get_pending_tasks(limit: int = 5) -> list[str]:
    """Retrieve pinned or high-importance memories as pending tasks."""
    try:
        from ragnarok.memory.db import MemoryDB
        db = MemoryDB()
        pinned = db.get_pinned(limit=limit)
        return [m.text[:100] for m in pinned]
    except Exception as exc:
        logger.debug("Pending tasks fetch failed: %s", exc)
        return []


# ── Morning Briefing Engine ───────────────────────────────────────────────────

class MorningBriefingEngine:
    """
    Generates daily morning briefings by combining:
      - Git status intelligence
      - Memory recall (pending tasks, recent interactions)
      - LLM-synthesized natural language summary
    """

    def __init__(
        self,
        repo_root: Path | None = None,
        on_briefing: Callable[[BriefingReport], None] | None = None,
    ) -> None:
        self._git = GitIntelligence(repo_root)
        self._on_briefing = on_briefing
        self._last_fired_date: str = ""

    def generate(self) -> BriefingReport:
        """Generate a complete morning briefing report."""
        today = datetime.now().strftime("%A, %B %d, %Y")
        git_summary = self._git.summarize()
        pending_tasks = _get_pending_tasks()
        memory_snippets = _get_recent_memory_snippets()

        # Use LLM to synthesize a natural spoken briefing
        llm_summary = self._synthesize_with_llm(
            today=today,
            git_summary=git_summary,
            pending_tasks=pending_tasks,
            memory_snippets=memory_snippets,
        )

        report = BriefingReport(
            date=today,
            git_summary=git_summary,
            pending_tasks=pending_tasks,
            upcoming_notes=[],
            recent_memory_snippets=memory_snippets,
            full_text=llm_summary,
        )

        if self._on_briefing:
            try:
                self._on_briefing(report)
            except Exception as exc:
                logger.error("Briefing callback error: %s", exc)

        return report

    @staticmethod
    def _synthesize_with_llm(
        today: str,
        git_summary: str,
        pending_tasks: list[str],
        memory_snippets: list[str],
    ) -> str:
        """Ask the LLM to write a concise morning briefing narrative."""
        try:
            from ragnarok import llm
            context_parts = [f"Today is {today}."]
            if git_summary:
                context_parts.append(f"Git status:\n{git_summary}")
            if pending_tasks:
                context_parts.append("Pending tasks:\n" + "\n".join(f"- {t}" for t in pending_tasks))
            if memory_snippets:
                context_parts.append("Recent memory:\n" + "\n".join(f"- {m}" for m in memory_snippets))

            prompt = (
                "You are Ragnarok, a privacy-first AI assistant. "
                "Generate a concise, friendly morning briefing (2-4 sentences max) "
                "based on the following context. Mention the most important items. "
                "Speak directly to the user.\n\n"
                + "\n\n".join(context_parts)
            )
            response = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=settings.model_small,
                temperature=0.7,
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("Briefing LLM synthesis failed: %s", exc)
            return (
                f"📅 Morning Briefing — {today}\n\n"
                + (git_summary or "No git activity.")
                + ("\n\nPending: " + "; ".join(pending_tasks[:3]) if pending_tasks else "")
            )

    def maybe_fire(self, target_time: str = "08:30") -> bool:
        """Fire the briefing if the current time matches target_time (HH:MM)."""
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        if self._last_fired_date == today_str:
            return False

        try:
            h, m = map(int, target_time.split(":"))
            if now.hour == h and now.minute == m:
                logger.info("Shadow Worker: firing morning briefing")
                self.generate()
                self._last_fired_date = today_str
                return True
        except Exception:
            pass
        return False


# ── Error Sentinel ────────────────────────────────────────────────────────────

_TRACEBACK_PATTERN = re.compile(
    r"(Traceback \(most recent call last\):.*?(?:Error|Exception)[^\n]*)",
    re.DOTALL,
)
_TEST_FAIL_PATTERN = re.compile(
    r"(FAILED\s+\S+\.py::\S+.*?(?:AssertionError|Error)[^\n]*)",
)


class ErrorSentinel:
    """
    Monitors log files and terminal output for Python errors/test failures.
    On detection: diagnoses the error using LLM and stores a proposed fix.
    """

    def __init__(
        self,
        watch_paths: list[Path] | None = None,
        on_alert: Callable[[ErrorAlert], None] | None = None,
    ) -> None:
        self._watch_paths = watch_paths or self._default_watch_paths()
        self._on_alert = on_alert
        self._file_positions: dict[Path, int] = {}
        self._recent_alerts: dict[str, float] = {}  # file: last_alert_time
        self._cooldown_sec = 60.0

    @staticmethod
    def _default_watch_paths() -> list[Path]:
        """Determine default paths to watch for errors."""
        paths = []
        # Project log directory
        try:
            paths.append(settings.log_dir)
        except Exception:
            pass
        # Common pytest output
        paths.extend([
            Path("logs"),
            Path(".pytest_cache"),
            Path("tmp"),
        ])
        # Filter to existing directories/files
        return [p for p in paths if p.exists()]

    def scan_once(self) -> list[ErrorAlert]:
        """Scan watched paths for new errors. Returns list of new alerts."""
        alerts = []
        for watch_path in self._watch_paths:
            if watch_path.is_dir():
                for log_file in watch_path.glob("**/*.log"):
                    alert = self._scan_file(log_file)
                    if alert:
                        alerts.append(alert)
            elif watch_path.is_file():
                alert = self._scan_file(watch_path)
                if alert:
                    alerts.append(alert)
        return alerts

    def _scan_file(self, path: Path) -> ErrorAlert | None:
        """Scan a single file for new error content since last check."""
        try:
            current_size = path.stat().st_size
            last_pos = self._file_positions.get(path, 0)

            if current_size <= last_pos:
                return None

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                new_content = f.read(65536)  # Read up to 64KB of new content

            self._file_positions[path] = current_size

            # Look for error patterns
            tb_match = _TRACEBACK_PATTERN.search(new_content)
            fail_match = _TEST_FAIL_PATTERN.search(new_content)
            match = tb_match or fail_match

            if not match:
                return None

            # Check cooldown
            path_key = str(path)
            last_alert = self._recent_alerts.get(path_key, 0)
            if time.time() - last_alert < self._cooldown_sec:
                return None

            self._recent_alerts[path_key] = time.time()
            snippet = match.group(1)[:600].strip()
            error_type = self._classify_error(snippet)

            alert = ErrorAlert(
                file_path=str(path),
                error_type=error_type,
                traceback_snippet=snippet,
            )

            # Diagnose with LLM
            alert.proposed_fix = self._diagnose_with_llm(snippet, str(path))

            # Store in memory as a pinned alert
            self._store_alert_in_memory(alert)

            if self._on_alert:
                try:
                    self._on_alert(alert)
                except Exception as exc:
                    logger.error("Error sentinel callback error: %s", exc)

            logger.warning(
                "Error Sentinel: %s detected in %s", error_type, path
            )
            return alert

        except Exception as exc:
            logger.debug("Error sentinel scan failed for %s: %s", path, exc)
            return None

    @staticmethod
    def _classify_error(snippet: str) -> str:
        """Classify the error type from the traceback."""
        patterns = {
            "AssertionError": "Test Failure",
            "ImportError": "Import Error",
            "ModuleNotFoundError": "Missing Module",
            "AttributeError": "Attribute Error",
            "TypeError": "Type Error",
            "ValueError": "Value Error",
            "RuntimeError": "Runtime Error",
            "KeyError": "Key Error",
            "FileNotFoundError": "File Not Found",
            "ConnectionError": "Connection Error",
        }
        for exc_type, label in patterns.items():
            if exc_type in snippet:
                return label
        return "Python Error"

    @staticmethod
    def _diagnose_with_llm(snippet: str, file_path: str) -> str:
        """Ask the LLM to diagnose and propose a fix for the error."""
        try:
            from ragnarok import llm
            prompt = (
                f"Diagnose this Python error from file '{file_path}' and suggest a concise fix.\n"
                f"Respond in 2-3 sentences: first the root cause, then the fix.\n\n"
                f"Error:\n```\n{snippet[:800]}\n```"
            )
            response = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=settings.model_small,
                temperature=0.1,
            )
            return response.content.strip()
        except Exception as exc:
            return f"LLM diagnosis unavailable: {exc}"

    @staticmethod
    def _store_alert_in_memory(alert: ErrorAlert) -> None:
        """Store the error alert as a pinned memory entry."""
        try:
            from ragnarok.memory.db import MemoryDB
            import numpy as np
            db = MemoryDB()
            text = (
                f"ERROR ALERT [{alert.error_type}] in {alert.file_path}: "
                f"{alert.traceback_snippet[:200]}"
            )
            if alert.proposed_fix:
                text += f"\nProposed fix: {alert.proposed_fix}"
            # Store with zero embedding (will be re-embedded by consolidation)
            db.insert(
                text=text,
                embedding=np.zeros(768, dtype=np.float32).tolist(),
                kind="episodic",
                importance=0.9,
                pinned=True,
                source="error_sentinel",
            )
        except Exception as exc:
            logger.debug("Error sentinel: could not store alert in memory: %s", exc)


# ── Shadow Worker Orchestrator ────────────────────────────────────────────────

class ShadowWorker:
    """
    Background worker that runs MorningBriefingEngine and ErrorSentinel
    in a daemon thread.
    """

    def __init__(
        self,
        brief_time: str | None = None,
        on_briefing: Callable[[BriefingReport], None] | None = None,
        on_error: Callable[[ErrorAlert], None] | None = None,
        watch_paths: list[Path] | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self._brief_time = brief_time or settings.daily_brief_time
        self._briefing_engine = MorningBriefingEngine(
            repo_root=repo_root, on_briefing=on_briefing
        )
        self._sentinel = ErrorSentinel(watch_paths=watch_paths, on_alert=on_error)
        self._thread: threading.Thread | None = None
        self._running = False
        self._poll_interval = 30  # seconds

    def start(self) -> None:
        """Start the shadow worker background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ragnarok-shadow-worker"
        )
        self._thread.start()
        logger.info("Shadow Worker started (briefing at %s, error sentinel active)", self._brief_time)

    def stop(self) -> None:
        self._running = False

    def _run_loop(self) -> None:
        """Main background loop."""
        while self._running:
            try:
                # Morning briefing check
                self._briefing_engine.maybe_fire(self._brief_time)
                # Error sentinel scan
                self._sentinel.scan_once()
            except Exception as exc:
                logger.error("Shadow Worker loop error: %s", exc)
            time.sleep(self._poll_interval)

    def force_briefing(self) -> BriefingReport:
        """Manually trigger a briefing regardless of time."""
        return self._briefing_engine.generate()

    def force_scan(self) -> list[ErrorAlert]:
        """Manually trigger an error sentinel scan."""
        return self._sentinel.scan_once()


# ── Module-level singleton ────────────────────────────────────────────────────

_shadow_worker: ShadowWorker | None = None


def get_shadow_worker(
    on_briefing: Callable[[BriefingReport], None] | None = None,
    on_error: Callable[[ErrorAlert], None] | None = None,
) -> ShadowWorker:
    global _shadow_worker
    if _shadow_worker is None:
        _shadow_worker = ShadowWorker(
            on_briefing=on_briefing,
            on_error=on_error,
        )
    return _shadow_worker
