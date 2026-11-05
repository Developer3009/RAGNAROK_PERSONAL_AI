"""
ragnarok/triggers/ambient_context.py
─────────────────────────────────────
Innovation #10 — Ambient Cognitive Context Stream (Passive Desktop Awareness)

Background daemon that passively monitors what the user is doing:
  • WindowObserver   — polls the foreground window title every N seconds
  • ScreenDiffDetector — only triggers OCR/LLM when pixels changed significantly
  • EpisodicRingBuffer — SQLite ring-buffer (max 100 episodes) for fast recall
  • ContextStreamAnalyzer — extracts text, detects patterns, inserts into memory

Runs in a daemon thread; call get_ambient_daemon().start() to activate.

Privacy guarantee: ALL processing is 100% local — no data leaves the machine.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Episode dataclass ─────────────────────────────────────────────────────────

@dataclass
class AmbientEpisode:
    timestamp: float
    app_name: str
    window_title: str
    ocr_summary: str
    embedding: bytes = b""     # serialized float32 numpy array (optional)


# ── EpisodicRingBuffer ────────────────────────────────────────────────────────

class EpisodicRingBuffer:
    """
    SQLite-backed ring buffer that stores up to `max_entries` ambient episodes.
    Oldest entries are auto-purged when capacity is exceeded.
    """

    def __init__(self, db_path: str, max_entries: int = 100) -> None:
        self.db_path = db_path
        self.max_entries = max_entries
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ambient_episodes (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL NOT NULL,
                    app_name      TEXT NOT NULL,
                    window_title  TEXT NOT NULL,
                    ocr_summary   TEXT NOT NULL,
                    embedding     BLOB
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ae_ts ON ambient_episodes(timestamp)"
            )

    def insert(self, episode: AmbientEpisode) -> None:
        """Insert episode; prune oldest if over capacity."""
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT INTO ambient_episodes "
                "(timestamp, app_name, window_title, ocr_summary, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    episode.timestamp,
                    episode.app_name,
                    episode.window_title,
                    episode.ocr_summary,
                    episode.embedding or b"",
                ),
            )
            # Prune oldest entries
            count = conn.execute(
                "SELECT COUNT(*) FROM ambient_episodes"
            ).fetchone()[0]
            if count > self.max_entries:
                excess = count - self.max_entries
                conn.execute(
                    "DELETE FROM ambient_episodes WHERE id IN "
                    "(SELECT id FROM ambient_episodes ORDER BY timestamp ASC LIMIT ?)",
                    (excess,),
                )

    def recent(self, limit: int = 10) -> list[AmbientEpisode]:
        """Return the N most recent episodes."""
        rows = self._conn().execute(
            "SELECT * FROM ambient_episodes ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            AmbientEpisode(
                timestamp=row["timestamp"],
                app_name=row["app_name"],
                window_title=row["window_title"],
                ocr_summary=row["ocr_summary"],
                embedding=row["embedding"] or b"",
            )
            for row in rows
        ]

    def count(self) -> int:
        return self._conn().execute(
            "SELECT COUNT(*) FROM ambient_episodes"
        ).fetchone()[0]

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ── WindowObserver ────────────────────────────────────────────────────────────

class WindowObserver:
    """
    Polls the currently active foreground window title.
    Uses win32gui on Windows; falls back to a stub on other platforms.
    """

    def get_active_window(self) -> tuple[str, str]:
        """Returns (app_name, window_title) for the foreground window."""
        try:
            import win32gui   # type: ignore[import]
            import win32process  # type: ignore[import]
            import psutil       # type: ignore[import]

            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                proc = psutil.Process(pid)
                app_name = proc.name()
            except Exception:
                app_name = "unknown"
            return app_name, title
        except ImportError:
            # Graceful fallback if pywin32/psutil unavailable
            return "unknown", "(win32gui not available)"
        except Exception as exc:
            logger.debug("WindowObserver error: %s", exc)
            return "unknown", ""


# ── ScreenDiffDetector ────────────────────────────────────────────────────────

class ScreenDiffDetector:
    """
    Captures a screenshot with mss and computes pixel diff vs last capture.
    Returns the new screenshot bytes only if diff > threshold, else None.
    """

    def __init__(self, diff_threshold: float = 0.05) -> None:
        self.diff_threshold = diff_threshold
        self._last_pixels: Any = None

    def capture_if_changed(self) -> bytes | None:
        """
        Returns PNG bytes of the primary monitor screenshot if significant
        change detected since last call, else None.
        """
        try:
            import mss
            import mss.tools
            import numpy as np

            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                sct_img = sct.grab(monitor)
                # Convert to numpy for fast diff
                new_pixels = np.frombuffer(sct_img.bgra, dtype=np.uint8).reshape(
                    sct_img.height, sct_img.width, 4
                )

            if self._last_pixels is None:
                self._last_pixels = new_pixels
                # On first capture, always return screenshot
                import io
                from PIL import Image
                img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            # Compute mean absolute diff (normalised 0-1)
            diff = float(np.mean(np.abs(new_pixels.astype(int) - self._last_pixels.astype(int)))) / 255.0
            self._last_pixels = new_pixels

            if diff > self.diff_threshold:
                import io
                from PIL import Image
                img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            return None

        except Exception as exc:
            logger.debug("ScreenDiffDetector error: %s", exc)
            return None


# ── ContextStreamAnalyzer ─────────────────────────────────────────────────────

class ContextStreamAnalyzer:
    """
    Takes a screenshot + window metadata and produces a short textual summary
    using the local Ollama LLM (or basic OCR if vision model unavailable).
    """

    def __init__(self, model: str | None = None) -> None:
        from ragnarok.config import settings  # late import
        self.model = model or settings.model_main

    def summarize(
        self,
        app_name: str,
        window_title: str,
        screenshot_bytes: bytes | None,
    ) -> str:
        """Return a ≤50-word summary of what the user is currently doing."""
        if not window_title and not screenshot_bytes:
            return ""

        base_summary = f"User is in {app_name}: '{window_title}'" if window_title else "User is active on desktop"

        if not screenshot_bytes:
            return base_summary

        # Try local LLM vision summary
        try:
            from ragnarok import llm
            import base64
            img_b64 = base64.b64encode(screenshot_bytes).decode()
            # Use Ollama's /api/generate endpoint directly for vision (llava model)
            import httpx
            from ragnarok.config import settings
            with httpx.Client(base_url=settings.ollama_url, timeout=15) as client:
                resp = client.post("/api/generate", json={
                    "model": "llava",
                    "prompt": (
                        "Describe what the user is doing in one sentence "
                        "(max 20 words). Focus on activity, not aesthetics."
                    ),
                    "images": [img_b64],
                    "stream": False,
                })
                if resp.status_code == 200:
                    text = resp.json().get("response", "").strip()
                    if text and len(text) > 5:
                        return text[:200]
        except Exception as exc:
            logger.debug("Vision LLM summary failed (%s); using title fallback", exc)

        return base_summary

    def detect_stuck_pattern(
        self, episodes: list[AmbientEpisode], same_title_threshold: int = 4
    ) -> str | None:
        """
        Detect if the user appears stuck: same window title for N+ consecutive episodes.
        Returns a proactive help hint or None.
        """
        if len(episodes) < same_title_threshold:
            return None
        recent_titles = [e.window_title for e in episodes[:same_title_threshold]]
        if len(set(recent_titles)) == 1 and recent_titles[0]:
            return (
                f"Proactive hint: You have been on '{recent_titles[0]}' for a while. "
                "Need help with this task?"
            )
        return None


# ── AmbientContextDaemon ──────────────────────────────────────────────────────

class AmbientContextDaemon:
    """
    Background daemon thread that runs the ambient monitoring pipeline.
    Call start() to launch; stop() to shut down gracefully.
    """

    def __init__(
        self,
        db_path: str,
        poll_interval_sec: float = 15.0,
        diff_threshold: float = 0.05,
        ring_buffer_size: int = 100,
    ) -> None:
        self.poll_interval_sec = poll_interval_sec
        self.ring_buffer = EpisodicRingBuffer(db_path, max_entries=ring_buffer_size)
        self.window_observer = WindowObserver()
        self.screen_detector = ScreenDiffDetector(diff_threshold=diff_threshold)
        self.analyzer = ContextStreamAnalyzer()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_proactive_hint_time: float = 0.0

    def start(self) -> None:
        """Launch the daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.info("AmbientContextDaemon already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ambient-context-daemon",
            daemon=True,
        )
        self._thread.start()
        logger.info("AmbientContextDaemon started (poll=%.1fs)", self.poll_interval_sec)

    def stop(self) -> None:
        """Signal the daemon to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval_sec + 2)
        self.ring_buffer.close()
        logger.info("AmbientContextDaemon stopped")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run_loop(self) -> None:
        """Main monitoring loop — runs until stop_event is set."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.warning("AmbientContextDaemon tick error: %s", exc)
            self._stop_event.wait(timeout=self.poll_interval_sec)

    def _tick(self) -> None:
        """Single observation cycle."""
        app_name, window_title = self.window_observer.get_active_window()
        screenshot = self.screen_detector.capture_if_changed()

        if not window_title and screenshot is None:
            return  # nothing interesting

        summary = self.analyzer.summarize(app_name, window_title, screenshot)
        if not summary:
            return

        episode = AmbientEpisode(
            timestamp=time.time(),
            app_name=app_name,
            window_title=window_title,
            ocr_summary=summary,
        )
        self.ring_buffer.insert(episode)
        logger.debug("AmbientContext: %s", summary)

        # Check for stuck pattern and inject hint into memory (cooldown: 5 min)
        recent = self.ring_buffer.recent(limit=6)
        hint = self.analyzer.detect_stuck_pattern(recent)
        if hint and (time.time() - self._last_proactive_hint_time) > 300:
            self._last_proactive_hint_time = time.time()
            self._inject_hint(hint)

    def _inject_hint(self, hint: str) -> None:
        """Inject a proactive hint into the Ragnarok memory system."""
        try:
            from ragnarok.memory.db import MemoryDB
            from ragnarok.config import settings
            db = MemoryDB(settings.db_dir)
            db.add(text=hint, source="ambient_context")
            logger.info("Injected proactive hint: %s", hint)
        except Exception as exc:
            logger.debug("Could not inject hint into memory: %s", exc)

    def get_context_summary(self, limit: int = 5) -> str:
        """Return a human-readable summary of recent ambient context."""
        episodes = self.ring_buffer.recent(limit=limit)
        if not episodes:
            return "No ambient context captured yet."
        lines = [f"Recent ambient context ({self.ring_buffer.count()} total episodes):"]
        for ep in reversed(episodes):
            ts = time.strftime("%H:%M:%S", time.localtime(ep.timestamp))
            lines.append(f"  [{ts}] {ep.app_name}: {ep.ocr_summary[:100]}")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────

_daemon: AmbientContextDaemon | None = None
_daemon_lock = threading.Lock()


def get_ambient_daemon() -> AmbientContextDaemon:
    """Return the module-level AmbientContextDaemon singleton."""
    global _daemon
    if _daemon is None:
        with _daemon_lock:
            if _daemon is None:
                from ragnarok.config import settings
                db_path = str(Path(settings.db_dir) / "ambient_context.db")
                poll = float(getattr(settings, "ambient_poll_interval_sec", 15.0))
                diff = float(getattr(settings, "ambient_diff_threshold", 0.05))
                buf_size = int(getattr(settings, "ambient_ring_buffer_size", 100))
                _daemon = AmbientContextDaemon(
                    db_path=db_path,
                    poll_interval_sec=poll,
                    diff_threshold=diff,
                    ring_buffer_size=buf_size,
                )
    return _daemon
