"""
ragnarok/triggers/watchdog.py
──────────────────────────────
Phase 6: Watchdog — detect hangs / unresponsive agent turns and take action.

The Watchdog monitors how long the agent has been "thinking" on a single turn.
If the deadline passes it:
  1. Fires an optional on_timeout callback (e.g., to kill the thread / signal).
  2. Writes an event to the audit log.
  3. Sets an internal 'tripped' flag so callers can inspect state.

Usage pattern::

    watchdog = Watchdog(timeout_sec=30.0, on_timeout=lambda: print("hung!"))
    watchdog.arm()          # Start the countdown
    agent.chat(user_input)  # Long-running call
    watchdog.disarm()       # Cancel if it finishes in time

The watchdog resets automatically when disarm() is called, ready for the
next turn.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class Watchdog:
    """
    Single-turn hang detector using a daemon timer thread.

    Args:
        timeout_sec:  Seconds before the watchdog trips. Default 60.
        on_timeout:   Optional zero-argument callback invoked on timeout.
                      Called from the watchdog thread — keep it lightweight.
        name:         Human-readable label (used in log messages).

    Thread safety:
        arm() / disarm() are safe to call from any thread.
        The on_timeout callback is invoked from the watchdog thread.
    """

    def __init__(
        self,
        timeout_sec: float = 60.0,
        on_timeout: Callable[[], None] | None = None,
        name: str = "ragnarok-watchdog",
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive.")
        self.timeout_sec = timeout_sec
        self.on_timeout = on_timeout
        self.name = name

        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._tripped = False
        self._armed_count = 0   # Monotonic arm counter for reentrancy guard

    # ── Public API ─────────────────────────────────────────────────────────────

    def arm(self) -> None:
        """Start the countdown. Cancels any in-progress timer first."""
        with self._lock:
            self._cancel_timer()
            self._tripped = False
            self._armed_count += 1
            current_count = self._armed_count
            self._timer = threading.Timer(
                self.timeout_sec,
                self._on_deadline,
                args=(current_count,),
            )
            self._timer.daemon = True
            self._timer.name = self.name
            self._timer.start()
            logger.debug("Watchdog armed (%.1fs).", self.timeout_sec)

    def disarm(self) -> None:
        """Cancel the countdown — call this when the agent finishes a turn."""
        with self._lock:
            self._cancel_timer()
            logger.debug("Watchdog disarmed.")

    def reset(self) -> None:
        """Alias for disarm(); resets the tripped flag."""
        with self._lock:
            self._cancel_timer()
            self._tripped = False

    @property
    def tripped(self) -> bool:
        """True if the watchdog fired (timed out). Reset by arm() or reset()."""
        with self._lock:
            return self._tripped

    @property
    def is_armed(self) -> bool:
        """True if the watchdog timer is currently running."""
        with self._lock:
            return bool(self._timer and self._timer.is_alive())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _cancel_timer(self) -> None:
        """Cancel the running timer if any. Must be called under self._lock."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_deadline(self, armed_count: int) -> None:
        """Invoked by the Timer thread when timeout expires."""
        with self._lock:
            # Stale fire guard: if arm() was called again before we fired, ignore
            if armed_count != self._armed_count:
                return
            self._tripped = True
            self._timer = None

        logger.warning(
            "Watchdog '%s' tripped! Agent turn exceeded %.1fs — possible hang.",
            self.name,
            self.timeout_sec,
        )

        if self.on_timeout:
            try:
                self.on_timeout()
            except Exception as exc:
                logger.exception(
                    "Watchdog on_timeout callback raised: %s", exc
                )
