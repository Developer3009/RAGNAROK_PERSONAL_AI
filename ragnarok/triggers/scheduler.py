"""
ragnarok/triggers/scheduler.py
──────────────────────────────
Phase 6: Cron-like trigger scheduler for always-on tasks.

Provides a background thread that fires ScheduledTrigger callbacks at
configured times (HH:MM daily) or on a fixed interval (every N seconds).

Design:
  - Lightweight: uses only stdlib (threading, time, datetime).
  - Thread-safe: all state mutations are under a lock.
  - Persistent: fires missed triggers if the process restarted late
    (configurable, off by default to avoid duplicate fires on slow boot).
  - Graceful shutdown: stop() joins the worker thread cleanly.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTrigger:
    """
    Describes one scheduled task.

    Attributes:
        name:             Human-readable identifier (used in logs and audit).
        callback:         Function to call when the trigger fires. Receives the
                          trigger name as its only argument.
        daily_at:         Optional HH:MM string. If set, fires once per day at
                          this wall-clock time.
        interval_sec:     Optional positive float. If set, fires every N seconds
                          (intended for short intervals / testing).
        enabled:          Whether this trigger is active.
        fire_missed:      If True, fires immediately on startup if the scheduled
                          time was missed today (only relevant for daily_at).
    """

    name: str
    callback: Callable[[str], None]
    daily_at: str | None = None          # "HH:MM"
    interval_sec: float | None = None    # e.g. 30.0
    enabled: bool = True
    fire_missed: bool = False

    # Internal tracking (not part of public API)
    _last_fired_date: str = field(default="", init=False, repr=False)
    _last_fired_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.daily_at is None and self.interval_sec is None:
            raise ValueError(
                f"Trigger '{self.name}' must specify daily_at or interval_sec."
            )
        if self.daily_at is not None and self.interval_sec is not None:
            raise ValueError(
                f"Trigger '{self.name}' cannot specify both daily_at and interval_sec."
            )
        if self.daily_at is not None:
            # Validate HH:MM format
            parts = self.daily_at.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Trigger '{self.name}': daily_at must be 'HH:MM', got '{self.daily_at}'."
                )
            try:
                h, m = int(parts[0]), int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError()
            except ValueError:
                raise ValueError(
                    f"Trigger '{self.name}': invalid daily_at '{self.daily_at}'."
                )
        if self.interval_sec is not None and self.interval_sec <= 0:
            raise ValueError(
                f"Trigger '{self.name}': interval_sec must be positive."
            )

    def _should_fire_daily(self, now: datetime) -> bool:
        """Return True if this daily trigger should fire right now."""
        if not self.enabled or self.daily_at is None:
            return False
        today_str = now.strftime("%Y-%m-%d")
        if self._last_fired_date == today_str:
            return False  # Already fired today
        h, m = (int(x) for x in self.daily_at.split(":"))
        scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        # Fire if we're past the scheduled minute for today
        if now >= scheduled_dt:
            return True
        return False

    def _should_fire_interval(self, now_ts: float) -> bool:
        """Return True if this interval trigger should fire."""
        if not self.enabled or self.interval_sec is None:
            return False
        return (now_ts - self._last_fired_ts) >= self.interval_sec

    def _mark_fired(self, now: datetime) -> None:
        self._last_fired_date = now.strftime("%Y-%m-%d")
        self._last_fired_ts = now.timestamp()


class TriggerScheduler:
    """
    Background scheduler that fires ScheduledTriggers at configured times.

    Usage::

        scheduler = TriggerScheduler()
        scheduler.add(ScheduledTrigger(
            name="daily_brief",
            callback=lambda name: print(f"Trigger: {name}"),
            daily_at="08:00",
        ))
        scheduler.start()
        # ... later ...
        scheduler.stop()

    Thread safety:
        All trigger mutations (add, remove, enable/disable) are protected by
        an internal RLock. Callbacks are invoked from the scheduler thread —
        keep them short or dispatch work to another thread.
    """

    _TICK_INTERVAL_SEC = 15.0   # How often the scheduler polls triggers

    def __init__(self, tick_interval_sec: float | None = None) -> None:
        self._triggers: list[ScheduledTrigger] = []
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick = tick_interval_sec or self._TICK_INTERVAL_SEC

    # ── Trigger Management ─────────────────────────────────────────────────────

    def add(self, trigger: ScheduledTrigger) -> None:
        """Register a new trigger."""
        with self._lock:
            if any(t.name == trigger.name for t in self._triggers):
                raise ValueError(f"Trigger '{trigger.name}' already registered.")
            self._triggers.append(trigger)
            logger.info("Trigger registered: %s", trigger.name)

    def remove(self, name: str) -> bool:
        """Remove a trigger by name. Returns True if found and removed."""
        with self._lock:
            before = len(self._triggers)
            self._triggers = [t for t in self._triggers if t.name != name]
            removed = len(self._triggers) < before
            if removed:
                logger.info("Trigger removed: %s", name)
            return removed

    def enable(self, name: str) -> None:
        """Enable a trigger by name."""
        with self._lock:
            for t in self._triggers:
                if t.name == name:
                    t.enabled = True
                    return
            raise KeyError(f"Trigger '{name}' not found.")

    def disable(self, name: str) -> None:
        """Disable a trigger by name (it stays registered but won't fire)."""
        with self._lock:
            for t in self._triggers:
                if t.name == name:
                    t.enabled = False
                    return
            raise KeyError(f"Trigger '{name}' not found.")

    def list_triggers(self) -> list[dict]:
        """Return a snapshot of all registered triggers."""
        with self._lock:
            return [
                {
                    "name": t.name,
                    "enabled": t.enabled,
                    "daily_at": t.daily_at,
                    "interval_sec": t.interval_sec,
                    "last_fired": t._last_fired_date or t._last_fired_ts or "never",
                }
                for t in self._triggers
            ]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("TriggerScheduler is already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ragnarok-scheduler",
            daemon=True,   # Dies with the main process
        )
        self._thread.start()
        logger.info("TriggerScheduler started (tick=%.1fs).", self._tick)

    def stop(self, timeout_sec: float = 5.0) -> None:
        """Signal the scheduler to stop and wait for its thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_sec)
            if self._thread.is_alive():
                logger.warning("TriggerScheduler thread did not stop within %.1fs.", timeout_sec)
            else:
                logger.info("TriggerScheduler stopped.")

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        logger.debug("TriggerScheduler loop started.")
        while not self._stop_event.is_set():
            self._tick_once()
            self._stop_event.wait(self._tick)
        logger.debug("TriggerScheduler loop exited.")

    def _tick_once(self) -> None:
        now = datetime.now()
        now_ts = now.timestamp()
        with self._lock:
            triggers_snapshot = list(self._triggers)

        for trigger in triggers_snapshot:
            should_fire = False
            if trigger.daily_at is not None:
                should_fire = trigger._should_fire_daily(now)
            elif trigger.interval_sec is not None:
                should_fire = trigger._should_fire_interval(now_ts)

            if should_fire:
                logger.info("Firing trigger: %s", trigger.name)
                trigger._mark_fired(now)
                try:
                    trigger.callback(trigger.name)
                except Exception as exc:
                    logger.exception(
                        "Trigger '%s' callback raised an exception: %s",
                        trigger.name,
                        exc,
                    )
