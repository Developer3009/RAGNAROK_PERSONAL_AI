"""
ragnarok/triggers/__init__.py
──────────────────────────────
Phase 6: Always-on triggers and crash recovery.

Exposes:
  - TriggerScheduler: cron-like scheduler for time-based tasks.
  - Watchdog:         detects crashes and requests auto-restart.
  - SessionCheckpoint: persist and restore agent session state across crashes.
"""

from ragnarok.triggers.scheduler import TriggerScheduler, ScheduledTrigger
from ragnarok.triggers.checkpoint import SessionCheckpoint
from ragnarok.triggers.watchdog import Watchdog

__all__ = [
    "TriggerScheduler",
    "ScheduledTrigger",
    "SessionCheckpoint",
    "Watchdog",
]
