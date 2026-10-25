"""
ragnarok/safety/rate_limiter.py
────────────────────────────────
Sliding-window rate limiter for Ragnarok tool calls.

Prevents expensive or sensitive skills from being hammered in a short window.
Each tool gets an independent sliding window tracked by call timestamps.

Example:
    limiter = RateLimiter()
    limiter.configure("fetch_web", max_calls=5, window_sec=60.0)
    ok, wait = limiter.check_and_record("fetch_web")
    if not ok:
        print(f"Rate limited. Try again in {wait:.1f}s")
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateRule:
    max_calls: int        # max allowed calls in the window
    window_sec: float     # window duration in seconds
    _timestamps: deque = field(default_factory=deque, repr=False)


class RateLimiter:
    """
    Sliding-window rate limiter per skill name.

    Default limits (applied when no explicit rule is configured):
        - fetch_web:    5 calls / 60 seconds
        - write_file:   10 calls / 60 seconds
        - search_files: 20 calls / 60 seconds
        - All others:   unlimited
    """

    _DEFAULTS: dict[str, tuple[int, float]] = {
        "fetch_web":    (5,  60.0),
        "write_file":   (10, 60.0),
        "search_files": (20, 60.0),
    }

    def __init__(self) -> None:
        self._rules: dict[str, RateRule] = {}
        # Apply defaults
        for tool, (max_calls, window) in self._DEFAULTS.items():
            self.configure(tool, max_calls=max_calls, window_sec=window)

    # ── Configuration ──────────────────────────────────────────────────────────

    def configure(self, tool_name: str, max_calls: int, window_sec: float) -> None:
        """Set or overwrite the rate rule for a tool."""
        self._rules[tool_name] = RateRule(max_calls=max_calls, window_sec=window_sec)
        logger.debug(
            "RateLimiter configured: tool=%s max_calls=%d window=%.0fs",
            tool_name, max_calls, window_sec,
        )

    def remove(self, tool_name: str) -> None:
        """Remove rate limit for a tool (make it unlimited)."""
        self._rules.pop(tool_name, None)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def check_and_record(self, tool_name: str) -> tuple[bool, float]:
        """
        Check if `tool_name` is within its rate limit, and record this call.

        Returns:
            (allowed: bool, retry_after_seconds: float)
            If allowed, retry_after_seconds is 0.0.
            If rate limited, retry_after_seconds is the time to wait before retrying.
        """
        rule = self._rules.get(tool_name)
        if rule is None:
            return True, 0.0  # No rule = unlimited

        now = time.monotonic()
        cutoff = now - rule.window_sec

        # Evict expired timestamps from the window
        while rule._timestamps and rule._timestamps[0] < cutoff:
            rule._timestamps.popleft()

        if len(rule._timestamps) >= rule.max_calls:
            # Calculate how long until the oldest call expires
            oldest = rule._timestamps[0]
            retry_after = (oldest + rule.window_sec) - now
            logger.warning(
                "Rate limit hit: tool=%s calls=%d/%d window=%.0fs retry_in=%.1fs",
                tool_name, len(rule._timestamps), rule.max_calls, rule.window_sec, retry_after,
            )
            return False, max(retry_after, 0.0)

        # Record this call
        rule._timestamps.append(now)
        return True, 0.0

    def current_count(self, tool_name: str) -> int:
        """Return the number of calls in the current window for a tool."""
        rule = self._rules.get(tool_name)
        if rule is None:
            return 0
        now = time.monotonic()
        cutoff = now - rule.window_sec
        while rule._timestamps and rule._timestamps[0] < cutoff:
            rule._timestamps.popleft()
        return len(rule._timestamps)

    def reset(self, tool_name: str | None = None) -> None:
        """Reset call counts for one tool or all tools."""
        if tool_name:
            rule = self._rules.get(tool_name)
            if rule:
                rule._timestamps.clear()
        else:
            for rule in self._rules.values():
                rule._timestamps.clear()

    def status(self) -> dict[str, dict]:
        """Return current window usage for all configured tools."""
        now = time.monotonic()
        result = {}
        for name, rule in self._rules.items():
            cutoff = now - rule.window_sec
            active = sum(1 for t in rule._timestamps if t >= cutoff)
            result[name] = {
                "calls_in_window": active,
                "max_calls": rule.max_calls,
                "window_sec": rule.window_sec,
            }
        return result
