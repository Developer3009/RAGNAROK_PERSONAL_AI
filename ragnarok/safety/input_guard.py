"""
ragnarok/safety/input_guard.py
───────────────────────────────
Input validation & throttling layer for the Ragnarok API.

Provides:
  - MAX_INPUT_LENGTH: hard cap on user chat message length (DoS / prompt-stuffing defence)
  - InputValidator: validates & sanitises incoming API text fields
  - IPRateLimiter: per-IP sliding-window rate limiter (independent of tool-level limiter)
  - validate_chat_input(req): FastAPI dependency combining both checks
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_INPUT_LENGTH = 8_000          # chars — blocks prompt-stuffing & context flooding
MAX_MEMORY_TEXT_LENGTH = 4_000    # chars for user-submitted memory text
_WINDOW_SEC = 60.0                # sliding window for IP-level rate limit
_MAX_REQUESTS_PER_WINDOW = 30     # per IP

# Patterns that look like null-byte / control-char injection
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── InputValidator ────────────────────────────────────────────────────────────

class InputValidator:
    """Validates and lightly sanitises text fields before they reach the LLM."""

    @staticmethod
    def validate_chat_message(text: str) -> str:
        """
        Validate and sanitise a user chat message.

        Raises:
            ValueError: if the message is empty, too long, or contains invalid characters.

        Returns:
            The sanitised message string.
        """
        if not text or not text.strip():
            raise ValueError("Message must not be empty.")
        if len(text) > MAX_INPUT_LENGTH:
            raise ValueError(
                f"Message too long ({len(text):,} chars). "
                f"Maximum allowed is {MAX_INPUT_LENGTH:,} characters."
            )
        # Strip null bytes and non-printable control characters
        sanitised = _CONTROL_CHARS_RE.sub("", text)
        return sanitised

    @staticmethod
    def validate_memory_text(text: str) -> str:
        """Validate and sanitise a memory text field."""
        if not text or not text.strip():
            raise ValueError("Memory text must not be empty.")
        if len(text) > MAX_MEMORY_TEXT_LENGTH:
            raise ValueError(
                f"Memory text too long ({len(text):,} chars). "
                f"Maximum is {MAX_MEMORY_TEXT_LENGTH:,} characters."
            )
        return _CONTROL_CHARS_RE.sub("", text)


# ── IPRateLimiter ─────────────────────────────────────────────────────────────

class IPRateLimiter:
    """
    Sliding-window rate limiter keyed by client IP address.
    Thread-safe for multi-threaded WSGI / ASGI servers.
    """

    def __init__(
        self,
        window_sec: float = _WINDOW_SEC,
        max_requests: int = _MAX_REQUESTS_PER_WINDOW,
    ) -> None:
        self._window = window_sec
        self._max = max_requests
        self._timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, ip: str) -> tuple[bool, float]:
        """
        Check whether `ip` is within the rate limit.

        Returns:
            (True, 0.0) if allowed.
            (False, retry_after_sec) if rate-limited.
        """
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            q = self._timestamps[ip]
            # Evict expired timestamps
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._max:
                retry_after = self._window - (now - q[0])
                return False, max(0.0, retry_after)
            q.append(now)
            return True, 0.0

    def reset(self, ip: str) -> None:
        with self._lock:
            self._timestamps.pop(ip, None)


# Module-level singleton
_ip_limiter = IPRateLimiter()


# ── FastAPI Dependencies ──────────────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """Extract the real client IP, honouring X-Forwarded-For (reverse proxy)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_ip_rate_limit(request: Request) -> None:
    """
    FastAPI dependency — raises HTTP 429 if the IP has exceeded the rate limit.
    """
    ip = get_client_ip(request)
    allowed, retry_after = _ip_limiter.is_allowed(ip)
    if not allowed:
        logger.warning("Rate limit exceeded for IP %s; retry_after=%.1fs", ip, retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many requests. Please retry after {retry_after:.0f} seconds."
            ),
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
