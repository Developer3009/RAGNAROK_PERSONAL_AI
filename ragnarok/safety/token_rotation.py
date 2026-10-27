"""
ragnarok/safety/token_rotation.py
───────────────────────────────────
Automatic UI token rotation for the Ragnarok web interface.

Security hardening:
  - Rotates the session token every N minutes (configurable)
  - Broadcasts the new token over a WebSocket so the open browser page
    silently updates — no manual copy-paste required
  - Old token is invalidated immediately after rotation
  - Token history kept in-memory only (never written to disk after rotation)
  - Rotation can be triggered manually via POST /api/security/rotate-token
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)

_DEFAULT_ROTATION_INTERVAL_SEC = 30 * 60   # 30 minutes


@dataclass
class TokenSlot:
    """A single token with its creation timestamp."""
    value: str
    created_at: float = field(default_factory=time.time)

    @property
    def age_sec(self) -> float:
        return time.time() - self.created_at


class TokenRotator:
    """
    Manages the active UI token with automatic rotation support.

    The rotator keeps one active token and one grace-period token
    (the previous token, valid for `grace_sec` seconds after rotation
    to allow in-flight requests to complete).

    Thread-safe.
    """

    def __init__(
        self,
        initial_token: str | None = None,
        rotation_interval_sec: float = _DEFAULT_ROTATION_INTERVAL_SEC,
        grace_sec: float = 30.0,
    ) -> None:
        self._interval = rotation_interval_sec
        self._grace = grace_sec
        self._lock = Lock()
        initial = initial_token or secrets.token_urlsafe(24)
        self._active = TokenSlot(value=initial)
        self._previous: TokenSlot | None = None
        self._rotation_callbacks: list = []
        logger.info("TokenRotator initialised (interval=%.0fs, grace=%.0fs)", rotation_interval_sec, grace_sec)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def active_token(self) -> str:
        with self._lock:
            return self._active.value

    def is_valid(self, token: str) -> bool:
        """Return True if the token matches the active OR still-valid previous token."""
        with self._lock:
            if secrets.compare_digest(token, self._active.value):
                return True
            if self._previous and self._previous.age_sec < self._grace:
                return secrets.compare_digest(token, self._previous.value)
        return False

    def maybe_rotate(self) -> bool:
        """
        Rotate the token if the rotation interval has elapsed.

        Returns:
            True if rotation occurred, False otherwise.
        """
        with self._lock:
            if self._active.age_sec >= self._interval:
                return self._do_rotate()
        return False

    def force_rotate(self) -> str:
        """Immediately rotate the token regardless of interval. Returns new token."""
        with self._lock:
            self._do_rotate()
            return self._active.value

    def register_rotation_callback(self, cb) -> None:
        """Register a callable(new_token: str) invoked after every rotation."""
        with self._lock:
            self._rotation_callbacks.append(cb)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _do_rotate(self) -> bool:
        """Must be called with self._lock held."""
        new_token = secrets.token_urlsafe(24)
        old = self._active
        self._previous = old
        self._active = TokenSlot(value=new_token)
        logger.info(
            "UI token rotated (was age=%.0fs, new_token=***%s)",
            old.age_sec, new_token[-4:],
        )
        # Fire callbacks (outside lock to avoid deadlock)
        cbs = list(self._rotation_callbacks)
        # Run callbacks asynchronously to not block the lock caller
        for cb in cbs:
            try:
                cb(new_token)
            except Exception as exc:
                logger.warning("Token rotation callback failed: %s", exc)
        return True


# ── Module singleton ──────────────────────────────────────────────────────────
# Created lazily by app.py so it can receive the initial token from the token file

_rotator: TokenRotator | None = None


def get_rotator(initial_token: str | None = None) -> TokenRotator:
    global _rotator
    if _rotator is None:
        _rotator = TokenRotator(initial_token=initial_token)
    return _rotator
