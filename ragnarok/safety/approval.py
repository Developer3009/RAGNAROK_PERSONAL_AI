"""
ragnarok/safety/approval.py
───────────────────────────
Interactive approval flow with default-to-deny timeout and parameter edit support.

Features:
  - Structured ApprovalRequest: skill, parameters, reason, risk level.
  - Interactive / programmatic decisions: "approve", "deny", "edit".
  - Thread-safe timeout handling: if no approval is received within `timeout_sec`,
    the request strictly defaults to DENIED.
  - Audit logging of approval lifecycle.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ragnarok.config import settings

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    EDITED = "edited"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    request_id: str
    tool_name: str
    args: dict[str, Any]
    risk_level: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ApprovalDecision:
    status: ApprovalStatus
    tool_name: str
    args: dict[str, Any]
    comment: str = ""

    @property
    def is_approved(self) -> bool:
        return self.status in (ApprovalStatus.APPROVED, ApprovalStatus.EDITED)


class ApprovalManager:
    """
    Manages user approval gates for high-risk actions.
    """

    def __init__(
        self,
        default_timeout_sec: float | None = None,
        approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
    ) -> None:
        self.timeout_sec = (
            default_timeout_sec
            if default_timeout_sec is not None
            else getattr(settings, "approval_timeout_sec", 60.0)
        )
        self.handler = approval_handler
        self._history: list[ApprovalDecision] = []
        self._pending: dict[str, tuple[ApprovalRequest, threading.Event, list[ApprovalDecision | None]]] = {}
        self._lock = threading.Lock()

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """Return list of currently active approval requests waiting for user decision."""
        with self._lock:
            return [req for req, _, _ in self._pending.values()]

    def resolve_request(
        self,
        request_id: str,
        status: ApprovalStatus | str,
        edited_args: dict[str, Any] | None = None,
        comment: str = "",
    ) -> bool:
        """Resolve a pending approval request programmatically or from UI."""
        with self._lock:
            if request_id not in self._pending:
                return False
            req, event, holder = self._pending[request_id]
            st = ApprovalStatus(status) if isinstance(status, str) else status
            holder[0] = ApprovalDecision(
                status=st,
                tool_name=req.tool_name,
                args=edited_args if (st == ApprovalStatus.EDITED and edited_args is not None) else req.args,
                comment=comment,
            )
            event.set()
            return True

    def request_approval(
        self,
        tool_name: str,
        args: dict[str, Any],
        risk_level: str = "high",
        reason: str = "",
        timeout_sec: float | None = None,
    ) -> ApprovalDecision:
        """
        Request approval for an action. Blocks up to `timeout_sec`.
        Defaults strictly to DENY if time expires or on error.
        """
        timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        req = ApprovalRequest(
            request_id=f"appr-{int(time.time() * 1000)}",
            tool_name=tool_name,
            args=dict(args),
            risk_level=risk_level,
            reason=reason or f"Execution of {risk_level}-risk tool '{tool_name}'",
        )

        logger.info(
            "APPROVAL REQUIRED: Tool='%s' Risk='%s' Timeout=%.1fs Args=%s",
            tool_name, risk_level, timeout, args
        )

        event = threading.Event()
        holder: list[ApprovalDecision | None] = [None]
        with self._lock:
            self._pending[req.request_id] = (req, event, holder)

        # If custom handler registered, run in worker thread
        if self.handler is not None:
            def _worker() -> None:
                try:
                    dec = self.handler(req)  # type: ignore[misc]
                    with self._lock:
                        if req.request_id in self._pending:
                            holder[0] = dec
                            event.set()
                except Exception as exc:
                    logger.error("Approval handler error: %s", exc)
                    with self._lock:
                        if req.request_id in self._pending:
                            holder[0] = ApprovalDecision(
                                status=ApprovalStatus.DENIED,
                                tool_name=tool_name,
                                args=args,
                                comment=f"Approval handler error: {exc}",
                            )
                            event.set()

            threading.Thread(target=_worker, daemon=True).start()

        # Wait on event or timeout
        signaled = event.wait(timeout=timeout)
        with self._lock:
            self._pending.pop(req.request_id, None)

        if signaled and holder[0] is not None:
            decision = holder[0]
        else:
            logger.warning(
                "Approval request for '%s' TIMED OUT after %.1fs -- defaulting to DENIED",
                tool_name, timeout
            )
            decision = ApprovalDecision(
                status=ApprovalStatus.TIMEOUT,
                tool_name=tool_name,
                args=args,
                comment=f"Timed out after {timeout}s without response -- defaulted to deny.",
            )

        self._history.append(decision)
        return decision
