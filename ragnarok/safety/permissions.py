"""
ragnarok/safety/permissions.py
───────────────────────────────
Role-based tool permission system for Ragnarok.

Defines:
  - PermissionLevel: risk tiers (low / medium / high / blocked)
  - ToolPolicy: explicit allow/deny rules per skill name
  - PermissionManager: evaluates whether a skill is allowed to run in context

Default policy:
  - All "low" risk tools allowed by default
  - "medium" tools require explicit allow or user approval
  - "high" tools require explicit allow AND approval callback
  - "blocked" tools are always rejected (no approval possible)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PermissionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"

    @classmethod
    def from_risk(cls, risk_str: str) -> "PermissionLevel":
        mapping = {
            "low": cls.LOW,
            "medium": cls.MEDIUM,
            "high": cls.HIGH,
            "blocked": cls.BLOCKED,
        }
        return mapping.get(risk_str.lower(), cls.MEDIUM)


@dataclass
class ToolPolicy:
    """
    Per-tool override policy.

    Attributes:
        allowed:   Explicit allow (True) or deny (False). None = use default risk-level policy.
        reason:    Human-readable reason for this policy entry.
        max_calls: Optional cap on how many times this tool may be called per session.
                   None means unlimited.
    """
    allowed: bool | None = None
    reason: str = ""
    max_calls: int | None = None


class PermissionManager:
    """
    Evaluates whether a tool call is permitted based on risk level and explicit policies.

    Usage:
        pm = PermissionManager()
        pm.deny("fetch_web", reason="No internet access in this session")
        pm.allow("write_file", reason="User explicitly approved file writes")

        result = pm.check("read_file", risk_level="low")
        if not result.permitted:
            print(result.reason)
    """

    def __init__(self) -> None:
        self._policies: dict[str, ToolPolicy] = {}
        self._session_call_counts: dict[str, int] = {}

    # ── Policy management ──────────────────────────────────────────────────────

    def allow(self, tool_name: str, reason: str = "", max_calls: int | None = None) -> None:
        """Explicitly allow a tool (overrides risk-level default)."""
        self._policies[tool_name] = ToolPolicy(allowed=True, reason=reason, max_calls=max_calls)
        logger.debug("PermissionManager: allowed '%s' reason=%r", tool_name, reason)

    def deny(self, tool_name: str, reason: str = "") -> None:
        """Explicitly deny a tool regardless of risk level."""
        self._policies[tool_name] = ToolPolicy(allowed=False, reason=reason)
        logger.info("PermissionManager: denied '%s' reason=%r", tool_name, reason)

    def reset(self) -> None:
        """Clear all per-session policies and call counts."""
        self._policies.clear()
        self._session_call_counts.clear()

    # ── Evaluation ─────────────────────────────────────────────────────────────

    @dataclass
    class CheckResult:
        permitted: bool
        reason: str
        requires_approval: bool = False

    def check(
        self,
        tool_name: str,
        risk_level: str = "low",
        args: dict[str, Any] | None = None,
    ) -> "PermissionManager.CheckResult":
        """
        Determine whether `tool_name` may be executed.

        Returns:
            CheckResult with permitted=True/False, reason, and requires_approval flag.
        """
        level = PermissionLevel.from_risk(risk_level)

        # 1. Always-blocked tools
        if level == PermissionLevel.BLOCKED:
            return self.CheckResult(
                permitted=False,
                reason=f"Tool '{tool_name}' is permanently blocked (risk=blocked).",
            )

        # 2. Explicit policy override
        policy = self._policies.get(tool_name)
        if policy is not None:
            if policy.allowed is False:
                return self.CheckResult(
                    permitted=False,
                    reason=policy.reason or f"Tool '{tool_name}' has been explicitly denied.",
                )
            if policy.allowed is True:
                # Check call cap
                if policy.max_calls is not None:
                    count = self._session_call_counts.get(tool_name, 0)
                    if count >= policy.max_calls:
                        return self.CheckResult(
                            permitted=False,
                            reason=(
                                f"Tool '{tool_name}' call cap reached "
                                f"({count}/{policy.max_calls} uses this session)."
                            ),
                        )
                # High risk always needs approval even when explicitly allowed
                return self.CheckResult(
                    permitted=True,
                    reason=policy.reason or "Explicitly allowed.",
                    requires_approval=(level == PermissionLevel.HIGH),
                )

        # 3. Default risk-level policy
        if level == PermissionLevel.LOW:
            return self.CheckResult(permitted=True, reason="Low-risk tool; auto-approved.")
        if level == PermissionLevel.MEDIUM:
            return self.CheckResult(
                permitted=True,
                reason="Medium-risk tool; allowed by default, monitor usage.",
                requires_approval=False,
            )
        if level == PermissionLevel.HIGH:
            return self.CheckResult(
                permitted=True,
                reason="High-risk tool; requires explicit approval.",
                requires_approval=True,
            )
        return self.CheckResult(permitted=False, reason=f"Unknown risk level: {risk_level}")

    def record_call(self, tool_name: str) -> None:
        """Increment the call counter for a tool (call after successful execution)."""
        self._session_call_counts[tool_name] = (
            self._session_call_counts.get(tool_name, 0) + 1
        )

    def call_count(self, tool_name: str) -> int:
        return self._session_call_counts.get(tool_name, 0)

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of current policies and call counts."""
        return {
            "policies": {
                name: {
                    "allowed": p.allowed,
                    "reason": p.reason,
                    "max_calls": p.max_calls,
                }
                for name, p in self._policies.items()
            },
            "call_counts": dict(self._session_call_counts),
        }
