"""
ragnarok/agent/self_correction.py
──────────────────────────────────
Self-correction, error diagnosis, and doom-loop detection for the agent loop.

Prevents the agent from getting stuck repeating identical failing tool calls,
analyzes runtime errors, and injects actionable corrective guidance into history.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from ragnarok.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CorrectionAdvice:
    error_type: str
    message: str
    suggested_alternative: str
    prevent_retry: bool = False

    def format_prompt(self) -> str:
        advice_lines = [
            f"[SELF-CORRECTION ADVICE - {self.error_type}]",
            f"Issue: {self.message}",
            f"Actionable Guidance: {self.suggested_alternative}",
        ]
        if self.prevent_retry:
            advice_lines.append(
                "CRITICAL: Do NOT attempt the exact same tool call with the same arguments. Choose a different tool or modify arguments."
            )
        return "\n".join(advice_lines)


class DoomLoopDetector:
    """
    Detects when an agent repeatedly executes the exact same failing or ineffective tool call.
    """

    def __init__(self, max_consecutive_failures: int = 2) -> None:
        self.max_consecutive_failures = max_consecutive_failures
        self._history: list[dict[str, Any]] = []

    def record_call(self, tool_name: str, args: dict[str, Any], result: str) -> None:
        args_hash = self._hash_args(args)
        is_error = self._is_error(result)
        self._history.append({
            "tool": tool_name,
            "args_hash": args_hash,
            "args": args,
            "is_error": is_error,
            "result_summary": result[:120],
        })

    def check_loop(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
        """
        Check if the proposed tool call repeats a recent consecutive failure.
        Returns (is_loop, warning_message).
        """
        if len(self._history) < self.max_consecutive_failures:
            return False, ""

        args_hash = self._hash_args(args)
        consecutive_matches = 0

        for entry in reversed(self._history):
            if entry["tool"] == tool_name and entry["args_hash"] == args_hash and entry["is_error"]:
                consecutive_matches += 1
            else:
                break

        if consecutive_matches >= self.max_consecutive_failures:
            warning = (
                f"DOOM LOOP DETECTED: You have already attempted '{tool_name}' with these exact arguments "
                f"{consecutive_matches} times and received errors. You are prohibited from calling '{tool_name}' "
                f"with these arguments again. You MUST choose an alternative approach or report the blocker to the user."
            )
            logger.warning("Doom loop flagged for tool=%s args=%s", tool_name, args)
            return True, warning

        return False, ""

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def _hash_args(args: dict[str, Any]) -> str:
        try:
            serialized = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            serialized = str(sorted(args.items()))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_error(result: str) -> bool:
        lowered = result.strip().lower()
        return (
            lowered.startswith("error")
            or "traceback" in lowered
            or "exception" in lowered
            or "failed" in lowered
            or "not found" in lowered
        )


class ErrorDiagnoser:
    """
    Analyzes error messages from tool executions or schema validations
    and produces actionable recovery advice.
    """

    @classmethod
    def diagnose(cls, tool_name: str, args: dict[str, Any], error_str: str) -> CorrectionAdvice:
        err = error_str.lower()

        # 1. File not found
        if "not found" in err or "no such file" in err:
            path = args.get("path") or args.get("filepath") or args.get("filename", "")
            return CorrectionAdvice(
                error_type="FILE_NOT_FOUND",
                message=f"Target path '{path}' does not exist on disk.",
                suggested_alternative=(
                    "Use 'list_dir' to inspect available files in the directory or 'search_files' "
                    "to locate the correct filename before attempting to read or modify it."
                ),
            )

        # 2. Allowlist / Security violation
        if "allowlist" in err or "permission" in err or "not allowed" in err or "traversal" in err:
            allowed = ", ".join(str(p) for p in settings.allowed_folders)
            return CorrectionAdvice(
                error_type="PERMISSION_DENIED",
                message="File access outside configured allowed directories was blocked.",
                suggested_alternative=(
                    f"Ragnarok restricts file operations to configured allowed folders: [{allowed}]. "
                    f"Ensure target paths reside inside these folders (e.g., 'data/knowledge/')."
                ),
                prevent_retry=True,
            )

        # 3. Schema validation / Missing arguments
        if "missing required" in err or "validation error" in err or "schema" in err:
            return CorrectionAdvice(
                error_type="SCHEMA_VALIDATION_ERROR",
                message=f"Invalid arguments provided for tool '{tool_name}'.",
                suggested_alternative=(
                    f"Review the required schema for '{tool_name}'. Verify argument names, types, "
                    f"and mandatory fields before calling."
                ),
            )

        # 4. Network / HTTP fetch errors
        if "http" in err or "connection" in err or "timeout" in err:
            url = args.get("url", "")
            return CorrectionAdvice(
                error_type="NETWORK_TIMEOUT",
                message=f"Failed to fetch network resource '{url}'.",
                suggested_alternative=(
                    "Verify the URL syntax. If offline or the target server is unreachable, "
                    "rely on existing local knowledge/files or ask the user."
                ),
            )

        # Generic / Unknown tool error
        return CorrectionAdvice(
            error_type="TOOL_EXECUTION_ERROR",
            message=f"Tool '{tool_name}' failed: {error_str[:120]}",
            suggested_alternative=(
                "Analyze the error message. Consider trying an alternative skill or adjusting "
                "the input arguments to resolve the issue."
            ),
        )
