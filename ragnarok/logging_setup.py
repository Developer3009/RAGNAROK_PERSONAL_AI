"""
ragnarok/logging_setup.py
─────────────────────────
Structured JSON logging for Ragnarok.
Every prompt, tool choice, arguments, result, and duration is recorded.
Call setup_logging() once at startup.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragnarok.config import settings


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        # Pass-through any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key.startswith("_rk_"):
                obj[key[4:]] = val
        return json.dumps(obj, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger:
      - stderr: human-readable (level INFO+)
      - logs/ragnarok.jsonl: structured JSON (level DEBUG+)

    Call once at the top of every entry-point script.
    """
    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Console handler (human-readable) ──────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                          datefmt="%H:%M:%S")
    )
    root.addHandler(console)

    # ── File handler (structured JSON) ────────────────────────────────────────
    log_file = log_dir / "ragnarok.jsonl"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)  # suppress httpx noise


def log_tool_event(
    logger: logging.Logger,
    event: str,
    tool_name: str,
    arguments: dict | None = None,
    result: Any = None,
    latency_ms: float | None = None,
    error: str | None = None,
) -> None:
    """
    Emit a structured log record for a tool-related event.
    This is the explainability record used in Phase 7.

    Args:
        event:      "call" | "result" | "error" | "schema_fail"
        tool_name:  Name of the tool being called.
        arguments:  Arguments passed to the tool.
        result:     Return value from the tool (serialisable).
        latency_ms: How long the tool took.
        error:      Error message if it failed.
    """
    extra: dict[str, Any] = {
        "_rk_event": event,
        "_rk_tool": tool_name,
    }
    if arguments is not None:
        extra["_rk_args"] = arguments
    if result is not None:
        extra["_rk_result"] = str(result)[:500]
    if latency_ms is not None:
        extra["_rk_latency_ms"] = round(latency_ms, 1)
    if error is not None:
        extra["_rk_error"] = error

    msg = f"tool.{event} name={tool_name}"
    if error:
        logger.error(msg, extra=extra)
    else:
        logger.info(msg, extra=extra)
