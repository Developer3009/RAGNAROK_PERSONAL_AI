"""
ragnarok/skills/write_file.py
──────────────────────────────
Skill: write_file
Safely writes content to an allowed file path.
Risk level: high (modifies filesystem).

Improvements:
  - Uses PathSandbox for path resolution and traversal protection.
  - Uses ResourceGuard to enforce maximum write size.
  - Creates parent directories only when inside the sandbox.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragnarok.config import settings
from ragnarok.safety.sandbox import PathSandbox, ResourceGuard, SandboxViolation

logger = logging.getLogger(__name__)

MANIFEST = {
    "name": "write_file",
    "description": "Write or overwrite content to a file in an allowed directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative to allowed directory or absolute).",
            },
            "content": {
                "type": "string",
                "description": "The exact text content to write.",
            },
        },
        "required": ["path", "content"],
    },
    "risk_level": "high",
}


def run(args: dict) -> str:
    path = str(args.get("path") or "").strip().strip("'\"")
    content = str(args.get("content") or "")

    if not path:
        return "Error: Path cannot be empty."

    sandbox = PathSandbox(allowed_dirs=settings.allowed_folders)
    guard = ResourceGuard()  # default: 512 KB write limit

    # ── Enforce write size limit before touching the filesystem ───────────────
    try:
        guard.check_write(content)
    except SandboxViolation as exc:
        return f"Error: {exc}"

    # ── Validate parent directory is inside sandbox ────────────────────────────
    target = Path(path).resolve()
    try:
        sandbox.validate(target.parent)
    except SandboxViolation as exc:
        return f"Error: {exc}"

    if target.is_dir():
        return f"Error: '{target.name}' is an existing directory, cannot overwrite."

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info("write_file path=%s bytes=%d", target, len(content.encode("utf-8")))
        return f"Successfully wrote {len(content)} characters to '{target.name}'."
    except PermissionError:
        return f"Error: Permission denied writing to '{target}'."
    except OSError as exc:
        return f"Error writing file: {exc}"


def execute(path: str, content: str) -> str:
    return run({"path": path, "content": content})
