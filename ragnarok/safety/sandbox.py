"""
ragnarok/safety/sandbox.py
───────────────────────────
Hardened path sandbox and resource guard for Ragnarok.

Provides:
  - PathSandbox: resolves and validates paths against an allowlist,
    rejects traversal attempts, symlink escapes, and hidden paths.
  - ResourceGuard: enforces per-operation character/byte size limits.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Max bytes allowed for a single file write through a sandboxed path
_DEFAULT_MAX_WRITE_BYTES = 512 * 1024   # 512 KB
# Max bytes for a single file read result
_DEFAULT_MAX_READ_BYTES = 64 * 1024     # 64 KB
# Max number of directory entries returned per list
_DEFAULT_MAX_LIST_ENTRIES = 200


class SandboxViolation(Exception):
    """Raised when a path or resource check fails."""


class PathSandbox:
    """
    Resolves and validates filesystem paths against a configured allowlist.

    All path checks are done after resolving to absolute realpath to prevent
    symlink escape, `..` traversal, and drive-relative tricks on Windows.

    Args:
        allowed_dirs: Iterable of allowed root directories (Path objects or strings).
    """

    def __init__(self, allowed_dirs: Sequence[Path | str]) -> None:
        self._allowed: list[Path] = [Path(d).resolve() for d in allowed_dirs]
        if not self._allowed:
            raise ValueError("PathSandbox requires at least one allowed directory.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def validate(self, raw_path: str | Path, *, must_exist: bool = False) -> Path:
        """
        Resolve and validate `raw_path` against the allowlist.

        Args:
            raw_path:   The path to validate (relative paths resolved from CWD).
            must_exist: If True, raise SandboxViolation if the path doesn't exist.

        Returns:
            The resolved, safe Path.

        Raises:
            SandboxViolation: If the path is outside the allowlist or violates other rules.
        """
        try:
            resolved = Path(raw_path).resolve()
        except (ValueError, OSError) as exc:
            raise SandboxViolation(f"Cannot resolve path '{raw_path}': {exc}") from exc

        # Hidden file/dir check (starts with dot on POSIX or Windows)
        parts = resolved.parts
        if any(part.startswith(".") and part not in (".", "..") for part in parts[1:]):
            raise SandboxViolation(
                f"Access to hidden path '{resolved}' is not permitted."
            )

        # Allowlist check
        if not self._in_allowlist(resolved):
            allowed_str = ", ".join(str(a) for a in self._allowed)
            raise SandboxViolation(
                f"Path '{resolved}' is outside allowed directories: [{allowed_str}]"
            )

        # Symlink escape check
        if resolved.is_symlink():
            link_target = Path(os.readlink(resolved)).resolve()
            if not self._in_allowlist(link_target):
                raise SandboxViolation(
                    f"Symlink '{resolved}' points outside allowed directories."
                )

        if must_exist and not resolved.exists():
            raise SandboxViolation(f"Path does not exist: '{resolved}'")

        return resolved

    def is_allowed(self, raw_path: str | Path) -> bool:
        """Return True if the path passes sandbox validation, False otherwise."""
        try:
            self.validate(raw_path)
            return True
        except SandboxViolation:
            return False

    @property
    def allowed_dirs(self) -> list[Path]:
        return list(self._allowed)

    def allowed_str(self) -> str:
        return ", ".join(str(a) for a in self._allowed)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _in_allowlist(self, resolved: Path) -> bool:
        return any(
            str(resolved).startswith(str(allowed))
            for allowed in self._allowed
        )


class ResourceGuard:
    """
    Enforces size limits on tool I/O to prevent runaway reads/writes.

    Args:
        max_write_bytes: Maximum bytes allowed for a single file write.
        max_read_bytes:  Maximum bytes allowed in a single file read result.
        max_list_entries: Maximum directory entries per listing.
    """

    def __init__(
        self,
        max_write_bytes: int = _DEFAULT_MAX_WRITE_BYTES,
        max_read_bytes: int = _DEFAULT_MAX_READ_BYTES,
        max_list_entries: int = _DEFAULT_MAX_LIST_ENTRIES,
    ) -> None:
        self.max_write_bytes = max_write_bytes
        self.max_read_bytes = max_read_bytes
        self.max_list_entries = max_list_entries

    def check_write(self, content: str | bytes) -> None:
        """Raise SandboxViolation if content is too large to write."""
        size = len(content.encode("utf-8") if isinstance(content, str) else content)
        if size > self.max_write_bytes:
            raise SandboxViolation(
                f"Write payload {size:,} bytes exceeds limit of {self.max_write_bytes:,} bytes."
            )

    def truncate_read(self, content: str) -> str:
        """Truncate read content to max_read_bytes, appending a notice if trimmed."""
        encoded = content.encode("utf-8")
        if len(encoded) <= self.max_read_bytes:
            return content
        truncated = encoded[: self.max_read_bytes].decode("utf-8", errors="replace")
        return truncated + f"\n\n[...truncated: output exceeded {self.max_read_bytes:,} bytes]"

    def truncate_listing(self, entries: list[str]) -> list[str]:
        """Cap directory listing at max_list_entries."""
        if len(entries) <= self.max_list_entries:
            return entries
        capped = entries[: self.max_list_entries]
        capped.append(
            f"[...{len(entries) - self.max_list_entries} more entries not shown]"
        )
        return capped
