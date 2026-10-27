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

        # Protected configuration / system files check (agent cannot view or modify its own permissions)
        protected_names = {"config.toml", "settings.toml", ".ragnarok_key", "security_audit.jsonl"}
        if resolved.name.lower() in protected_names or resolved.suffix.lower() in (".enc", ".key"):
            raise SandboxViolation(
                f"Access to protected system file '{resolved.name}' is strictly forbidden for agent tools."
            )

        # Windows Alternate Data Stream (ADS) and reserved device name defense
        if ":" in resolved.name:
            raise SandboxViolation(
                f"Invalid characters or NTFS Alternate Data Stream detected in '{resolved.name}'."
            )
        _RESERVED_DEVICE_NAMES = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if resolved.stem.upper() in _RESERVED_DEVICE_NAMES or resolved.name.upper() in _RESERVED_DEVICE_NAMES:
            raise SandboxViolation(
                f"Access to reserved Windows system device name '{resolved.name}' is strictly blocked."
            )

        # Allowlist check
        if not self._in_allowlist(resolved):
            allowed_str = ", ".join(str(a) for a in self._allowed)
            raise SandboxViolation(
                f"Path '{resolved}' is outside allowed directories (not within any allowed directory): [{allowed_str}]"
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
        """Check genuine path containment using relative_to, avoiding prefix collision attacks."""
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                continue
        return False


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


class SubprocessSandbox:
    """
    Executes commands in an isolated, restricted subprocess environment.

    Safety controls:
      - Execution timeout to prevent infinite loops / runaway processes.
      - Enforces working directory within PathSandbox allowed folders.
      - Environment variable sanitization (strips credentials, API keys, tokens).
      - Network isolation: sets offline/isolation flags and can reject network calls.
      - Limits stdout/stderr output size via ResourceGuard.
    """

    def __init__(
        self,
        path_sandbox: PathSandbox | None = None,
        resource_guard: ResourceGuard | None = None,
        default_timeout_sec: float = 5.0,
    ) -> None:
        self.sandbox = path_sandbox
        self.guard = resource_guard or ResourceGuard()
        self.default_timeout_sec = default_timeout_sec

    def _sanitize_env(self) -> dict[str, str]:
        """Produce an isolated minimal environment devoid of secrets."""
        safe_keys = {
            "SYSTEMROOT",
            "PATH",
            "TEMP",
            "TMP",
            "WINDIR",
            "COMSPEC",
            "PYTHONPATH",
            "PYTHONIOENCODING",
        }
        clean_env: dict[str, str] = {}
        for k, v in os.environ.items():
            if k.upper() in safe_keys:
                clean_env[k] = v
        # Ensure offline marker
        clean_env["RAGNAROK_SANDBOX"] = "1"
        clean_env["NO_PROXY"] = "*"
        return clean_env

    def run_command(
        self,
        command: Sequence[str],
        cwd: str | Path | None = None,
        timeout_sec: float | None = None,
        allow_network: bool = False,
    ) -> tuple[int, str, str]:
        """
        Execute command with strict safety limits.

        Returns:
            tuple of (returncode, stdout, stderr)

        Raises:
            SandboxViolation: on path escape, timeout, or disallowed network attempt.
        """
        import subprocess

        timeout = timeout_sec if timeout_sec is not None else self.default_timeout_sec

        # Validate working directory
        working_dir: Path | None = None
        if cwd:
            if self.sandbox:
                working_dir = self.sandbox.validate(cwd, must_exist=True)
            else:
                working_dir = Path(cwd).resolve()

        # Check for disallowed commands
        disallowed = {"rmdir /s", "del /f", "format", "mkfs", "dd"}
        cmd_str = " ".join(command).lower()
        if any(d in cmd_str for d in disallowed):
            raise SandboxViolation(f"Command '{cmd_str}' contains dangerous operation.")

        if not allow_network:
            # Check for direct network tooling
            net_tools = {"curl", "wget", "nc", "ncat", "telnet", "ssh"}
            if any(tool in command[0].lower() for tool in net_tools):
                raise SandboxViolation(
                    f"Network access disabled in sandbox: cannot execute '{command[0]}'."
                )

        env = self._sanitize_env()

        try:
            proc = subprocess.run(
                command,
                cwd=str(working_dir) if working_dir else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = self.guard.truncate_read(proc.stdout)
            stderr = self.guard.truncate_read(proc.stderr)
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired as exc:
            raise SandboxViolation(
                f"Subprocess exceeded safety time limit of {timeout}s: infinite loop or stall aborted."
            ) from exc
        except Exception as exc:
            raise SandboxViolation(f"Subprocess execution failed: {exc}") from exc

