"""
tests/test_phase5_safety.py
────────────────────────────
Phase 5 tests: Sandboxing, Permissions, Encryption, Rate Limiting, Audit Logging.

All tests are unit tests — no live Ollama or network required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragnarok.safety.audit import AuditEvent, AuditLogger, AuditRecord
from ragnarok.safety.encryption import EncryptionError, SecretStore
from ragnarok.safety.permissions import (
    PermissionLevel,
    PermissionManager,
    ToolPolicy,
)
from ragnarok.safety.rate_limiter import RateLimiter
from ragnarok.safety.sandbox import (
    PathSandbox,
    ResourceGuard,
    SandboxViolation,
)
from ragnarok.agent.loop import AgentLoop
from ragnarok.skills.registry import SkillRegistry


# ─────────────────────────────────────────────────────────────────────────────
# PathSandbox & ResourceGuard Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPathSandbox:
    def test_allowed_path_resolves(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        safe_file = allowed / "doc.txt"
        safe_file.write_text("safe")

        sandbox = PathSandbox(allowed_dirs=[allowed])
        validated = sandbox.validate(safe_file)
        assert validated == safe_file.resolve()
        assert sandbox.is_allowed(safe_file) is True

    def test_disallowed_path_raises_violation(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        forbidden = tmp_path / "forbidden"
        forbidden.mkdir()
        bad_file = forbidden / "secret.txt"

        sandbox = PathSandbox(allowed_dirs=[allowed])
        assert sandbox.is_allowed(bad_file) is False
        with pytest.raises(SandboxViolation) as exc_info:
            sandbox.validate(bad_file)
        assert "outside allowed directories" in str(exc_info.value)

    def test_traversal_attack_blocked(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        target = allowed / ".." / "outside.txt"

        sandbox = PathSandbox(allowed_dirs=[allowed])
        assert sandbox.is_allowed(target) is False
        with pytest.raises(SandboxViolation):
            sandbox.validate(target)

    def test_hidden_file_blocked(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        hidden = allowed / ".hidden_file"

        sandbox = PathSandbox(allowed_dirs=[allowed])
        with pytest.raises(SandboxViolation) as exc_info:
            sandbox.validate(hidden)
        assert "hidden" in str(exc_info.value).lower()

    def test_must_exist_flag(self, tmp_path):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        non_existent = allowed / "ghost.txt"

        sandbox = PathSandbox(allowed_dirs=[allowed])
        with pytest.raises(SandboxViolation) as exc_info:
            sandbox.validate(non_existent, must_exist=True)
        assert "does not exist" in str(exc_info.value).lower()


class TestResourceGuard:
    def test_check_write_under_limit(self):
        guard = ResourceGuard(max_write_bytes=100)
        guard.check_write("Hello world")  # 11 bytes < 100

    def test_check_write_over_limit_raises(self):
        guard = ResourceGuard(max_write_bytes=10)
        with pytest.raises(SandboxViolation) as exc_info:
            guard.check_write("A" * 20)
        assert "exceeds limit" in str(exc_info.value)

    def test_truncate_read(self):
        guard = ResourceGuard(max_read_bytes=10)
        out = guard.truncate_read("A" * 30)
        assert len(out) > 10
        assert "[...truncated" in out

    def test_truncate_listing(self):
        guard = ResourceGuard(max_list_entries=3)
        files = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
        capped = guard.truncate_listing(files)
        assert len(capped) == 4
        assert "[...2 more" in capped[-1]


# ─────────────────────────────────────────────────────────────────────────────
# PermissionManager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissionManager:
    def test_low_risk_auto_approved(self):
        pm = PermissionManager()
        res = pm.check("get_time", risk_level="low")
        assert res.permitted is True
        assert res.requires_approval is False

    def test_high_risk_requires_approval(self):
        pm = PermissionManager()
        res = pm.check("write_file", risk_level="high")
        assert res.permitted is True
        assert res.requires_approval is True

    def test_blocked_risk_denied(self):
        pm = PermissionManager()
        res = pm.check("format_disk", risk_level="blocked")
        assert res.permitted is False
        assert "permanently blocked" in res.reason

    def test_explicit_deny_overrides_low_risk(self):
        pm = PermissionManager()
        pm.deny("get_time", reason="Maintenance mode")
        res = pm.check("get_time", risk_level="low")
        assert res.permitted is False
        assert "Maintenance mode" in res.reason

    def test_explicit_allow_with_max_calls_cap(self):
        pm = PermissionManager()
        pm.allow("fetch_web", reason="Testing cap", max_calls=2)

        res1 = pm.check("fetch_web", risk_level="medium")
        assert res1.permitted is True
        pm.record_call("fetch_web")

        res2 = pm.check("fetch_web", risk_level="medium")
        assert res2.permitted is True
        pm.record_call("fetch_web")

        res3 = pm.check("fetch_web", risk_level="medium")
        assert res3.permitted is False
        assert "call cap reached" in res3.reason

    def test_reset_clears_policies_and_counts(self):
        pm = PermissionManager()
        pm.deny("tool_a")
        pm.record_call("tool_a")
        pm.reset()
        assert pm.call_count("tool_a") == 0
        res = pm.check("tool_a", risk_level="low")
        assert res.permitted is True


# ─────────────────────────────────────────────────────────────────────────────
# RateLimiter Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allowed_under_limit(self):
        rl = RateLimiter()
        rl.configure("test_tool", max_calls=3, window_sec=10.0)
        assert rl.check_and_record("test_tool")[0] is True
        assert rl.check_and_record("test_tool")[0] is True
        assert rl.check_and_record("test_tool")[0] is True
        assert rl.current_count("test_tool") == 3

    def test_rate_limited_over_limit(self):
        rl = RateLimiter()
        rl.configure("test_tool", max_calls=2, window_sec=60.0)
        assert rl.check_and_record("test_tool")[0] is True
        assert rl.check_and_record("test_tool")[0] is True
        ok, retry_after = rl.check_and_record("test_tool")
        assert ok is False
        assert retry_after > 0.0

    def test_unconfigured_tool_is_unlimited(self):
        rl = RateLimiter()
        for _ in range(15):
            ok, _ = rl.check_and_record("untracked_tool")
            assert ok is True

    def test_reset_clears_timestamps(self):
        rl = RateLimiter()
        rl.configure("t", max_calls=1, window_sec=60.0)
        rl.check_and_record("t")
        assert rl.check_and_record("t")[0] is False
        rl.reset("t")
        assert rl.check_and_record("t")[0] is True


# ─────────────────────────────────────────────────────────────────────────────
# SecretStore / Encryption Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSecretStore:
    def test_encrypt_decrypt_roundtrip(self, tmp_path):
        key_file = tmp_path / "test.key"
        store = SecretStore(key_path=key_file)
        secret = "super_sensitive_api_key_12345"

        ciphertext = store.encrypt(secret)
        assert ciphertext != secret
        assert ciphertext.startswith("gAAAAA")
        assert store.is_encrypted(ciphertext) is True

        decrypted = store.decrypt(ciphertext)
        assert decrypted == secret

    def test_invalid_token_raises_encryption_error(self, tmp_path):
        store = SecretStore(key_path=tmp_path / "k.key")
        with pytest.raises(EncryptionError):
            store.decrypt("gAAAAABbadtoken12345")

    def test_from_passphrase_deterministic(self):
        s1 = SecretStore.from_passphrase("my_master_password")
        s2 = SecretStore.from_passphrase("my_master_password")
        token = s1.encrypt("confidential")
        assert s2.decrypt(token) == "confidential"

    def test_empty_string_handling(self, tmp_path):
        store = SecretStore(key_path=tmp_path / "k.key")
        assert store.encrypt("") == ""
        assert store.decrypt("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# AuditLogger Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLogger:
    def test_logs_events_to_jsonl(self, tmp_path):
        logger = AuditLogger(log_dir=tmp_path)
        logger.log_tool_call("read_file", {"path": "a.txt"}, risk_level="low")
        logger.log_tool_success("read_file", latency_ms=12.5, result_len=150)
        logger.log_denied("fetch_web", reason="Policy block")

        records = logger.tail(10)
        assert len(records) == 3
        assert records[0]["event"] == AuditEvent.TOOL_CALL
        assert records[0]["tool_name"] == "read_file"
        assert records[1]["event"] == AuditEvent.TOOL_SUCCESS
        assert records[1]["latency_ms"] == 12.5
        assert records[2]["event"] == AuditEvent.TOOL_DENIED
        assert records[2]["permitted"] is False

    def test_count_by_event(self, tmp_path):
        logger = AuditLogger(log_dir=tmp_path)
        logger.log_tool_call("t1", {})
        logger.log_tool_call("t2", {})
        logger.log_approval_denied("t2")

        counts = logger.count_by_event()
        assert counts.get(AuditEvent.TOOL_CALL) == 2
        assert counts.get(AuditEvent.APPROVAL_DENIED) == 1


# ─────────────────────────────────────────────────────────────────────────────
# AgentLoop Phase 5 Safety Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoopSafetyIntegration:
    def _make_registry(self):
        reg = SkillRegistry()
        reg.load_all()
        return reg

    def test_rate_limited_tool_call_blocked_in_loop(self, tmp_path):
        from ragnarok.llm import ChatToolResult, ToolCall

        reg = self._make_registry()
        rl = RateLimiter()
        rl.configure("get_time", max_calls=1, window_sec=60.0)
        audit = AuditLogger(log_dir=tmp_path)

        loop = AgentLoop(
            registry=reg,
            rate_limiter=rl,
            audit_logger=audit,
            enable_planning=False,
        )

        # 1st call succeeds, 2nd call is rate-limited
        tc1 = ChatToolResult(tool_calls=[ToolCall(name="get_time", arguments={})])
        tc2 = ChatToolResult(tool_calls=[ToolCall(name="get_time", arguments={})])
        final = ChatToolResult(final_content="Done.")

        with patch("ragnarok.agent.loop.llm.chat_with_tools", side_effect=[tc1, tc2, final]):
            res = loop.chat("What time is it twice?")

        counts = audit.count_by_event()
        assert counts.get(AuditEvent.TOOL_RATE_LIMITED, 0) == 1
        assert res.answer == "Done."

    def test_permission_denied_tool_blocked_in_loop(self, tmp_path):
        from ragnarok.llm import ChatToolResult, ToolCall

        reg = self._make_registry()
        pm = PermissionManager()
        pm.deny("get_time", reason="Tool forbidden by policy")
        audit = AuditLogger(log_dir=tmp_path)

        loop = AgentLoop(
            registry=reg,
            permission_manager=pm,
            audit_logger=audit,
            enable_planning=False,
        )

        tc = ChatToolResult(tool_calls=[ToolCall(name="get_time", arguments={})])
        final = ChatToolResult(final_content="Denied response.")

        with patch("ragnarok.agent.loop.llm.chat_with_tools", side_effect=[tc, final]):
            res = loop.chat("Get time")

        counts = audit.count_by_event()
        assert counts.get(AuditEvent.PERMISSION_DENIED, 0) == 1
        assert res.answer == "Denied response."
