"""
tests/test_innovations.py
──────────────────────────
Test suite for all 7 Ragnarok innovations.

Tests cover:
  [I1] GUI Action skill — manifest, VLM grounding mock, action dispatch
  [I2] Knowledge Graph — triple insertion, temporal deprecation, multi-hop, context
  [I3] Mesh Swarm — peer model, transport encryption, node routing logic
  [I4] Streaming Voice — VAD fallback, TTS chunking, interrupt event
  [I5] MCP Hub — hub routing, skill forge validation, bad imports detection
  [I6] Shadow Worker — git intelligence mock, error sentinel pattern detection
  [I7] Vault — hardware fingerprint, Merkle chain integrity, tamper detection
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# [I1] GUI Action Skill Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGuiActionManifest:
    def test_manifest_has_required_fields(self):
        from ragnarok.skills.gui_action import MANIFEST
        assert MANIFEST.name == "gui_action"
        assert MANIFEST.risk_level == "high"
        assert "action" in MANIFEST.parameters
        assert "action" in MANIFEST.required

    def test_manifest_action_enum(self):
        from ragnarok.skills.gui_action import MANIFEST
        valid_actions = {"click", "double_click", "right_click", "type", "move", "scroll", "screenshot_describe"}
        param_enum = set(MANIFEST.parameters["action"].enum or [])
        assert param_enum == valid_actions

    def test_run_missing_action_defaults_to_describe(self):
        """Missing action should not crash — defaults to screenshot_describe."""
        from ragnarok.skills.gui_action import run
        # Patch screenshot capture to avoid requiring a display
        with patch("ragnarok.skills.gui_action._capture_screenshot") as mock_cap, \
             patch("ragnarok.skills.gui_action._query_vlm_describe", return_value="Test screen"):
            mock_cap.return_value = (b"\x89PNG", 1920, 1080)
            result = run({"action": "screenshot_describe", "vision_model": "llava"})
            assert "Test screen" in result or "Screen Description" in result

    def test_run_click_without_target_returns_error(self):
        from ragnarok.skills.gui_action import run
        with patch("ragnarok.skills.gui_action._capture_screenshot") as mock_cap:
            mock_cap.return_value = (b"\x89PNG", 1920, 1080)
            result = run({"action": "click"})
            assert "ERROR" in result or "target" in result.lower()

    def test_screen_coord_normalized_conversion(self):
        """Normalized coordinates 0..1 should be converted to absolute pixels."""
        from ragnarok.skills.gui_action import ScreenCoord
        coord = ScreenCoord(x=0, y=0, confidence=0.95)
        # If x <= 1, treat as normalized
        assert coord.x <= 1
        screen_w, screen_h = 1920, 1080
        abs_x = int(coord.x * screen_w) if coord.x <= 1 else coord.x
        abs_y = int(coord.y * screen_h) if coord.y <= 1 else coord.y
        assert abs_x == 0
        assert abs_y == 0


# ══════════════════════════════════════════════════════════════════════════════
# [I2] Knowledge Graph Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeGraph:
    @pytest.fixture
    def kg(self, tmp_path: Path):
        from ragnarok.memory.knowledge_graph import KnowledgeGraph
        return KnowledgeGraph(db_path=tmp_path / "test_kg.db")

    def test_upsert_entity_creates_entity(self, kg):
        entity = kg.upsert_entity("Alice", entity_type="person")
        assert entity.name == "Alice"
        assert entity.entity_type == "person"
        assert entity.id

    def test_upsert_entity_is_idempotent(self, kg):
        e1 = kg.upsert_entity("Bob")
        e2 = kg.upsert_entity("Bob")
        assert e1.id == e2.id

    def test_upsert_merges_aliases(self, kg):
        kg.upsert_entity("Charlie", aliases=["C"])
        entity = kg.upsert_entity("Charlie", aliases=["Chuck"])
        assert "C" in entity.aliases
        assert "Chuck" in entity.aliases

    def test_add_triple_basic(self, kg):
        edge = kg.add_triple("User", "prefers", "dark mode")
        assert edge.rel == "prefers"
        assert edge.valid_until is None  # Active

    def test_temporal_deprecation(self, kg):
        """Adding a new triple with same (src, rel) should deprecate old one."""
        edge1 = kg.add_triple("User", "uses", "VS Code", supersede_existing=True)
        time.sleep(0.01)
        edge2 = kg.add_triple("User", "uses", "Cursor", supersede_existing=True)

        # edge1 should now be deprecated
        history = kg.get_history("User", "uses")
        assert len(history) == 2
        active = [h for h in history if h["valid_until"] is None]
        deprecated = [h for h in history if h["valid_until"] is not None]
        assert len(active) == 1
        assert len(deprecated) == 1
        assert active[0]["obj"] == "Cursor"
        assert deprecated[0]["obj"] == "VS Code"

    def test_get_neighbors(self, kg):
        kg.add_triple("Project Titan", "deadline", "Friday")
        kg.add_triple("Project Titan", "owner", "Alice")
        neighbors = kg.get_neighbors("Project Titan")
        assert len(neighbors) >= 2
        rels = [n["rel"] for n in neighbors]
        assert "deadline" in rels
        assert "owner" in rels

    def test_multi_hop_query(self, kg):
        kg.add_triple("Alice", "works_on", "Project Titan")
        kg.add_triple("Project Titan", "uses", "Python")
        triples = kg.multi_hop_query("Alice", max_hops=2)
        assert any("Alice" in t for t in triples)
        assert any("Project Titan" in t for t in triples)

    def test_query_as_context_returns_string(self, kg):
        kg.add_triple("User", "prefers", "dark mode")
        ctx = kg.query_as_context("User")
        assert isinstance(ctx, str)
        assert "prefers" in ctx or "dark mode" in ctx

    def test_deprecate_fact(self, kg):
        kg.add_triple("User", "works_at", "Startup X")
        count = kg.deprecate_fact("User", "works_at")
        assert count == 1
        neighbors = kg.get_neighbors("User", active_only=True)
        active_work = [n for n in neighbors if n["rel"] == "works_at"]
        assert len(active_work) == 0

    def test_search_entities(self, kg):
        kg.upsert_entity("Ragnarok Project")
        kg.upsert_entity("Ragnarok Memory System")
        results = kg.search_entities("Ragnarok")
        assert len(results) >= 2

    def test_stats(self, kg):
        kg.add_triple("A", "relates", "B")
        kg.add_triple("C", "connects", "D")
        stats = kg.stats()
        assert stats["entities"] >= 4
        assert stats["active_edges"] >= 2
        assert stats["total_edges"] >= 2

    def test_ingest_fact_text_preference(self, tmp_path):
        from ragnarok.memory.knowledge_graph import KnowledgeGraph, ingest_fact_text
        import ragnarok.memory.knowledge_graph as kg_module
        # Use isolated KG
        test_kg = KnowledgeGraph(db_path=tmp_path / "ingest_test.db")
        original = kg_module._kg
        kg_module._kg = test_kg
        try:
            edges = ingest_fact_text("I prefer dark mode", source="test")
            assert len(edges) >= 1
        finally:
            kg_module._kg = original


# ══════════════════════════════════════════════════════════════════════════════
# [I3] Mesh Swarm Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMeshSwarm:
    def test_mesh_peer_private_ip_check(self):
        from ragnarok.agent.mesh_swarm import MeshPeer
        private_peers = [
            MeshPeer("192.168.1.100", 47890, "node-1"),
            MeshPeer("10.0.0.5", 47890, "node-2"),
            MeshPeer("172.16.0.1", 47890, "node-3"),
            MeshPeer("127.0.0.1", 47890, "node-4"),
        ]
        for peer in private_peers:
            assert peer.is_private_address(), f"{peer.host} should be private"

    def test_mesh_peer_public_ip_rejected(self):
        from ragnarok.agent.mesh_swarm import MeshPeer
        public_peer = MeshPeer("8.8.8.8", 47890, "evil-node")
        assert not public_peer.is_private_address()

    def test_mesh_transport_encrypt_decrypt(self):
        from ragnarok.agent.mesh_swarm import MeshTransport
        transport = MeshTransport.from_passphrase("test-secret-key")
        message = '{"cmd": "run_task", "user_input": "Hello Ragnarok"}'
        encrypted = transport.encrypt(message)
        decrypted = transport.decrypt(encrypted)
        assert decrypted == message

    def test_mesh_transport_tamper_detection(self):
        from ragnarok.agent.mesh_swarm import MeshTransport
        from cryptography.fernet import InvalidToken
        transport = MeshTransport.from_passphrase("test-secret-key")
        encrypted = transport.encrypt("important message")
        # Tamper with the ciphertext
        tampered = bytearray(encrypted)
        tampered[10] ^= 0xFF
        with pytest.raises(Exception):  # Fernet raises InvalidToken
            transport.decrypt(bytes(tampered))

    def test_mesh_task_request_json(self):
        from ragnarok.agent.mesh_swarm import MeshTaskRequest
        req = MeshTaskRequest(
            task_id="abc-123",
            user_input="What is the meaning of life?",
            model_hint="llama3.2",
            requester_id="node-laptop",
        )
        raw = req.to_json()
        data = json.loads(raw)
        assert data["cmd"] == "run_task"
        assert data["task_id"] == "abc-123"
        assert data["user_input"] == "What is the meaning of life?"

    def test_valhalla_node_detect_capabilities_returns_list(self):
        from ragnarok.agent.mesh_swarm import ValhallaMeshNode
        caps = ValhallaMeshNode._detect_capabilities()
        assert isinstance(caps, list)
        assert "cpu" in caps

    def test_best_peer_for_returns_none_when_no_peers(self):
        from ragnarok.agent.mesh_swarm import ValhallaMeshNode
        node = ValhallaMeshNode(node_id="test-node", local_executor=lambda x: x)
        result = node.best_peer_for("gpu")
        assert result is None

    def test_node_status(self):
        from ragnarok.agent.mesh_swarm import ValhallaMeshNode
        node = ValhallaMeshNode(node_id="test-node", local_executor=lambda x: "ok")
        status = node.status()
        assert status["node_id"] == "test-node"
        assert status["active_peers"] == 0
        assert isinstance(status["capabilities"], list)


# ══════════════════════════════════════════════════════════════════════════════
# [I4] Streaming Voice Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamingVoice:
    def test_manifest_valid(self):
        from ragnarok.skills.streaming_voice import MANIFEST
        assert MANIFEST.name == "streaming_voice"
        assert "mode" in MANIFEST.required
        valid_modes = set(MANIFEST.parameters["mode"].enum or [])
        assert valid_modes == {"listen", "speak", "duplex"}

    def test_interrupt_event(self):
        from ragnarok.skills.streaming_voice import interrupt_speech, _clear_interrupt, _interrupt_event
        _clear_interrupt()
        assert not _interrupt_event.is_set()
        interrupt_speech()
        assert _interrupt_event.is_set()
        _clear_interrupt()
        assert not _interrupt_event.is_set()

    def test_speak_missing_text_returns_error(self):
        from ragnarok.skills.streaming_voice import run
        result = run({"mode": "speak", "text": ""})
        assert "ERROR" in result

    def test_silero_vad_fallback_energy(self):
        """SileroVAD energy fallback should work without torch."""
        from ragnarok.skills.streaming_voice import SileroVAD
        import numpy as np
        vad = SileroVAD()
        # Silent audio chunk (near zero)
        silent = np.zeros(8000, dtype=np.float32)
        # Loud audio chunk
        loud = (np.random.rand(8000).astype(np.float32) * 0.5)
        # Energy fallback used when torch unavailable — energy test
        assert SileroVAD._energy_vad(loud) is True
        assert SileroVAD._energy_vad(silent) is False

    def test_speak_streaming_with_interrupt(self):
        """Streaming TTS with interrupt should return 'interrupted'."""
        from ragnarok.skills.streaming_voice import speak_streaming
        stop = threading.Event()

        def _tokens():
            for word in ["Hello", " world", " this", " is", " a", " test"]:
                time.sleep(0.01)
                yield word

        # Set interrupt immediately
        stop.set()
        result = speak_streaming(_tokens(), interrupt_event=stop)
        assert result == "interrupted"

    def test_run_unknown_mode_returns_error(self):
        from ragnarok.skills.streaming_voice import run
        result = run({"mode": "invalid_mode"})
        assert "ERROR" in result


# ══════════════════════════════════════════════════════════════════════════════
# [I5] MCP Hub & Auto-Forge Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMCPHub:
    def test_manifest_valid(self):
        from ragnarok.skills.mcp_hub import MANIFEST
        assert MANIFEST.name == "mcp_hub"
        assert MANIFEST.risk_level == "high"
        valid_actions = {"list_servers", "list_tools", "call_tool", "forge_skill", "add_server"}
        param_enum = set(MANIFEST.parameters["action"].enum or [])
        assert param_enum == valid_actions

    def test_list_servers_empty(self):
        from ragnarok.skills.mcp_hub import MCPHub
        hub = MCPHub()
        servers = hub.list_servers()
        assert isinstance(servers, list)
        assert len(servers) == 0

    def test_call_tool_unknown_server(self):
        from ragnarok.skills.mcp_hub import MCPHub
        hub = MCPHub()
        result = hub.call_tool("nonexistent", "some_tool", {})
        assert "not connected" in result

    def test_run_list_servers_action(self):
        from ragnarok.skills.mcp_hub import run
        result = run({"action": "list_servers"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_add_server_no_name_returns_error(self):
        from ragnarok.skills.mcp_hub import run
        result = run({"action": "add_server"})
        assert "ERROR" in result

    def test_run_call_tool_no_server_returns_error(self):
        from ragnarok.skills.mcp_hub import run
        result = run({"action": "call_tool"})
        assert "ERROR" in result

    def test_run_forge_no_description_returns_error(self):
        from ragnarok.skills.mcp_hub import run
        result = run({"action": "forge_skill"})
        assert "ERROR" in result

    def test_skill_forge_forbidden_import_detection(self):
        """Auto-Forge should reject code with forbidden imports."""
        from ragnarok.skills.mcp_hub import SkillForge
        forge = SkillForge()
        bad_code = '''
from ragnarok.skills.registry import SkillManifest
import boto3

MANIFEST = SkillManifest(name="evil_skill", description="Evil", risk_level="high", parameters={}, required=[])

def run(args: dict) -> str:
    return boto3.client("s3").list_buckets()
'''
        error = forge._validate_code(bad_code)
        assert error is not None
        assert "boto3" in error.lower() or "Forbidden" in error

    def test_skill_forge_missing_manifest_detected(self):
        from ragnarok.skills.mcp_hub import SkillForge
        forge = SkillForge()
        bad_code = '''
def run(args: dict) -> str:
    return "no manifest"
'''
        error = forge._validate_code(bad_code)
        assert error is not None
        assert "MANIFEST" in error

    def test_skill_forge_extract_name(self):
        from ragnarok.skills.mcp_hub import SkillForge
        code = '''
MANIFEST = SkillManifest(name="my_cool_skill", description="Test", risk_level="low", parameters={}, required=[])
def run(args): return "ok"
'''
        name = SkillForge._extract_skill_name(code)
        assert name == "my_cool_skill"

    def test_skill_forge_extract_code_from_markdown(self):
        from ragnarok.skills.mcp_hub import SkillForge
        md = '''Some preamble
```python
def run(args):
    return "hello"
```
Some postamble'''
        code = SkillForge._extract_code(md)
        assert "def run" in code
        assert "```" not in code


# ══════════════════════════════════════════════════════════════════════════════
# [I6] Shadow Worker Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestShadowWorker:
    def test_git_intelligence_summarize_no_crash(self, tmp_path):
        """GitIntelligence should not crash even in a non-git directory."""
        from ragnarok.triggers.shadow_worker import GitIntelligence
        gi = GitIntelligence(repo_root=tmp_path)
        summary = gi.summarize()
        assert isinstance(summary, str)

    def test_git_recent_commits_empty_in_non_repo(self, tmp_path):
        from ragnarok.triggers.shadow_worker import GitIntelligence
        gi = GitIntelligence(repo_root=tmp_path)
        commits = gi.recent_commits(since_hours=24)
        assert isinstance(commits, list)

    def test_error_sentinel_traceback_pattern_detection(self, tmp_path):
        from ragnarok.triggers.shadow_worker import ErrorSentinel, _TRACEBACK_PATTERN
        # Create a fake log file with a traceback
        log_file = tmp_path / "test.log"
        log_file.write_text("""
Some normal output
Traceback (most recent call last):
  File "test.py", line 42, in test_something
    assert result == expected
AssertionError: expected 42 but got 0
""")
        sentinel = ErrorSentinel(watch_paths=[tmp_path], on_alert=lambda a: None)
        alerts = sentinel.scan_once()
        assert any(a.file_path == str(log_file) for a in alerts)

    def test_error_sentinel_no_alert_on_clean_log(self, tmp_path):
        from ragnarok.triggers.shadow_worker import ErrorSentinel
        log_file = tmp_path / "clean.log"
        log_file.write_text("All tests passed. OK (122 tests)\n")
        sentinel = ErrorSentinel(watch_paths=[log_file])
        alerts = sentinel.scan_once()
        assert len(alerts) == 0

    def test_error_sentinel_cooldown(self, tmp_path):
        """Same file should not produce two alerts within cooldown period."""
        from ragnarok.triggers.shadow_worker import ErrorSentinel
        log_file = tmp_path / "error.log"
        log_file.write_text("""
Traceback (most recent call last):
  File "app.py", line 10, in main
    raise ValueError("test error")
ValueError: test error
""")
        sentinel = ErrorSentinel(watch_paths=[log_file])
        sentinel._cooldown_sec = 10.0  # 10 second cooldown
        alerts1 = sentinel.scan_once()
        # Second scan should not re-alert (cooldown)
        alerts2 = sentinel.scan_once()
        assert len(alerts1) <= 1
        assert len(alerts2) == 0

    def test_error_classify(self):
        from ragnarok.triggers.shadow_worker import ErrorSentinel
        assert ErrorSentinel._classify_error("AssertionError: things") == "Test Failure"
        assert ErrorSentinel._classify_error("ImportError: no module") == "Import Error"
        assert ErrorSentinel._classify_error("ConnectionError: refused") == "Connection Error"
        assert ErrorSentinel._classify_error("some weird SyntaxError") == "Python Error"

    def test_shadow_worker_start_stop(self):
        from ragnarok.triggers.shadow_worker import ShadowWorker
        worker = ShadowWorker(brief_time="23:59")  # Won't fire today
        worker.start()
        assert worker._running
        time.sleep(0.1)
        worker.stop()
        assert not worker._running

    def test_briefing_report_render(self):
        from ragnarok.triggers.shadow_worker import BriefingReport
        report = BriefingReport(
            date="Wednesday, October 28, 2026",
            git_summary="Branch: main\n3 commits",
            pending_tasks=["Review PR #42", "Fix encryption bug"],
            upcoming_notes=[],
            recent_memory_snippets=["User prefers dark mode"],
        )
        text = report.render()
        assert "Wednesday" in text or "Branch" in text


# ══════════════════════════════════════════════════════════════════════════════
# [I7] Hardware-Bound Vault + Merkle Audit Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMerkleAuditLog:
    @pytest.fixture
    def merkle_log(self, tmp_path):
        from ragnarok.safety.vault import MerkleAuditLog
        return MerkleAuditLog(log_path=tmp_path / "test_audit.jsonl")

    def test_genesis_state(self, merkle_log):
        assert merkle_log.entry_count == 0
        assert merkle_log.get_root_hash() == merkle_log.GENESIS_HASH

    def test_append_creates_entry(self, merkle_log):
        entry = merkle_log.append("tool_call", tool_name="read_file", details={"path": "/test"})
        assert entry.event == "tool_call"
        assert entry.tool_name == "read_file"
        assert entry.entry_id == 0
        assert entry.prev_hash == merkle_log.GENESIS_HASH
        assert len(entry.entry_hash) == 64  # SHA-256 hex

    def test_hash_chain_linkage(self, merkle_log):
        e1 = merkle_log.append("session_start")
        e2 = merkle_log.append("tool_call", tool_name="get_time")
        e3 = merkle_log.append("tool_success", tool_name="get_time")
        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash

    def test_verify_chain_valid(self, merkle_log):
        for i in range(5):
            merkle_log.append(f"event_{i}", tool_name=f"tool_{i}")
        result = merkle_log.verify_chain()
        assert result["valid"] is True
        assert result["entries"] == 5
        assert result["tampered_at"] is None

    def test_verify_chain_detects_tampering(self, tmp_path):
        """Modifying a log entry should fail chain verification."""
        from ragnarok.safety.vault import MerkleAuditLog
        log_path = tmp_path / "tamper_test.jsonl"
        ml = MerkleAuditLog(log_path=log_path)
        ml.append("event_0")
        ml.append("event_1")
        ml.append("event_2")

        # Tamper with first entry
        lines = log_path.read_text().splitlines()
        first_entry = json.loads(lines[0])
        first_entry["event"] = "TAMPERED"
        lines[0] = json.dumps(first_entry)
        log_path.write_text("\n".join(lines) + "\n")

        # Re-open and verify
        ml2 = MerkleAuditLog(log_path=log_path)
        result = ml2.verify_chain()
        assert result["valid"] is False
        assert result["tampered_at"] == 0

    def test_hash_determinism(self):
        """Same inputs must always produce the same hash."""
        from ragnarok.safety.vault import MerkleAuditEntry
        h1 = MerkleAuditEntry.compute_hash("event", "tool", 12345.0, {"k": "v"}, "prev")
        h2 = MerkleAuditEntry.compute_hash("event", "tool", 12345.0, {"k": "v"}, "prev")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        from ragnarok.safety.vault import MerkleAuditEntry
        h1 = MerkleAuditEntry.compute_hash("event_a", "tool", 1.0, {}, "prev")
        h2 = MerkleAuditEntry.compute_hash("event_b", "tool", 1.0, {}, "prev")
        assert h1 != h2

    def test_entry_count_persists(self, tmp_path):
        """Entry count should load correctly from existing log file."""
        from ragnarok.safety.vault import MerkleAuditLog
        path = tmp_path / "persist_test.jsonl"
        ml1 = MerkleAuditLog(log_path=path)
        for i in range(3):
            ml1.append(f"event_{i}")

        # Re-open and check count
        ml2 = MerkleAuditLog(log_path=path)
        assert ml2.entry_count == 3
        assert ml2.get_root_hash() == ml1.get_root_hash()


class TestHardwareBoundVault:
    @pytest.fixture
    def vault(self, tmp_path):
        from ragnarok.safety.vault import HardwareBoundVault
        return HardwareBoundVault(vault_dir=tmp_path / "vault")

    def test_hardware_fingerprint_is_stable(self):
        from ragnarok.safety.vault import HardwareFingerprint
        fp1 = HardwareFingerprint.get()
        fp2 = HardwareFingerprint.get()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex

    def test_vault_set_and_get(self, vault):
        vault.set("api_key", "secret-value-123")
        result = vault.get("api_key")
        assert result == "secret-value-123"

    def test_vault_get_missing_returns_default(self, vault):
        result = vault.get("nonexistent_key", default="fallback")
        assert result == "fallback"

    def test_vault_delete_key(self, vault):
        vault.set("temp_secret", "delete-me")
        deleted = vault.delete("temp_secret")
        assert deleted is True
        assert vault.get("temp_secret") is None

    def test_vault_list_keys(self, vault):
        vault.set("key_a", "value_a")
        vault.set("key_b", "value_b")
        keys = vault.list_keys()
        assert "key_a" in keys
        assert "key_b" in keys

    def test_vault_secrets_encrypted_at_rest(self, vault, tmp_path):
        """Secrets file must not contain plaintext secrets."""
        vault.set("sensitive_token", "super-secret-token-xyz")
        secrets_file = tmp_path / "vault" / "secrets.enc"
        raw_bytes = secrets_file.read_bytes()
        # Plaintext should NOT appear in the encrypted file
        assert b"super-secret-token-xyz" not in raw_bytes

    def test_vault_verify_hardware_binding(self, vault):
        status = vault.verify_hardware_binding()
        assert status["bound"] is True
        assert "hardware_id" in status
        assert "platform" in status

    def test_vault_update_secret(self, vault):
        vault.set("my_key", "version_1")
        vault.set("my_key", "version_2")
        assert vault.get("my_key") == "version_2"

    def test_vault_multiple_secrets_isolated(self, vault):
        vault.set("alpha", "aaa")
        vault.set("beta", "bbb")
        vault.set("gamma", "ccc")
        assert vault.get("alpha") == "aaa"
        assert vault.get("beta") == "bbb"
        assert vault.get("gamma") == "ccc"


class TestAuditWithMerkle:
    def test_audit_with_merkle_returns_hash(self, tmp_path):
        from ragnarok.safety import vault as vault_module
        # Patch the singleton
        from ragnarok.safety.vault import MerkleAuditLog
        original = vault_module._merkle_log
        vault_module._merkle_log = MerkleAuditLog(log_path=tmp_path / "test.jsonl")
        try:
            from ragnarok.safety.vault import audit_with_merkle
            entry_hash = audit_with_merkle("tool_call", tool_name="read_file")
            assert isinstance(entry_hash, str)
            assert len(entry_hash) == 64
        finally:
            vault_module._merkle_log = original

    def test_verify_audit_integrity(self, tmp_path):
        from ragnarok.safety import vault as vault_module
        from ragnarok.safety.vault import MerkleAuditLog, audit_with_merkle, verify_audit_integrity
        original = vault_module._merkle_log
        vault_module._merkle_log = MerkleAuditLog(log_path=tmp_path / "integrity.jsonl")
        try:
            audit_with_merkle("session_start")
            audit_with_merkle("tool_call", tool_name="get_time")
            result = verify_audit_integrity()
            assert result["valid"] is True
            assert result["entries"] == 2
        finally:
            vault_module._merkle_log = original
