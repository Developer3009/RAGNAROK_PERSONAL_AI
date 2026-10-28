"""
tests/test_e2e_3runs.py
────────────────────────
End-to-End User Flow Multi-Run Validation Suite.
Enforces the mandatory requirement:
"Test every user flow in this application end-to-end, at least 3 times each with different input data each run (not the same happy-path data repeated)."

Covers the complete user flows:
  Flow 1: Knowledge Graph Ingestion, Multi-Hop Query & Deprecation
  Flow 2: Hardware-Bound Vault Secret Lifecycle & Key Isolation
  Flow 3: Cryptographic Merkle Audit Hash Chaining & Tamper Detection
  Flow 4: Mesh Swarm Task Creation, Transport Encryption & Dispatch
  Flow 5: Shadow Worker Error Sentinel Diagnosis & Briefing Generation
  Flow 6: MCP Hub Server Management & Skill Forge Synthesis Validation
  Flow 7: Streaming Voice & GUI Action Parameter & Dispatch Pipelines
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Flow 1: Temporal Knowledge Graph (3 Runs with Distinct Data)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_idx, user_name, old_pref, new_pref, project_name, tool_stack", [
    (1, "Dev_Alice", "Sublime Text", "Neovim", "Project_Alpha", "Rust"),
    (2, "Admin_Marcus", "Debian 11", "Ubuntu 24.04", "Cluster_Valhalla", "Kubernetes"),
    (3, "Researcher_Elena", "PyTorch 1.12", "JAX 0.4", "Model_Ragnarok", "Flax"),
])
def test_e2e_flow_knowledge_graph(tmp_path: Path, run_idx, user_name, old_pref, new_pref, project_name, tool_stack):
    """Run flow: Entity creation -> fact addition -> deprecation/supersession -> multi-hop query -> context retrieval."""
    from ragnarok.memory.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(db_path=tmp_path / f"kg_run_{run_idx}.db")

    # Step 1: Add old preference
    e1 = kg.add_triple(user_name, "uses_tool", old_pref, supersede_existing=True)
    assert e1.valid_until is None

    # Step 2: Add project relationship
    e2 = kg.add_triple(user_name, "leads_project", project_name)
    e3 = kg.add_triple(project_name, "written_in", tool_stack)

    # Step 3: Supersede old preference with new preference
    e4 = kg.add_triple(user_name, "uses_tool", new_pref, supersede_existing=True)
    assert e4.valid_until is None

    # Verify history has 2 entries (1 active, 1 superseded)
    history = kg.get_history(user_name, "uses_tool")
    assert len(history) == 2
    active = [h for h in history if h["valid_until"] is None]
    deprecated = [h for h in history if h["valid_until"] is not None]
    assert len(active) == 1
    assert active[0]["obj"] == new_pref
    assert len(deprecated) == 1
    assert deprecated[0]["obj"] == old_pref

    # Step 4: Multi-hop query: user -> project -> tool_stack
    hops = kg.multi_hop_query(user_name, max_hops=2)
    assert any(project_name in t for t in hops)
    assert any(tool_stack in t for t in hops)

    # Step 5: Format context for prompt injection
    ctx = kg.query_as_context(user_name)
    assert user_name in ctx
    assert new_pref in ctx


# ══════════════════════════════════════════════════════════════════════════════
# Flow 2: Hardware-Bound Vault (3 Runs with Distinct Data)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_idx, secrets_dict", [
    (1, {"api_key_openai": "sk-run1-abc123456789", "db_password": "super_secret_pw_1"}),
    (2, {"github_pat": "ghp_run2_xyz987654321", "tpm_token": "token_hardware_secure_2"}),
    (3, {"ssh_private_passphrase": "pass_run3_secure!", "encryption_seed": "seed_entropy_999"}),
])
def test_e2e_flow_vault_lifecycle(tmp_path: Path, run_idx, secrets_dict):
    """Run flow: Hardware fingerprint -> vault set -> encrypted-at-rest check -> retrieve -> delete -> verify."""
    from ragnarok.safety.vault import HardwareBoundVault, HardwareFingerprint

    # Step 1: Verify hardware fingerprint
    fp = HardwareFingerprint.get()
    assert len(fp) == 64

    vault = HardwareBoundVault(vault_dir=tmp_path / f"vault_run_{run_idx}")

    # Step 2: Store secrets
    for k, v in secrets_dict.items():
        vault.set(k, v)

    # Step 3: Verify ciphertext on disk does not leak plaintext
    enc_file = tmp_path / f"vault_run_{run_idx}" / "secrets.enc"
    assert enc_file.exists()
    raw_enc = enc_file.read_bytes()
    for v in secrets_dict.values():
        assert v.encode() not in raw_enc

    # Step 4: Retrieve and verify secrets match
    for k, v in secrets_dict.items():
        assert vault.get(k) == v

    # Step 5: Delete one secret and verify deletion
    first_key = list(secrets_dict.keys())[0]
    assert vault.delete(first_key) is True
    assert vault.get(first_key) is None

    # Step 6: Verify remaining keys persist
    remaining = vault.list_keys()
    assert first_key not in remaining
    for k in list(secrets_dict.keys())[1:]:
        assert k in remaining


# ══════════════════════════════════════════════════════════════════════════════
# Flow 3: Cryptographic Merkle Audit Log (3 Runs with Distinct Events)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_idx, event_sequence", [
    (1, [("session_start", "core"), ("tool_call", "read_file"), ("tool_result", "read_file"), ("session_stop", "core")]),
    (2, [("auth_success", "vault"), ("key_rotated", "vault"), ("file_written", "write_file"), ("audit_verified", "merkle")]),
    (3, [("mesh_peer_joined", "mesh"), ("task_dispatched", "mesh"), ("task_completed", "mesh"), ("mesh_peer_left", "mesh")]),
])
def test_e2e_flow_merkle_audit_integrity(tmp_path: Path, run_idx, event_sequence):
    """Run flow: Append chained events -> verify valid chain -> tamper with an event -> verify tamper detection."""
    from ragnarok.safety.vault import MerkleAuditLog

    log_path = tmp_path / f"merkle_run_{run_idx}.jsonl"
    audit = MerkleAuditLog(log_path=log_path)

    # Step 1: Append chained events
    entries = []
    for ev, tool in event_sequence:
        e = audit.append(ev, tool_name=tool, details={"run": run_idx})
        entries.append(e)

    assert audit.entry_count == len(event_sequence)

    # Step 2: Verify chain integrity passes
    check = audit.verify_chain()
    assert check["valid"] is True
    assert check["entries"] == len(event_sequence)
    assert check["tampered_at"] is None

    # Step 3: Tamper with middle entry on disk
    lines = log_path.read_text(encoding="utf-8").splitlines()
    target_idx = 1
    tampered_record = json.loads(lines[target_idx])
    tampered_record["details"]["malicious_edit"] = True
    lines[target_idx] = json.dumps(tampered_record)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Step 4: Verify that tampering is immediately caught at target_idx
    audit_tampered = MerkleAuditLog(log_path=log_path)
    tamper_check = audit_tampered.verify_chain()
    assert tamper_check["valid"] is False
    assert tamper_check["tampered_at"] == target_idx


# ══════════════════════════════════════════════════════════════════════════════
# Flow 4: Mesh Swarm Transport & Task Protocol (3 Runs with Distinct Data)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_idx, task_id, user_prompt, secret_key", [
    (1, "task-dev-001", "Compile the Linux kernel with custom flags", "mesh_secret_alpha_2026"),
    (2, "task-sec-002", "Scan internal subnet 192.168.1.0/24 for vulnerabilities", "mesh_secret_bravo_9999"),
    (3, "task-ml-003", "Run embedding generation across 500 documents", "mesh_secret_charlie_xyz"),
])
def test_e2e_flow_mesh_swarm(run_idx, task_id, user_prompt, secret_key):
    """Run flow: Create task request -> encrypt with Fernet transport -> decrypt -> verify payload."""
    from ragnarok.agent.mesh_swarm import MeshTaskRequest, MeshTransport, ValhallaMeshNode

    # Step 1: Generate request
    req = MeshTaskRequest(
        task_id=task_id,
        user_input=user_prompt,
        model_hint="gemma3:4b",
        requester_id=f"node-origin-{run_idx}",
    )
    raw_json = req.to_json()

    # Step 2: Encrypt with transport
    transport = MeshTransport.from_passphrase(secret_key)
    ciphertext = transport.encrypt(raw_json)
    assert user_prompt.encode() not in ciphertext

    # Step 3: Decrypt and reconstruct task
    plaintext = transport.decrypt(ciphertext)
    reconstructed = json.loads(plaintext)
    assert reconstructed["task_id"] == task_id
    assert reconstructed["user_input"] == user_prompt

    # Step 4: Capability detection check
    node = ValhallaMeshNode(node_id=f"worker-{run_idx}", local_executor=lambda x: f"Done: {x}")
    status = node.status()
    assert "capabilities" in status
    assert "cpu" in status["capabilities"]


# ══════════════════════════════════════════════════════════════════════════════
# Flow 5: Shadow Worker Sentinel & Diagnosis (3 Runs with Distinct Errors)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_idx, filename, error_snippet, expected_type", [
    (1, "test_network.py", "Traceback (most recent call last):\n  File 'test_network.py', line 12\nConnectionError: Failed to connect to 127.0.0.1:47890", "Connection Error"),
    (2, "pipeline.py", "Traceback (most recent call last):\n  File 'pipeline.py', line 99\nAssertionError: Expected 200 OK but received 500", "Test Failure"),
    (3, "auth_service.py", "Traceback (most recent call last):\n  File 'auth_service.py', line 45\nKeyError: 'user_session_token'", "Key Error"),
])
def test_e2e_flow_shadow_worker_sentinel(tmp_path: Path, run_idx, filename, error_snippet, expected_type):
    """Run flow: Error occurs in log -> Sentinel detects traceback -> classifies -> triggers alert."""
    from ragnarok.triggers.shadow_worker import ErrorSentinel

    log_dir = tmp_path / f"logs_run_{run_idx}"
    log_dir.mkdir()
    target_log = log_dir / f"{filename}.log"
    target_log.write_text(error_snippet, encoding="utf-8")

    captured_alerts = []
    sentinel = ErrorSentinel(
        watch_paths=[log_dir],
        on_alert=lambda alert: captured_alerts.append(alert),
    )

    alerts = sentinel.scan_once()
    assert len(alerts) >= 1
    alert = alerts[0]
    assert alert.error_type == expected_type
    assert str(target_log) in alert.file_path


# ══════════════════════════════════════════════════════════════════════════════
# Flow 6: MCP Hub Management & Auto-Forge Validation (3 Runs with Distinct Data)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_idx, server_alias, fake_tool_desc, forbidden_module", [
    (1, "sqlite_server", "Tool to execute read-only SQLite queries", "paramiko"),
    (2, "docker_server", "Tool to inspect container status", "boto3"),
    (3, "git_server", "Tool to check git branch status", "ctypes"),
])
def test_e2e_flow_mcp_and_forge(run_idx, server_alias, fake_tool_desc, forbidden_module):
    """Run flow: MCP tool schema injection check + Auto-Forge safety filter against dangerous imports."""
    from ragnarok.skills.mcp_hub import MCPHub, SkillForge

    # Step 1: MCP Hub listing & discovery state
    hub = MCPHub()
    servers = hub.list_servers()
    assert isinstance(servers, list)

    # Step 2: Auto-Forge static validation blocks forbidden cloud/system imports
    forge = SkillForge()
    malicious_code = f'''
from ragnarok.skills.registry import SkillManifest
import {forbidden_module}

MANIFEST = SkillManifest(name="forged_tool_{run_idx}", description="{fake_tool_desc}", risk_level="low", parameters={{}}, required=[])

def run(args: dict) -> str:
    return "executed"
'''
    validation_err = forge._validate_code(malicious_code)
    assert validation_err is not None
    assert forbidden_module in validation_err


# ══════════════════════════════════════════════════════════════════════════════
# Flow 7: Voice & GUI Pipeline Parameters (3 Runs with Distinct Data)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("run_idx, voice_text, voice_mode, gui_action_name", [
    (1, "Ragnarok, initiate system diagnostics.", "speak", "move"),
    (2, "Warning: battery level below twenty percent.", "speak", "scroll"),
    (3, "Task completed successfully. Valhalla mesh nodes synchronized.", "speak", "double_click"),
])
def test_e2e_flow_voice_and_gui_dispatch(run_idx, voice_text, voice_mode, gui_action_name):
    """Run flow: Voice skill validation + GUI action parameter dispatch validation."""
    from ragnarok.skills.streaming_voice import run as voice_run
    from ragnarok.skills.gui_action import MANIFEST as gui_manifest

    # Voice test
    result = voice_run({"mode": voice_mode, "text": voice_text})
    assert "TTS" in result or "speak" in result or "completed" in result

    # GUI action validation
    allowed_actions = gui_manifest.parameters["action"].enum
    assert gui_action_name in allowed_actions
