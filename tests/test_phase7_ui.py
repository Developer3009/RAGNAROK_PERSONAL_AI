"""
tests/test_phase7_ui.py
───────────────────────
Unit and integration tests for Phase 7: Local interface and transparency.
Tests:
  - Token authentication (headers & query parameters)
  - System status & hardware telemetry
  - Chat endpoint with explainability traces
  - Memory Manager CRUD (create, search, edit, pin, confirm-delete, export)
  - Approval workflow (pending queue, approve, deny, parameter edit)
  - Execution log timeline
  - Settings & permissions configuration
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from ragnarok.ui.app import app, ACTIVE_TOKEN, state, trace_store
from ragnarok.safety.approval import ApprovalRequest, ApprovalStatus
from ragnarok.agent.loop import AgentResult


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-Ragnarok-Token": ACTIVE_TOKEN}


# ── 1. Security & Authentication Tests ────────────────────────────────────────

class TestAuthSecurity:
    def test_missing_token_rejected(self, client):
        res = client.get("/api/status")
        assert res.status_code == 401
        assert "Invalid or missing" in res.json()["detail"]

    def test_invalid_token_rejected(self, client):
        res = client.get("/api/status", headers={"X-Ragnarok-Token": "bogus-token-1234"})
        assert res.status_code == 401

    def test_valid_header_token_accepted(self, client, auth_headers):
        res = client.get("/api/status", headers=auth_headers)
        assert res.status_code == 200

    def test_valid_query_token_accepted(self, client):
        res = client.get(f"/api/status?token={ACTIVE_TOKEN}")
        assert res.status_code == 200

    def test_index_page_injects_token(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "Ragnarok AI" in res.text
        assert ACTIVE_TOKEN in res.text


# ── 2. System Status & Telemetry Tests ─────────────────────────────────────────

class TestSystemStatus:
    def test_status_fields(self, client, auth_headers):
        res = client.get("/api/status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert "cpu_percent" in data
        assert "ram_used_mb" in data
        assert "models" in data
        assert "main" in data["models"]
        assert "memory_count" in data
        assert "allowed_folders" in data
        assert "registered_skills" in data
        assert isinstance(data["registered_skills"], list)


# ── 3. Chat & Explainability Traces Tests ─────────────────────────────────────

class TestChatEndpoint:
    def test_chat_interaction_and_trace(self, client, auth_headers):
        mock_result = AgentResult(
            answer="Ragnarok operates entirely on your local machine.",
            steps_taken=2,
            tool_calls_made=["get_time"],
            latency_ms=120.5,
        )
        with patch.object(state.agent, "chat", return_value=mock_result):
            res = client.post(
                "/api/chat",
                headers=auth_headers,
                json={"message": "Where are you running?"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["answer"] == "Ragnarok operates entirely on your local machine."
            assert data["steps_taken"] == 2
            assert data["tool_calls"] == ["get_time"]
            assert data["latency_ms"] == pytest.approx(120.5, 5.0)

            # Confirm trace recorded in trace_store
            traces = trace_store.get_all(limit=5)
            assert len(traces) > 0
            assert traces[0]["user_query"] == "Where are you running?"


# ── 4. Memory Manager CRUD Tests ──────────────────────────────────────────────

class TestMemoryManager:
    def test_add_and_list_memory(self, client, auth_headers):
        # 1. Add memory
        res = client.post(
            "/api/memory",
            headers=auth_headers,
            json={
                "text": "User prefers concise answers with zero fluff.",
                "kind": "preference",
                "importance": 0.8,
                "pinned": False,
            },
        )
        assert res.status_code == 200
        mid = res.json()["id"]

        # 2. List memories
        list_res = client.get("/api/memory", headers=auth_headers)
        assert list_res.status_code == 200
        items = list_res.json()
        found = next((m for m in items if m["id"] == mid), None)
        assert found is not None
        assert "concise" in found["text"]
        assert found["kind"] == "preference"

        # 3. Update memory
        up_res = client.put(
            f"/api/memory/{mid}",
            headers=auth_headers,
            json={"text": "User strictly prefers bullet points.", "importance": 0.9},
        )
        assert up_res.status_code == 200

        # Verify update
        check_res = client.get("/api/memory", headers=auth_headers)
        updated = next((m for m in check_res.json() if m["id"] == mid), None)
        assert updated is not None
        assert "bullet points" in updated["text"]
        assert updated["importance"] == 0.9

        # 4. Pin and Unpin
        pin_res = client.post(f"/api/memory/{mid}/pin?pinned=true", headers=auth_headers)
        assert pin_res.status_code == 200
        assert pin_res.json()["pinned"] is True

        # 5. Delete without confirmation must fail
        del_fail = client.delete(f"/api/memory/{mid}", headers=auth_headers)
        assert del_fail.status_code == 400
        assert "confirmation" in del_fail.json()["detail"].lower()

        # 6. Delete with confirmation succeeds
        del_ok = client.delete(f"/api/memory/{mid}?confirm=true", headers=auth_headers)
        assert del_ok.status_code == 200
        assert del_ok.json()["status"] == "deleted"

    def test_export_memories_json(self, client, auth_headers):
        res = client.get("/api/memory/export", headers=auth_headers)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/json"
        assert "ragnarok_memories.json" in res.headers["content-disposition"]
        data = res.json()
        assert isinstance(data, list)


# ── 5. Approvals Workflow Tests ───────────────────────────────────────────────

class TestApprovalsWorkflow:
    def test_pending_approval_approve_flow(self, client, auth_headers):
        mgr = state.approval_manager

        # Inject a pending request directly
        req_id = f"test-appr-{int(time.time() * 1000)}"
        req = ApprovalRequest(
            request_id=req_id,
            tool_name="write_file",
            args={"path": "notes/secret.txt", "content": "top secret"},
            risk_level="high",
            reason="Writing file to disk",
        )
        import threading
        event = threading.Event()
        holder = [None]
        with mgr._lock:
            mgr._pending[req_id] = (req, event, holder)

        # 1. Inspect pending via API
        list_res = client.get("/api/approvals", headers=auth_headers)
        assert list_res.status_code == 200
        pending = list_res.json()["pending"]
        assert any(p["request_id"] == req_id for p in pending)

        # 2. Approve via API
        resp = client.post(
            f"/api/approvals/{req_id}/respond",
            headers=auth_headers,
            json={"status": "approved", "comment": "Operator verified path is safe"},
        )
        assert resp.status_code == 200
        assert resp.json()["resolved_status"] == "approved"

        # Verify event was signaled and holder populated
        assert event.is_set()
        assert holder[0].status == ApprovalStatus.APPROVED
        assert holder[0].args["path"] == "notes/secret.txt"

    def test_pending_approval_edit_flow(self, client, auth_headers):
        mgr = state.approval_manager
        req_id = f"test-edit-{int(time.time() * 1000)}"
        req = ApprovalRequest(
            request_id=req_id,
            tool_name="write_file",
            args={"path": "data/untrusted.txt", "content": "payload"},
            risk_level="high",
        )
        import threading
        event = threading.Event()
        holder = [None]
        with mgr._lock:
            mgr._pending[req_id] = (req, event, holder)

        # Edit parameters and approve
        edited_path = "data/safe_name.txt"
        resp = client.post(
            f"/api/approvals/{req_id}/respond",
            headers=auth_headers,
            json={
                "status": "edited",
                "edited_args": {"path": edited_path, "content": "payload"},
                "comment": "Sanitized path",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["resolved_status"] == "edited"
        assert event.is_set()
        assert holder[0].status == ApprovalStatus.EDITED
        assert holder[0].args["path"] == edited_path


# ── 6. Execution Logs & Explainability Tests ──────────────────────────────────

class TestExecutionLogs:
    def test_get_execution_logs(self, client, auth_headers):
        res = client.get("/api/logs", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "execution_traces" in data
        assert "recent_tool_events" in data
        assert isinstance(data["execution_traces"], list)


# ── 7. Settings & Roles Tests ─────────────────────────────────────────────────

class TestSettingsEndpoint:
    def test_get_and_update_settings(self, client, auth_headers):
        # 1. Get settings
        res = client.get("/api/settings", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
        assert "skills" in data
        assert "permissions" in data
        assert len(data["skills"]) > 0

        # 2. Update role
        up_res = client.post(
            "/api/settings",
            headers=auth_headers,
            json={"active_role": "admin"},
        )
        assert up_res.status_code == 200
        assert up_res.json()["active_role"] == "admin"
        assert state.permission_manager.active_role == "admin"

        # Restore role back to operator
        client.post(
            "/api/settings",
            headers=auth_headers,
            json={"active_role": "operator"},
        )
