"""
tests/test_agenda_e2e_3runs.py
──────────────────────────────
End-to-end tests for the Agenda & Calendar feature.

Covers:
  Flow 1 — Calendar CRUD          (3 runs × different data)
  Flow 2 — Todo CRUD              (3 runs × different data)
  Flow 3 — Email AI parsing       (3 runs × different email bodies, mocked LLM)
  Flow 4 — REST API endpoints     (3 runs × different payloads via TestClient)

Global rule: every flow is exercised ≥ 3 times with distinct data.
Root cause must be fixed before any item is marked as passing.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def tmp_agenda_db(tmp_path):
    """Create a fresh AgendaDB pointing at a temp directory for each test."""
    from ragnarok.memory.agenda import AgendaDB
    db = AgendaDB(db_path=tmp_path / "test_agenda.db")
    yield db
    db.close()


@pytest.fixture()
def test_client(tmp_path, monkeypatch):
    """Return a FastAPI TestClient with a fresh AgendaDB singleton."""
    from ragnarok.memory.agenda import AgendaDB
    test_db = AgendaDB(db_path=tmp_path / "api_test_agenda.db")

    # Patch get_agenda_db to return our test instance for the whole module
    import ragnarok.memory.agenda as agenda_mod
    import ragnarok.ui.app as app_mod
    monkeypatch.setattr(agenda_mod, "_agenda_db", test_db)

    # Import after patching
    from fastapi.testclient import TestClient
    from ragnarok.ui.app import app, ACTIVE_TOKEN

    client = TestClient(app, raise_server_exceptions=False)
    yield client, ACTIVE_TOKEN, test_db
    test_db.close()


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 1 — Calendar CRUD  (3 distinct runs)
# ════════════════════════════════════════════════════════════════════════════════

CALENDAR_EVENT_DATA = [
    {
        "title": "Team Standup",
        "start_time": "2026-11-10T09:00:00",
        "end_time": "2026-11-10T09:30:00",
        "description": "Daily sync meeting",
        "location": "Conference Room A",
        "color": "#00dbe9",
        "source": "manual",
    },
    {
        "title": "Doctor Appointment",
        "start_time": "2026-11-15T14:30:00",
        "end_time": "2026-11-15T15:30:00",
        "description": "Annual checkup",
        "location": "City Clinic, 2nd Floor",
        "color": "#ff6b6b",
        "source": "manual",
    },
    {
        "title": "Sprint Review Deadline",
        "start_time": "2026-11-28T18:00:00",
        "end_time": "2026-11-28T19:00:00",
        "description": "Q4 Sprint 3 final review",
        "location": "Main Conference Hall",
        "color": "#a855f7",
        "source": "agent",
    },
]


@pytest.mark.parametrize("event_data", CALENDAR_EVENT_DATA)
def test_calendar_crud(tmp_agenda_db, event_data):
    """Flow 1: Create → Read → Update → Delete a calendar event."""
    db = tmp_agenda_db

    # CREATE
    ev = db.create_event(
        title=event_data["title"],
        start_time=event_data["start_time"],
        end_time=event_data["end_time"],
        description=event_data["description"],
        location=event_data["location"],
        color=event_data["color"],
        source=event_data["source"],
    )
    assert ev.id, "Created event must have an ID"
    assert ev.title == event_data["title"]
    assert ev.start_time == event_data["start_time"]
    assert ev.location == event_data["location"]
    assert ev.color == event_data["color"]

    # READ by ID
    fetched = db.get_event(ev.id)
    assert fetched is not None
    assert fetched.id == ev.id
    assert fetched.title == ev.title

    # LIST — event should appear
    events = db.list_events()
    ids = [e.id for e in events]
    assert ev.id in ids

    # LIST with date filter
    events_filtered = db.list_events(
        start=event_data["start_time"][:10] + "T00:00:00",
        end=event_data["start_time"][:10] + "T23:59:59",
    )
    assert any(e.id == ev.id for e in events_filtered), "Event should appear in date-filtered list"

    # UPDATE
    new_title = "UPDATED_" + event_data["title"]
    new_location = "Virtual / Zoom"
    updated = db.update_event(ev.id, title=new_title, location=new_location)
    assert updated is not None
    assert updated.title == new_title
    assert updated.location == new_location
    assert updated.start_time == ev.start_time, "start_time must be preserved"

    # Verify update persisted
    re_fetched = db.get_event(ev.id)
    assert re_fetched.title == new_title

    # DELETE
    deleted = db.delete_event(ev.id)
    assert deleted is True

    # Verify deletion
    gone = db.get_event(ev.id)
    assert gone is None

    # Deleting again returns False
    second_delete = db.delete_event(ev.id)
    assert second_delete is False


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 2 — Todo CRUD  (3 distinct runs)
# ════════════════════════════════════════════════════════════════════════════════

TODO_TASK_DATA = [
    {
        "title": "Prepare quarterly budget report",
        "due_date": "2026-11-14T17:00:00",
        "priority": "high",
        "description": "Include Q3 actual vs forecast",
        "tags": ["finance", "q4"],
    },
    {
        "title": "Review pull request #142",
        "due_date": "2026-11-11T12:00:00",
        "priority": "urgent",
        "description": "Critical bug fix PR needs review",
        "tags": ["engineering", "review"],
    },
    {
        "title": "Order office supplies",
        "due_date": "2026-11-20T18:00:00",
        "priority": "low",
        "description": "Pens, notebooks, sticky notes",
        "tags": ["admin"],
    },
]


@pytest.mark.parametrize("task_data", TODO_TASK_DATA)
def test_todo_crud(tmp_agenda_db, task_data):
    """Flow 2: Create → Read → Update → Toggle → Delete a to-do task."""
    db = tmp_agenda_db

    # CREATE
    task = db.create_task(
        title=task_data["title"],
        due_date=task_data["due_date"],
        priority=task_data["priority"],
        description=task_data["description"],
        tags=task_data["tags"],
    )
    assert task.id, "Created task must have an ID"
    assert task.title == task_data["title"]
    assert task.priority == task_data["priority"]
    assert task.status == "pending"
    assert task.tags == task_data["tags"]

    # READ by ID
    fetched = db.get_task(task.id)
    assert fetched is not None
    assert fetched.id == task.id
    assert fetched.tags == task_data["tags"]

    # LIST — task should appear
    tasks = db.list_tasks()
    ids = [t.id for t in tasks]
    assert task.id in ids

    # LIST with status filter
    pending_tasks = db.list_tasks(status="pending")
    assert any(t.id == task.id for t in pending_tasks)

    # UPDATE
    new_title = "UPDATED_" + task_data["title"]
    updated = db.update_task(task.id, title=new_title, priority="medium")
    assert updated is not None
    assert updated.title == new_title
    assert updated.priority == "medium"
    assert updated.due_date == task.due_date, "due_date must be preserved"

    # TOGGLE to completed
    toggled = db.toggle_task(task.id)
    assert toggled is not None
    assert toggled.status == "completed"
    assert toggled.completed_at > 0

    # TOGGLE back to pending
    toggled_back = db.toggle_task(task.id)
    assert toggled_back is not None
    assert toggled_back.status == "pending"
    assert toggled_back.completed_at == 0.0

    # Completed filter should NOT contain it now
    completed_tasks = db.list_tasks(status="completed")
    assert not any(t.id == task.id for t in completed_tasks)

    # DELETE
    deleted = db.delete_task(task.id)
    assert deleted is True

    # Verify deletion
    gone = db.get_task(task.id)
    assert gone is None

    # Double delete returns False
    assert db.delete_task(task.id) is False


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 3 — Email AI Parsing  (3 distinct runs, mocked LLM)
# ════════════════════════════════════════════════════════════════════════════════

EMAIL_PARSE_CASES = [
    # Run 1: Meeting invite + action item
    (
        "Hi team, please join the Product Review meeting on 2026-11-18 at 2:00 PM in Room 101. "
        "The meeting ends at 3:30 PM. Also, please send the deck by 2026-11-17 at 5:00 PM.",
        {
            "events": [{"title": "Product Review", "start_time": "2026-11-18T14:00:00", "end_time": "2026-11-18T15:30:00", "location": "Room 101", "description": ""}],
            "todos": [{"title": "Send the deck", "due_date": "2026-11-17T17:00:00", "priority": "medium", "description": ""}],
        },
    ),
    # Run 2: Deadline reminder only
    (
        "Reminder: The contract signing is scheduled for 2026-11-25T10:30:00 in the Legal Office. "
        "Please submit your feedback on the contract draft by 2026-11-22T23:59:59. This is urgent.",
        {
            "events": [{"title": "Contract Signing", "start_time": "2026-11-25T10:30:00", "end_time": "2026-11-25T11:30:00", "location": "Legal Office", "description": ""}],
            "todos": [{"title": "Submit feedback on contract draft", "due_date": "2026-11-22T23:59:59", "priority": "urgent", "description": ""}],
        },
    ),
    # Run 3: Multiple tasks, no events
    (
        "Quick follow-ups: 1) Review the marketing proposal by 2026-11-12T12:00:00 (high priority). "
        "2) Update the project timeline by 2026-11-14T17:00:00. 3) Schedule team offsite for Q1.",
        {
            "events": [],
            "todos": [
                {"title": "Review the marketing proposal", "due_date": "2026-11-12T12:00:00", "priority": "high", "description": ""},
                {"title": "Update the project timeline", "due_date": "2026-11-14T17:00:00", "priority": "medium", "description": ""},
                {"title": "Schedule team offsite for Q1", "due_date": "2026-11-30T23:59:59", "priority": "low", "description": ""},
            ],
        },
    ),
]


@pytest.mark.parametrize("email_body,mock_json", EMAIL_PARSE_CASES)
def test_email_ai_parsing(tmp_agenda_db, email_body, mock_json):
    """Flow 3: Parse email with mocked LLM, verify events/tasks are created."""
    from ragnarok.agent.email_scheduler import EmailScheduler

    # Mock the LLM chat result
    mock_result = MagicMock()
    mock_result.content = json.dumps(mock_json)

    with patch("ragnarok.agent.email_scheduler.llm") as mock_llm:
        mock_llm.chat.return_value = mock_result
        scheduler = EmailScheduler(agenda_db=tmp_agenda_db)
        result = scheduler.parse_and_schedule(email_body)

    expected_events = mock_json["events"]
    expected_todos  = mock_json["todos"]

    assert len(result["events_created"]) == len(expected_events), (
        f"Expected {len(expected_events)} event(s), got {len(result['events_created'])}"
    )
    assert len(result["todos_created"]) == len(expected_todos), (
        f"Expected {len(expected_todos)} todo(s), got {len(result['todos_created'])}"
    )
    assert result["parse_errors"] == [], f"Unexpected parse errors: {result['parse_errors']}"

    # Verify persisted in DB
    all_events = tmp_agenda_db.list_events()
    all_tasks  = tmp_agenda_db.list_tasks()

    created_event_ids = {e["id"] for e in result["events_created"]}
    created_task_ids  = {t["id"] for t in result["todos_created"]}

    for eid in created_event_ids:
        assert any(e.id == eid for e in all_events), f"Event {eid} not found in DB"
    for tid in created_task_ids:
        assert any(t.id == tid for t in all_tasks), f"Task {tid} not found in DB"

    # Source should be "email"
    for ev_dict in result["events_created"]:
        assert ev_dict["source"] == "email"
    for td_dict in result["todos_created"]:
        assert td_dict["source"] == "email"


def test_email_parsing_handles_markdown_fences(tmp_agenda_db):
    """EmailScheduler must strip ```json ... ``` fences from LLM output."""
    from ragnarok.agent.email_scheduler import EmailScheduler

    wrapped_json = "```json\n{\"events\": [], \"todos\": [{\"title\": \"Fenced Task\", \"due_date\": \"2026-11-10T09:00:00\", \"priority\": \"low\", \"description\": \"\"}]}\n```"
    mock_result = MagicMock()
    mock_result.content = wrapped_json

    with patch("ragnarok.agent.email_scheduler.llm") as mock_llm:
        mock_llm.chat.return_value = mock_result
        scheduler = EmailScheduler(agenda_db=tmp_agenda_db)
        result = scheduler.parse_and_schedule("Please do fenced task")

    assert len(result["todos_created"]) == 1
    assert result["todos_created"][0]["title"] == "Fenced Task"


def test_email_parsing_handles_llm_error(tmp_agenda_db):
    """EmailScheduler must return structured error dict on LLM failure."""
    from ragnarok.agent.email_scheduler import EmailScheduler

    with patch("ragnarok.agent.email_scheduler.llm") as mock_llm:
        mock_llm.chat.side_effect = RuntimeError("LLM offline")
        scheduler = EmailScheduler(agenda_db=tmp_agenda_db)
        result = scheduler.parse_and_schedule("Meeting tomorrow")

    assert result["events_created"] == []
    assert result["todos_created"] == []
    assert len(result["parse_errors"]) > 0
    assert "LLM error" in result["parse_errors"][0]


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 4 — REST API Endpoints  (3 runs × different data via TestClient)
# ════════════════════════════════════════════════════════════════════════════════

API_EVENT_PAYLOADS = [
    {"title": "API Event Alpha", "start_time": "2026-11-10T10:00:00", "end_time": "2026-11-10T11:00:00", "location": "HQ Lab", "description": "Test run 1", "color": "#00dbe9", "source": "manual"},
    {"title": "API Event Beta",  "start_time": "2026-11-15T14:00:00", "end_time": "2026-11-15T15:30:00", "location": "Virtual",  "description": "Test run 2", "color": "#ff6b6b", "source": "manual"},
    {"title": "API Event Gamma", "start_time": "2026-11-22T09:30:00", "end_time": "2026-11-22T10:30:00", "location": "Studio B",  "description": "Test run 3", "color": "#a855f7", "source": "agent"},
]

API_TODO_PAYLOADS = [
    {"title": "API Task Alpha", "due_date": "2026-11-12T17:00:00", "priority": "high",   "description": "Test task run 1"},
    {"title": "API Task Beta",  "due_date": "2026-11-18T12:00:00", "priority": "urgent", "description": "Test task run 2"},
    {"title": "API Task Gamma", "due_date": "2026-11-25T23:59:00", "priority": "low",    "description": "Test task run 3"},
]


@pytest.mark.parametrize("payload,todo_payload", list(zip(API_EVENT_PAYLOADS, API_TODO_PAYLOADS)))
def test_rest_api_full_flow(test_client, payload, todo_payload):
    """Flow 4: Full REST CRUD for event AND todo via FastAPI TestClient."""
    client, token, _db = test_client
    headers = {"X-Ragnarok-Token": token, "Origin": "http://localhost:8000"}

    # ── Calendar Event: CREATE ────────────────────────────────────────────────
    r = client.post("/api/calendar/events", json=payload, headers=headers)
    assert r.status_code == 200, f"Create event failed: {r.text}"
    event = r.json()["event"]
    event_id = event["id"]
    assert event["title"] == payload["title"]
    assert event["location"] == payload["location"]

    # ── Calendar Event: LIST ──────────────────────────────────────────────────
    r = client.get("/api/calendar/events", headers=headers)
    assert r.status_code == 200
    ids = [e["id"] for e in r.json()["events"]]
    assert event_id in ids

    # ── Calendar Event: LIST with date filter ─────────────────────────────────
    start_date = payload["start_time"][:10] + "T00:00:00"
    end_date   = payload["start_time"][:10] + "T23:59:59"
    r = client.get(f"/api/calendar/events?start={start_date}&end={end_date}", headers=headers)
    assert r.status_code == 200
    assert any(e["id"] == event_id for e in r.json()["events"])

    # ── Calendar Event: UPDATE ────────────────────────────────────────────────
    r = client.put(f"/api/calendar/events/{event_id}", json={"title": "UPDATED_" + payload["title"], "location": "Remote"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["event"]["title"] == "UPDATED_" + payload["title"]

    # ── Calendar Event: DELETE ────────────────────────────────────────────────
    r = client.delete(f"/api/calendar/events/{event_id}", headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"

    # ── Calendar Event: DELETE non-existent → 404 ────────────────────────────
    r = client.delete(f"/api/calendar/events/{event_id}", headers=headers)
    assert r.status_code == 404

    # ── Todo: CREATE ──────────────────────────────────────────────────────────
    r = client.post("/api/todos", json=todo_payload, headers=headers)
    assert r.status_code == 200, f"Create todo failed: {r.text}"
    task = r.json()["task"]
    task_id = task["id"]
    assert task["title"] == todo_payload["title"]
    assert task["priority"] == todo_payload["priority"]
    assert task["status"] == "pending"

    # ── Todo: LIST ────────────────────────────────────────────────────────────
    r = client.get("/api/todos", headers=headers)
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()["tasks"]]
    assert task_id in ids

    # ── Todo: LIST filtered by status ─────────────────────────────────────────
    r = client.get("/api/todos?status=pending", headers=headers)
    assert r.status_code == 200
    assert any(t["id"] == task_id for t in r.json()["tasks"])

    # ── Todo: TOGGLE to completed ─────────────────────────────────────────────
    r = client.post(f"/api/todos/{task_id}/toggle", headers=headers)
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "completed"

    # ── Todo: TOGGLE back to pending ──────────────────────────────────────────
    r = client.post(f"/api/todos/{task_id}/toggle", headers=headers)
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "pending"

    # ── Todo: UPDATE ──────────────────────────────────────────────────────────
    r = client.put(f"/api/todos/{task_id}", json={"title": "UPDATED_" + todo_payload["title"], "priority": "medium"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["task"]["title"] == "UPDATED_" + todo_payload["title"]
    assert r.json()["task"]["priority"] == "medium"

    # ── Todo: DELETE ──────────────────────────────────────────────────────────
    r = client.delete(f"/api/todos/{task_id}", headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"

    # ── Todo: DELETE non-existent → 404 ───────────────────────────────────────
    r = client.delete(f"/api/todos/{task_id}", headers=headers)
    assert r.status_code == 404


def test_rest_api_email_parse_mocked(test_client):
    """Flow 4 extra: /api/agenda/parse-email with mocked LLM."""
    client, token, _db = test_client
    headers = {"X-Ragnarok-Token": token, "Origin": "http://localhost:8000"}

    mock_json = {
        "events": [{"title": "Board Meeting", "start_time": "2026-11-20T10:00:00", "end_time": "2026-11-20T11:00:00", "location": "Boardroom", "description": ""}],
        "todos":  [{"title": "Prepare agenda", "due_date": "2026-11-19T17:00:00", "priority": "high", "description": ""}],
    }
    mock_result = MagicMock()
    mock_result.content = json.dumps(mock_json)

    with patch("ragnarok.agent.email_scheduler.llm") as mock_llm:
        mock_llm.chat.return_value = mock_result
        r = client.post("/api/agenda/parse-email", json={"email_text": "Board Meeting on Nov 20 at 10 AM. Prepare agenda by Nov 19 5 PM."}, headers=headers)

    assert r.status_code == 200
    data = r.json()
    assert len(data["events_created"]) == 1
    assert len(data["todos_created"]) == 1
    assert data["events_created"][0]["title"] == "Board Meeting"
    assert data["todos_created"][0]["title"] == "Prepare agenda"


def test_rest_api_unauthorized(test_client):
    """REST endpoints must reject requests with wrong/missing token."""
    client, _, _db = test_client
    headers_no_token = {"Origin": "http://localhost:8000"}

    r = client.get("/api/calendar/events", headers=headers_no_token)
    assert r.status_code == 401

    r = client.post("/api/todos", json={"title": "hack", "due_date": "2026-11-10T10:00:00", "priority": "low"}, headers=headers_no_token)
    assert r.status_code == 401


# ════════════════════════════════════════════════════════════════════════════════
#  Agenda Manager Skill  (unit tests)
# ════════════════════════════════════════════════════════════════════════════════

def test_agenda_manager_skill_manifest():
    """Skill manifest must be valid and have required fields."""
    from ragnarok.skills.agenda_manager import MANIFEST
    assert MANIFEST.name == "agenda_manager"
    assert "action" in MANIFEST.parameters
    assert "action" in MANIFEST.required
    assert MANIFEST.risk_level == "low"


@pytest.mark.parametrize("action,details,expected_contains", [
    (
        "create_event",
        json.dumps({"title": "Skill Test Event", "start_time": "2026-11-10T09:00:00", "end_time": "2026-11-10T10:00:00"}),
        "Skill Test Event",
    ),
    (
        "create_todo",
        json.dumps({"title": "Skill Test Task", "due_date": "2026-11-15T17:00:00", "priority": "high"}),
        "Skill Test Task",
    ),
    (
        "list_events",
        "",
        "event",
    ),
])
def test_agenda_manager_skill_run(tmp_path, action, details, expected_contains):
    """Skill run() must execute actions and return readable strings."""
    import ragnarok.memory.agenda as agenda_mod
    from ragnarok.memory.agenda import AgendaDB

    test_db = AgendaDB(db_path=tmp_path / "skill_test.db")
    original = agenda_mod._agenda_db
    agenda_mod._agenda_db = test_db

    try:
        from ragnarok.skills.agenda_manager import run
        result = run({"action": action, "details": details})
        assert isinstance(result, str)
        assert expected_contains.lower() in result.lower(), (
            f"Expected '{expected_contains}' in result. Got: {result}"
        )
    finally:
        agenda_mod._agenda_db = original
        test_db.close()


def test_agenda_manager_unknown_action():
    """Skill must return an error string for unknown actions."""
    from ragnarok.skills.agenda_manager import run
    result = run({"action": "fly_to_moon", "details": ""})
    assert "Unknown action" in result or "unknown" in result.lower() or "❌" in result


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 5 — Date & Time Watcher Queries  (3 distinct runs)
# ════════════════════════════════════════════════════════════════════════════════

WATCHER_DATE_TIME_CASES = [
    # Run 1: Morning slot on Day 1
    (
        "2026-11-10T06:00:00",
        "2026-11-10T12:00:00",
        {"title": "Morning Sync", "due_date": "2026-11-10T09:00:00", "priority": "high"},
        {"title": "Evening Code Review", "due_date": "2026-11-10T19:00:00", "priority": "low"},
    ),
    # Run 2: Afternoon slot on Day 2
    (
        "2026-11-15T12:00:00",
        "2026-11-15T18:00:00",
        {"title": "Afternoon Demo", "due_date": "2026-11-15T14:30:00", "priority": "urgent"},
        {"title": "Next Day Task", "due_date": "2026-11-16T10:00:00", "priority": "medium"},
    ),
    # Run 3: Evening slot on Day 3
    (
        "2026-11-20T18:00:00",
        "2026-11-20T23:59:59",
        {"title": "Late Night Deploy", "due_date": "2026-11-20T22:00:00", "priority": "urgent"},
        {"title": "Morning Coffee", "due_date": "2026-11-20T08:00:00", "priority": "low"},
    ),
]


@pytest.mark.parametrize("start_iso,end_iso,matching_task,non_matching_task", WATCHER_DATE_TIME_CASES)
def test_todo_date_time_range_filter(tmp_agenda_db, start_iso, end_iso, matching_task, non_matching_task):
    """Test date/time filtering for to-do tasks (used by Date & Time Watcher)."""
    db = tmp_agenda_db

    t_in = db.create_task(
        title=matching_task["title"],
        due_date=matching_task["due_date"],
        priority=matching_task["priority"],
    )
    t_out = db.create_task(
        title=non_matching_task["title"],
        due_date=non_matching_task["due_date"],
        priority=non_matching_task["priority"],
    )

    filtered = db.list_tasks(start=start_iso, end=end_iso)
    filtered_ids = [t.id for t in filtered]

    assert t_in.id in filtered_ids, f"Matching task {t_in.title} should be in time range {start_iso} - {end_iso}"
    assert t_out.id not in filtered_ids, f"Non-matching task {t_out.title} should NOT be in time range {start_iso} - {end_iso}"


@pytest.mark.parametrize("start_iso,end_iso,matching_task,non_matching_task", WATCHER_DATE_TIME_CASES)
def test_rest_api_todos_date_time_filter(test_client, start_iso, end_iso, matching_task, non_matching_task):
    """Test REST endpoint /api/todos with start and end query params."""
    client, token, _db = test_client
    headers = {"X-Ragnarok-Token": token, "Origin": "http://localhost:8000"}

    # Create both tasks via API
    r1 = client.post("/api/todos", json=matching_task, headers=headers)
    assert r1.status_code == 200
    in_id = r1.json()["task"]["id"]

    r2 = client.post("/api/todos", json=non_matching_task, headers=headers)
    assert r2.status_code == 200
    out_id = r2.json()["task"]["id"]

    # Query with date/time range
    r = client.get(f"/api/todos?start={start_iso}&end={end_iso}", headers=headers)
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()["tasks"]]

    assert in_id in ids
    assert out_id not in ids

