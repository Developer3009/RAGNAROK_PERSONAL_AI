"""
ragnarok/skills/agenda_manager.py
──────────────────────────────────
Ragnarok skill: Agenda Manager

Provides agent/voice access to the AgendaDB (calendar events + to-do tasks)
and the AI email scheduler. Follows the standard SkillManifest + run() pattern.

Actions:
  create_event    — create a new calendar event (details: JSON string)
  list_events     — list upcoming events (optional details: {"start": ..., "end": ...})
  update_event    — update an event (details: JSON with "id" + fields to change)
  delete_event    — delete an event (details: {"id": "..."})
  create_todo     — create a new to-do task (details: JSON string)
  list_todos      — list tasks (details: optional {"status": "pending|completed|..."})
  toggle_todo     — toggle task completion (details: {"id": "..."})
  delete_todo     — delete a task (details: {"id": "..."})
  schedule_email  — parse raw email text and auto-schedule (details: email body string)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from ragnarok.memory.agenda import get_agenda_db
from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

MANIFEST = SkillManifest(
    name="agenda_manager",
    description=(
        "Manage calendar events and to-do tasks. "
        "Create, list, update, complete, or delete events and tasks. "
        "Also schedule items automatically from raw email text."
    ),
    risk_level="low",
    parameters={
        "action": {
            "type": "string",
            "description": "Operation to perform",
            "enum": [
                "create_event",
                "list_events",
                "update_event",
                "delete_event",
                "create_todo",
                "list_todos",
                "toggle_todo",
                "delete_todo",
                "schedule_email",
            ],
        },
        "details": {
            "type": "string",
            "description": (
                "JSON string with event/task fields for mutations, "
                "or raw email text for schedule_email action. "
                "For list_events: optional JSON with 'start' and/or 'end' ISO dates. "
                "For list_todos: optional JSON with 'status' filter."
            ),
        },
    },
    required=["action"],
)


def run(args: dict) -> str:  # noqa: PLR0912
    """Execute an agenda management action and return a human-readable result."""
    action: str = args.get("action", "").strip()
    details_raw: str = args.get("details", "") or ""

    db = get_agenda_db()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_details() -> dict:
        if not details_raw.strip():
            return {}
        try:
            return json.loads(details_raw)
        except json.JSONDecodeError:
            return {}

    # ── Dispatch ──────────────────────────────────────────────────────────────

    try:
        if action == "create_event":
            d = _parse_details()
            ev = db.create_event(
                title=d.get("title", "New Event"),
                start_time=d.get("start_time", ""),
                end_time=d.get("end_time", ""),
                description=d.get("description", ""),
                location=d.get("location", ""),
                color=d.get("color", "#00dbe9"),
                source=d.get("source", "agent"),
            )
            return f"✅ Event created: '{ev.title}' (id={ev.id}) at {ev.start_time}"

        elif action == "list_events":
            d = _parse_details()
            events = db.list_events(
                start=d.get("start"),
                end=d.get("end"),
                limit=d.get("limit", 50),
            )
            if not events:
                return "📅 No events found."
            lines = ["📅 Upcoming events:"]
            for ev in events:
                lines.append(f"  • [{ev.id[:8]}] {ev.title} — {ev.start_time} → {ev.end_time}")
            return "\n".join(lines)

        elif action == "update_event":
            d = _parse_details()
            event_id = d.pop("id", None)
            if not event_id:
                return "❌ 'id' is required for update_event."
            ev = db.update_event(event_id, **d)
            if ev is None:
                return f"❌ Event '{event_id}' not found."
            return f"✅ Event updated: '{ev.title}' (id={ev.id})"

        elif action == "delete_event":
            d = _parse_details()
            event_id = d.get("id", details_raw.strip())
            ok = db.delete_event(event_id)
            if not ok:
                return f"❌ Event '{event_id}' not found."
            return f"🗑️ Event '{event_id}' deleted."

        elif action == "create_todo":
            d = _parse_details()
            task = db.create_task(
                title=d.get("title", "New Task"),
                due_date=d.get("due_date", ""),
                priority=d.get("priority", "medium"),
                description=d.get("description", ""),
                scheduled_time=d.get("scheduled_time", ""),
                tags=d.get("tags", []),
                source=d.get("source", "agent"),
            )
            return f"✅ Task created: '{task.title}' [{task.priority}] due {task.due_date} (id={task.id})"

        elif action == "list_todos":
            d = _parse_details()
            tasks = db.list_tasks(
                status=d.get("status"),
                priority=d.get("priority"),
                start=d.get("start"),
                end=d.get("end"),
                limit=d.get("limit", 50),
            )
            if not tasks:
                return "✅ No tasks found."
            lines = ["📋 To-do tasks:"]
            for t in tasks:
                status_icon = "✅" if t.status == "completed" else "⏳"
                lines.append(f"  {status_icon} [{t.id[:8]}] {t.title} [{t.priority}] — due {t.due_date}")
            return "\n".join(lines)

        elif action == "toggle_todo":
            d = _parse_details()
            task_id = d.get("id", details_raw.strip())
            task = db.toggle_task(task_id)
            if task is None:
                return f"❌ Task '{task_id}' not found."
            icon = "✅" if task.status == "completed" else "⏳"
            return f"{icon} Task '{task.title}' is now {task.status}."

        elif action == "delete_todo":
            d = _parse_details()
            task_id = d.get("id", details_raw.strip())
            ok = db.delete_task(task_id)
            if not ok:
                return f"❌ Task '{task_id}' not found."
            return f"🗑️ Task '{task_id}' deleted."

        elif action == "schedule_email":
            # Import here to avoid circular import at module load time
            from ragnarok.agent.email_scheduler import EmailScheduler
            scheduler = EmailScheduler(agenda_db=db)
            result = scheduler.parse_and_schedule(details_raw)
            ev_count = len(result["events_created"])
            td_count = len(result["todos_created"])
            errs = result["parse_errors"]
            msg = f"📧 Email parsed: {ev_count} event(s), {td_count} task(s) scheduled."
            if errs:
                msg += f" Warnings: {'; '.join(errs)}"
            return msg

        else:
            return (
                f"❌ Unknown action '{action}'. "
                "Valid actions: create_event, list_events, update_event, delete_event, "
                "create_todo, list_todos, toggle_todo, delete_todo, schedule_email."
            )

    except Exception as exc:  # noqa: BLE001
        logger.exception("agenda_manager skill error (action=%s): %s", action, exc)
        return f"❌ Agenda error: {exc}"
