"""
ragnarok/agent/email_scheduler.py
──────────────────────────────────
AI-powered email parser that extracts calendar events and to-do tasks
from raw email text using the local Ollama LLM (100% offline).

Flow:
  1. Inject current datetime into prompt for relative-date resolution
  2. Call llm.chat() with strict JSON schema prompt
  3. Extract .content from ChatResult
  4. Parse JSON (handle markdown code fences gracefully)
  5. Create CalendarEvent / TodoTask objects via AgendaDB
  6. Return a summary dict with counts and created IDs
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

from ragnarok import llm
from ragnarok.memory.agenda import AgendaDB, CalendarEvent, TodoTask, get_agenda_db

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

_PARSE_PROMPT_TEMPLATE = """You are a calendar assistant. Extract all calendar events and to-do tasks from the email text below.

Current datetime: {datetime_now}

Return ONLY valid JSON (no markdown, no explanation, no code fences):
{{
  "events": [
    {{
      "title": "...",
      "start_time": "YYYY-MM-DDTHH:MM:SS",
      "end_time": "YYYY-MM-DDTHH:MM:SS",
      "location": "...",
      "description": "..."
    }}
  ],
  "todos": [
    {{
      "title": "...",
      "due_date": "YYYY-MM-DDTHH:MM:SS",
      "priority": "low|medium|high|urgent",
      "description": "..."
    }}
  ]
}}

Rules:
- For relative dates like "tomorrow", "next Friday", resolve them using the current datetime above.
- If no end_time is specified, add 1 hour to start_time.
- If no due date is specified for a todo, use end of current day (23:59:59).
- If no events or todos found, return empty arrays.
- Dates MUST be in ISO 8601 format: YYYY-MM-DDTHH:MM:SS

EMAIL TEXT:
{email_text}
"""

# ── EmailScheduler ────────────────────────────────────────────────────────────

class EmailScheduler:
    """
    Parses raw email text with a local LLM to extract and persist
    calendar events and to-do tasks into AgendaDB.
    """

    def __init__(self, agenda_db: AgendaDB | None = None) -> None:
        self._db = agenda_db or get_agenda_db()

    def parse_and_schedule(self, email_text: str) -> dict[str, Any]:
        """
        Parse *email_text* with the local LLM, create extracted items in AgendaDB,
        and return a summary dict with created event/task details.

        Returns:
            {
              "events_created": [...],
              "todos_created":  [...],
              "parse_errors":   [...],
              "raw_llm_output": "..."
            }
        """
        datetime_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        prompt = _PARSE_PROMPT_TEMPLATE.format(
            datetime_now=datetime_now,
            email_text=email_text.strip(),
        )

        raw_output = ""
        parse_errors: list[str] = []

        try:
            result = llm.chat(prompt)
            # llm.chat() returns a ChatResult object — use .content attribute
            raw_output = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            logger.error("LLM chat failed during email parse: %s", exc)
            return {
                "events_created": [],
                "todos_created": [],
                "parse_errors": [f"LLM error: {exc}"],
                "raw_llm_output": "",
            }

        # Parse JSON — strip markdown code fences if present
        json_str = self._extract_json(raw_output)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed: %s\nRaw output: %s", exc, raw_output)
            return {
                "events_created": [],
                "todos_created": [],
                "parse_errors": [f"JSON parse error: {exc}"],
                "raw_llm_output": raw_output,
            }

        events_created: list[dict[str, Any]] = []
        todos_created: list[dict[str, Any]] = []

        # Create calendar events
        for ev_data in parsed.get("events", []):
            try:
                ev = self._db.create_event(
                    title=ev_data.get("title", "Untitled Event"),
                    start_time=ev_data.get("start_time", datetime_now),
                    end_time=ev_data.get("end_time", datetime_now),
                    description=ev_data.get("description", ""),
                    location=ev_data.get("location", ""),
                    source="email",
                )
                events_created.append(asdict(ev))
            except Exception as exc:
                parse_errors.append(f"Event creation failed: {exc}")
                logger.warning("Failed to create event from %s: %s", ev_data, exc)

        # Create to-do tasks
        for task_data in parsed.get("todos", []):
            try:
                task = self._db.create_task(
                    title=task_data.get("title", "Untitled Task"),
                    due_date=task_data.get("due_date", datetime_now),
                    priority=task_data.get("priority", "medium"),
                    description=task_data.get("description", ""),
                    source="email",
                )
                todos_created.append(asdict(task))
            except Exception as exc:
                parse_errors.append(f"Task creation failed: {exc}")
                logger.warning("Failed to create task from %s: %s", task_data, exc)

        logger.info(
            "Email parse complete: %d events, %d todos created",
            len(events_created),
            len(todos_created),
        )
        return {
            "events_created": events_created,
            "todos_created": todos_created,
            "parse_errors": parse_errors,
            "raw_llm_output": raw_output,
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences and extract the JSON portion."""
        # Remove ```json ... ``` or ``` ... ``` fences
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            return fenced.group(1).strip()
        # Find the first { ... } block
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return text[brace_start : brace_end + 1]
        return text.strip()
