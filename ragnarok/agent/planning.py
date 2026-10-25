"""
ragnarok/agent/planning.py
──────────────────────────
Task planning and goal decomposition for Ragnarok.

Provides:
  - PlanStep: Individual actionable unit of work with status tracking.
  - Plan: Ordered sequence of steps representing a strategy to achieve a goal.
  - Planner: Heuristic + LLM-assisted task decomposition and replanning.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ragnarok import llm
from ragnarok.config import settings

logger = logging.getLogger(__name__)

StepStatus = Literal["pending", "in_progress", "completed", "failed", "skipped"]
PlanStatus = Literal["created", "in_progress", "completed", "failed"]


@dataclass
class PlanStep:
    step_id: int
    description: str
    status: StepStatus = "pending"
    tool: str | None = None
    result: str | None = None
    error: str | None = None
    notes: str = ""

    def mark_in_progress(self) -> None:
        self.status = "in_progress"

    def mark_completed(self, result: str | None = None) -> None:
        self.status = "completed"
        if result is not None:
            self.result = result

    def mark_failed(self, error: str | None = None) -> None:
        self.status = "failed"
        if error is not None:
            self.error = error

    def mark_skipped(self, reason: str | None = None) -> None:
        self.status = "skipped"
        if reason:
            self.notes = f"Skipped: {reason}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    status: PlanStatus = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_step(self, description: str, tool: str | None = None) -> PlanStep:
        step_id = len(self.steps) + 1
        step = PlanStep(step_id=step_id, description=description, tool=tool)
        self.steps.append(step)
        self.updated_at = time.time()
        return step

    def get_step(self, step_id: int) -> PlanStep | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def active_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status in ("pending", "in_progress"):
                return step
        return None

    def advance_step(self, step_id: int | None = None, result: str | None = None) -> PlanStep | None:
        """Mark step completed and move to next."""
        target = self.get_step(step_id) if step_id is not None else self.active_step()
        if target:
            target.mark_completed(result)
            self.updated_at = time.time()

        next_s = self.active_step()
        if next_s:
            next_s.mark_in_progress()
            self.status = "in_progress"
        else:
            self.status = "completed"
        return next_s

    def fail_current_step(self, step_id: int | None = None, error: str | None = None) -> None:
        target = self.get_step(step_id) if step_id is not None else self.active_step()
        if target:
            target.mark_failed(error)
            self.status = "failed"
            self.updated_at = time.time()

    def format_progress(self) -> str:
        """Format a human-readable summary of plan progress."""
        if not self.steps:
            return f"Plan for: '{self.goal}' (No steps defined)"
        lines = [f"Plan: {self.goal} [{self.status.upper()}]"]
        for s in self.steps:
            mark = {
                "completed": "[x]",
                "in_progress": "[>]",
                "failed": "[!]",
                "skipped": "[-]",
                "pending": "[ ]",
            }.get(s.status, "[ ]")
            tool_info = f" (tool: {s.tool})" if s.tool else ""
            lines.append(f"  {mark} Step {s.step_id}: {s.description}{tool_info}")
        return "\n".join(lines)

    def format_prompt_context(self) -> str:
        """Format concise progress context for system prompt injection."""
        if not self.steps:
            return ""
        lines = ["CURRENT EXECUTION PLAN:"]
        for s in self.steps:
            status_tag = s.status.upper()
            lines.append(f"- Step {s.step_id} [{status_tag}]: {s.description}")
        active = self.active_step()
        if active:
            lines.append(f"Current focus: Step {active.step_id} - {active.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        steps_raw = data.get("steps", [])
        steps = [PlanStep.from_dict(s) for s in steps_raw]
        return cls(
            goal=data.get("goal", ""),
            steps=steps,
            current_step_index=data.get("current_step_index", 0),
            status=data.get("status", "created"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class Planner:
    """
    Decomposes user queries into multi-step execution plans and manages replanning.
    """

    # Trigger keywords indicating a multi-step task
    _COMPLEX_PATTERNS = [
        r"\band\s+then\b",
        r"\bfirst\b.*\bthen\b",
        r"\bstep\s+by\s+step\b",
        r"\bplan\b",
        r"\bworkflow\b",
        r"\bpipeline\b",
        r"\banalyze\s+and\s+write\b",
        r"\bfind\s+and\s+replace\b",
        r"\bsearch.*and\s+save\b",
        r"\bread.*and\s+create\b",
        r"\bcompile\b",
    ]

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    def should_plan(self, user_input: str) -> bool:
        """Determine whether user request warrants explicit planning."""
        text = user_input.strip().lower()
        if len(text.split()) > 15:
            return True
        for pattern in self._COMPLEX_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def create_plan(
        self,
        goal: str,
        available_tools: list[str] | None = None,
    ) -> Plan:
        """
        Create a multi-step plan for a goal.
        Attempts LLM-based decomposition using chat_small; falls back to heuristic.
        """
        plan = Plan(goal=goal)

        if self.use_llm:
            try:
                steps = self._decompose_with_llm(goal, available_tools)
                if steps:
                    for s in steps:
                        plan.add_step(s.get("description", ""), tool=s.get("tool"))
                    if plan.steps:
                        plan.steps[0].mark_in_progress()
                        plan.status = "in_progress"
                        return plan
            except Exception as exc:
                logger.warning("LLM planning decomposition failed: %s; falling back to heuristic", exc)

        # Heuristic fallback
        self._decompose_heuristic(plan, goal, available_tools)
        if plan.steps:
            plan.steps[0].mark_in_progress()
            plan.status = "in_progress"
        return plan

    def replan_on_failure(self, plan: Plan, failed_step_id: int, error_message: str) -> Plan:
        """
        Dynamically adjust plan after a step failure:
        Inserts diagnostic or recovery step before retrying or alternative strategy.
        """
        failed_step = plan.get_step(failed_step_id)
        if not failed_step:
            return plan

        failed_step.mark_failed(error_message)

        # Generate recovery step
        recovery_desc = f"Diagnose/resolve failure in step {failed_step_id}: {error_message[:80]}"
        new_step_id = len(plan.steps) + 1
        recovery_step = PlanStep(
            step_id=new_step_id,
            description=recovery_desc,
            status="in_progress",
            notes=f"Auto-inserted recovery for step {failed_step_id}",
        )
        plan.steps.append(recovery_step)
        plan.status = "in_progress"
        plan.updated_at = time.time()
        logger.info("Replanned: added recovery step %d for failed step %d", new_step_id, failed_step_id)
        return plan

    def _decompose_with_llm(self, goal: str, available_tools: list[str] | None) -> list[dict[str, Any]]:
        tool_str = ", ".join(available_tools) if available_tools else "general local tools"
        prompt = (
            f"You are a task planner. Break the user goal into 2 to 4 concise, sequential steps.\n"
            f"Goal: {goal}\n"
            f"Available tools: {tool_str}\n\n"
            f"Return ONLY valid JSON with this exact structure:\n"
            f'[{{"description": "step description", "tool": "optional_tool_name_or_null"}}]\n'
            f"Do not include explanation or markdown formatting."
        )

        resp = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.model_small,
            temperature=0.0,
        )
        content = resp.content.strip()

        # Clean markdown codeblocks if model wrapped in ```json
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        data = json.loads(content)
        if isinstance(data, list) and len(data) > 0:
            return data
        return []

    def _decompose_heuristic(
        self,
        plan: Plan,
        goal: str,
        available_tools: list[str] | None,
    ) -> None:
        """Rule-based plan decomposition fallback."""
        lowered = goal.lower()
        if "search" in lowered or "find" in lowered:
            plan.add_step("Search for relevant files or patterns", tool="search_files")
            if "read" in lowered or "check" in lowered:
                plan.add_step("Inspect contents of discovered files", tool="read_file")
            if "write" in lowered or "save" in lowered or "note" in lowered:
                plan.add_step("Save results or notes", tool="write_file")
            else:
                plan.add_step("Synthesize findings and present answer to user")
        elif "read" in lowered and ("write" in lowered or "save" in lowered):
            plan.add_step("Read source file", tool="read_file")
            plan.add_step("Process information and write target output", tool="write_file")
        elif "list" in lowered:
            plan.add_step("List directory contents", tool="list_dir")
            plan.add_step("Analyze file structure and summarize")
        else:
            plan.add_step("Analyze request and gather necessary context")
            plan.add_step("Execute primary actions")
            plan.add_step("Verify results and formulate final response")
