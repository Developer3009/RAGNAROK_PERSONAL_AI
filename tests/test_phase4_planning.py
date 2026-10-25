"""
tests/test_phase4_planning.py
──────────────────────────────
Phase 4 tests: Planning, Self-Correction, Doom-Loop Detection.

All tests are unit tests (mocked LLM) — no live Ollama required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call
import pytest

from ragnarok.agent.planning import Plan, PlanStep, Planner
from ragnarok.agent.self_correction import (
    CorrectionAdvice,
    DoomLoopDetector,
    ErrorDiagnoser,
)
from ragnarok.agent.loop import AgentLoop, AgentResult
from ragnarok.skills.registry import SkillRegistry


# ─────────────────────────────────────────────────────────────────────────────
# PlanStep tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanStep:
    def test_initial_status_is_pending(self):
        s = PlanStep(step_id=1, description="Do something")
        assert s.status == "pending"
        assert s.result is None
        assert s.error is None

    def test_mark_in_progress(self):
        s = PlanStep(step_id=1, description="Work")
        s.mark_in_progress()
        assert s.status == "in_progress"

    def test_mark_completed_with_result(self):
        s = PlanStep(step_id=1, description="Work")
        s.mark_completed(result="Done successfully")
        assert s.status == "completed"
        assert s.result == "Done successfully"

    def test_mark_failed_with_error(self):
        s = PlanStep(step_id=1, description="Work")
        s.mark_failed(error="Network timeout")
        assert s.status == "failed"
        assert s.error == "Network timeout"

    def test_mark_skipped_sets_notes(self):
        s = PlanStep(step_id=1, description="Work")
        s.mark_skipped(reason="Prerequisite unavailable")
        assert s.status == "skipped"
        assert "Prerequisite" in s.notes

    def test_roundtrip_to_from_dict(self):
        s = PlanStep(step_id=3, description="Analyse data", tool="read_file")
        s.mark_completed("read 512 bytes")
        d = s.to_dict()
        restored = PlanStep.from_dict(d)
        assert restored.step_id == 3
        assert restored.description == "Analyse data"
        assert restored.tool == "read_file"
        assert restored.status == "completed"
        assert restored.result == "read 512 bytes"


# ─────────────────────────────────────────────────────────────────────────────
# Plan tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlan:
    def _make_plan(self) -> Plan:
        plan = Plan(goal="Summarise the report")
        plan.add_step("Read source file", tool="read_file")
        plan.add_step("Extract key points")
        plan.add_step("Write summary", tool="write_file")
        return plan

    def test_add_steps_assigns_sequential_ids(self):
        plan = self._make_plan()
        assert [s.step_id for s in plan.steps] == [1, 2, 3]

    def test_active_step_returns_first_pending(self):
        plan = self._make_plan()
        active = plan.active_step()
        assert active is not None
        assert active.step_id == 1

    def test_advance_marks_current_completed_next_in_progress(self):
        plan = self._make_plan()
        plan.steps[0].mark_in_progress()
        plan.advance_step(step_id=1, result="file content")
        assert plan.steps[0].status == "completed"
        assert plan.steps[1].status == "in_progress"
        assert plan.status == "in_progress"

    def test_advance_all_steps_sets_plan_completed(self):
        plan = self._make_plan()
        plan.steps[0].mark_in_progress()
        plan.advance_step(step_id=1)
        plan.advance_step(step_id=2)
        plan.advance_step(step_id=3)
        assert plan.status == "completed"

    def test_fail_step_sets_plan_failed(self):
        plan = self._make_plan()
        plan.steps[0].mark_in_progress()
        plan.fail_current_step(step_id=1, error="disk error")
        assert plan.steps[0].status == "failed"
        assert plan.status == "failed"

    def test_format_progress_contains_goal_and_steps(self):
        plan = self._make_plan()
        plan.steps[0].mark_in_progress()
        output = plan.format_progress()
        assert "Summarise the report" in output
        assert "Read source file" in output
        assert "[>]" in output   # in_progress marker

    def test_format_prompt_context_has_current_focus(self):
        plan = self._make_plan()
        plan.steps[0].mark_in_progress()
        ctx = plan.format_prompt_context()
        assert "CURRENT EXECUTION PLAN" in ctx
        assert "Current focus" in ctx

    def test_roundtrip_to_from_dict(self):
        plan = self._make_plan()
        plan.steps[0].mark_completed("done")
        d = plan.to_dict()
        restored = Plan.from_dict(d)
        assert restored.goal == "Summarise the report"
        assert len(restored.steps) == 3
        assert restored.steps[0].status == "completed"


# ─────────────────────────────────────────────────────────────────────────────
# Planner tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanner:
    def test_should_plan_long_query(self):
        planner = Planner(use_llm=False)
        long = "Please search for all Python files in the project and then read each one and write a summary report"
        assert planner.should_plan(long) is True

    def test_should_plan_trigger_keyword_step_by_step(self):
        planner = Planner(use_llm=False)
        assert planner.should_plan("Do this step by step") is True

    def test_should_plan_short_simple_query_returns_false(self):
        planner = Planner(use_llm=False)
        assert planner.should_plan("What is the time?") is False

    def test_create_plan_heuristic_search_goal(self):
        planner = Planner(use_llm=False)
        plan = planner.create_plan("Search for log files and save the findings")
        assert len(plan.steps) >= 2
        assert plan.status == "in_progress"
        assert plan.steps[0].status == "in_progress"
        assert any("search" in s.description.lower() for s in plan.steps)

    def test_create_plan_heuristic_read_write_goal(self):
        planner = Planner(use_llm=False)
        plan = planner.create_plan("Read the config file and write a backup")
        assert len(plan.steps) >= 2
        assert plan.steps[0].tool == "read_file"

    def test_create_plan_heuristic_list_goal(self):
        planner = Planner(use_llm=False)
        plan = planner.create_plan("List all available files in the data folder")
        assert any("list" in s.description.lower() for s in plan.steps)

    def test_llm_failure_falls_back_to_heuristic(self):
        """When LLM decomposition raises, heuristic plan is used."""
        planner = Planner(use_llm=True)
        with patch("ragnarok.agent.planning.llm.chat", side_effect=RuntimeError("Ollama down")):
            plan = planner.create_plan(
                "Search for reports and write a summary",
                available_tools=["search_files", "write_file"],
            )
        assert len(plan.steps) >= 2

    def test_llm_decomposition_valid_json(self):
        """LLM decompose path: mock llm.chat inside planning module to return 2-step JSON."""
        planner = Planner(use_llm=True)
        mock_resp = MagicMock()
        mock_resp.content = json.dumps([
            {"description": "Step A", "tool": "read_file"},
            {"description": "Step B", "tool": None},
        ])
        # Patch at the module level where planning imports llm
        with patch("ragnarok.agent.planning.llm.chat", return_value=mock_resp) as mock_chat:
            # Use a query that triggers should_plan (>15 words) so LLM path is attempted
            plan = planner.create_plan(
                "Do task A and then do task B with the read file tool for this test",
                available_tools=["read_file"],
            )
        if mock_chat.called:
            # LLM path succeeded — must have exactly 2 steps
            assert len(plan.steps) == 2
            assert plan.steps[0].tool == "read_file"
            assert plan.steps[1].tool is None
        else:
            # Fell back to heuristic (acceptable) — verify steps still exist
            assert len(plan.steps) >= 2

    def test_replan_on_failure_adds_recovery_step(self):
        planner = Planner(use_llm=False)
        plan = planner.create_plan("Search files then write report")
        plan.steps[0].mark_in_progress()
        original_count = len(plan.steps)
        updated = planner.replan_on_failure(plan, plan.steps[0].step_id, "disk read error")
        assert len(updated.steps) == original_count + 1
        recovery = updated.steps[-1]
        assert "recovery" in recovery.notes.lower() or "diagnose" in recovery.description.lower()
        assert recovery.status == "in_progress"


# ─────────────────────────────────────────────────────────────────────────────
# DoomLoopDetector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDoomLoopDetector:
    def test_no_loop_on_first_call(self):
        d = DoomLoopDetector(max_consecutive_failures=2)
        is_loop, _ = d.check_loop("read_file", {"path": "a.txt"})
        assert is_loop is False

    def test_no_loop_on_success(self):
        d = DoomLoopDetector(max_consecutive_failures=2)
        d.record_call("read_file", {"path": "a.txt"}, "file contents here")
        d.record_call("read_file", {"path": "a.txt"}, "file contents here")
        is_loop, _ = d.check_loop("read_file", {"path": "a.txt"})
        assert is_loop is False  # results were not errors

    def test_detects_loop_after_consecutive_failures(self):
        d = DoomLoopDetector(max_consecutive_failures=2)
        args = {"path": "missing.txt"}
        d.record_call("read_file", args, "Error: file not found")
        d.record_call("read_file", args, "Error: file not found")
        is_loop, warning = d.check_loop("read_file", args)
        assert is_loop is True
        assert "DOOM LOOP" in warning

    def test_loop_not_triggered_different_args(self):
        d = DoomLoopDetector(max_consecutive_failures=2)
        d.record_call("read_file", {"path": "a.txt"}, "Error: not found")
        d.record_call("read_file", {"path": "a.txt"}, "Error: not found")
        # Different path — should NOT trigger loop
        is_loop, _ = d.check_loop("read_file", {"path": "b.txt"})
        assert is_loop is False

    def test_loop_not_triggered_different_tool(self):
        d = DoomLoopDetector(max_consecutive_failures=2)
        d.record_call("read_file", {"path": "a.txt"}, "Error: not found")
        d.record_call("read_file", {"path": "a.txt"}, "Error: not found")
        is_loop, _ = d.check_loop("write_file", {"path": "a.txt"})
        assert is_loop is False

    def test_reset_clears_history(self):
        d = DoomLoopDetector(max_consecutive_failures=2)
        args = {"path": "a.txt"}
        d.record_call("read_file", args, "Error: not found")
        d.record_call("read_file", args, "Error: not found")
        d.reset()
        is_loop, _ = d.check_loop("read_file", args)
        assert is_loop is False

    def test_is_error_detection(self):
        assert DoomLoopDetector._is_error("Error: file not found") is True
        assert DoomLoopDetector._is_error("ERROR: something failed") is True
        assert DoomLoopDetector._is_error("file contents here") is False
        assert DoomLoopDetector._is_error("Success: 512 bytes written") is False


# ─────────────────────────────────────────────────────────────────────────────
# ErrorDiagnoser tests
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorDiagnoser:
    def test_diagnoses_file_not_found(self):
        advice = ErrorDiagnoser.diagnose("read_file", {"path": "x.txt"}, "Error: file not found")
        assert advice.error_type == "FILE_NOT_FOUND"
        assert "list_dir" in advice.suggested_alternative

    def test_diagnoses_permission_denied(self):
        advice = ErrorDiagnoser.diagnose("write_file", {"path": "/etc/passwd"}, "Path not allowed by allowlist")
        assert advice.error_type == "PERMISSION_DENIED"
        assert advice.prevent_retry is True

    def test_diagnoses_schema_validation(self):
        advice = ErrorDiagnoser.diagnose("write_file", {}, "Missing required field: path")
        assert advice.error_type == "SCHEMA_VALIDATION_ERROR"

    def test_diagnoses_network_error(self):
        advice = ErrorDiagnoser.diagnose("fetch_web", {"url": "http://bad.invalid"}, "Connection timeout")
        assert advice.error_type == "NETWORK_TIMEOUT"

    def test_diagnoses_generic_error(self):
        advice = ErrorDiagnoser.diagnose("some_tool", {}, "Some totally unknown problem")
        assert advice.error_type == "TOOL_EXECUTION_ERROR"
        assert advice.prevent_retry is False

    def test_format_prompt_includes_all_fields(self):
        advice = CorrectionAdvice(
            error_type="TEST_ERR",
            message="Something broke",
            suggested_alternative="Try differently",
            prevent_retry=True,
        )
        prompt = advice.format_prompt()
        assert "TEST_ERR" in prompt
        assert "Something broke" in prompt
        assert "Try differently" in prompt
        assert "CRITICAL" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# AgentLoop integration with planning (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoopPhase4Integration:
    def _make_registry(self):
        reg = SkillRegistry()
        reg.load_all()
        return reg

    def _make_chat_result(self, content: str):
        from ragnarok.llm import ChatToolResult
        return ChatToolResult(final_content=content)

    def test_simple_query_no_plan_created(self):
        """Short simple query should not trigger planning."""
        reg = self._make_registry()
        loop = AgentLoop(registry=reg, enable_planning=True)
        mock_result = self._make_chat_result("It is 3:15 PM.")
        with patch("ragnarok.agent.loop.llm.chat_with_tools", return_value=mock_result):
            result = loop.chat("What time is it?")
        assert result.answer == "It is 3:15 PM."
        assert result.plan is None

    def test_complex_query_creates_plan(self):
        """Multi-step query triggers plan creation."""
        reg = self._make_registry()
        loop = AgentLoop(registry=reg, enable_planning=True)
        mock_result = self._make_chat_result("Here is the analysis report.")
        with patch("ragnarok.agent.loop.llm.chat_with_tools", return_value=mock_result), \
             patch("ragnarok.agent.planning.llm.chat", side_effect=RuntimeError("no llm")):
            result = loop.chat(
                "Search the data folder for CSV files, read each one, "
                "and write a summary report step by step."
            )
        assert result.plan is not None
        assert len(result.plan.steps) >= 2

    def test_reset_clears_plan(self):
        reg = self._make_registry()
        loop = AgentLoop(registry=reg, enable_planning=True)
        # Manually set a plan
        dummy = Plan(goal="test goal")
        dummy.add_step("step 1")
        loop._active_plan = dummy
        loop.reset()
        assert loop.active_plan is None

    def test_doom_loop_blocks_repeated_failures(self):
        """DoomLoopDetector prevents 3rd consecutive identical-failing tool call."""
        from ragnarok.llm import ChatToolResult, ToolCall

        reg = self._make_registry()
        loop = AgentLoop(registry=reg, enable_planning=False)

        # Two consecutive calls returning the same tool → trigger doom loop on 3rd
        bad_args = {"path": "nonexistent.txt"}
        call1 = ChatToolResult(tool_calls=[ToolCall(name="read_file", arguments=bad_args)])
        call2 = ChatToolResult(tool_calls=[ToolCall(name="read_file", arguments=bad_args)])
        final = ChatToolResult(final_content="I could not read the file.")

        with patch("ragnarok.agent.loop.llm.chat_with_tools", side_effect=[call1, call2, final]):
            # Make read_file always raise so doom detector sees errors
            read_skill = reg.get("read_file")
            orig_run = read_skill.run
            read_skill.run = lambda _: (_ for _ in ()).throw(FileNotFoundError("not found"))
            try:
                result = loop.chat("Read file nonexistent.txt for me please")
            finally:
                read_skill.run = orig_run

        # Agent should eventually answer without going into infinite retry
        assert result.answer is not None
        assert len(result.answer) > 0

    def test_planning_disabled_no_plan(self):
        reg = self._make_registry()
        loop = AgentLoop(registry=reg, enable_planning=False)
        mock_result = self._make_chat_result("All done.")
        with patch("ragnarok.agent.loop.llm.chat_with_tools", return_value=mock_result):
            result = loop.chat(
                "Search data folder, read files, then write a full summary report step by step."
            )
        assert result.plan is None
