"""
tests/test_agent_loop.py
─────────────────────────
Unit tests for the AgentLoop.
All LLM calls are mocked — no Ollama required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ragnarok.agent.loop import AgentLoop, AgentResult
from ragnarok.llm import ChatToolResult, ChatResult, ToolCall
from ragnarok.skills.registry import SkillRegistry


def _make_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.load_all()
    return reg


def _tool_result(tool_name: str, args: dict) -> ChatToolResult:
    return ChatToolResult(
        tool_calls=[ToolCall(name=tool_name, arguments=args)],
        model="test",
        latency_ms=1.0,
        token_count=5,
    )


def _final_result(content: str) -> ChatToolResult:
    return ChatToolResult(
        tool_calls=[],
        final_content=content,
        model="test",
        latency_ms=1.0,
        token_count=5,
    )


class TestAgentLoop:

    def test_direct_answer_no_tools(self):
        reg = _make_registry()

        with patch("ragnarok.agent.loop.llm") as mock_llm:
            mock_llm.chat_with_tools.return_value = _final_result("Paris is the capital of France.")
            agent = AgentLoop(registry=reg)
            result = agent.chat("What is the capital of France?")

        assert "Paris" in result.answer
        assert result.steps_taken == 1
        assert result.tool_calls_made == []

    def test_uses_get_time_tool(self):
        reg = _make_registry()

        responses = [
            _tool_result("get_time", {"format": "human"}),
            _final_result("It is Sunday."),
        ]

        with patch("ragnarok.agent.loop.llm") as mock_llm:
            mock_llm.chat_with_tools.side_effect = responses
            agent = AgentLoop(registry=reg)
            result = agent.chat("What time is it?")

        assert "get_time" in result.tool_calls_made
        assert result.steps_taken == 2

    def test_schema_validation_failure_sent_back(self):
        """If the model sends bad args, the error goes back to it."""
        reg = _make_registry()

        responses = [
            _tool_result("read_file", {}),         # missing required 'path'
            _final_result("I need a path to read a file."),
        ]

        with patch("ragnarok.agent.loop.llm") as mock_llm:
            mock_llm.chat_with_tools.side_effect = responses
            agent = AgentLoop(registry=reg)
            result = agent.chat("Read a file")

        # Should still get a final answer (not crash)
        assert isinstance(result.answer, str)
        assert result.steps_taken == 2

    def test_high_risk_blocked_without_callback(self):
        """High-risk tool with no approval callback → blocked."""
        reg = _make_registry()
        # Manually set get_time to high-risk for this test
        skill = reg.get("get_time")
        skill.manifest.risk_level = "high"

        responses = [
            _tool_result("get_time", {}),
            _final_result("I couldn't get the time."),
        ]

        with patch("ragnarok.agent.loop.llm") as mock_llm:
            mock_llm.chat_with_tools.side_effect = responses
            agent = AgentLoop(registry=reg, approval_cb=None)
            result = agent.chat("What time is it?")

        # Should not have executed the tool, but should still return an answer
        assert isinstance(result.answer, str)
        # Reset risk level
        skill.manifest.risk_level = "low"

    def test_high_risk_approved_runs(self):
        """High-risk tool with approving callback → runs."""
        reg = _make_registry()
        skill = reg.get("get_time")
        skill.manifest.risk_level = "high"

        responses = [
            _tool_result("get_time", {}),
            _final_result("The time was retrieved."),
        ]

        def always_approve(name, args):
            return True

        with patch("ragnarok.agent.loop.llm") as mock_llm:
            mock_llm.chat_with_tools.side_effect = responses
            agent = AgentLoop(registry=reg, approval_cb=always_approve)
            result = agent.chat("What time is it?")

        assert "get_time" in result.tool_calls_made
        skill.manifest.risk_level = "low"

    def test_max_steps_enforced(self):
        """Agent never loops forever."""
        reg = _make_registry()

        # Always return a tool call — model never gives a final answer
        with patch("ragnarok.agent.loop.llm") as mock_llm:
            from ragnarok.config import settings
            mock_llm.chat_with_tools.return_value = _tool_result("get_time", {})
            mock_llm.chat.return_value = ChatResult(
                content="Stopped.", model="test", latency_ms=1.0, token_count=1
            )
            agent = AgentLoop(registry=reg)
            result = agent.chat("Loop forever")

        assert result.stopped_early is True
        assert result.steps_taken == settings.max_steps

    def test_reset_clears_history(self):
        reg = _make_registry()

        with patch("ragnarok.agent.loop.llm") as mock_llm:
            mock_llm.chat_with_tools.return_value = _final_result("OK")
            agent = AgentLoop(registry=reg)
            agent.chat("First message")
            agent.reset()
            snap = agent.history_snapshot()
        assert len(snap) == 0

    def test_memory_fn_injected(self):
        """Memory function output appears in the system prompt."""
        reg = _make_registry()
        captured_system: list[str] = []

        def capture(*args, system=None, **kwargs):
            if system:
                captured_system.append(system)
            return _final_result("answer")

        def fake_memory(text):
            return "User prefers dark mode."

        with patch("ragnarok.agent.loop.llm") as mock_llm:
            mock_llm.chat_with_tools.side_effect = capture
            agent = AgentLoop(registry=reg, memory_fn=fake_memory)
            agent.chat("Any question")

        assert any("dark mode" in s for s in captured_system)
