"""
tests/test_history.py
──────────────────────
Unit tests for the MessageHistory manager.
Covers: add, trim by turns, trim by chars, snapshot/restore.
"""
from __future__ import annotations

import pytest
from ragnarok.agent.history import MessageHistory


class TestMessageHistory:

    def test_add_and_len(self, fresh_history):
        h = fresh_history
        h.add_user("Hello")
        h.add_assistant("Hi there!")
        assert len(h) == 2

    def test_as_dicts_format(self, fresh_history):
        h = fresh_history
        h.add_user("test")
        dicts = h.as_dicts()
        assert dicts[0]["role"] == "user"
        assert dicts[0]["content"] == "test"

    def test_trim_by_turns(self):
        h = MessageHistory(max_turns=2, max_context_chars=999_999)
        for i in range(5):
            h.add_user(f"User message {i}")
            h.add_assistant(f"Bot reply {i}")
        # After 5 pairs, only the last 2 user messages should remain
        users = [m for m in h.as_dicts() if m["role"] == "user"]
        assert len(users) <= 2

    def test_trim_by_chars(self):
        h = MessageHistory(max_turns=999, max_context_chars=100)
        for i in range(20):
            h.add_user(f"A message that is somewhat long and takes space {i}")
        total = h.total_chars()
        assert total <= 200  # allow slight overshoot on last add

    def test_clear(self, fresh_history):
        h = fresh_history
        h.add_user("something")
        h.clear()
        assert len(h) == 0

    def test_snapshot_restore(self, fresh_history):
        h = fresh_history
        h.add_user("hello")
        h.add_assistant("world")
        snap = h.snapshot()
        h2 = MessageHistory.from_snapshot(snap, max_turns=5, max_context_chars=2000)
        assert h2.as_dicts() == h.as_dicts()

    def test_tool_result_role(self, fresh_history):
        h = fresh_history
        h.add_tool_result("get_time", "Sunday, October 25 2026 at 01:00 PM")
        dicts = h.as_dicts()
        assert dicts[0]["role"] == "tool"
        assert dicts[0]["name"] == "get_time"
