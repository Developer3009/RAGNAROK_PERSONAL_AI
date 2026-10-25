"""
tests/conftest.py
──────────────────
Shared fixtures for all Ragnarok tests.
"""
from __future__ import annotations

import pytest

from ragnarok.agent.history import MessageHistory
from ragnarok.skills.registry import SkillRegistry


@pytest.fixture
def fresh_history():
    """A new MessageHistory for each test."""
    return MessageHistory(max_turns=5, max_context_chars=2000)


@pytest.fixture
def loaded_registry():
    """SkillRegistry with all built-in skills loaded."""
    reg = SkillRegistry()
    reg.load_all()
    return reg
