"""
tests/test_skills.py
─────────────────────
Unit tests for the skill registry and all built-in skills.
Tests run without Ollama (pure logic — no model calls needed).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ragnarok.skills.registry import SkillRegistry, SkillManifest


# ── Registry tests ─────────────────────────────────────────────────────────────

class TestSkillRegistry:

    def test_loads_builtin_skills(self, loaded_registry):
        assert len(loaded_registry) >= 2  # at least get_time + read_file

    def test_get_known_skill(self, loaded_registry):
        skill = loaded_registry.get("get_time")
        assert skill is not None
        assert skill.manifest.name == "get_time"

    def test_get_unknown_returns_none(self, loaded_registry):
        assert loaded_registry.get("does_not_exist") is None

    def test_as_ollama_tools_format(self, loaded_registry):
        tools = loaded_registry.as_ollama_tools()
        assert isinstance(tools, list)
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]

    def test_validate_call_missing_required(self, loaded_registry):
        valid, msg = loaded_registry.validate_call("read_file", {})
        assert not valid
        assert "path" in msg

    def test_validate_call_unknown_tool(self, loaded_registry):
        valid, msg = loaded_registry.validate_call("nonexistent", {})
        assert not valid
        assert "Unknown" in msg

    def test_validate_call_valid(self, loaded_registry):
        valid, msg = loaded_registry.validate_call("get_time", {"format": "iso"})
        assert valid
        assert msg == ""


# ── get_time tests ─────────────────────────────────────────────────────────────

class TestGetTime:

    def test_returns_string(self, loaded_registry):
        skill = loaded_registry.get("get_time")
        result = skill.run({})
        assert isinstance(result, str)
        assert len(result) > 5

    def test_iso_format(self, loaded_registry):
        skill = loaded_registry.get("get_time")
        result = skill.run({"format": "iso"})
        # ISO-8601: should contain 'T' and '-'
        assert "T" in result or "-" in result

    def test_human_format(self, loaded_registry):
        skill = loaded_registry.get("get_time")
        result = skill.run({"format": "human"})
        assert "2026" in result or "202" in result  # year present


# ── read_file tests ────────────────────────────────────────────────────────────

class TestReadFile:

    def test_blocked_outside_allowed(self, loaded_registry):
        skill = loaded_registry.get("read_file")
        result = skill.run({"path": "C:/Windows/System32/drivers/etc/hosts"})
        assert "Error" in result or "outside" in result

    def test_file_not_found(self, loaded_registry, tmp_path):
        # tmp_path is not in allowed_folders → error
        skill = loaded_registry.get("read_file")
        result = skill.run({"path": str(tmp_path / "ghost.txt")})
        assert "Error" in result

    def test_reads_allowed_file(self, loaded_registry):
        from ragnarok.config import settings
        inbox: Path = settings.inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        test_file = inbox / "_test_read.txt"
        test_file.write_text("Hello from test!", encoding="utf-8")
        try:
            skill = loaded_registry.get("read_file")
            result = skill.run({"path": str(test_file)})
            assert "Hello from test!" in result
        finally:
            test_file.unlink(missing_ok=True)

    def test_truncation(self, loaded_registry):
        from ragnarok.config import settings
        inbox: Path = settings.inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        test_file = inbox / "_test_trunc.txt"
        test_file.write_text("X" * 10_000, encoding="utf-8")
        try:
            skill = loaded_registry.get("read_file")
            result = skill.run({"path": str(test_file), "max_chars": 100})
            assert "truncated" in result
            assert len(result) < 500  # should be way shorter than 10k
        finally:
            test_file.unlink(missing_ok=True)


# ── add_note tests ─────────────────────────────────────────────────────────────

class TestAddNote:

    def test_saves_note(self, loaded_registry):
        skill = loaded_registry.get("add_note")
        result = skill.run({"content": "This is a test note", "tag": "test"})
        assert "saved" in result.lower()

    def test_empty_content_error(self, loaded_registry):
        skill = loaded_registry.get("add_note")
        result = skill.run({"content": ""})
        assert "Error" in result

    def test_too_long_error(self, loaded_registry):
        skill = loaded_registry.get("add_note")
        result = skill.run({"content": "A" * 6000})
        assert "Error" in result


# ── search_files tests ────────────────────────────────────────────────────────

class TestSearchFiles:

    def test_returns_string(self, loaded_registry):
        skill = loaded_registry.get("search_files")
        result = skill.run({"pattern": "*.txt"})
        assert isinstance(result, str)

    def test_no_results_message(self, loaded_registry):
        skill = loaded_registry.get("search_files")
        result = skill.run({"pattern": "*.zzzneverexists_xyz"})
        assert "No files" in result

    def test_finds_gitkeep(self, loaded_registry):
        skill = loaded_registry.get("search_files")
        result = skill.run({"pattern": ".gitkeep"})
        # .gitkeep files were placed in data/ dirs
        assert isinstance(result, str)  # just verify it runs cleanly
