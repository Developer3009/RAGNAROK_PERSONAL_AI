"""
Unit tests and end-to-end verification for all Ragnarok skills and tools.
Tests every tool flow at least 3 times with different input data each run.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from ragnarok.skills import (
    write_file,
    list_dir,
    fetch_web,
    read_file,
    add_note,
    remember,
    search_files,
    get_time,
)
from ragnarok.skills.registry import SkillRegistry, registry as skill_registry
from ragnarok.config import settings


# ── 1. get_time: 3 runs with different formats ────────────────────────────────

def test_get_time_run1_human():
    res = get_time.execute("human")
    assert isinstance(res, str)
    assert any(str(yr) in res for yr in range(2024, 2030))


def test_get_time_run2_iso():
    res = get_time.execute("iso")
    assert isinstance(res, str)
    assert "T" in res and "-" in res


def test_get_time_run3_default_and_run():
    res = get_time.run({})
    assert isinstance(res, str)
    assert len(res) > 5


# ── 2. write_file: 3 runs with different data scenarios ────────────────────────

def test_write_file_run1_create():
    target = settings.knowledge_dir / "test_run1_alpha.txt"
    try:
        res = write_file.execute(str(target), "Alpha payload 12345")
        assert "Successfully wrote" in res
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "Alpha payload 12345"
    finally:
        target.unlink(missing_ok=True)


def test_write_file_run2_overwrite():
    target = settings.knowledge_dir / "test_run2_beta.txt"
    try:
        write_file.execute(str(target), "Initial content")
        res = write_file.execute(str(target), "Updated beta payload 67890")
        assert "Successfully wrote" in res
        assert target.read_text(encoding="utf-8") == "Updated beta payload 67890"
    finally:
        target.unlink(missing_ok=True)


def test_write_file_run3_blocked_outside():
    res = write_file.execute("C:/Windows/System32/evil_test.txt", "Malicious content")
    assert "outside allowed directories" in res or "not within any allowed directory" in res


# ── 3. read_file: 3 runs with different data scenarios ─────────────────────────

def test_read_file_run1_normal():
    target = settings.knowledge_dir / "test_read_run1.txt"
    target.write_text("Hello Ragnarok Skill Test!", encoding="utf-8")
    try:
        res = read_file.execute(str(target))
        assert res == "Hello Ragnarok Skill Test!"
    finally:
        target.unlink(missing_ok=True)


def test_read_file_run2_truncation():
    target = settings.knowledge_dir / "test_read_run2.txt"
    target.write_text("A" * 500, encoding="utf-8")
    try:
        res = read_file.execute(str(target), max_chars=40)
        assert "truncated" in res
        assert len(res) < 200
    finally:
        target.unlink(missing_ok=True)


def test_read_file_run3_not_found():
    res = read_file.execute(str(settings.knowledge_dir / "nonexistent_file_999.txt"))
    assert "Error" in res


# ── 4. add_note: 3 runs with different categories and contents ────────────────

def test_add_note_run1_preference():
    res = add_note.execute("User prefers dark mode UI", tag="preference")
    assert "Note saved" in res
    assert "preference" in res


def test_add_note_run2_task():
    res = add_note.execute("Submit project documentation by Friday", tag="task-mgmt")
    assert "Note saved" in res
    assert "task_mgmt" in res or "task-mgmt" in res or "task" in res


def test_add_note_run3_empty_error():
    res = add_note.execute("", tag="empty")
    assert "Error" in res and "empty" in res


# ── 5. remember: 3 runs with different memories and importance ─────────────────

def test_remember_run1_personal():
    res = remember.execute("User's name is Divyansh", importance=0.95, tag="profile")
    assert "Remembered" in res
    assert "Divyansh" in res


def test_remember_run2_technical():
    res = remember.execute("Primary database port is 5432", importance=0.8, tag="infra")
    assert "Remembered" in res
    assert "5432" in res


def test_remember_run3_empty_fails():
    res = remember.execute("", importance=0.5, tag="test")
    assert "Error" in res


# ── 6. search_files: 3 runs with different criteria ───────────────────────────

def test_search_files_run1_pattern():
    res = search_files.execute(pattern="*.txt", max_results=10)
    assert isinstance(res, str)


def test_search_files_run2_content():
    target = settings.knowledge_dir / "search_sample_target.txt"
    target.write_text("UniqueQueryKey777", encoding="utf-8")
    try:
        res = search_files.execute(pattern="*.txt", contains="UniqueQueryKey777")
        assert "search_sample_target.txt" in res
    finally:
        target.unlink(missing_ok=True)


def test_search_files_run3_no_match():
    res = search_files.execute(pattern="*.completely_absent_ext_12345")
    assert "No files found" in res


# ── 7. list_dir: 3 runs with different directories and limits ─────────────────

def test_list_dir_run1_default():
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    res = list_dir.execute()
    assert "Error" not in res


def test_list_dir_run2_capped():
    res = list_dir.execute(str(settings.knowledge_dir), max_entries=2)
    assert isinstance(res, str)
    assert "outside" not in res


def test_list_dir_run3_blocked():
    res = list_dir.execute("C:/Windows")
    assert "outside allowed directories" in res or "not within any allowed directory" in res


# ── 8. fetch_web: 3 runs with different URLs and mocks ────────────────────────

@patch("httpx.Client.get")
def test_fetch_web_run1_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = "Sample Web Page Body"
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    res = fetch_web.execute("https://example.com/api/test")
    assert res == "Sample Web Page Body"


@patch("httpx.Client.get")
def test_fetch_web_run2_truncation(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = "Z" * 200
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    res = fetch_web.execute("https://example.com/long", max_length=50)
    assert "Truncated at 50 characters" in res
    assert len(res) < 150


def test_fetch_web_run3_invalid_protocol():
    res = fetch_web.execute("file:///etc/passwd")
    assert "Error: URL must start with http:// or https://" in res


# ── 9. registry: 3 runs with different schema validations ──────────────────────

def test_registry_run1_type_coercion():
    skill_registry.load_all()
    # Integer as numeric string should pass validation
    ok, err = skill_registry.validate_call("read_file", {"path": "data/foo.txt", "max_chars": "500"})
    assert ok
    assert err == ""


def test_registry_run2_missing_param():
    skill_registry.load_all()
    ok, err = skill_registry.validate_call("write_file", {"path": "data/bar.txt"})
    assert not ok
    assert "Missing required parameter 'content'" in err


def test_registry_run3_unknown_skill():
    skill_registry.load_all()
    ok, err = skill_registry.validate_call("teleport_skill", {})
    assert not ok
    assert "Unknown skill 'teleport_skill'" in err
