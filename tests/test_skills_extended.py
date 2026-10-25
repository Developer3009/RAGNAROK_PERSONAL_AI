"""
Unit tests for the new Phase 3 skills: write_file, list_dir, and fetch_web.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
from ragnarok.skills import write_file, list_dir, fetch_web
from ragnarok.config import settings


def test_write_file_allowed(tmp_path):
    target = settings.knowledge_dir / "test_write.txt"
    try:
        res = write_file.execute(str(target), "Hello World!")
        assert "Successfully wrote" in res
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "Hello World!"
    finally:
        if target.exists():
            target.unlink()


def test_write_file_blocked():
    res = write_file.execute("C:/Windows/System32/evil.txt", "Bad payload")
    assert "not within any allowed directory" in res


def test_list_dir_allowed():
    res = list_dir.execute("data/knowledge")
    assert "gitkeep" in res or "FILE" in res or "empty" in res


def test_list_dir_blocked():
    res = list_dir.execute("C:/Windows")
    assert "not within any allowed directory" in res


def test_fetch_web_invalid_url():
    res = fetch_web.execute("ftp://invalid.url")
    assert "Error: URL must start with" in res


@patch("httpx.Client.get")
def test_fetch_web_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = "Mocked Web Content"
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    res = fetch_web.execute("https://example.com/test")
    assert res == "Mocked Web Content"
