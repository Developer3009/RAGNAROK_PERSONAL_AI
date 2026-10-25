"""
tests/test_llm.py
──────────────────
Unit tests for ragnarok/llm.py.
Mocks httpx so these pass without a running Ollama instance.
Integration tests (marked @pytest.mark.integration) need Ollama live.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ragnarok import llm
from ragnarok.llm import ChatResult, ChatToolResult, ToolCall


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


# ── chat() tests ───────────────────────────────────────────────────────────────

class TestChat:

    def test_returns_chat_result(self):
        fake_data = {
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "Hello world"},
            "eval_count": 10,
        }
        with patch("ragnarok.llm._make_client") as mock_ctx:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.return_value = _mock_response(fake_data)
            mock_ctx.return_value = client

            result = llm.chat([{"role": "user", "content": "hi"}])

        assert isinstance(result, ChatResult)
        assert result.content == "Hello world"
        assert result.token_count == 10

    def test_system_prepended(self):
        fake_data = {
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "ok"},
            "eval_count": 5,
        }
        captured_payload = {}

        def capture_post(url, json=None, **kw):
            captured_payload.update(json or {})
            return _mock_response(fake_data)

        with patch("ragnarok.llm._make_client") as mock_ctx:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.side_effect = capture_post
            mock_ctx.return_value = client

            llm.chat([{"role": "user", "content": "test"}], system="Be brief.")

        msgs = captured_payload.get("messages", [])
        assert msgs[0]["role"] == "system"
        assert "Be brief." in msgs[0]["content"]


# ── chat_with_tools() tests ───────────────────────────────────────────────────

class TestChatWithTools:

    def test_parses_tool_call(self):
        fake_data = {
            "model": "llama3.2:3b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_time", "arguments": {"format": "iso"}}}
                ],
            },
            "eval_count": 8,
        }
        with patch("ragnarok.llm._make_client") as mock_ctx:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.return_value = _mock_response(fake_data)
            mock_ctx.return_value = client

            result = llm.chat_with_tools(
                messages=[{"role": "user", "content": "What time is it?"}],
                tools=[{"type": "function", "function": {"name": "get_time"}}],
            )

        assert result.has_tool_calls
        assert result.tool_calls[0].name == "get_time"
        assert result.tool_calls[0].arguments == {"format": "iso"}

    def test_no_tool_call_returns_content(self):
        fake_data = {
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "Paris.", "tool_calls": []},
            "eval_count": 5,
        }
        with patch("ragnarok.llm._make_client") as mock_ctx:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.return_value = _mock_response(fake_data)
            mock_ctx.return_value = client

            result = llm.chat_with_tools(
                messages=[{"role": "user", "content": "Capital of France?"}],
                tools=[],
            )

        assert not result.has_tool_calls
        assert result.final_content == "Paris."

    def test_string_arguments_parsed(self):
        """Model sometimes returns arguments as a JSON string instead of dict."""
        fake_data = {
            "model": "llama3.2:3b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_time", "arguments": '{"format": "human"}'}}
                ],
            },
            "eval_count": 4,
        }
        with patch("ragnarok.llm._make_client") as mock_ctx:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.post.return_value = _mock_response(fake_data)
            mock_ctx.return_value = client

            result = llm.chat_with_tools(
                messages=[{"role": "user", "content": "time?"}],
                tools=[],
            )

        assert result.tool_calls[0].arguments == {"format": "human"}


# ── is_ollama_alive() ─────────────────────────────────────────────────────────

class TestIsAlive:

    def test_returns_true_on_200(self):
        with patch("ragnarok.llm._make_client") as mock_ctx:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            resp = MagicMock()
            resp.status_code = 200
            client.get.return_value = resp
            mock_ctx.return_value = client
            assert llm.is_ollama_alive() is True

    def test_returns_false_on_connect_error(self):
        import httpx
        with patch("ragnarok.llm._make_client") as mock_ctx:
            client = MagicMock()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            client.get.side_effect = httpx.ConnectError("refused")
            mock_ctx.return_value = client
            assert llm.is_ollama_alive() is False


# ── Integration tests (require live Ollama) ───────────────────────────────────

@pytest.mark.integration
def test_live_chat():
    """Requires ollama running with llama3.2:3b pulled."""
    result = llm.chat([{"role": "user", "content": "Say exactly: PONG"}])
    assert "PONG" in result.content.upper()


@pytest.mark.integration
def test_live_embed():
    """Requires ollama running with nomic-embed-text pulled."""
    vector = llm.embed("test embedding")
    assert isinstance(vector, list)
    assert len(vector) > 100
