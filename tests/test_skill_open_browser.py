"""Unit and end-to-end tests for the open_browser skill and voice execution."""

from unittest.mock import patch, MagicMock
import pytest

from ragnarok.skills import open_browser
from ragnarok.skills.registry import SkillRegistry
from ragnarok.agent.loop import AgentLoop
from starlette.testclient import TestClient
from ragnarok.ui.app import app, ACTIVE_TOKEN


def test_open_browser_manifest():
    """Verify open_browser manifest adheres to registry schema."""
    registry = SkillRegistry()
    registry.load_all()
    skill = registry.get("open_browser")
    assert skill is not None
    assert skill.manifest.name == "open_browser"
    assert "search_query" in skill.manifest.parameters


@pytest.mark.parametrize("run_idx,command,expected_query,expected_target", [
    (1, 'open youtube and search "working of LLM"', "working of LLM", "youtube"),
    (2, 'open youtube and search working of LLM', "working of LLM", "youtube"),
    (3, 'search youtube for local AI agents offline', "local AI agents offline", "youtube"),
    (4, 'search quantum computing architecture on youtube', "quantum computing architecture", "youtube"),
    (5, 'open youtube', None, "youtube"),
])
def test_voice_command_parsing(run_idx, command, expected_query, expected_target):
    """Test voice parsing across distinct speech variations."""
    parsed = open_browser.parse_voice_command(command)
    assert parsed is not None
    assert parsed.get("target") == expected_target
    if expected_query:
        assert parsed.get("search_query") == expected_query
    else:
        assert "youtube.com" in parsed.get("url", "")


@pytest.mark.parametrize("run_idx,args,expected_url_part", [
    (1, {"search_query": "working of LLM", "target": "youtube"}, "youtube.com/results?search_query=working+of+LLM"),
    (2, {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}, "youtube.com/watch?v=dQw4w9WgXcQ"),
    (3, {"search_query": "deep learning 2026", "target": "web"}, "google.com/search?q=deep+learning+2026"),
])
def test_open_browser_execution_3x_runs(run_idx, args, expected_url_part):
    """Test 3 distinct execution flows with different inputs."""
    with patch("webbrowser.open") as mock_open:
        res = open_browser.run(args)
        assert "Opening" in res or "navigated" in res
        mock_open.assert_called_once()
        called_url = mock_open.call_args[0][0]
        assert expected_url_part in called_url


@pytest.mark.parametrize("run_idx,prompt,expected_search", [
    (1, 'open youtube and search "working of LLM"', "working of LLM"),
    (2, 'open youtube and search neural networks architecture', "neural networks architecture"),
    (3, 'search youtube for local Ollama autonomous agent', "local Ollama autonomous agent"),
])
def test_agent_voice_command_execution_end_to_end_3x(run_idx, prompt, expected_search):
    """Test AgentLoop end-to-end 3x with different voice queries."""
    with patch("webbrowser.open") as mock_open:
        agent = AgentLoop()
        res = agent.chat(prompt)
        assert res.steps_taken == 1
        assert "open_browser" in res.tool_calls_made
        assert expected_search in res.answer
        mock_open.assert_called_once()
        called_url = mock_open.call_args[0][0]
        assert "youtube.com/results?search_query=" in called_url


@pytest.mark.parametrize("run_idx,voice_input", [
    (1, 'open youtube and search "working of LLM"'),
    (2, 'open youtube and search transformers attention mechanism'),
    (3, 'search youtube for python fast api microservices'),
])
def test_ui_api_chat_voice_command_end_to_end_3x(run_idx, voice_input):
    """Test /api/chat endpoint with voice command queries 3x."""
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    with patch("webbrowser.open") as mock_open:
        resp = client.post(
            "/api/chat",
            json={"message": voice_input},
            headers={"X-Ragnarok-Token": ACTIVE_TOKEN},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "open_browser" in data["tool_calls"]
        assert "youtube.com/results" in data["answer"]
        mock_open.assert_called_once()


def test_ui_voice_endpoints():
    """Test /api/voice/listen and /api/voice/speak endpoints."""
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    with patch("ragnarok.skills.voice_listen.run", return_value='open youtube and search "working of LLM"') as mock_listen:
        res = client.post("/api/voice/listen?duration_sec=3.0", headers={"X-Ragnarok-Token": ACTIVE_TOKEN})
        assert res.status_code == 200
        assert "working of LLM" in res.json()["transcript"]

    with patch("ragnarok.skills.voice_speak.run", return_value="Spoken successfully") as mock_speak:
        res = client.post(
            "/api/voice/speak",
            json={"text": "Opening YouTube and searching for working of LLM"},
            headers={"X-Ragnarok-Token": ACTIVE_TOKEN},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "success"


@pytest.mark.parametrize(
    "run_num, prompt, expected_url_part",
    [
        (1, "open yt", "https://www.youtube.com"),
        (2, "open yt and search neural network architecture", "youtube.com/results?search_query=neural+network+architecture"),
        (3, "play bohemian rhapsody on yt", "youtube.com/results?search_query=bohemian+rhapsody"),
    ],
)
def test_yt_aliases_and_url_fallback_opens_browser_3x(run_num, prompt, expected_url_part):
    """
    Test End-to-End 3x:
    Validates that 'yt' commands and YouTube search voice commands ALWAYS open YouTube in the browser.
    """
    with patch("webbrowser.open") as mock_open:
        agent = AgentLoop()
        res = agent.chat(prompt)
        assert res.steps_taken >= 1
        assert "open_browser" in res.tool_calls_made
        mock_open.assert_called_once()
        called_url = mock_open.call_args[0][0]
        assert expected_url_part in called_url


def test_agent_auto_opens_browser_when_llm_gives_url():
    """
    Test that if the LLM finishes without tool calls but includes a YouTube URL in answer,
    AgentLoop detects the URL and triggers open_browser anyway so the user doesn't just get a URL.
    """
    fake_answer = "Sure, you can watch it here: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    fake_llm_result = MagicMock()
    fake_llm_result.has_tool_calls = False
    fake_llm_result.final_content = fake_answer

    with patch("ragnarok.llm.chat_with_tools", return_value=fake_llm_result), \
         patch("webbrowser.open") as mock_open:
        agent = AgentLoop()
        res = agent.chat("can you find me that video?")
        assert "open_browser" in res.tool_calls_made
        mock_open.assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ", new=2)

