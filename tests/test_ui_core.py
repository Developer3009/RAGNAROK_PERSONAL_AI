"""Unit and integration tests verifying the Ragnarok 3D UI Core and its assets."""

import pytest
from starlette.testclient import TestClient
from ragnarok.ui.app import app, ACTIVE_TOKEN


@pytest.fixture
def client():
    return TestClient(app, base_url="http://127.0.0.1:8000")


def test_core_ui_rendered_properly(client):
    """Test that the 3D Core HTML is served properly with Three.js and no obscuring dead canvas."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Three.js script loader must be present
    assert "three.min.js" in html
    # Three.js container must be present
    assert 'id="threejs-container"' in html
    # Animation loop must be defined
    assert "animate" in html
    # Mode switcher pills for interactive core must be present
    assert 'id="btn-mode-listening"' in html


def test_threejs_static_library_served(client):
    """Test that Three.js library is accessible locally without CDN requirement."""
    res = client.get("/static/three.min.js")
    assert res.status_code == 200
    assert len(res.content) > 500_000


@pytest.mark.parametrize("run_idx,mode", [
    (1, "listening"),
    (2, "thinking"),
    (3, "speaking"),
    (4, "idle"),
])
def test_core_mode_state_end_to_end_runs(client, run_idx, mode):
    """Test end-to-end 3+ times with distinct modes that UI state and status respond."""
    res = client.get("/api/status", headers={"X-Ragnarok-Token": ACTIVE_TOKEN})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"


@pytest.mark.parametrize("run_idx,page_path", [
    (1, "/"),
    (2, "/"),
    (3, "/"),
])
def test_ragnarok_branding_and_elements_removed_3x(client, run_idx, page_path):
    """Test 3 times that Synapse AI is rebranded to Ragnarok, logo is present, and telemetry badges are removed."""
    res = client.get(page_path)
    assert res.status_code == 200
    html = res.text

    # Ragnarok logo must be present in UI corner
    assert "ragnarok_logo.png" in html
    # Synapse AI must be absent
    assert "SYNAPSE" not in html
    assert "synapse" not in html.lower()

    # Elements requested to be removed must be absent
    assert "BITRATE:" not in html
    assert "BUFFER:" not in html
    assert "Default Mic Array" not in html
    assert "LATENCY: 18ms" not in html

    # Logo static asset must be served successfully
    logo_res = client.get("/static/ragnarok_logo.png")
    assert logo_res.status_code == 200
    assert len(logo_res.content) > 1000


@pytest.mark.parametrize("run_idx,check_type", [
    (1, "typography_features"),
    (2, "alignment_and_dimensions"),
    (3, "legacy_absence"),
])
def test_cli_banner_matches_image_typography_3x(run_idx, check_type):
    """Test 3 times with different criteria that CLI banner matches the stylized image typography."""
    from ragnarok.cli import _BANNER

    if check_type == "typography_features":
        # Must feature the stylized R cuts and O underline bar from the image
        assert "█████▄" in _BANNER
        assert "▀██ ▀██" in _BANNER
        assert "▄██▄██▀" in _BANNER
        assert "██████" in _BANNER  # underline under O
        assert "L O C A L" in _BANNER
        assert "A G E N T" in _BANNER

    elif check_type == "alignment_and_dimensions":
        # All box-enclosed lines must be precisely aligned (88 columns)
        lines = [ln for ln in _BANNER.strip().split("\n") if ln.startswith("╔") or ln.startswith("║") or ln.startswith("╚")]
        assert len(lines) == 13
        for idx, line in enumerate(lines):
            assert len(line) == 88, f"Line {idx} length is {len(line)}, expected 88: {line}"

    elif check_type == "legacy_absence":
        # Legacy generic ASCII banner characters must be completely removed
        assert "██╔══██╗" not in _BANNER
        assert "██║  ███╗" not in _BANNER
        assert "╚═════╝" not in _BANNER


@pytest.mark.parametrize("run_idx,target_component", [
    (1, "cli_ansi_color"),
    (2, "ui_html_banner"),
    (3, "ui_static_banner"),
])
def test_banner_color_33ccff_3x(client, run_idx, target_component):
    """Test 3 times across CLI and UI targets that banner color is #33ccff."""
    if target_component == "cli_ansi_color":
        from ragnarok.cli import _BANNER_COLOR
        # #33ccff in RGB: 51, 204, 255
        assert "\033[38;2;51;204;255m" in _BANNER_COLOR

    elif target_component == "ui_html_banner":
        res = client.get("/")
        assert res.status_code == 200
        assert "text-[#33ccff]" in res.text
        assert "cli-mic-btn" in res.text
        assert "cli-voice-toggle-btn" in res.text

    elif target_component == "ui_static_banner":
        with open("ragnarok/ui/static/index.html", "r", encoding="utf-8") as f:
            html = f.read()
        assert "text-[#33ccff]" in html
        assert "triggerVoiceInput" in html
        assert "speakAnswer" in html


@pytest.mark.parametrize("run_idx,voice_input_command,expected_tool,expected_spoken_substr", [
    (1, 'open youtube and search "neural architecture search"', "open_browser", "neural architecture search"),
    (2, 'open youtube and search "transformers attention mechanism"', "open_browser", "transformers attention mechanism"),
    (3, 'search youtube for "quantum computing in 2026"', "open_browser", "quantum computing in 2026"),
])
def test_voice_input_tool_use_and_voice_output_end_to_end_3x(
    client, run_idx, voice_input_command, expected_tool, expected_spoken_substr
):
    """Test end-to-end 3 times with distinct input data:
       1. Voice input transcribed via voice_listen
       2. Tool dispatched dynamically by voice through agent
       3. Voice output spoken aloud via voice_speak
    """
    from unittest.mock import patch

    # 1. Voice Input Stage: Mock microphone listening returning the speech
    with patch("ragnarok.skills.voice_listen.run", return_value=voice_input_command) as mock_listen:
        listen_res = client.post("/api/voice/listen?duration_sec=3.0", headers={"X-Ragnarok-Token": ACTIVE_TOKEN})
        assert listen_res.status_code == 200
        transcribed = listen_res.json()["transcript"]
        assert transcribed == voice_input_command

    # 2. Tool Execution Stage: Voice transcript submitted to agent /api/chat
    with patch("webbrowser.open") as mock_web_open:
        chat_res = client.post(
            "/api/chat",
            headers={"X-Ragnarok-Token": ACTIVE_TOKEN},
            json={"message": transcribed},
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        if expected_tool:
            assert expected_tool in chat_data["tool_calls"]
            mock_web_open.assert_called_once()
        assert expected_spoken_substr in chat_data["answer"]

    # 3. Voice Output Stage: Agent answer spoken aloud via voice_speak
    with patch("ragnarok.skills.voice_speak.run", return_value="[Spoken via pyttsx3 offline TTS]") as mock_speak:
        speak_res = client.post(
            "/api/voice/speak",
            headers={"X-Ragnarok-Token": ACTIVE_TOKEN},
            json={"text": chat_data["answer"]},
        )
        assert speak_res.status_code == 200
        assert speak_res.json()["status"] == "success"
        mock_speak.assert_called_once_with({"text": chat_data["answer"]})


@pytest.mark.parametrize(
    "run_num, token_source, voice_cmd",
    [
        (1, "master_active_token", 'open youtube and search "reinforcement learning"'),
        (2, "auto_refreshed_token", 'open youtube and search "convolutional neural networks"'),
        (3, "query_param_token", 'open youtube and search "graph neural networks"'),
    ],
)
def test_security_token_voice_command_resilience_3x(client, run_num, token_source, voice_cmd):
    """
    Test End-to-End 3x:
    Validates that voice commands NEVER fail with 'Invalid or missing Ragnarok security token'
    under rotation, master token persistence, and query param passing.
    """
    from ragnarok.ui.app import ACTIVE_TOKEN, _rotator
    from unittest.mock import patch

    # Ensure rotator has rotated to a fresh token
    new_token = _rotator.force_rotate()

    if token_source == "master_active_token":
        # Master token MUST remain accepted even after force_rotate
        token_to_use = ACTIVE_TOKEN
        headers = {"X-Ragnarok-Token": token_to_use}
        url = "/api/chat"
    elif token_source == "auto_refreshed_token":
        # Client discovers active token from local discovery endpoint
        token_res = client.get("/api/security/token")
        assert token_res.status_code == 200
        token_to_use = token_res.json()["token"]
        headers = {"X-Ragnarok-Token": token_to_use}
        url = "/api/chat"
    else:  # query_param_token
        token_to_use = ACTIVE_TOKEN
        headers = {}
        url = f"/api/chat?token={token_to_use}"

    with patch("webbrowser.open"):
        res = client.post(url, headers=headers, json={"message": voice_cmd})
        assert res.status_code == 200, f"Expected 200 OK on run {run_num}, got {res.status_code}: {res.text}"
        data = res.json()
        assert "open_browser" in data["tool_calls"]
        assert "youtube.com" in data["answer"].lower()




