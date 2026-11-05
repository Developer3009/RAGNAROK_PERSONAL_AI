"""
tests/test_screen_sharing_e2e_3runs.py
──────────────────────────────────────
End-to-end tests for the Laptop Screen Sharing & Multimodal Vision feature.

Covers:
  Flow 1 — Image Frame Inspection & Metadata Extraction (3 runs × distinct image sizes & formats)
  Flow 2 — Multimodal Vision Analysis with Mocked Ollama (3 runs × distinct queries & models)
  Flow 3 — REST API Screen Frame Streaming & Status Tracking (3 runs × distinct frames)
  Flow 4 — REST API Full Screen Analysis Endpoint (3 runs × distinct base64 payloads)
  Flow 5 — capture_screen Skill Integration with Active Shared Screen (3 runs)

Global rule: Every flow is exercised ≥ 3 times with distinct data.
"""

from __future__ import annotations

import base64
import io
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ════════════════════════════════════════════════════════════════════════════════
#  Helpers & Fixtures
# ════════════════════════════════════════════════════════════════════════════════

def _create_test_image_bytes(width: int, height: int, color: tuple, fmt: str = "PNG") -> bytes:
    """Generate in-memory test image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def test_client():
    """Return a FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from ragnarok.ui.app import app, ACTIVE_TOKEN, _latest_screen_frame

    # Reset buffer before each test
    _latest_screen_frame["image_bytes"] = None
    _latest_screen_frame["timestamp"] = 0.0
    _latest_screen_frame["width"] = 0
    _latest_screen_frame["height"] = 0

    client = TestClient(app, raise_server_exceptions=False)
    yield client, ACTIVE_TOKEN


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 1 — Image Metadata Extraction  (3 runs with distinct images)
# ════════════════════════════════════════════════════════════════════════════════

METADATA_TEST_CASES = [
    # Run 1: Full HD 1080p PNG (Cyan)
    (1920, 1080, (0, 219, 233), "PNG"),
    # Run 2: 720p JPEG (Deep Grey / Dark Theme)
    (1280, 720, (18, 19, 24), "JPEG"),
    # Run 3: Compact 800x600 PNG (Amber / Code Editor)
    (800, 600, (255, 179, 0), "PNG"),
]


@pytest.mark.parametrize("width,height,color,fmt", METADATA_TEST_CASES)
def test_screen_metadata_extraction(width, height, color, fmt):
    """Flow 1: Test image metadata extraction from screen frame bytes."""
    from ragnarok.vision.screen_analyzer import get_screen_metadata

    image_bytes = _create_test_image_bytes(width, height, color, fmt)
    meta = get_screen_metadata(image_bytes)

    assert meta["width"] == width
    assert meta["height"] == height
    assert meta["format"] == fmt
    assert meta["size_bytes"] == len(image_bytes)
    assert meta["size_bytes"] > 0


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 2 — Multimodal Vision Analysis  (3 runs with distinct queries)
# ════════════════════════════════════════════════════════════════════════════════

VISION_ANALYSIS_CASES = [
    # Run 1: Code debugging query on 1080p
    (
        "Find any syntax errors or exceptions in this code.",
        "llava",
        1920, 1080, (0, 100, 150),
        "Identified a missing closing parenthesis on line 42 in app.py.",
    ),
    # Run 2: Document / text summary query on 720p
    (
        "Summarize the main points of this document.",
        "minicpm-v",
        1280, 720, (50, 50, 50),
        "The document is a technical architecture diagram explaining local Ollama inference.",
    ),
    # Run 3: Terminal error inspection query on 800x600
    (
        "What error message is displayed in the terminal window?",
        "llama3.2-vision",
        800, 600, (10, 10, 10),
        "The terminal displays 'HTTP 200 OK — service successfully listening on port 8000'.",
    ),
]


@pytest.mark.parametrize("question,model,width,height,color,expected_reply", VISION_ANALYSIS_CASES)
def test_screen_vision_analysis_mocked(question, model, width, height, color, expected_reply):
    """Flow 2: Test multimodal vision call to Ollama."""
    from ragnarok.vision.screen_analyzer import analyze_screen_frame

    image_bytes = _create_test_image_bytes(width, height, color, "PNG")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "model": model,
        "message": {"role": "assistant", "content": expected_reply},
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp):
        res = analyze_screen_frame(image_bytes, question=question, vision_model=model)

    assert res["status"] == "success"
    assert res["answer"] == expected_reply
    assert res["model"] == model
    assert res["metadata"]["width"] == width
    assert res["metadata"]["height"] == height
    assert res["latency_ms"] >= 0.0


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 3 — Stream Frame & Status Tracking  (3 runs with distinct frames)
# ════════════════════════════════════════════════════════════════════════════════

STREAM_FRAME_CASES = [
    # Run 1: 1920x1080 stream update
    (1920, 1080, (20, 40, 60)),
    # Run 2: 1440x900 stream update
    (1440, 900, (60, 40, 20)),
    # Run 3: 1280x800 stream update
    (1280, 800, (40, 60, 20)),
]


@pytest.mark.parametrize("width,height,color", STREAM_FRAME_CASES)
def test_rest_api_stream_frame_and_status(test_client, width, height, color):
    """Flow 3: Test periodic stream frame ingestion and screen status query."""
    client, token = test_client
    headers = {"X-Ragnarok-Token": token, "Origin": "http://localhost:8000"}

    # Generate test frame
    raw_bytes = _create_test_image_bytes(width, height, color, "JPEG")
    b64 = "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode("ascii")

    # Post frame
    r = client.post(
        "/api/vision/stream-frame",
        json={"image_base64": b64, "width": width, "height": height},
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "received"
    assert data["width"] == width
    assert data["height"] == height

    # Check status endpoint
    r_stat = client.get("/api/vision/screen-status", headers=headers)
    assert r_stat.status_code == 200
    stat = r_stat.json()
    assert stat["has_frame"] is True
    assert stat["active"] is True
    assert stat["width"] == width
    assert stat["height"] == height
    assert stat["source"] == "browser_share"
    assert stat["age_sec"] is not None and stat["age_sec"] < 5.0


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 4 — Full REST API Screen Analysis  (3 runs with distinct requests)
# ════════════════════════════════════════════════════════════════════════════════

API_ANALYZE_CASES = [
    # Run 1: VS Code debug inspection
    (
        "Explain the function highlighted on screen.",
        "llava",
        1920, 1080,
        "The function is calculate_metrics() which computes latency percentiles.",
    ),
    # Run 2: Browser dashboard inspection
    (
        "What are the CPU and RAM values shown in the top strip?",
        "minicpm-v",
        1366, 768,
        "CPU is at 12% and RAM usage is 2,410 MB.",
    ),
    # Run 3: Graph / Chart query
    (
        "Is the trend in the chart going up or down?",
        "llama3.2-vision",
        1024, 768,
        "The response time graph shows a steady downward trend from 120ms to 45ms.",
    ),
]


@pytest.mark.parametrize("question,model,width,height,mock_reply", API_ANALYZE_CASES)
def test_rest_api_analyze_screen(test_client, question, model, width, height, mock_reply):
    """Flow 4: Test POST /api/vision/analyze with base64 image."""
    client, token = test_client
    headers = {"X-Ragnarok-Token": token, "Origin": "http://localhost:8000"}

    raw_bytes = _create_test_image_bytes(width, height, (30, 30, 80), "PNG")
    b64 = "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "model": model,
        "message": {"role": "assistant", "content": mock_reply},
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp):
        r = client.post(
            "/api/vision/analyze",
            json={"image_base64": b64, "question": question, "vision_model": model},
            headers=headers,
        )

    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "success"
    assert res["answer"] == mock_reply
    assert res["model"] == model
    assert res["metadata"]["width"] == width
    assert res["metadata"]["height"] == height


# ════════════════════════════════════════════════════════════════════════════════
#  Flow 5 — capture_screen Skill Integration with Active Shared Screen
# ════════════════════════════════════════════════════════════════════════════════

SKILL_INTEGRATION_CASES = [
    # Run 1: Using active shared laptop screen (1080p)
    (
        1920, 1080,
        "What code is on screen?",
        "The screen shows Python code in VS Code defining FastAPI routes.",
    ),
    # Run 2: Using active shared laptop screen (720p)
    (
        1280, 720,
        "What terminal command just ran?",
        "The terminal shows pytest running 27 tests all passing.",
    ),
    # Run 3: Using active shared laptop screen (900p)
    (
        1440, 900,
        "Summarize the web page open.",
        "The web page is the Ragnarok Multimodal Vision dashboard with screen sharing active.",
    ),
]


@pytest.mark.parametrize("width,height,question,mock_desc", SKILL_INTEGRATION_CASES)
def test_capture_screen_uses_shared_laptop_frame(width, height, question, mock_desc):
    """Flow 5: Verify capture_screen skill uses the actively shared screen frame from the UI."""
    from ragnarok.skills import capture_screen
    import ragnarok.ui.app as app_mod

    # Populate active shared frame in app
    img_bytes = _create_test_image_bytes(width, height, (10, 80, 120), "PNG")
    app_mod._latest_screen_frame["image_bytes"] = img_bytes
    app_mod._latest_screen_frame["timestamp"] = time.time()
    app_mod._latest_screen_frame["width"] = width
    app_mod._latest_screen_frame["height"] = height
    app_mod._latest_screen_frame["source"] = "browser_share"

    with patch.object(capture_screen, "_describe_image", return_value=mock_desc):
        # We don't want it to call _capture_screenshot if a shared frame is active
        with patch.object(capture_screen, "_capture_screenshot", side_effect=AssertionError("Should have used shared screen!")):
            result = capture_screen.run({"question": question, "vision_model": "llava"})

    assert mock_desc in result
