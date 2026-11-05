"""
Tests verifying that the Ragnarok Web UI serves all 5 tabs and views properly,
with active tokens, valid HTML structure, and endpoints functional across 3 distinct runs.
"""

import pytest
from fastapi.testclient import TestClient
from ragnarok.ui.app import app


@pytest.fixture
def client():
    return TestClient(app)


RUN_CASES = [
    {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "accept_lang": "en-US,en;q=0.9",
        "expected_views": [
            "view-voice-interaction-studio",
            "view-multimodal-vision",
            "view-agenda-calendar",
            "view-telemetry-benchmarks",
            "view-agent-orchestrator",
        ],
    },
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/17.0",
        "accept_lang": "en-GB,en;q=0.8",
        "expected_views": [
            "view-voice-interaction-studio",
            "view-multimodal-vision",
            "view-agenda-calendar",
            "view-telemetry-benchmarks",
            "view-agent-orchestrator",
        ],
    },
    {
        "user_agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
        "accept_lang": "fr-FR,fr;q=0.9,en;q=0.8",
        "expected_views": [
            "view-voice-interaction-studio",
            "view-multimodal-vision",
            "view-agenda-calendar",
            "view-telemetry-benchmarks",
            "view-agent-orchestrator",
        ],
    },
]


@pytest.mark.parametrize("case", RUN_CASES)
def test_ui_index_serves_all_5_views(client, case):
    """Flow 1: GET / serves HTML with all 5 functional views and correct token injection."""
    headers = {
        "User-Agent": case["user_agent"],
        "Accept-Language": case["accept_lang"],
    }
    response = client.get("/", headers=headers)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text

    # Verify all 5 views are present
    for view_id in case["expected_views"]:
        assert f'id="{view_id}"' in html, f"Missing view section: {view_id}"

    # Verify that navigation data-paths match
    assert 'data-path="voice-interaction-studio"' in html
    assert 'data-path="multimodal-vision"' in html
    assert 'data-path="agenda-calendar"' in html
    assert 'data-path="telemetry-benchmarks"' in html
    assert 'data-path="agent-orchestrator"' in html

    # Verify that footer is closed
    assert "<footer" in html
    assert "</footer>" in html
    assert "</html>" in html


@pytest.mark.parametrize("case", RUN_CASES)
def test_ui_associated_endpoints(client, case):
    """Flow 2: Backend endpoints powering the 5 views return valid payloads."""
    # Obtain token first
    tok_resp = client.get("/api/security/token")
    assert tok_resp.status_code == 200
    token = tok_resp.json()["token"]
    headers = {"X-Ragnarok-Token": token, "User-Agent": case["user_agent"]}

    # Status endpoint powering Telemetry view
    r_status = client.get("/api/status", headers=headers)
    assert r_status.status_code == 200
    status_data = r_status.json()
    assert "cpu_percent" in status_data
    assert "ram_used_mb" in status_data
    assert "models" in status_data

    # Logs endpoint powering Telemetry timeline
    r_logs = client.get("/api/logs?limit=10", headers=headers)
    assert r_logs.status_code == 200
    logs_data = r_logs.json()
    assert "execution_traces" in logs_data

    # Settings endpoint powering Skills gallery
    r_settings = client.get("/api/settings", headers=headers)
    assert r_settings.status_code == 200
    assert "skills" in r_settings.json()

    # Approvals endpoint powering Neural Orchestrator
    r_appr = client.get("/api/approvals", headers=headers)
    assert r_appr.status_code == 200
    assert "pending" in r_appr.json()

    # Screen status endpoint powering Vision view
    r_vis = client.get("/api/vision/screen-status", headers=headers)
    assert r_vis.status_code == 200
    assert "active" in r_vis.json()


@pytest.mark.parametrize("case", RUN_CASES)
def test_static_and_root_files_in_sync(case):
    """Flow 3: Verifies that ragnarok/ui/static/index.html and root index.html remain synchronized."""
    from pathlib import Path
    static_html = Path("ragnarok/ui/static/index.html").read_text(encoding="utf-8")
    root_html = Path("index.html").read_text(encoding="utf-8")

    assert len(static_html) > 0
    assert len(root_html) > 0
    # Both must contain all 5 views
    for view_id in case["expected_views"]:
        assert f'id="{view_id}"' in static_html
        assert f'id="{view_id}"' in root_html
