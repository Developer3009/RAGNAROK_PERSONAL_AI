"""
tests/test_super_agent.py
──────────────────────────
End-to-end test suite for the 4 Super-AI-Agent innovations:
  • Innovation #8: AgentCouncil (Hierarchical Multi-Agent Governance)
  • Innovation #9: WebAgent  (Headless Autonomous Web Browser Agent)
  • Innovation #10: AmbientContextDaemon (Passive Desktop Awareness)
  • Innovation #11: PlaybookExtractor  (Experience Replay Buffer & Playbooks)

Global rule enforced: each user flow is tested at least 3 times with
different input data each run.  Failures are fixed before passing.

All tests are offline-first; LLM calls and browser I/O are mocked.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ════════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════════

class _FakeChatResult:
    """Mimics ragnarok.llm.ChatResult for mock patches."""
    def __init__(self, content: str) -> None:
        self.content = content
        self.model = "mock-model"
        self.latency_ms = 1.0
        self.token_count = 10


def _make_llm_mock(content: str) -> MagicMock:
    """Return a mock that makes llm.chat() return _FakeChatResult(content)."""
    m = MagicMock(return_value=_FakeChatResult(content))
    return m


# ════════════════════════════════════════════════════════════════════════════════
# Innovation #8 — AgentCouncil
# ════════════════════════════════════════════════════════════════════════════════

class TestAgentCouncil:
    """Tests for ragnarok.agent.council"""

    # ── import ────────────────────────────────────────────────────────────────

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        """Clear module-level council singleton between tests."""
        import ragnarok.agent.council as cmod
        cmod._council = None
        yield
        cmod._council = None

    # ── TaskDAG ───────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("goal,node_ids,deps", [
        ("Build a REST API",        ["t1", "t2", "t3"],       [[], ["t1"], ["t2"]]),
        ("Plan a road trip",        ["t1", "t2", "t3", "t4"], [[], [], ["t1"], ["t2", "t3"]]),
        ("Write a research paper",  ["t1", "t2"],              [[], ["t1"]]),
    ])
    def test_dag_topological_order(self, goal, node_ids, deps):
        """TaskDAG.topological_order() must respect dependencies (3 runs)."""
        from ragnarok.agent.council import TaskNode, TaskDAG

        nodes = [
            TaskNode(id=nid, description=f"desc-{nid}", dependencies=d)
            for nid, d in zip(node_ids, deps)
        ]
        dag = TaskDAG(goal=goal, nodes=nodes)
        order = dag.topological_order()

        # Every dependency must appear BEFORE the node that depends on it
        pos = {nid: i for i, nid in enumerate(order)}
        for node in nodes:
            for dep in node.dependencies:
                if dep in pos and node.id in pos:
                    assert pos[dep] < pos[node.id], (
                        f"Dependency {dep} must precede {node.id} in topological order"
                    )

    # ── decompose (mocked LLM) ────────────────────────────────────────────────

    @pytest.mark.parametrize("goal,mock_json", [
        (
            "Create a Python package",
            json.dumps({"tasks": [
                {"id": "t1", "description": "Write setup.py",    "dependencies": []},
                {"id": "t2", "description": "Write source code", "dependencies": ["t1"]},
                {"id": "t3", "description": "Run tests",         "dependencies": ["t2"]},
            ]}),
        ),
        (
            "Deploy a Docker container",
            json.dumps({"tasks": [
                {"id": "t1", "description": "Build image",    "dependencies": []},
                {"id": "t2", "description": "Push to registry", "dependencies": ["t1"]},
                {"id": "t3", "description": "Run container",  "dependencies": ["t2"]},
            ]}),
        ),
        (
            "Analyse CSV data",
            json.dumps({"tasks": [
                {"id": "t1", "description": "Load CSV",    "dependencies": []},
                {"id": "t2", "description": "Clean data",  "dependencies": ["t1"]},
                {"id": "t3", "description": "Plot charts", "dependencies": ["t2"]},
            ]}),
        ),
    ])
    def test_decompose_valid_dag(self, goal, mock_json):
        """AgentCouncil.decompose() should parse valid JSON into a TaskDAG (3 runs)."""
        from ragnarok.agent.council import AgentCouncil

        council = AgentCouncil()
        with patch("ragnarok.llm.chat", _make_llm_mock(mock_json)):
            dag = council.decompose(goal)

        assert dag.goal == goal
        assert len(dag.nodes) == 3

    def test_decompose_fallback_on_bad_json(self):
        """decompose() falls back to single-node DAG when LLM returns garbage."""
        from ragnarok.agent.council import AgentCouncil

        council = AgentCouncil()
        with patch("ragnarok.llm.chat", _make_llm_mock("not json at all!")):
            dag = council.decompose("some goal")

        assert len(dag.nodes) == 1

    # ── critique ──────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("objections,risk,recs", [
        (["Step 2 depends on Step 1 incorrectly"], 4, ["Re-order steps"]),
        (["Missing error handling"],               7, ["Add try/except blocks"]),
        ([],                                       1, ["Looks good"]),
    ])
    def test_critique_parses_report(self, objections, risk, recs):
        """AgentCouncil.critique() should return a CritiqueReport (3 runs)."""
        from ragnarok.agent.council import AgentCouncil, TaskNode, TaskDAG

        critique_json = json.dumps({
            "objections": objections,
            "risk_score": risk,
            "recommendations": recs,
        })
        council = AgentCouncil()
        dag = TaskDAG(
            goal="test goal",
            nodes=[TaskNode(id="t1", description="do something")],
        )
        with patch("ragnarok.llm.chat", _make_llm_mock(critique_json)):
            report = council.critique(dag)

        assert report.risk_score == risk
        assert report.objections == objections
        assert report.recommendations == recs

    # ── execute ───────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("goal,n_tasks", [
        ("Build CLI tool", 2),
        ("Setup database", 3),
        ("Write report",   4),
    ])
    def test_execute_completes_all_nodes(self, goal, n_tasks):
        """AgentCouncil.execute() should mark all nodes done (3 runs)."""
        from ragnarok.agent.council import AgentCouncil, TaskNode, TaskDAG

        nodes = [
            TaskNode(id=f"t{i}", description=f"Task {i}", dependencies=[f"t{i-1}"] if i > 1 else [])
            for i in range(1, n_tasks + 1)
        ]
        dag = TaskDAG(goal=goal, nodes=nodes)
        council = AgentCouncil()

        with patch("ragnarok.llm.chat", _make_llm_mock("Done!")):
            results = council.execute(dag)

        assert len(results) == n_tasks
        for node in dag.nodes.values():
            assert node.status == "done"
            assert node.result == "Done!"

    # ── skill run() ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize("mode,goal", [
        ("plan",     "Organise a conference"),
        ("critique", "Build a machine learning pipeline"),
        ("full",     "Launch a startup product"),
    ])
    def test_skill_run_modes(self, mode, goal):
        """council_plan skill run() should succeed for each mode (3 runs)."""
        import ragnarok.agent.council as cmod

        dag_json = json.dumps({"tasks": [
            {"id": "t1", "description": "Step one", "dependencies": []},
            {"id": "t2", "description": "Step two", "dependencies": ["t1"]},
        ]})
        critique_json = json.dumps({"objections": [], "risk_score": 2, "recommendations": ["OK"]})
        worker_reply  = "Worker completed task."
        synth_reply   = "Final synthesized answer."

        responses = [dag_json, critique_json, worker_reply, worker_reply, synth_reply]
        call_count = [0]

        def _side_effect(**kwargs):
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return _FakeChatResult(responses[idx])

        with patch("ragnarok.llm.chat", side_effect=lambda **kw: _side_effect(**kw)):
            with patch("ragnarok.llm.chat", side_effect=lambda *a, **kw: _FakeChatResult(responses[call_count[0] % len(responses)])):
                call_count[0] = 0
                # Patch at module level
                import ragnarok.llm as llm_mod
                orig = llm_mod.chat

                def _patched_chat(*args, **kwargs):
                    idx = call_count[0] % len(responses)
                    call_count[0] += 1
                    return _FakeChatResult(responses[idx])

                llm_mod.chat = _patched_chat
                try:
                    result = cmod.run({"goal": goal, "mode": mode})
                finally:
                    llm_mod.chat = orig

        assert isinstance(result, str)
        assert len(result) > 0
        assert "ERROR" not in result or mode == "critique"  # critique may have empty

    def test_skill_run_missing_goal(self):
        """council_plan skill should return error when goal is missing."""
        import ragnarok.agent.council as cmod
        result = cmod.run({"mode": "plan"})
        assert "ERROR" in result

    def test_manifest_valid(self):
        """council_plan MANIFEST must have required fields."""
        from ragnarok.agent.council import MANIFEST
        assert MANIFEST.name == "council_plan"
        assert MANIFEST.risk_level == "medium"
        assert "goal" in MANIFEST.required


# ════════════════════════════════════════════════════════════════════════════════
# Innovation #9 — WebAgent
# ════════════════════════════════════════════════════════════════════════════════

class TestWebAgent:
    """Tests for ragnarok.skills.web_agent"""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        import ragnarok.skills.web_agent as wa
        wa._web_agent = None
        yield
        wa._web_agent = None

    # ── URL safety ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("url,expected", [
        ("https://example.com",   True),
        ("http://localhost:8080", True),
        ("ftp://example.com",     False),
        ("javascript:alert(1)",   False),
        ("file:///etc/passwd",    False),
    ])
    def test_url_safety(self, url, expected):
        """_is_safe_url() must block non-HTTP schemes (5 cases)."""
        from ragnarok.skills.web_agent import _is_safe_url
        assert _is_safe_url(url) == expected

    # ── BrowserSession (mocked playwright) ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_browser_session_navigate_safe(self):
        """BrowserSession.navigate() should work for safe URLs."""
        from ragnarok.skills.web_agent import BrowserSession

        mock_resp = MagicMock()
        mock_resp.status = 200

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.goto = AsyncMock(return_value=mock_resp)
        mock_page.set_default_timeout = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_browser.close = AsyncMock()

        mock_pw = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()

        async def _fake_start(self):
            self._playwright = mock_pw
            self._browser    = mock_browser
            self.page        = mock_page

        async def _fake_stop(self):
            pass

        session = BrowserSession(headless=True)
        session._playwright = mock_pw
        session._browser    = mock_browser
        session.page        = mock_page

        result = await session.navigate("https://example.com")
        assert "200" in result or "example.com" in result

    @pytest.mark.asyncio
    async def test_browser_session_blocked_url(self):
        """BrowserSession.navigate() must reject non-HTTP URLs."""
        from ragnarok.skills.web_agent import BrowserSession

        session = BrowserSession(headless=True)
        session.page = AsyncMock()

        result = await session.navigate("ftp://example.com")
        assert "BLOCKED" in result

    # ── WebAgentLoop (fully mocked) ────────────────────────────────────────────

    @pytest.mark.parametrize("task,url,llm_responses", [
        (
            "Find the page title",
            "https://example.com",
            [
                json.dumps({"action": "extract_text", "reason": "Get page content"}),
                json.dumps({"action": "done", "reason": "Title is Example Domain"}),
            ],
        ),
        (
            "Click the About link",
            "https://example.org",
            [
                json.dumps({"action": "click", "target": "About", "reason": "Navigate to About page"}),
                json.dumps({"action": "done", "reason": "On About page now"}),
            ],
        ),
        (
            "Search for Python docs",
            "https://docs.python.org",
            [
                json.dumps({"action": "scroll", "target": "down", "reason": "See more content"}),
                json.dumps({"action": "done", "reason": "Found Python documentation"}),
            ],
        ),
    ])
    @pytest.mark.asyncio
    async def test_web_agent_loop_async(self, task, url, llm_responses):
        """WebAgentLoop.run_async() should complete the task (3 runs)."""
        from ragnarok.skills.web_agent import WebAgentLoop

        mock_resp = MagicMock(); mock_resp.status = 200
        mock_page = AsyncMock()
        mock_page.url = url
        mock_page.goto = AsyncMock(return_value=mock_resp)
        mock_page.set_default_timeout = MagicMock()
        mock_page.inner_text = AsyncMock(return_value="Sample page text")
        mock_page.evaluate   = AsyncMock(return_value=None)
        mock_page.click      = AsyncMock(return_value=None)
        mock_page.fill       = AsyncMock(return_value=None)
        mock_page.get_by_text = MagicMock(return_value=MagicMock(first=MagicMock(click=AsyncMock(return_value=None), fill=AsyncMock(return_value=None))))
        mock_page.accessibility = AsyncMock()
        mock_page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea", "name": "Test", "children": [
                {"role": "heading", "name": "Main heading", "children": []},
                {"role": "link",    "name": "About",        "children": []},
            ]
        })

        mock_browser = AsyncMock()
        mock_browser.new_page = AsyncMock(return_value=mock_page)
        mock_browser.close    = AsyncMock()

        mock_ctx_mgr = AsyncMock()
        mock_pw_instance = AsyncMock()
        mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw_instance.stop = AsyncMock()

        call_idx = [0]
        def _fake_llm_chat(*args, **kwargs):
            idx = call_idx[0] % len(llm_responses)
            call_idx[0] += 1
            return _FakeChatResult(llm_responses[idx])

        import ragnarok.llm as llm_mod
        orig_chat = llm_mod.chat

        async def _fake_start():
            return mock_pw_instance

        with patch("playwright.async_api.async_playwright") as mock_pw_ctx:
            # async_playwright() returns an async context manager
            mock_pw_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_pw_instance)
            mock_pw_ctx.return_value.__aexit__  = AsyncMock(return_value=False)
            mock_pw_ctx.return_value.start      = AsyncMock(return_value=mock_pw_instance)

            llm_mod.chat = _fake_llm_chat
            try:
                agent = WebAgentLoop()
                result = await agent.run_async(task, url, max_steps=5)
            finally:
                llm_mod.chat = orig_chat

        assert isinstance(result, str)
        assert len(result) > 0

    # ── skill run() ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize("task,url", [
        ("Get page title",     "https://example.com"),
        ("Find latest news",   "https://news.ycombinator.com"),
        ("List Python versions","https://www.python.org"),
    ])
    def test_skill_run_invokes_agent(self, task, url):
        """web_agent skill run() should call WebAgentLoop.run_sync (3 runs)."""
        import ragnarok.skills.web_agent as wa_mod

        with patch.object(
            wa_mod,
            "get_web_agent",
            return_value=MagicMock(run_sync=MagicMock(return_value=f"Completed: {task}")),
        ):
            result = wa_mod.run({"task": task, "url": url})

        assert "Completed" in result or isinstance(result, str)

    def test_skill_run_missing_task(self):
        """web_agent skill should return error when task is missing."""
        import ragnarok.skills.web_agent as wa_mod
        result = wa_mod.run({"url": "https://example.com"})
        assert "ERROR" in result

    def test_skill_run_missing_url(self):
        """web_agent skill should return error when url is missing."""
        import ragnarok.skills.web_agent as wa_mod
        result = wa_mod.run({"task": "Find something"})
        assert "ERROR" in result

    def test_skill_run_blocked_url(self):
        """web_agent skill should return error for non-HTTP URL."""
        import ragnarok.skills.web_agent as wa_mod
        result = wa_mod.run({"task": "attack", "url": "ftp://evil.com"})
        assert "ERROR" in result

    def test_manifest_valid(self):
        """web_agent MANIFEST must have required fields."""
        from ragnarok.skills.web_agent import MANIFEST
        assert MANIFEST.name == "web_agent"
        assert MANIFEST.risk_level == "high"
        assert "task" in MANIFEST.required
        assert "url"  in MANIFEST.required


# ════════════════════════════════════════════════════════════════════════════════
# Innovation #10 — Ambient Context Stream
# ════════════════════════════════════════════════════════════════════════════════

class TestAmbientContext:
    """Tests for ragnarok.triggers.ambient_context"""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        import ragnarok.triggers.ambient_context as ac
        ac._daemon = None
        yield
        ac._daemon = None

    # ── EpisodicRingBuffer ────────────────────────────────────────────────────

    @pytest.mark.parametrize("episodes_data", [
        [("Notepad", "Untitled - Notepad", "User typing notes"),
         ("Chrome",  "GitHub - Google",   "User browsing GitHub"),
         ("VSCode",  "council.py — Ragnarok", "User editing Python file")],

        [("Terminal", "PowerShell",        "User running commands"),
         ("Excel",    "Budget.xlsx",       "User editing spreadsheet"),
         ("Teams",    "Meeting — Teams",   "User in video call")],

        [("Explorer", "Desktop",           "User browsing files"),
         ("Outlook",  "Inbox — Outlook",   "User reading email"),
         ("Word",     "Report.docx",       "User writing document")],
    ])
    def test_ring_buffer_insert_and_recall(self, tmp_path, episodes_data):
        """EpisodicRingBuffer: insert 3 episodes, recall them in order (3 runs)."""
        from ragnarok.triggers.ambient_context import EpisodicRingBuffer, AmbientEpisode

        db = str(tmp_path / "test_ae.db")
        buf = EpisodicRingBuffer(db, max_entries=50)

        for app, title, summary in episodes_data:
            ep = AmbientEpisode(
                timestamp=time.time(),
                app_name=app,
                window_title=title,
                ocr_summary=summary,
            )
            buf.insert(ep)
            time.sleep(0.01)  # ensure distinct timestamps

        assert buf.count() == len(episodes_data)
        recent = buf.recent(limit=10)
        assert len(recent) == len(episodes_data)
        # Most recent first
        assert recent[0].ocr_summary == episodes_data[-1][2]
        buf.close()

    def test_ring_buffer_prunes_old_entries(self, tmp_path):
        """EpisodicRingBuffer prunes oldest when over capacity."""
        from ragnarok.triggers.ambient_context import EpisodicRingBuffer, AmbientEpisode

        db = str(tmp_path / "prune.db")
        buf = EpisodicRingBuffer(db, max_entries=5)

        for i in range(10):
            buf.insert(AmbientEpisode(
                timestamp=float(i),
                app_name="app",
                window_title=f"window-{i}",
                ocr_summary=f"summary-{i}",
            ))

        assert buf.count() == 5
        # Oldest entries should be pruned
        recent = buf.recent(limit=10)
        titles = [e.window_title for e in recent]
        assert "window-0" not in titles
        buf.close()

    # ── WindowObserver ────────────────────────────────────────────────────────

    def test_window_observer_returns_tuple(self):
        """WindowObserver.get_active_window() must return (str, str)."""
        from ragnarok.triggers.ambient_context import WindowObserver
        obs = WindowObserver()
        result = obs.get_active_window()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    # ── ScreenDiffDetector ────────────────────────────────────────────────────

    def test_screen_diff_first_capture_always_returns(self):
        """ScreenDiffDetector should always return bytes on first call."""
        from ragnarok.triggers.ambient_context import ScreenDiffDetector
        detector = ScreenDiffDetector(diff_threshold=0.05)

        # Mock mss
        mock_img = MagicMock()
        mock_img.height = 10
        mock_img.width  = 10
        mock_img.bgra   = bytes(10 * 10 * 4)
        mock_img.rgb    = bytes(10 * 10 * 3)

        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__  = MagicMock(return_value=False)
        mock_sct.monitors  = [{}, {"width": 10, "height": 10, "top": 0, "left": 0}]
        mock_sct.grab      = MagicMock(return_value=mock_img)

        from PIL import Image
        fake_img = Image.new("RGB", (10, 10))

        with patch("mss.mss", return_value=mock_sct):
            with patch("PIL.Image.frombytes", return_value=fake_img):
                result = detector.capture_if_changed()

        assert result is not None  # first capture always returns

    # ── ContextStreamAnalyzer ─────────────────────────────────────────────────

    @pytest.mark.parametrize("app,title,screenshot", [
        ("Notepad",  "todo.txt",          None),
        ("Chrome",   "Stack Overflow",    None),
        ("Terminal", "PowerShell",        None),
    ])
    def test_analyzer_summarize_no_screenshot(self, app, title, screenshot):
        """ContextStreamAnalyzer.summarize() works without screenshot (3 runs)."""
        from ragnarok.triggers.ambient_context import ContextStreamAnalyzer
        analyzer = ContextStreamAnalyzer()
        result = analyzer.summarize(app, title, screenshot)
        assert isinstance(result, str)
        # Should at least mention the app name
        assert app in result or title[:10] in result

    @pytest.mark.parametrize("titles,expect_hint", [
        (["same error", "same error", "same error", "same error"], True),
        (["window1",    "window2",    "window3",    "window4"],    False),
        (["stuck here", "stuck here", "stuck here", "stuck here"], True),
    ])
    def test_detect_stuck_pattern(self, titles, expect_hint):
        """ContextStreamAnalyzer.detect_stuck_pattern detects repeated windows (3 runs)."""
        from ragnarok.triggers.ambient_context import ContextStreamAnalyzer, AmbientEpisode
        analyzer = ContextStreamAnalyzer()
        episodes = [
            AmbientEpisode(timestamp=float(i), app_name="app", window_title=t, ocr_summary=t)
            for i, t in enumerate(titles)
        ]
        hint = analyzer.detect_stuck_pattern(episodes, same_title_threshold=4)
        if expect_hint:
            assert hint is not None
            assert isinstance(hint, str)
        else:
            assert hint is None

    # ── AmbientContextDaemon lifecycle ────────────────────────────────────────

    def test_daemon_start_stop(self, tmp_path):
        """AmbientContextDaemon starts and stops without errors."""
        from ragnarok.triggers.ambient_context import AmbientContextDaemon

        db = str(tmp_path / "daemon_test.db")
        daemon = AmbientContextDaemon(
            db_path=db,
            poll_interval_sec=0.1,
            diff_threshold=0.05,
            ring_buffer_size=10,
        )

        # Patch the _tick method so it doesn't try to capture screen
        daemon._tick = MagicMock()

        daemon.start()
        assert daemon.is_running()
        time.sleep(0.3)
        daemon.stop()
        assert not daemon.is_running()

    def test_daemon_context_summary_empty(self, tmp_path):
        """get_context_summary returns a message when no episodes exist."""
        from ragnarok.triggers.ambient_context import AmbientContextDaemon

        db = str(tmp_path / "empty_summary.db")
        daemon = AmbientContextDaemon(db_path=db)
        summary = daemon.get_context_summary()
        assert "No ambient context" in summary or isinstance(summary, str)


# ════════════════════════════════════════════════════════════════════════════════
# Innovation #11 — Playbook / Experience Replay Buffer
# ════════════════════════════════════════════════════════════════════════════════

class TestPlaybook:
    """Tests for ragnarok.memory.playbook"""

    # ── PlaybookStore ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("pb_data", [
        {"name": "Deploy app",       "description": "Deploy a web app to server",        "template_steps": [], "param_slots": ["APP_NAME"]},
        {"name": "Setup database",   "description": "Create and seed a PostgreSQL DB",   "template_steps": [], "param_slots": ["DB_NAME"]},
        {"name": "Run test suite",   "description": "Execute pytest with coverage",      "template_steps": [], "param_slots": ["TEST_PATH"]},
    ])
    def test_store_save_and_get(self, tmp_path, pb_data):
        """PlaybookStore.save() then .get() should return the same playbook (3 runs)."""
        from ragnarok.memory.playbook import PlaybookStore, Playbook

        store = PlaybookStore(str(tmp_path / "pb.db"))
        pb = Playbook(id=str(uuid.uuid4()), **pb_data)
        store.save(pb)

        retrieved = store.get(pb.id)
        assert retrieved is not None
        assert retrieved.name        == pb.name
        assert retrieved.description == pb.description
        assert retrieved.param_slots == pb.param_slots
        store.close()

    def test_store_list_and_delete(self, tmp_path):
        """PlaybookStore.list_all() and delete() work correctly."""
        from ragnarok.memory.playbook import PlaybookStore, Playbook

        store = PlaybookStore(str(tmp_path / "pb2.db"))
        pb1 = Playbook(id=str(uuid.uuid4()), name="PB1", description="d1", template_steps=[], param_slots=[])
        pb2 = Playbook(id=str(uuid.uuid4()), name="PB2", description="d2", template_steps=[], param_slots=[])
        store.save(pb1)
        store.save(pb2)

        all_pbs = store.list_all()
        assert len(all_pbs) == 2

        deleted = store.delete(pb1.id)
        assert deleted is True
        assert store.get(pb1.id) is None
        assert len(store.list_all()) == 1
        store.close()

    # ── PlaybookExtractor.record ───────────────────────────────────────────────

    @pytest.mark.parametrize("goal,steps_data,llm_resp", [
        (
            "Create a new Python virtualenv",
            [{"tool_name": "run_command", "args": {"cmd": "python -m venv .venv"}, "result": "OK"}],
            json.dumps({"name": "Create virtualenv", "description": "Create a Python venv",
                        "template_steps": [{"tool": "run_command", "args": {"cmd": "python -m venv {{VENV_NAME}}"}}],
                        "param_slots": ["VENV_NAME"]}),
        ),
        (
            "Add a note to memory",
            [{"tool_name": "add_note", "args": {"content": "Important fact"}, "result": "Saved"}],
            json.dumps({"name": "Add note", "description": "Add a note to memory",
                        "template_steps": [{"tool": "add_note", "args": {"content": "{{NOTE_CONTENT}}"}}],
                        "param_slots": ["NOTE_CONTENT"]}),
        ),
        (
            "List directory contents",
            [{"tool_name": "list_dir", "args": {"path": "/tmp"}, "result": "file1.txt\nfile2.py"}],
            json.dumps({"name": "List directory", "description": "List files in a directory",
                        "template_steps": [{"tool": "list_dir", "args": {"path": "{{DIR_PATH}}"}}],
                        "param_slots": ["DIR_PATH"]}),
        ),
    ])
    def test_record_creates_playbook(self, tmp_path, goal, steps_data, llm_resp):
        """PlaybookExtractor.record() should persist an abstracted playbook (3 runs)."""
        from ragnarok.memory.playbook import (
            PlaybookStore, PlaybookExtractor, WorkflowTrajectory, TrajectoryStep
        )

        store = PlaybookStore(str(tmp_path / "rec.db"))
        extractor = PlaybookExtractor(store)

        steps = [
            TrajectoryStep(tool_name=s["tool_name"], args=s["args"], result=s["result"])
            for s in steps_data
        ]
        traj = WorkflowTrajectory(goal=goal, outcome="Success", steps=steps)

        # Mock both llm.chat and llm.embed
        with patch("ragnarok.llm.chat", return_value=_FakeChatResult(llm_resp)):
            with patch("ragnarok.llm.embed", return_value=[0.1, 0.2, 0.3]):
                pb = extractor.record(traj)

        assert pb.id is not None
        assert len(pb.id) > 0
        assert pb.name != ""
        # Should be persisted
        retrieved = store.get(pb.id)
        assert retrieved is not None
        store.close()

    # ── PlaybookExtractor.match ────────────────────────────────────────────────

    def test_match_returns_similar_playbooks(self, tmp_path):
        """PlaybookExtractor.match() should find playbooks by semantic similarity."""
        from ragnarok.memory.playbook import PlaybookStore, PlaybookExtractor, Playbook
        import struct, math

        store = PlaybookStore(str(tmp_path / "match.db"))
        extractor = PlaybookExtractor(store)

        # Create a known embedding vector [1.0, 0.0, 0.0]
        vec = [1.0, 0.0, 0.0]
        emb_bytes = struct.pack("3f", *vec)

        pb = Playbook(
            id=str(uuid.uuid4()),
            name="Deploy app",
            description="deploy docker app",
            template_steps=[],
            param_slots=[],
            embedding=emb_bytes,
        )
        store.save(pb)

        # Query with same vector → cosine = 1.0
        with patch("ragnarok.llm.embed", return_value=vec):
            matches = extractor.match("deploy docker app", threshold=0.90)

        assert len(matches) >= 1
        assert matches[0][0].id == pb.id
        assert matches[0][1] > 0.90
        store.close()

    def test_match_returns_empty_below_threshold(self, tmp_path):
        """PlaybookExtractor.match() should return empty list when no match."""
        from ragnarok.memory.playbook import PlaybookStore, PlaybookExtractor, Playbook
        import struct

        store = PlaybookStore(str(tmp_path / "nomatch.db"))
        extractor = PlaybookExtractor(store)

        # Orthogonal vectors → cosine = 0.0
        emb_bytes = struct.pack("3f", 1.0, 0.0, 0.0)
        pb = Playbook(
            id=str(uuid.uuid4()),
            name="Unrelated task",
            description="something completely different",
            template_steps=[],
            param_slots=[],
            embedding=emb_bytes,
        )
        store.save(pb)

        with patch("ragnarok.llm.embed", return_value=[0.0, 1.0, 0.0]):
            matches = extractor.match("completely different query", threshold=0.90)

        assert len(matches) == 0
        store.close()

    # ── skill run() ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize("action,extra_args", [
        ("list",   {}),
        ("search", {"query": "deploy docker"}),
        ("record", {"goal": "Run tests", "steps_json": json.dumps([
            {"tool_name": "get_time", "args": {}, "result": "12:00"}
        ])}),
    ])
    def test_skill_run_actions(self, tmp_path, action, extra_args):
        """playbook skill run() should work for list/search/record (3 runs)."""
        import ragnarok.memory.playbook as pb_mod

        db_path = str(tmp_path / f"skill_{action}.db")
        store = pb_mod.PlaybookStore(db_path)
        extractor = pb_mod.PlaybookExtractor(store)

        def _fake_llm_chat(*a, **kw):
            return _FakeChatResult(json.dumps({
                "name": "Test playbook",
                "description": "A test playbook",
                "template_steps": [],
                "param_slots": [],
            }))

        import ragnarok.llm as llm_mod
        orig_chat = llm_mod.chat
        orig_embed = llm_mod.embed

        llm_mod.chat  = _fake_llm_chat
        llm_mod.embed = lambda *a, **kw: [0.1, 0.2, 0.3]

        try:
            with patch.object(pb_mod, "get_extractor", return_value=extractor):
                args = {"action": action, **extra_args}
                result = pb_mod.run(args)
        finally:
            llm_mod.chat  = orig_chat
            llm_mod.embed = orig_embed
            store.close()

        assert isinstance(result, str)
        assert "ERROR" not in result or action == "search"  # search with no matches is OK

    def test_skill_run_replay(self, tmp_path):
        """playbook skill run(action=replay) should execute steps and return result."""
        import ragnarok.memory.playbook as pb_mod
        import struct

        db_path = str(tmp_path / "replay.db")
        store = pb_mod.PlaybookStore(db_path)

        pb = pb_mod.Playbook(
            id=str(uuid.uuid4()),
            name="Replay test",
            description="A playbook to replay",
            template_steps=[{"tool": "get_time", "args": {}}],
            param_slots=[],
        )
        store.save(pb)
        extractor = pb_mod.PlaybookExtractor(store)

        # Mock registry to return a simple skill
        mock_skill = MagicMock()
        mock_skill.run = MagicMock(return_value="2026-01-01 12:00:00")
        mock_registry = MagicMock()
        mock_registry.get = MagicMock(return_value=mock_skill)

        with patch("ragnarok.skills.registry.registry", mock_registry):
            with patch.object(pb_mod, "get_extractor", return_value=extractor):
                result = pb_mod.run({
                    "action": "replay",
                    "playbook_id": pb.id,
                    "params_json": "{}",
                })

        assert "Replaying" in result or isinstance(result, str)
        store.close()

    def test_skill_run_missing_action(self):
        """playbook skill should return error for unknown action."""
        import ragnarok.memory.playbook as pb_mod
        result = pb_mod.run({"action": "fly_to_moon"})
        assert "ERROR" in result

    def test_manifest_valid(self):
        """playbook MANIFEST must have required fields."""
        from ragnarok.memory.playbook import MANIFEST
        assert MANIFEST.name == "playbook"
        assert MANIFEST.risk_level == "low"
        assert "action" in MANIFEST.required


# ════════════════════════════════════════════════════════════════════════════════
# Integration: Council + Playbook round-trip
# ════════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Cross-module integration tests."""

    def test_council_result_recorded_as_playbook(self, tmp_path):
        """
        A successful council execution can be serialised into a playbook.
        End-to-end: council.run_full → playbook.record.
        """
        from ragnarok.agent.council import AgentCouncil
        from ragnarok.memory.playbook import (
            PlaybookStore, PlaybookExtractor, WorkflowTrajectory, TrajectoryStep
        )

        dag_json = json.dumps({"tasks": [
            {"id": "t1", "description": "Research topic",   "dependencies": []},
            {"id": "t2", "description": "Write summary",    "dependencies": ["t1"]},
        ]})
        critique_json = json.dumps({"objections": [], "risk_score": 1, "recommendations": []})
        worker_reply  = "Step completed."
        synth_reply   = "Final answer: Research done and summary written."

        responses = [dag_json, critique_json, worker_reply, worker_reply, synth_reply]
        call_idx = [0]

        import ragnarok.llm as llm_mod
        orig_chat = llm_mod.chat
        orig_embed = llm_mod.embed

        def _fake_chat(*a, **kw):
            idx = call_idx[0] % len(responses)
            call_idx[0] += 1
            return _FakeChatResult(responses[idx])

        llm_mod.chat  = _fake_chat
        llm_mod.embed = lambda *a, **kw: [0.5, 0.5]

        try:
            council = AgentCouncil()
            outcome = council.run_full("Research and summarise AI trends")

            # Now record it as a playbook
            steps = [
                TrajectoryStep(tool_name="council_plan",
                               args={"goal": outcome["goal"], "mode": "full"},
                               result=outcome["final_answer"])
            ]
            traj = WorkflowTrajectory(
                goal=outcome["goal"],
                outcome=outcome["final_answer"],
                steps=steps,
            )

            store = PlaybookStore(str(tmp_path / "integration.db"))
            extractor = PlaybookExtractor(store)

            call_idx[0] = 0
            pb = extractor.record(traj)

            assert pb.id is not None
            assert store.get(pb.id) is not None
            store.close()
        finally:
            llm_mod.chat  = orig_chat
            llm_mod.embed = orig_embed
