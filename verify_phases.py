"""
Verification script for:
  Phase 0: Setup and scoping
  Phase 1: Core agent loop with Ollama
  Phase 2: Persistent memory, Skills and tool calling

Each phase is verified with 3 independent runs using distinct input data.
Run:  .venv/Scripts/python verify_phases.py
"""

import sys
import time
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from ragnarok.config import settings
from ragnarok import llm
from ragnarok.agent.loop import AgentLoop
from ragnarok.skills.registry import SkillRegistry
from ragnarok.memory.db import MemoryDB
from ragnarok.memory.recall import MemoryStore
from ragnarok.memory.working import WorkingMemoryManager

PASS = "  [PASS]"
FAIL = "  [FAIL]"


# ─────────────────────────────────────────────────────────────
# PHASE 0 — Setup & Scoping
# ─────────────────────────────────────────────────────────────

def test_phase_0_setup():
    print("\n" + "=" * 64)
    print("PHASE 0: SETUP & SCOPING — 3 VERIFICATION RUNS")
    print("=" * 64)

    # Run 1: Config correctness and directory structure
    print("\n[Run 1/3] Configuration integrity & directory structure...")
    assert settings.ollama_url.startswith("http"), f"Bad base URL: {settings.ollama_url}"
    assert settings.model_main == "llama3.2:3b", f"Unexpected main model: {settings.model_main}"
    assert settings.model_embed == "nomic-embed-text", f"Unexpected embed model: {settings.model_embed}"
    assert settings.knowledge_dir.exists(), f"Missing dir: {settings.knowledge_dir}"
    assert settings.db_dir.exists(), f"Missing db dir: {settings.db_dir}"
    assert settings.log_dir.exists(), f"Missing log dir: {settings.log_dir}"
    print(f"    ollama_url     = {settings.ollama_url}")
    print(f"    model_main     = {settings.model_main}")
    print(f"    model_embed    = {settings.model_embed}")
    print(f"    knowledge_dir  = {settings.knowledge_dir}")
    print(f"    allowed_folders= {[str(p) for p in settings.allowed_folders]}")
    print(PASS, "Run 1 — config and directory structure verified")

    # Run 2: Ollama connectivity ping
    print("\n[Run 2/3] Ollama service connectivity test...")
    t0 = time.perf_counter()
    try:
        with httpx.Client(base_url=settings.ollama_url, timeout=10.0) as client:
            resp = client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        print(FAIL, f"Ollama unreachable: {exc}")
        raise
    ping_ms = (time.perf_counter() - t0) * 1000
    print(f"    Ollama latency : {ping_ms:.1f}ms")
    print(f"    Pulled models  : {models}")
    assert settings.model_main in models, f"'{settings.model_main}' not found in Ollama models: {models}"
    assert settings.model_embed in models, f"'{settings.model_embed}' not found in Ollama models: {models}"
    print(PASS, "Run 2 — Ollama is reachable and required models are present")

    # Run 3: Embedding model sanity (vector dimension, determinism)
    print("\n[Run 3/3] Embedding model sanity (dimension & LRU cache)...")
    texts = [
        "Phase zero verification sentence alpha.",
        "Setup scoping dimensional consistency check beta.",
    ]
    vecs = []
    for txt in texts:
        t0 = time.perf_counter()
        vec = llm.embed(txt)
        emb_ms = (time.perf_counter() - t0) * 1000
        assert len(vec) > 0, "Empty embedding vector!"
        vecs.append(vec)
        print(f"    '{txt[:40]}...' → dim={len(vec)}, {emb_ms:.1f}ms")
    # Cache hit — second call for same text must be near-instant
    t0 = time.perf_counter()
    vec_cached = llm.embed(texts[0])
    cache_ms = (time.perf_counter() - t0) * 1000
    assert vec_cached == vecs[0], "Cache returned different vector!"
    assert cache_ms < 5.0, f"Cache hit took {cache_ms:.1f}ms — expected <5ms"
    print(f"    Cache hit latency: {cache_ms:.2f}ms ✓")
    assert len(vecs[0]) == len(vecs[1]), "Embedding dimensions inconsistent!"
    print(PASS, "Run 3 — embedding model and LRU cache verified")


# ─────────────────────────────────────────────────────────────
# PHASE 1 — Core Agent Loop
# ─────────────────────────────────────────────────────────────

def test_phase_1_agent_loop():
    print("\n" + "=" * 64)
    print("PHASE 1: CORE AGENT LOOP & TOOL CALLING — 3 RUNS")
    print("=" * 64)

    reg = SkillRegistry()
    reg.load_all()
    loaded_tools = list(reg._skills.keys())
    print(f"    Registered skills: {loaded_tools}")

    # Run 1: Pure conversational response (no tool use)
    print("\n[Run 1/3] Conversational response (no tool)...")
    loop1 = AgentLoop(registry=reg)
    query1 = "Name three advantages of running AI models locally on-device."
    print(f"    Query: {query1}")
    res1 = loop1.chat(query1)
    print(f"    Answer ({len(res1.answer)} chars): {res1.answer.strip()[:120]}...")
    print(f"    Steps={res1.steps_taken} | Latency={res1.latency_ms:.0f}ms | Tools={res1.tool_calls_made}")
    assert len(res1.answer) > 10, "Response too short"
    assert res1.steps_taken >= 1
    print(PASS, "Run 1 — conversational response OK")

    # Run 2: Tool-calling — get_time skill must be invoked
    print("\n[Run 2/3] Tool invocation flow (get_time)...")
    loop2 = AgentLoop(registry=reg)
    query2 = "What is today's date and the current time?"
    print(f"    Query: {query2}")
    res2 = loop2.chat(query2)
    print(f"    Answer: {res2.answer.strip()[:120]}...")
    print(f"    Steps={res2.steps_taken} | Tools={res2.tool_calls_made}")
    assert "get_time" in res2.tool_calls_made, (
        f"Expected 'get_time' in tool calls, got: {res2.tool_calls_made}"
    )
    print(PASS, "Run 2 — get_time tool invoked and answer returned")

    # Run 3: High-risk approval gate — approval denied, loop recovers gracefully
    print("\n[Run 3/3] High-risk tool approval gate (deny flow)...")
    denial_log: list[str] = []

    def deny_all(tool_name: str, args: dict) -> bool:
        denial_log.append(tool_name)
        return False

    skill_gt = reg.get("get_time")
    original_risk = skill_gt.manifest.risk_level
    try:
        skill_gt.manifest.risk_level = "high"
        loop3 = AgentLoop(registry=reg, approval_cb=deny_all)
        query3 = "Tell me the current system clock time please."
        res3 = loop3.chat(query3)
        print(f"    Denial log: {denial_log}")
        print(f"    Agent reply: {res3.answer.strip()[:120]}...")
        assert "get_time" in denial_log, (
            f"Approval callback was NOT invoked. denial_log={denial_log}"
        )
        assert len(res3.answer) > 0, "Agent gave empty reply after denial"
    finally:
        skill_gt.manifest.risk_level = original_risk

    print(PASS, "Run 3 — high-risk approval gate blocks correctly, agent recovers")


# ─────────────────────────────────────────────────────────────
# PHASE 2 — Persistent Memory & Skills
# ─────────────────────────────────────────────────────────────

def test_phase_2_memory_and_skills(tmp_path: Path):
    print("\n" + "=" * 64)
    print("PHASE 2: PERSISTENT MEMORY & EXTENDED SKILLS — 3 RUNS")
    print("=" * 64)

    reg = SkillRegistry()
    reg.load_all()

    # Run 1: Vector recall — store a fact, then query it back
    print("\n[Run 1/3] Vector recall (store → embed → query)...")
    db1 = MemoryDB(db_path=tmp_path / "run1.db")
    store1 = MemoryStore(db=db1)

    mid1 = store1.remember(
        "The user prefers concise bullet-point summaries over long paragraphs.",
        kind="semantic",
        source="run1_verification",
        importance=0.85,
    )
    print(f"    Stored memory ID: {mid1}")

    results = store1.recall("What format does the user like for output?", top_k=3)
    print(f"    Recall returned {len(results)} result(s)")
    assert len(results) >= 1, "Recall returned no results!"
    top = results[0]
    print(f"    Top match: '{top.text[:60]}' | score={top.score:.4f}")
    assert "concise" in top.text.lower() or "summaries" in top.text.lower(), \
        f"Unexpected recall content: {top.text}"
    assert top.score > 0.5, f"Recall score too low: {top.score}"
    print(PASS, "Run 1 — vector recall returned correct memory")

    # Run 2: Deduplication + write_file + read_file skills
    print("\n[Run 2/3] Deduplication & file skills (write_file / read_file)...")
    db2 = MemoryDB(db_path=tmp_path / "run2.db")
    store2 = MemoryStore(db=db2)

    text_a = "Project submission deadline is December 20th, 2026."
    text_b = "The project deadline falls on December 20th 2026."
    id_a = store2.remember(text_a, kind="fact")
    id_b = store2.remember(text_b, kind="fact")
    print(f"    First insert ID : {id_a}")
    print(f"    Second insert ID: {id_b}  (deduplication may reuse {id_a})")
    active = store2.list_all()
    print(f"    Active records in store: {len(active)}")
    assert len(active) == 1, f"Expected 1 deduplicated record, found {len(active)}"
    print("    Deduplication ✓ — near-duplicate merged into one record")

    # write_file skill
    write_skill = reg.get("write_file")
    assert write_skill is not None, "write_file skill not found in registry!"
    write_result = write_skill.run({
        "path": "data/knowledge/verify_run2.txt",
        "content": "Phase 2 Run 2: file skill verification passed.",
    })
    print(f"    write_file result: {write_result}")
    assert "saved" in write_result.lower() or "wrote" in write_result.lower() or "note saved" in write_result.lower(), \
        f"Unexpected write result: {write_result}"

    # read_file skill
    read_skill = reg.get("read_file")
    assert read_skill is not None, "read_file skill not found in registry!"
    read_result = read_skill.run({"path": "data/knowledge/verify_run2.txt"})
    print(f"    read_file content: {read_result.strip()[:80]}")
    assert "Phase 2 Run 2" in read_result, f"read_file returned unexpected content: {read_result}"
    print(PASS, "Run 2 — deduplication and file skills verified")

    # Run 3: Working memory crash recovery + list_dir + add_note skills
    print("\n[Run 3/3] Working memory recovery & list_dir / add_note skills...")
    wm_dir = tmp_path / "wm"
    wm_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    wm = WorkingMemoryManager(checkpoint_dir=wm_dir)
    wm.memory.set_goal("Analyse quarterly sales data and summarise key metrics", task_id="task-q4-2026")
    wm.memory.update_context("files_processed", ["q1.csv", "q2.csv", "q3.csv"])
    wm.memory.update_context("status", "in_progress")
    wm.checkpoint()
    print(f"    Checkpointed task_id={wm.memory.task_id!r} to {wm_dir}")

    # Simulate restart: fresh manager loads checkpoint
    wm2 = WorkingMemoryManager(checkpoint_dir=wm_dir)
    loaded = wm2.load_checkpoint()
    assert loaded, "load_checkpoint() returned False — no checkpoint found!"
    assert wm2.memory.task_id == "task-q4-2026"
    assert wm2.memory.context["status"] == "in_progress"
    assert wm2.memory.context["files_processed"] == ["q1.csv", "q2.csv", "q3.csv"]
    print(f"    Recovered: task_id={wm2.memory.task_id!r}, goal={wm2.memory.goal!r}")

    # list_dir skill
    list_skill = reg.get("list_dir")
    assert list_skill is not None, "list_dir skill not found in registry!"
    list_result = list_skill.run({"path": "data"})
    print(f"    list_dir output:\n{list_result}")
    assert "knowledge" in list_result, f"'knowledge' not found in list_dir output: {list_result}"

    # add_note skill
    note_skill = reg.get("add_note")
    assert note_skill is not None, "add_note skill not found in registry!"
    note_result = note_skill.run({
        "title": "verify_run3_note",
        "content": "Working memory recovery and list_dir verified on run 3.",
    })
    print(f"    add_note result: {note_result}")
    assert "note saved" in note_result.lower() or "saved" in note_result.lower(), \
        f"Unexpected note result: {note_result}"
    print(PASS, "Run 3 — working memory recovery, list_dir, and add_note verified")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.perf_counter()

    errors: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_p = Path(tmp_d)

        for phase_name, phase_fn, kwargs in [
            ("Phase 0: Setup & Scoping", test_phase_0_setup, {}),
            ("Phase 1: Core Agent Loop", test_phase_1_agent_loop, {}),
            ("Phase 2: Memory & Skills", test_phase_2_memory_and_skills, {"tmp_path": tmp_p}),
        ]:
            try:
                phase_fn(**kwargs)
            except Exception as exc:
                errors.append((phase_name, str(exc)))
                print(f"\n{FAIL} {phase_name} raised: {exc}")

    elapsed = time.perf_counter() - t_total

    print("\n" + "=" * 64)
    if not errors:
        print(f"ALL 3 PHASES PASSED (3 runs each) in {elapsed:.1f}s")
    else:
        print(f"FAILED: {len(errors)} PHASE(S) FAILED out of 3:")
        for name, err in errors:
            print(f"   [FAIL] {name}: {err}")
    print("=" * 64)

    sys.exit(0 if not errors else 1)
