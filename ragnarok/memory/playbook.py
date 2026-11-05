"""
ragnarok/memory/playbook.py
────────────────────────────
Innovation #11 — Experience Replay Buffer & Self-Reflective "Playbooks"

Records proven multi-step workflows as parameterised "playbooks":
  • TrajectoryStep    — one tool call within a workflow
  • WorkflowTrajectory — full sequence of steps for a completed goal
  • PlaybookStore      — SQLite persistence with parameterised template slots
  • PlaybookExtractor  — records, matches, and replays playbooks
  • MANIFEST + run()  — skill integration

Usage as skill:
    run({"action": "list"})
    run({"action": "search", "query": "deploy docker container"})
    run({"action": "record", "goal": "...", "steps_json": "[...]"})
    run({"action": "replay", "playbook_id": "...", "params_json": "{}"})
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """One step (tool call) in a workflow trajectory."""
    tool_name: str
    args: dict
    result: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkflowTrajectory:
    """A complete workflow from goal to outcome."""
    goal: str
    outcome: str
    steps: list[TrajectoryStep]
    tools_used: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    def __post_init__(self) -> None:
        if not self.tools_used:
            self.tools_used = list({s.tool_name for s in self.steps})


@dataclass
class Playbook:
    """
    An abstracted, reusable recipe extracted from a WorkflowTrajectory.
    Parameter slots use double-brace syntax: {{PARAM_NAME}}.
    """
    id: str
    name: str
    description: str
    template_steps: list[dict]   # steps with {{SLOT}} placeholders
    param_slots: list[str]       # list of slot names
    created_at: float = field(default_factory=time.time)
    use_count: int = 0
    embedding: bytes = b""


# ── PlaybookStore ─────────────────────────────────────────────────────────────

class PlaybookStore:
    """
    SQLite-backed store for Playbooks.
    Thread-safe via per-thread connections.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS playbooks (
                    id             TEXT PRIMARY KEY,
                    name           TEXT NOT NULL,
                    description    TEXT NOT NULL,
                    template_steps TEXT NOT NULL,
                    param_slots    TEXT NOT NULL,
                    created_at     REAL NOT NULL,
                    use_count      INTEGER DEFAULT 0,
                    embedding      BLOB
                );
                CREATE TABLE IF NOT EXISTS trajectory_steps (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    playbook_id TEXT NOT NULL,
                    step_order  INTEGER NOT NULL,
                    tool_name   TEXT NOT NULL,
                    args_json   TEXT NOT NULL,
                    result      TEXT NOT NULL,
                    timestamp   REAL NOT NULL
                );
            """)

    # --- CRUD ----------------------------------------------------------------

    def save(self, pb: Playbook) -> None:
        """Insert or update a playbook."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO playbooks
                   (id, name, description, template_steps, param_slots,
                    created_at, use_count, embedding)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    pb.id, pb.name, pb.description,
                    json.dumps(pb.template_steps),
                    json.dumps(pb.param_slots),
                    pb.created_at, pb.use_count,
                    pb.embedding or b"",
                ),
            )

    def get(self, pb_id: str) -> Playbook | None:
        row = self._conn().execute(
            "SELECT * FROM playbooks WHERE id = ?", (pb_id,)
        ).fetchone()
        return self._row_to_pb(row) if row else None

    def list_all(self) -> list[Playbook]:
        rows = self._conn().execute(
            "SELECT * FROM playbooks ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_pb(r) for r in rows]

    def increment_use(self, pb_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE playbooks SET use_count = use_count + 1 WHERE id = ?",
                (pb_id,),
            )

    def _row_to_pb(self, row: sqlite3.Row) -> Playbook:
        return Playbook(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            template_steps=json.loads(row["template_steps"]),
            param_slots=json.loads(row["param_slots"]),
            created_at=row["created_at"],
            use_count=row["use_count"],
            embedding=row["embedding"] or b"",
        )

    def delete(self, pb_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM playbooks WHERE id = ?", (pb_id,))
            conn.execute("DELETE FROM trajectory_steps WHERE playbook_id = ?", (pb_id,))
        return cur.rowcount > 0

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ── PlaybookExtractor ─────────────────────────────────────────────────────────

class PlaybookExtractor:
    """
    Records trajectories → abstracts them into Playbooks → matches & replays.
    """

    def __init__(self, store: PlaybookStore, model: str | None = None) -> None:
        from ragnarok.config import settings  # late import
        self.store = store
        self.model = model or settings.model_main

    # ── private LLM helper ────────────────────────────────────────────────────

    def _chat(self, system: str, user: str) -> str:
        """Single-turn local LLM call."""
        from ragnarok import llm  # late import
        try:
            result = llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                model=self.model,
            )
            # llm.chat returns ChatResult; extract .content
            text = result.content if hasattr(result, "content") else str(result)
            return (text or "").strip()
        except Exception as exc:
            logger.warning("Playbook LLM call failed: %s", exc)
            return ""

    def _embed(self, text: str) -> bytes:
        """Compute embedding for similarity search. Returns raw float32 bytes."""
        try:
            from ragnarok import llm  # late import
            vec = llm.embed(text)   # returns list[float]
            if vec:
                import struct
                return struct.pack(f"{len(vec)}f", *vec)
        except Exception as exc:
            logger.debug("Embedding failed: %s", exc)
        return b""

    def _cosine(self, a: bytes, b: bytes) -> float:
        """Cosine similarity between two float32 byte arrays."""
        import struct
        import math
        if not a or not b:
            return 0.0
        try:
            n = len(a) // 4
            if len(b) // 4 != n:
                return 0.0
            va = struct.unpack(f"{n}f", a)
            vb = struct.unpack(f"{n}f", b)
            dot = sum(x * y for x, y in zip(va, vb))
            mag_a = math.sqrt(sum(x * x for x in va))
            mag_b = math.sqrt(sum(x * x for x in vb))
            if mag_a == 0 or mag_b == 0:
                return 0.0
            return dot / (mag_a * mag_b)
        except Exception:
            return 0.0

    # ── record ────────────────────────────────────────────────────────────────

    def record(self, trajectory: WorkflowTrajectory) -> Playbook:
        """
        Abstract a trajectory into a reusable Playbook using the LLM,
        then persist it in the store.
        """
        steps_text = json.dumps(
            [{"tool": s.tool_name, "args": s.args, "result": s.result[:100]}
             for s in trajectory.steps],
            indent=2,
        )

        system = """You are a workflow abstraction expert.
Given a completed task trajectory, extract a reusable playbook.
Respond ONLY with valid JSON (no markdown):
{
  "name": "short name (max 6 words)",
  "description": "one sentence describing when to use this playbook",
  "template_steps": [
    {"tool": "...", "args": {"key": "{{PARAM_NAME}}"}, "note": "..."},
    ...
  ],
  "param_slots": ["PARAM_NAME", ...]
}
Replace specific values with {{SLOT_NAME}} placeholders for reusability."""

        raw = self._chat(
            system,
            f"Goal: {trajectory.goal}\nOutcome: {trajectory.outcome}\n\nSteps:\n{steps_text}",
        )

        # Parse
        json_str = raw
        if "```" in raw:
            for part in raw.split("```"):
                if "{" in part:
                    json_str = part.strip().lstrip("json").strip()
                    break

        try:
            data = json.loads(json_str)
            pb = Playbook(
                id=str(uuid.uuid4()),
                name=data.get("name", trajectory.goal[:40]),
                description=data.get("description", trajectory.outcome[:100]),
                template_steps=data.get("template_steps", []),
                param_slots=data.get("param_slots", []),
            )
        except Exception as exc:
            logger.warning("Playbook LLM abstraction failed (%s); creating raw playbook", exc)
            pb = Playbook(
                id=str(uuid.uuid4()),
                name=trajectory.goal[:40],
                description=trajectory.outcome[:100],
                template_steps=[
                    {"tool": s.tool_name, "args": s.args}
                    for s in trajectory.steps
                ],
                param_slots=[],
            )

        # Embed description for similarity search
        pb.embedding = self._embed(pb.name + " " + pb.description)
        self.store.save(pb)
        logger.info("Recorded playbook '%s' (id=%s)", pb.name, pb.id)
        return pb

    # ── match ─────────────────────────────────────────────────────────────────

    def match(
        self, goal: str, threshold: float = 0.80
    ) -> list[tuple[Playbook, float]]:
        """
        Semantic search for playbooks similar to the given goal.
        Returns list of (playbook, score) sorted by score descending.
        """
        query_emb = self._embed(goal)
        playbooks = self.store.list_all()
        scored = []
        for pb in playbooks:
            score = self._cosine(query_emb, pb.embedding)
            if score >= threshold:
                scored.append((pb, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── replay ────────────────────────────────────────────────────────────────

    def replay(self, pb: Playbook, params: dict[str, str]) -> str:
        """
        Fill parameter slots in the playbook template and execute the steps.
        Returns a summary of execution results.
        """
        self.store.increment_use(pb.id)
        results: list[str] = [f"Replaying playbook: '{pb.name}'"]

        for i, step in enumerate(pb.template_steps, start=1):
            tool_name = step.get("tool", "")
            raw_args = step.get("args", {})

            # Fill template slots
            filled_args: dict[str, Any] = {}
            for k, v in raw_args.items():
                if isinstance(v, str):
                    for slot, replacement in params.items():
                        v = v.replace(f"{{{{{slot}}}}}", replacement)
                filled_args[k] = v

            # Dispatch via skill registry
            try:
                from ragnarok.skills.registry import registry
                skill = registry.get(tool_name)
                if skill:
                    result = skill.run(filled_args)
                else:
                    result = f"[skill '{tool_name}' not found — skipping]"
            except Exception as exc:
                result = f"[step {i} error: {exc}]"

            results.append(f"  Step {i} ({tool_name}): {result[:200]}")
            logger.debug("Playbook replay step %d (%s): done", i, tool_name)

        return "\n".join(results)


# ── Singleton ─────────────────────────────────────────────────────────────────

_extractor: PlaybookExtractor | None = None


def get_extractor() -> PlaybookExtractor:
    """Return the module-level PlaybookExtractor singleton."""
    global _extractor
    if _extractor is None:
        from ragnarok.config import settings
        db_path = str(Path(settings.db_dir) / "playbooks.db")
        store = PlaybookStore(db_path)
        _extractor = PlaybookExtractor(store)
    return _extractor


# ── Skill registration ────────────────────────────────────────────────────────

MANIFEST = SkillManifest(
    name="playbook",
    description=(
        "Experience Replay Buffer & Self-Reflective Playbooks. "
        "Records proven multi-step workflows as reusable parameterised recipes, "
        "semantically matches new goals to past successes, and fast-replays "
        "proven workflows. Actions: list | search | record | replay."
    ),
    risk_level="low",
    required_permissions=["memory"],
    parameters={
        "action": {
            "type": "string",
            "description": "Operation: list, search, record, or replay",
            "enum": ["list", "search", "record", "replay"],
        },
        "query": {
            "type": "string",
            "description": "Search query (for action=search)",
        },
        "goal": {
            "type": "string",
            "description": "Goal description (for action=record)",
        },
        "steps_json": {
            "type": "string",
            "description": (
                "JSON array of steps for action=record. "
                "Each step: {\"tool_name\": \"...\", \"args\": {...}, \"result\": \"...\"}"
            ),
        },
        "playbook_id": {
            "type": "string",
            "description": "Playbook ID (for action=replay)",
        },
        "params_json": {
            "type": "string",
            "description": "JSON dict of slot→value pairs (for action=replay)",
        },
    },
    required=["action"],
)


def run(args: dict) -> str:
    """
    Skill entry-point for playbook.

    Supported actions:
        list              — list all saved playbooks
        search            — semantic search (requires 'query')
        record            — record a new playbook (requires 'goal', 'steps_json')
        replay            — replay a playbook (requires 'playbook_id', optional 'params_json')

    Returns:
        str: human-readable result.
    """
    action = str(args.get("action", "list")).lower()
    extractor = get_extractor()
    store = extractor.store

    try:
        # ── list ──────────────────────────────────────────────────────────────
        if action == "list":
            playbooks = store.list_all()
            if not playbooks:
                return "No playbooks saved yet."
            lines = ["Saved Playbooks:"]
            for pb in playbooks:
                ts = time.strftime("%Y-%m-%d", time.localtime(pb.created_at))
                lines.append(
                    f"  [{pb.id[:8]}] '{pb.name}' — {pb.description[:60]} "
                    f"(used {pb.use_count}x, created {ts})"
                )
            return "\n".join(lines)

        # ── search ────────────────────────────────────────────────────────────
        elif action == "search":
            query = str(args.get("query", "")).strip()
            if not query:
                return "ERROR: 'query' required for action=search."
            matches = extractor.match(query, threshold=0.60)
            if not matches:
                return f"No playbooks found matching '{query}'."
            lines = [f"Playbooks matching '{query}':"]
            for pb, score in matches:
                lines.append(
                    f"  [{pb.id[:8]}] '{pb.name}' (score={score:.2f}) — {pb.description[:80]}"
                )
            return "\n".join(lines)

        # ── record ────────────────────────────────────────────────────────────
        elif action == "record":
            goal = str(args.get("goal", "")).strip()
            steps_json = str(args.get("steps_json", "[]"))
            if not goal:
                return "ERROR: 'goal' required for action=record."
            try:
                raw_steps = json.loads(steps_json)
            except json.JSONDecodeError as exc:
                return f"ERROR: invalid steps_json — {exc}"

            steps = [
                TrajectoryStep(
                    tool_name=s.get("tool_name", ""),
                    args=s.get("args", {}),
                    result=s.get("result", ""),
                    timestamp=s.get("timestamp", time.time()),
                )
                for s in raw_steps
            ]
            trajectory = WorkflowTrajectory(
                goal=goal,
                outcome="Recorded manually via playbook skill",
                steps=steps,
            )
            pb = extractor.record(trajectory)
            return (
                f"Playbook recorded!\n"
                f"  ID: {pb.id}\n"
                f"  Name: {pb.name}\n"
                f"  Slots: {pb.param_slots}\n"
                f"  Steps: {len(pb.template_steps)}"
            )

        # ── replay ────────────────────────────────────────────────────────────
        elif action == "replay":
            pb_id = str(args.get("playbook_id", "")).strip()
            params_json = str(args.get("params_json", "{}"))
            if not pb_id:
                return "ERROR: 'playbook_id' required for action=replay."

            # Fuzzy match by prefix if full UUID not provided
            pb = store.get(pb_id)
            if not pb:
                # Try prefix match
                for candidate in store.list_all():
                    if candidate.id.startswith(pb_id):
                        pb = candidate
                        break
            if not pb:
                return f"ERROR: Playbook '{pb_id}' not found."

            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            return extractor.replay(pb, params)

        else:
            return f"ERROR: Unknown action '{action}'. Use: list | search | record | replay"

    except Exception as exc:
        logger.exception("Playbook skill run failed: %s", exc)
        return f"ERROR: {exc}"
