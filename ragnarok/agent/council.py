"""
ragnarok/agent/council.py
─────────────────────────
Innovation #8 — Hierarchical Multi-Agent Governance (Orchestrator + Critic Swarm)

Architecture:
  • ARCHITECT role: decomposes a goal into a JSON DAG of sub-tasks
  • CRITIC   role: reviews the plan and surfaces objections / risks
  • WORKER   role: executes each node using existing skills / LLM
  • SYNTHESIZER role: condenses worker outputs into a coherent final answer

The module also registers itself as a Ragnarok skill (MANIFEST + run) so the
agent loop can invoke it via tool-call like any other skill.

Usage as skill:
    run({"goal": "...", "mode": "plan|critique|execute|full"})
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

# ── Roles ─────────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    ARCHITECT   = "architect"
    CRITIC      = "critic"
    WORKER      = "worker"
    SYNTHESIZER = "synthesizer"


# ── Task DAG ──────────────────────────────────────────────────────────────────

@dataclass
class TaskNode:
    """A single node in the execution DAG."""
    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"   # pending | running | done | failed
    result: str = ""


class TaskDAG:
    """Directed-acyclic-graph of TaskNodes with topological execution ordering."""

    def __init__(self, goal: str, nodes: list[TaskNode]) -> None:
        self.goal = goal
        self.nodes: dict[str, TaskNode] = {n.id: n for n in nodes}

    # --- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "tasks": [
                {
                    "id": n.id,
                    "description": n.description,
                    "dependencies": n.dependencies,
                    "status": n.status,
                    "result": n.result,
                }
                for n in self.nodes.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskDAG":
        nodes = [
            TaskNode(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
                status=t.get("status", "pending"),
                result=t.get("result", ""),
            )
            for t in data.get("tasks", [])
        ]
        return cls(goal=data.get("goal", ""), nodes=nodes)

    # --- traversal -----------------------------------------------------------

    def topological_order(self) -> list[str]:
        """Kahn's algorithm — returns node ids in dependency-safe execution order."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.id] = in_degree[node.id] + 1

        # Recompute properly
        in_degree = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in self.nodes:
                    in_degree[node.id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for child in self.nodes.values():
                if nid in child.dependencies:
                    in_degree[child.id] -= 1
                    if in_degree[child.id] == 0:
                        queue.append(child.id)
        return order


# ── Critique Report ───────────────────────────────────────────────────────────

@dataclass
class CritiqueReport:
    objections: list[str]
    risk_score: int        # 0-10
    recommendations: list[str]

    def to_str(self) -> str:
        lines = [f"Risk score: {self.risk_score}/10"]
        if self.objections:
            lines.append("Objections:")
            lines.extend(f"  • {o}" for o in self.objections)
        if self.recommendations:
            lines.append("Recommendations:")
            lines.extend(f"  • {r}" for r in self.recommendations)
        return "\n".join(lines)


# ── AgentCouncil ──────────────────────────────────────────────────────────────

class AgentCouncil:
    """
    Orchestrates a four-role council to plan, criticise, execute, and synthesise
    a user goal entirely offline via local Ollama LLM calls.
    """

    def __init__(self, model: str | None = None, max_dag_depth: int = 5) -> None:
        from ragnarok.config import settings  # late import avoids circular deps
        self.model = model or settings.model_main
        self.max_dag_depth = max_dag_depth

    # ── private LLM helper ────────────────────────────────────────────────────

    def _chat(self, system: str, user: str) -> str:
        """Single-turn LLM call. Returns stripped text response."""
        from ragnarok import llm
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
            logger.warning("Council LLM call failed: %s", exc)
            return f"[LLM unavailable: {exc}]"

    # ── ARCHITECT: decompose goal into DAG ───────────────────────────────────

    def decompose(self, goal: str) -> TaskDAG:
        """
        Ask the ARCHITECT to decompose the goal into a JSON task DAG.
        Falls back to a single-node DAG when the LLM returns malformed JSON.
        """
        system = textwrap.dedent("""\
            You are the ARCHITECT agent in a multi-agent council.
            Your job: decompose the user's goal into 3–6 concrete sub-tasks
            arranged as a dependency graph.

            Respond ONLY with valid JSON in this format (no markdown fences):
            {
              "tasks": [
                {"id": "t1", "description": "...", "dependencies": []},
                {"id": "t2", "description": "...", "dependencies": ["t1"]},
                ...
              ]
            }

            Rules:
            - Use short alphanumeric ids like t1, t2, t3.
            - dependencies must list ids of tasks that must finish first.
            - Keep each description under 20 words.
            - Maximum depth: """ + str(self.max_dag_depth))

        raw = self._chat(system, f"Goal: {goal}")

        # Parse JSON — be forgiving of markdown code fences
        json_str = raw
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                if "{" in part:
                    json_str = part.strip().lstrip("json").strip()
                    break

        try:
            data = json.loads(json_str)
            nodes = [
                TaskNode(
                    id=t.get("id", str(uuid.uuid4())[:4]),
                    description=t.get("description", ""),
                    dependencies=t.get("dependencies", []),
                )
                for t in data.get("tasks", [])
            ]
            if not nodes:
                raise ValueError("empty tasks")
        except Exception as exc:
            logger.warning("ARCHITECT produced invalid JSON (%s); using fallback DAG", exc)
            nodes = [TaskNode(id="t1", description=goal[:200])]

        return TaskDAG(goal=goal, nodes=nodes)

    # ── CRITIC: review the plan ───────────────────────────────────────────────

    def critique(self, dag: TaskDAG) -> CritiqueReport:
        """
        Ask the CRITIC agent to evaluate the plan and return a CritiqueReport.
        """
        plan_json = json.dumps(dag.to_dict(), indent=2)
        system = textwrap.dedent("""\
            You are the CRITIC agent in a multi-agent council.
            Evaluate the following plan for risks, gaps, and logical flaws.

            Respond ONLY with valid JSON (no markdown):
            {
              "objections": ["...", "..."],
              "risk_score": 3,
              "recommendations": ["...", "..."]
            }
            risk_score is 0 (safe) to 10 (dangerous). Keep each item under 15 words.""")

        raw = self._chat(system, f"Plan to critique:\n{plan_json}")

        json_str = raw
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                if "{" in part:
                    json_str = part.strip().lstrip("json").strip()
                    break

        try:
            data = json.loads(json_str)
            return CritiqueReport(
                objections=data.get("objections", []),
                risk_score=int(data.get("risk_score", 3)),
                recommendations=data.get("recommendations", []),
            )
        except Exception as exc:
            logger.warning("CRITIC produced invalid JSON (%s); using fallback report", exc)
            return CritiqueReport(
                objections=["Could not parse critique"],
                risk_score=5,
                recommendations=["Review plan manually"],
            )

    # ── WORKER: execute each node ─────────────────────────────────────────────

    def execute(self, dag: TaskDAG) -> list[tuple[str, str]]:
        """
        Execute nodes in topological order using WORKER agent.
        Returns list of (node_id, result) pairs.
        """
        system = textwrap.dedent("""\
            You are a WORKER agent. Your job: carry out the specific sub-task described.
            Respond concisely — give the result or a clear action description.
            Max 150 words per response.""")

        order = dag.topological_order()
        results: list[tuple[str, str]] = []
        context_so_far = ""

        for nid in order:
            node = dag.nodes.get(nid)
            if not node:
                continue
            node.status = "running"
            prompt = f"Sub-task: {node.description}"
            if context_so_far:
                prompt += f"\n\nContext from previous steps:\n{context_so_far[-800:]}"

            result = self._chat(system, prompt)
            node.result = result
            node.status = "done"
            context_so_far += f"\n[{nid}] {result}"
            results.append((nid, result))
            logger.debug("Council WORKER completed node %s", nid)

        return results

    # ── SYNTHESIZER: condense results ─────────────────────────────────────────

    def synthesize(self, goal: str, results: list[tuple[str, str]]) -> str:
        """
        Ask the SYNTHESIZER to condense all worker results into a final answer.
        """
        results_text = "\n".join(f"[{nid}] {res}" for nid, res in results)
        system = textwrap.dedent("""\
            You are the SYNTHESIZER agent.
            Combine the sub-task results below into a clear, concise final answer
            for the original goal. Max 300 words. No bullet points — flowing prose.""")

        return self._chat(system, f"Goal: {goal}\n\nSub-task results:\n{results_text}")

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_full(self, goal: str) -> dict[str, Any]:
        """
        Runs the complete ARCHITECT → CRITIC → WORKER → SYNTHESIZER pipeline.
        Returns a dict with all intermediate artifacts.
        """
        t0 = time.time()
        dag = self.decompose(goal)
        critique = self.critique(dag)
        worker_results = self.execute(dag)
        final_answer = self.synthesize(goal, worker_results)

        return {
            "goal": goal,
            "dag": dag.to_dict(),
            "critique": {
                "objections": critique.objections,
                "risk_score": critique.risk_score,
                "recommendations": critique.recommendations,
            },
            "worker_results": worker_results,
            "final_answer": final_answer,
            "elapsed_sec": round(time.time() - t0, 2),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_council: AgentCouncil | None = None


def get_council() -> AgentCouncil:
    """Return the module-level AgentCouncil singleton."""
    global _council
    if _council is None:
        _council = AgentCouncil()
    return _council


# ── Skill registration ────────────────────────────────────────────────────────

MANIFEST = SkillManifest(
    name="council_plan",
    description=(
        "Hierarchical multi-agent council: decomposes a goal into a DAG of sub-tasks, "
        "critiques the plan for risks, executes each node with WORKER agents, and "
        "synthesizes a final answer. Modes: plan | critique | execute | full."
    ),
    risk_level="medium",
    required_permissions=["llm"],
    parameters={
        "goal": {
            "type": "string",
            "description": "The high-level goal or question to process",
        },
        "mode": {
            "type": "string",
            "description": "Operation mode: plan, critique, execute, or full",
            "enum": ["plan", "critique", "execute", "full"],
        },
    },
    required=["goal"],
)


def run(args: dict) -> str:
    """
    Skill entry-point for council_plan.

    Args:
        args: dict with keys:
            goal (str, required): The goal to process.
            mode (str, optional): plan | critique | execute | full  [default: full]

    Returns:
        str: JSON-encoded result or plain-text answer.
    """
    goal = str(args.get("goal", "")).strip()
    mode = str(args.get("mode", "full")).lower()

    if not goal:
        return "ERROR: 'goal' parameter is required."

    council = get_council()

    try:
        if mode == "plan":
            dag = council.decompose(goal)
            return json.dumps(dag.to_dict(), indent=2)

        elif mode == "critique":
            dag = council.decompose(goal)
            report = council.critique(dag)
            return report.to_str()

        elif mode == "execute":
            dag = council.decompose(goal)
            worker_results = council.execute(dag)
            return council.synthesize(goal, worker_results)

        else:  # full
            result = council.run_full(goal)
            # Return the final answer plus a summary header
            lines = [
                f"=== COUNCIL RESULT ===",
                f"Goal: {goal}",
                f"DAG tasks: {len(result['dag']['tasks'])}",
                f"Risk score: {result['critique']['risk_score']}/10",
                f"Elapsed: {result['elapsed_sec']}s",
                "",
                "=== FINAL ANSWER ===",
                result["final_answer"],
            ]
            return "\n".join(lines)

    except Exception as exc:
        logger.exception("Council skill run failed: %s", exc)
        return f"ERROR: Council failed — {exc}"
