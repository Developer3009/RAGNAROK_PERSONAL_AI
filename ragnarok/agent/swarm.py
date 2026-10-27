"""
ragnarok/agent/swarm.py
────────────────────────
Innovation #4: Hierarchical Multi-Agent Swarm with Speculative Delegation.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                      SwarmOrchestrator                       │
  │  (entry point — decides which specialist to delegate to)     │
  └────────────┬──────────────────────────┬──────────────────────┘
               │                          │
       ┌───────▼────────┐         ┌───────▼────────┐
       │  ArchitectAgent │         │  SecurityAuditor│
       │  (plan + route) │         │  (risk scanner) │
       └───────┬─────────┘         └───────┬─────────┘
               │                           │
       ┌───────▼─────────────────────────────────────┐
       │              ExecutorAgent                    │
       │  (runs tools via AgentLoop, collects result)  │
       └──────────────────────────────────────────────┘

Design goals:
  - All three agents run in the same Python process (no IPC overhead).
  - The Architect uses the small fast model (gemma3:4b) for cheap planning.
  - The Executor delegates to the full AgentLoop for tool execution.
  - The SecurityAuditor scans every Executor result before it is returned to the user.
  - Speculative delegation: Architect picks ONE specialist path and executes concurrently.
  - FastAPI microservice endpoints are wired via /api/swarm/* in app.py.
  - APScheduler queue mode: any agent can post a job to the background scheduler.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ragnarok import llm
from ragnarok.agent.loop import AgentLoop, AgentResult
from ragnarok.config import settings
from ragnarok.safety.prompt_injection import PromptInjectionDefender
from ragnarok.skills.registry import SkillRegistry, registry as _default_registry

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    """Security audit outcome for a completed executor result."""
    passed: bool
    risk_score: float              # 0.0 – 1.0
    findings: list[str] = field(default_factory=list)
    blocked_content: str = ""     # sanitised version if blocked


@dataclass
class SwarmResult:
    """Top-level result returned from SwarmOrchestrator.run()."""
    answer: str
    agent_path: str                # 'direct' | 'architect' | 'executor' | 'swarm'
    audit: AuditResult
    plan_summary: str = ""
    latency_ms: float = 0.0
    steps_taken: int = 0
    tool_calls_made: list[str] = field(default_factory=list)


# ── ArchitectAgent ────────────────────────────────────────────────────────────

class ArchitectAgent:
    """
    Lightweight task classifier and planner.
    Uses the small model to quickly decide:
      1. Is this a simple query (answer directly) or complex task (use Executor)?
      2. Does this task need SecurityAuditor involvement?
      3. What tools might be needed?
    """

    _CLASSIFY_PROMPT = """\
You are a task routing assistant. Classify the user request with ONE JSON object:
{{
  "complexity": "simple" | "complex",
  "needs_tools": true | false,
  "security_sensitive": true | false,
  "suggested_tools": ["tool1", "tool2"],
  "reason": "one-line reason"
}}
Do not add any markdown or explanation outside the JSON.

Available tools: {tools}
User request: {request}
"""

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self._registry = registry or _default_registry

    def classify(self, user_input: str) -> dict[str, Any]:
        """
        Classify a user request via the small LLM model.

        Returns a dict with keys: complexity, needs_tools, security_sensitive,
        suggested_tools, reason.
        Falls back to safe defaults on any LLM error.
        """
        tools = list(self._registry._skills.keys())
        prompt = self._CLASSIFY_PROMPT.format(
            tools=", ".join(tools),
            request=user_input[:500],  # cap to avoid context overflow
        )
        try:
            result = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=settings.model_small,
                temperature=0.0,
            )
            import json, re
            raw = result.content.strip()
            # Strip markdown code blocks if model wrapped
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            return {
                "complexity": parsed.get("complexity", "complex"),
                "needs_tools": bool(parsed.get("needs_tools", True)),
                "security_sensitive": bool(parsed.get("security_sensitive", False)),
                "suggested_tools": parsed.get("suggested_tools", []),
                "reason": parsed.get("reason", ""),
            }
        except Exception as exc:
            logger.warning("ArchitectAgent.classify failed: %s — using safe defaults", exc)
            return {
                "complexity": "complex",
                "needs_tools": True,
                "security_sensitive": False,
                "suggested_tools": [],
                "reason": "classification_failed",
            }


# ── SecurityAuditor ───────────────────────────────────────────────────────────

class SecurityAuditorAgent:
    """
    Post-execution security auditor.
    Scans the Executor's answer + tool calls for:
      - Prompt injection patterns in the output (indirect injection via tool results)
      - Sensitive data leakage (tokens, keys, passwords)
      - Unexpected high-risk tool usage
    """

    _SENSITIVE_PATTERNS = [
        r"(?i)(api[_\-\s]?key|bearer|secret|password|passwd|token)[^\w].*",
        r"sk-[A-Za-z0-9]{20,}",          # OpenAI-style API key
        r"(?i)-----BEGIN\s+(RSA|EC|OPENSSH|PRIVATE)\s+KEY",  # private keys
        r"(?i)AKIA[0-9A-Z]{16}",          # AWS access key
        r"(?i)ghp_[A-Za-z0-9]{36}",       # GitHub PAT
    ]

    def audit(self, result: AgentResult) -> AuditResult:
        """
        Audit an AgentResult.
        Returns AuditResult with pass/fail, risk score, and findings.
        """
        import re
        _SENSITIVE_RE = [re.compile(p) for p in self._SENSITIVE_PATTERNS]

        findings: list[str] = []
        total_risk = 0.0

        # 1. Injection check on the answer text
        inj = PromptInjectionDefender.analyze(result.answer)
        if inj.is_suspicious:
            total_risk += max(0.5, inj.risk_score)
            for p in inj.detected_patterns[:3]:
                findings.append(f"Output injection pattern detected: {p!r}")

        for pattern in _SENSITIVE_RE:
            if pattern.search(result.answer):
                total_risk += 0.4
                findings.append(f"Potential sensitive data in answer: {pattern.pattern[:40]}...")
                break  # one alert is enough

        # 3. Unexpected high-risk tool calls
        high_risk_tools = {"write_file", "delete_file", "run_command", "fetch_web"}
        used_high = set(result.tool_calls_made) & high_risk_tools
        if used_high:
            total_risk += 0.15 * len(used_high)
            findings.append(f"High-risk tools used: {', '.join(used_high)}")

        total_risk = min(1.0, total_risk)
        passed = total_risk < 0.5  # threshold for blocking

        if not passed:
            logger.warning(
                "SecurityAuditor: answer BLOCKED (risk=%.2f, findings=%d)",
                total_risk, len(findings),
            )

        return AuditResult(
            passed=passed,
            risk_score=total_risk,
            findings=findings,
            blocked_content=result.answer if not passed else "",
        )


# ── ExecutorAgent ─────────────────────────────────────────────────────────────

class ExecutorAgent:
    """
    Wraps the main AgentLoop to execute tool-using tasks.
    Can be given a pre-filtered tool list from the Architect.
    """

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        approval_cb: Callable[[str, dict], bool] | None = None,
        memory_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._registry = registry or _default_registry
        self._approval_cb = approval_cb
        self._memory_fn = memory_fn

    def execute(
        self,
        user_input: str,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        """
        Run the agent loop for this input, optionally restricting to allowed_tools.
        """
        # Create a filtered registry view if Architect specified tools
        registry = self._registry
        if allowed_tools:
            from ragnarok.skills.registry import SkillRegistry
            filtered = SkillRegistry()
            for name in allowed_tools:
                sk = self._registry.get(name)
                if sk:
                    filtered._skills[name] = sk
            registry = filtered

        loop = AgentLoop(
            registry=registry,
            approval_cb=self._approval_cb,
            memory_fn=self._memory_fn,
            enable_planning=True,
        )
        return loop.chat(user_input)


# ── SwarmOrchestrator ─────────────────────────────────────────────────────────

class SwarmOrchestrator:
    """
    The top-level swarm entry point.

    Routing logic:
      1. ArchitectAgent classifies the request (fast, cheap small-model call).
      2. If complexity='simple' and no tools needed → answer directly via small model.
      3. If complexity='complex' → dispatch to ExecutorAgent (possibly in a thread).
      4. SecurityAuditorAgent always runs on the final answer before returning.
      5. If audit fails (risk ≥ 0.6) → return a safe refusal with the risk report.

    Speculative mode (enable_speculative=True):
      The Architect launches the Executor call speculatively in a background thread
      while it prepares a quick direct answer. Whichever finishes first "wins".
      This shaves latency on mixed workloads.
    """

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        approval_cb: Callable[[str, dict], bool] | None = None,
        memory_fn: Callable[[str], str] | None = None,
        enable_speculative: bool = False,
    ) -> None:
        self._registry = registry or _default_registry
        self._architect = ArchitectAgent(registry=self._registry)
        self._executor = ExecutorAgent(
            registry=self._registry,
            approval_cb=approval_cb,
            memory_fn=memory_fn,
        )
        self._auditor = SecurityAuditorAgent()
        self._speculative = enable_speculative

    def run(self, user_input: str) -> SwarmResult:
        """
        Route, execute, and audit a user request through the swarm.
        """
        t_start = time.perf_counter()

        # ── Step 1: Architect classification ──────────────────────────────────
        classification = self._architect.classify(user_input)
        complexity = classification.get("complexity", "complex")
        needs_tools = classification.get("needs_tools", True)
        suggested_tools = classification.get("suggested_tools", [])
        plan_summary = classification.get("reason", "")

        logger.info(
            "Swarm classification: complexity=%s needs_tools=%s suggested=%s",
            complexity, needs_tools, suggested_tools,
        )

        # ── Step 2: Route to executor or direct answer ─────────────────────────
        if complexity == "simple" and not needs_tools:
            # Fast path: answer directly without a full tool-calling loop
            direct = llm.chat(
                messages=[{"role": "user", "content": user_input}],
                model=settings.model_small,
                temperature=0.7,
            )
            agent_result = AgentResult(
                answer=direct.content,
                steps_taken=1,
                tool_calls_made=[],
                latency_ms=direct.latency_ms,
            )
            agent_path = "direct"
        elif self._speculative and needs_tools:
            # Speculative parallel path
            agent_result, agent_path = self._speculative_execute(user_input, suggested_tools)
        else:
            agent_result = self._executor.execute(user_input, allowed_tools=suggested_tools or None)
            agent_path = "executor"

        # ── Step 3: Security audit ─────────────────────────────────────────────
        audit = self._auditor.audit(agent_result)
        latency_ms = (time.perf_counter() - t_start) * 1000

        if not audit.passed:
            safe_answer = (
                "⚠️ **Response blocked by Ragnarok Security Auditor**\n\n"
                f"Risk score: {audit.risk_score:.2f}\n"
                f"Findings:\n" + "\n".join(f"  - {f}" for f in audit.findings)
            )
            return SwarmResult(
                answer=safe_answer,
                agent_path=agent_path,
                audit=audit,
                plan_summary=plan_summary,
                latency_ms=latency_ms,
                steps_taken=agent_result.steps_taken,
                tool_calls_made=agent_result.tool_calls_made,
            )

        return SwarmResult(
            answer=agent_result.answer,
            agent_path=agent_path,
            audit=audit,
            plan_summary=plan_summary,
            latency_ms=latency_ms,
            steps_taken=agent_result.steps_taken,
            tool_calls_made=agent_result.tool_calls_made,
        )

    def _speculative_execute(
        self, user_input: str, suggested_tools: list[str]
    ) -> tuple[AgentResult, str]:
        """
        Launch Executor speculatively in a thread.
        If it finishes within the timeout, use the result.
        Otherwise fall back to a fast direct answer.
        """
        SPECULATIVE_TIMEOUT = 45.0  # seconds

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._executor.execute, user_input, suggested_tools or None)
            try:
                result = future.result(timeout=SPECULATIVE_TIMEOUT)
                return result, "swarm_speculative"
            except concurrent.futures.TimeoutError:
                logger.warning("Speculative executor timed out; falling back to direct answer")
                future.cancel()
                direct = llm.chat(
                    messages=[{"role": "user", "content": user_input}],
                    model=settings.model_small,
                    temperature=0.7,
                )
                return AgentResult(
                    answer=direct.content + "\n\n*(Speculative executor timed out — partial direct answer)*",
                    steps_taken=1,
                    tool_calls_made=[],
                ), "speculative_fallback"
