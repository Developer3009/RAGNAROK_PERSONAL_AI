"""
ragnarok/agent/loop.py
───────────────────────
The core agent loop for Ragnarok.

Flow per turn:
  1. Add user message to history.
  2. Optionally create a multi-step Plan if the request is complex (Phase 4).
  3. Inject recalled memories + active plan context into system prompt.
  4. Call LLM with tool list.
  5. If LLM wants a tool:
       a. Validate call against schema — on failure, send error back to LLM.
       b. Rate limiter check — sliding window per tool (Phase 5).
       c. Permission check — explicit allow/deny/max_calls policy (Phase 5).
       d. Doom-loop check — block repeated identical failures (Phase 4).
       e. Risk check — high-risk pauses for approval (Phase 5 audit logged).
       f. Execute tool safely, log audit event & latency.
       g. Record call in RateLimiter, PermissionManager, and DoomLoopDetector.
       h. Generate CorrectionAdvice on error (Phase 4).
       i. Advance plan step on success (Phase 4).
       j. Append tool result to history.
       k. Loop back to step 4 (up to max_steps).
  6. When LLM gives a final answer, return it.
  7. Never loop forever — enforce max_steps hard cap.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from ragnarok import llm
from ragnarok.agent.history import MessageHistory
from ragnarok.agent.planning import Plan, Planner
from ragnarok.agent.self_correction import DoomLoopDetector, ErrorDiagnoser
from ragnarok.config import settings
from ragnarok.logging_setup import log_tool_event
from ragnarok.safety.audit import AuditLogger, audit_logger as _default_audit_logger
from ragnarok.safety.permissions import PermissionManager
from ragnarok.safety.rate_limiter import RateLimiter
from ragnarok.skills.registry import SkillRegistry, registry as _default_registry

logger = logging.getLogger(__name__)

# System prompt template — memories and plan context are injected at placeholders
_SYSTEM_TEMPLATE = """\
You are Ragnarok, a local AI assistant running entirely on this device.
You have access to tools — use them when they help answer the user's request.
Be concise, accurate, and honest about uncertainty.

{memories}{plan_context}"""


@dataclass
class AgentResult:
    answer: str
    steps_taken: int
    tool_calls_made: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    stopped_early: bool = False   # True if max_steps was hit
    plan: Plan | None = None      # The plan used this turn (if any)


class AgentLoop:
    """
    Stateful agent that manages one conversation session.

    Args:
        registry:           SkillRegistry to use (defaults to module-level singleton).
        approval_cb:        Optional callback(skill_name, args) → bool for high-risk
                            approval. If None, high-risk tools are blocked.
        memory_fn:          Optional callable(user_text) → str that returns injected
                            memory context.
        enable_planning:    Whether to use the Planner for complex multi-step tasks
                            (Phase 4). Defaults to True.
        permission_manager: Optional PermissionManager instance for ACL checks (Phase 5).
        rate_limiter:       Optional RateLimiter instance for call throttling (Phase 5).
        audit_logger:       Optional AuditLogger instance for audit trail (Phase 5).
    """

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        approval_cb: Callable[[str, dict], bool] | None = None,
        memory_fn: Callable[[str], str] | None = None,
        enable_planning: bool = True,
        permission_manager: PermissionManager | None = None,
        rate_limiter: RateLimiter | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._registry = registry or _default_registry
        self._approval_cb = approval_cb
        self._memory_fn = memory_fn
        self._enable_planning = enable_planning
        self._permissions = permission_manager or PermissionManager()
        self._rate_limiter = rate_limiter or RateLimiter()
        self._audit = audit_logger or _default_audit_logger

        self._history = MessageHistory()
        self._planner = Planner(use_llm=True)
        self._doom_detector = DoomLoopDetector(max_consecutive_failures=2)
        self._active_plan: Plan | None = None

        self._audit.log_session_start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> AgentResult:
        """
        Process one user turn and return the agent's final answer.
        """
        t_start = time.perf_counter()
        self._history.add_user(user_input)

        # ── Phase 4: Optionally create a plan ──────────────────────────────────
        if self._enable_planning and self._planner.should_plan(user_input):
            available_tools = list(self._registry._skills.keys())
            try:
                self._active_plan = self._planner.create_plan(
                    user_input, available_tools=available_tools
                )
                logger.info(
                    "Plan created: %d steps for goal=%r",
                    len(self._active_plan.steps), user_input[:60],
                )
            except Exception as exc:
                logger.warning("Plan creation failed: %s — continuing without plan", exc)
                self._active_plan = None
        else:
            self._active_plan = None

        system_prompt = self._build_system_prompt(user_input)
        tools = self._registry.as_ollama_tools()

        steps = 0
        tool_calls_made: list[str] = []
        self._doom_detector.reset()

        while steps < settings.max_steps:
            steps += 1
            logger.debug("Agent loop step %d/%d", steps, settings.max_steps)

            result = llm.chat_with_tools(
                messages=self._history.as_dicts(),
                tools=tools,
                system=system_prompt,
            )

            if not result.has_tool_calls:
                answer = result.final_content or "(no response)"
                self._history.add_assistant(answer)
                if self._active_plan:
                    self._active_plan.advance_step(result=answer[:120])
                return AgentResult(
                    answer=answer,
                    steps_taken=steps,
                    tool_calls_made=tool_calls_made,
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                    plan=self._active_plan,
                )

            # ── Process tool calls ─────────────────────────────────────────────
            for tc in result.tool_calls:
                tool_name = tc.name
                args = tc.arguments
                tool_calls_made.append(tool_name)

                skill = self._registry.get(tool_name)
                risk_level = skill.manifest.risk_level if skill else "medium"

                logger.info("Tool requested: %s args=%s", tool_name, args)
                log_tool_event(logger, "call", tool_name, arguments=args)
                self._audit.log_tool_call(tool_name, args, risk_level=risk_level)

                # 1. Validate schema
                valid, err_msg = self._registry.validate_call(tool_name, args)
                if not valid:
                    log_tool_event(logger, "schema_fail", tool_name, error=err_msg)
                    self._audit.log_tool_error(tool_name, f"Schema fail: {err_msg}")
                    advice = ErrorDiagnoser.diagnose(tool_name, args, err_msg)
                    self._history.add_tool_result(
                        tool_name,
                        f"ERROR: {err_msg}\n{advice.format_prompt()}"
                    )
                    continue

                # 2. Phase 5: Rate Limiting
                rate_ok, retry_after = self._rate_limiter.check_and_record(tool_name)
                if not rate_ok:
                    rule = self._rate_limiter._rules.get(tool_name)
                    window_sec = rule.window_sec if rule else 60.0
                    limit = rule.max_calls if rule else 0
                    self._audit.log_rate_limited(tool_name, window_sec, limit)
                    msg = (
                        f"ERROR: Rate limit exceeded for tool '{tool_name}'. "
                        f"Please retry after {retry_after:.1f} seconds."
                    )
                    self._history.add_tool_result(tool_name, msg)
                    continue

                # 3. Phase 5: Permission & Policy Check
                perm_check = self._permissions.check(tool_name, risk_level=risk_level, args=args)
                if not perm_check.permitted:
                    self._audit.log_permission_denied(tool_name, perm_check.reason)
                    self._history.add_tool_result(
                        tool_name,
                        f"ERROR: Permission denied: {perm_check.reason}"
                    )
                    continue

                # 4. Phase 4: Doom-loop check before execution
                is_loop, loop_warning = self._doom_detector.check_loop(tool_name, args)
                if is_loop:
                    log_tool_event(logger, "doom_loop", tool_name, error=loop_warning)
                    self._audit.log_doom_loop(tool_name)
                    self._history.add_tool_result(tool_name, loop_warning)
                    if self._active_plan:
                        self._active_plan.fail_current_step(error=f"Doom loop: {tool_name}")
                    continue

                # 5. Risk & Approval check
                if perm_check.requires_approval or risk_level == "high":
                    self._audit.log_approval_requested(tool_name, risk_level)
                    approved = self._request_approval(tool_name, args)
                    if not approved:
                        self._audit.log_approval_denied(tool_name)
                        msg = f"Action '{tool_name}' was not approved. Skipping."
                        log_tool_event(logger, "denied", tool_name, error=msg)
                        self._history.add_tool_result(tool_name, msg)
                        continue
                    self._audit.log_approval_granted(tool_name)

                # 6. Execute tool
                t_tool = time.perf_counter()
                try:
                    tool_result = skill.run(args)
                    latency_ms = (time.perf_counter() - t_tool) * 1000
                    log_tool_event(
                        logger, "result", tool_name,
                        result=tool_result, latency_ms=latency_ms
                    )
                    self._audit.log_tool_success(tool_name, latency_ms, len(tool_result))
                    self._permissions.record_call(tool_name)
                    self._doom_detector.record_call(tool_name, args, tool_result)
                    if self._active_plan:
                        self._active_plan.advance_step(result=tool_result[:120])
                except Exception as exc:
                    latency_ms = (time.perf_counter() - t_tool) * 1000
                    tool_result = f"ERROR: Tool '{tool_name}' raised an exception: {exc}"
                    log_tool_event(
                        logger, "error", tool_name, error=str(exc), latency_ms=latency_ms
                    )
                    self._audit.log_tool_error(tool_name, str(exc), latency_ms)
                    logger.exception("Tool %s failed", tool_name)
                    advice = ErrorDiagnoser.diagnose(tool_name, args, str(exc))
                    tool_result = f"{tool_result}\n{advice.format_prompt()}"
                    self._doom_detector.record_call(tool_name, args, tool_result)
                    if self._active_plan:
                        self._active_plan = self._planner.replan_on_failure(
                            self._active_plan,
                            self._active_plan.active_step().step_id if self._active_plan.active_step() else -1,
                            str(exc),
                        )

                self._history.add_tool_result(tool_name, tool_result)

            if self._active_plan:
                system_prompt = self._build_system_prompt(user_input)

        # ── Max steps exceeded ─────────────────────────────────────────────────
        logger.warning("Agent hit max_steps=%d — forcing final answer", settings.max_steps)
        final = llm.chat(
            messages=self._history.as_dicts(),
            system=system_prompt + "\n\nYou have reached the step limit. Give the best answer you can now.",
        )
        self._history.add_assistant(final.content)
        return AgentResult(
            answer=final.content,
            steps_taken=steps,
            tool_calls_made=tool_calls_made,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            stopped_early=True,
            plan=self._active_plan,
        )

    def reset(self) -> None:
        """Clear conversation history, plan, and reset safety session states."""
        self._history.clear()
        self._active_plan = None
        self._doom_detector.reset()
        self._permissions.reset()
        self._rate_limiter.reset()
        self._audit.log_session_reset()

    def history_snapshot(self) -> list[dict]:
        return self._history.snapshot()

    @property
    def active_plan(self) -> Plan | None:
        return self._active_plan

    @property
    def permissions(self) -> PermissionManager:
        return self._permissions

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self, user_input: str) -> str:
        memories = ""
        if self._memory_fn:
            try:
                memories = self._memory_fn(user_input)
                if memories:
                    memories = f"Relevant memories:\n{memories}\n\n"
            except Exception as exc:
                logger.warning("Memory recall failed: %s", exc)

        plan_context = ""
        if self._active_plan and self._active_plan.steps:
            plan_context = f"\n{self._active_plan.format_prompt_context()}\n"

        return _SYSTEM_TEMPLATE.format(memories=memories, plan_context=plan_context)

    def _request_approval(self, tool_name: str, args: dict) -> bool:
        if self._approval_cb is None:
            logger.warning(
                "High-risk tool '%s' blocked — no approval callback configured.", tool_name
            )
            return False
        try:
            return self._approval_cb(tool_name, args)
        except Exception as exc:
            logger.error("Approval callback failed: %s", exc)
            return False
