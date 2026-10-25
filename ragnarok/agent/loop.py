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
       b. Doom-loop check — block repeated identical failures (Phase 4).
       c. Check risk level — high-risk pauses for approval.
       d. Execute tool, log result.
       e. Record call in DoomLoopDetector; generate CorrectionAdvice on error (Phase 4).
       f. Advance plan step on success (Phase 4).
       g. Append tool result to history.
       h. Loop back to step 4 (up to max_steps).
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
        registry:       SkillRegistry to use (defaults to module-level singleton).
        approval_cb:    Optional callback(skill_name, args) → bool for high-risk
                        approval. If None, high-risk tools are blocked.
        memory_fn:      Optional callable(user_text) → str that returns injected
                        memory context. Wired up in Phase 2.
        enable_planning: Whether to use the Planner for complex multi-step tasks
                        (Phase 4). Defaults to True.
    """

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        approval_cb: Callable[[str, dict], bool] | None = None,
        memory_fn: Callable[[str], str] | None = None,
        enable_planning: bool = True,
    ) -> None:
        self._registry = registry or _default_registry
        self._approval_cb = approval_cb
        self._memory_fn = memory_fn
        self._enable_planning = enable_planning
        self._history = MessageHistory()
        self._planner = Planner(use_llm=True)
        self._doom_detector = DoomLoopDetector(max_consecutive_failures=2)
        self._active_plan: Plan | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> AgentResult:
        """
        Process one user turn and return the agent's final answer.

        Args:
            user_input: The raw text from the user.

        Returns:
            AgentResult with the answer and execution metadata.
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
                # Model gave a final answer — mark active plan step as completed
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

                logger.info("Tool requested: %s args=%s", tool_name, args)
                log_tool_event(logger, "call", tool_name, arguments=args)

                # 1. Validate schema
                valid, err_msg = self._registry.validate_call(tool_name, args)
                if not valid:
                    log_tool_event(logger, "schema_fail", tool_name, error=err_msg)
                    advice = ErrorDiagnoser.diagnose(tool_name, args, err_msg)
                    self._history.add_tool_result(
                        tool_name,
                        f"ERROR: {err_msg}\n{advice.format_prompt()}"
                    )
                    continue

                # 2. Phase 4: Doom-loop check before execution
                is_loop, loop_warning = self._doom_detector.check_loop(tool_name, args)
                if is_loop:
                    log_tool_event(logger, "doom_loop", tool_name, error=loop_warning)
                    self._history.add_tool_result(tool_name, loop_warning)
                    if self._active_plan:
                        self._active_plan.fail_current_step(
                            error=f"Doom loop: {tool_name}"
                        )
                    continue

                # 3. Risk check
                skill = self._registry.get(tool_name)
                if skill and skill.manifest.risk_level == "high":
                    approved = self._request_approval(tool_name, args)
                    if not approved:
                        msg = f"Action '{tool_name}' was not approved. Skipping."
                        log_tool_event(logger, "denied", tool_name, error=msg)
                        self._history.add_tool_result(tool_name, msg)
                        continue

                # 4. Execute
                t_tool = time.perf_counter()
                try:
                    tool_result = skill.run(args)
                    latency_ms = (time.perf_counter() - t_tool) * 1000
                    log_tool_event(
                        logger, "result", tool_name,
                        result=tool_result, latency_ms=latency_ms
                    )
                    # Record success in doom detector and advance plan step
                    self._doom_detector.record_call(tool_name, args, tool_result)
                    if self._active_plan:
                        self._active_plan.advance_step(result=tool_result[:120])
                except Exception as exc:
                    latency_ms = (time.perf_counter() - t_tool) * 1000
                    tool_result = f"ERROR: Tool '{tool_name}' raised an exception: {exc}"
                    log_tool_event(
                        logger, "error", tool_name, error=str(exc), latency_ms=latency_ms
                    )
                    logger.exception("Tool %s failed", tool_name)
                    # Phase 4: diagnose error and inject advice
                    advice = ErrorDiagnoser.diagnose(tool_name, args, str(exc))
                    tool_result = f"{tool_result}\n{advice.format_prompt()}"
                    self._doom_detector.record_call(tool_name, args, tool_result)
                    if self._active_plan:
                        self._active_plan = self._planner.replan_on_failure(
                            self._active_plan, self._active_plan.active_step().step_id
                            if self._active_plan.active_step() else -1,
                            str(exc)
                        )

                self._history.add_tool_result(tool_name, tool_result)

            # Refresh system prompt with updated plan context after each step
            if self._active_plan:
                system_prompt = self._build_system_prompt(user_input)

        # ── Max steps exceeded ─────────────────────────────────────────────────
        logger.warning(
            "Agent hit max_steps=%d — forcing final answer", settings.max_steps
        )
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
        """Clear conversation history and plan (start a new session)."""
        self._history.clear()
        self._active_plan = None
        self._doom_detector.reset()

    def history_snapshot(self) -> list[dict]:
        return self._history.snapshot()

    @property
    def active_plan(self) -> Plan | None:
        """Return the current active plan, if any."""
        return self._active_plan

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
        """Ask the approval callback (or block if none configured)."""
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
