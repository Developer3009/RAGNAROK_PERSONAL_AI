"""
ragnarok/agent/loop.py
───────────────────────
The core agent loop for Ragnarok.

Flow per turn:
  1. Add user message to history.
  2. Inject recalled memories into system prompt.
  3. Call LLM with tool list.
  4. If LLM wants a tool:
       a. Validate call against schema — on failure, send error back to LLM.
       b. Check risk level — high-risk pauses for approval.
       c. Execute tool, log result.
       d. Append tool result to history.
       e. Loop back to step 3 (up to max_steps).
  5. When LLM gives a final answer, return it.
  6. Never loop forever — enforce max_steps hard cap.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from ragnarok import llm
from ragnarok.agent.history import MessageHistory
from ragnarok.config import settings
from ragnarok.logging_setup import log_tool_event
from ragnarok.skills.registry import SkillRegistry, registry as _default_registry

logger = logging.getLogger(__name__)

# System prompt template — memories are injected at {memories}
_SYSTEM_TEMPLATE = """\
You are Ragnarok, a local AI assistant running entirely on this device.
You have access to tools — use them when they help answer the user's request.
Be concise, accurate, and honest about uncertainty.

{memories}
"""


@dataclass
class AgentResult:
    answer: str
    steps_taken: int
    tool_calls_made: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    stopped_early: bool = False   # True if max_steps was hit


class AgentLoop:
    """
    Stateful agent that manages one conversation session.

    Args:
        registry:       SkillRegistry to use (defaults to module-level singleton).
        approval_cb:    Optional callback(skill_name, args) → bool for high-risk
                        approval. If None, high-risk tools are blocked.
        memory_fn:      Optional callable(user_text) → str that returns injected
                        memory context. Wired up in Phase 2.
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
        self._history = MessageHistory()

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

        system_prompt = self._build_system_prompt(user_input)
        tools = self._registry.as_ollama_tools()

        steps = 0
        tool_calls_made: list[str] = []

        while steps < settings.max_steps:
            steps += 1
            logger.debug("Agent loop step %d/%d", steps, settings.max_steps)

            result = llm.chat_with_tools(
                messages=self._history.as_dicts(),
                tools=tools,
                system=system_prompt,
            )

            if not result.has_tool_calls:
                # Model gave a final answer
                answer = result.final_content or "(no response)"
                self._history.add_assistant(answer)
                return AgentResult(
                    answer=answer,
                    steps_taken=steps,
                    tool_calls_made=tool_calls_made,
                    latency_ms=(time.perf_counter() - t_start) * 1000,
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
                    self._history.add_tool_result(tool_name, f"ERROR: {err_msg}")
                    continue

                # 2. Risk check
                skill = self._registry.get(tool_name)
                if skill and skill.manifest.risk_level == "high":
                    approved = self._request_approval(tool_name, args)
                    if not approved:
                        msg = f"Action '{tool_name}' was not approved. Skipping."
                        log_tool_event(logger, "denied", tool_name, error=msg)
                        self._history.add_tool_result(tool_name, msg)
                        continue

                # 3. Execute
                t_tool = time.perf_counter()
                try:
                    tool_result = skill.run(args)
                    latency_ms = (time.perf_counter() - t_tool) * 1000
                    log_tool_event(logger, "result", tool_name, result=tool_result, latency_ms=latency_ms)
                except Exception as exc:
                    latency_ms = (time.perf_counter() - t_tool) * 1000
                    tool_result = f"ERROR: Tool '{tool_name}' raised an exception: {exc}"
                    log_tool_event(logger, "error", tool_name, error=str(exc), latency_ms=latency_ms)
                    logger.exception("Tool %s failed", tool_name)

                self._history.add_tool_result(tool_name, tool_result)

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
        )

    def reset(self) -> None:
        """Clear conversation history (start a new session)."""
        self._history.clear()

    def history_snapshot(self) -> list[dict]:
        return self._history.snapshot()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self, user_input: str) -> str:
        memories = ""
        if self._memory_fn:
            try:
                memories = self._memory_fn(user_input)
                if memories:
                    memories = f"Relevant memories:\n{memories}\n"
            except Exception as exc:
                logger.warning("Memory recall failed: %s", exc)
        return _SYSTEM_TEMPLATE.format(memories=memories)

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
