"""
ragnarok/cli.py
────────────────
Command-line interface for Ragnarok.
Run with:  python -m ragnarok.cli

Features:
  - Chat loop with the agent
  - Special commands: /help /reset /history /status /quit
  - Streams tokens to console for responsiveness
"""

from __future__ import annotations

import sys
import logging

from ragnarok import llm
from ragnarok.agent.loop import AgentLoop
from ragnarok.config import settings
from ragnarok.logging_setup import setup_logging
from ragnarok.memory.recall import MemoryStore
from ragnarok.memory.facts import extract_and_save_facts
from ragnarok.skills.registry import registry

logger = logging.getLogger(__name__)

_BANNER = r"""
+===========================================================+
|          R A G N A R O K   A G E N T  v0.3               |
|   Local AI  *  Phases 0-4 (Planning)  *  Ollama           |
+===========================================================+
Type your message and press Enter.
Special commands: /help  /plan  /memory  /remember  /reset  /history  /status  /quit
"""

_HELP = """
/help              -- show this message
/plan              -- show the current active execution plan and its step status
/memory            -- list active memories stored in SQLite
/remember <fact>   -- manually save a lasting semantic fact to memory
/reset             -- start a new conversation (clear history + active plan)
/history           -- show current conversation turn count and char total
/status            -- show Ollama status, loaded models, memory count, skill count
/quit              -- exit Ragnarok
"""


def _print_status(mem_store: MemoryStore | None = None) -> None:
    alive = llm.is_ollama_alive()
    if alive:
        models = llm.list_models()
        print(f"  Ollama:   [OK] online at {settings.ollama_url}")
        print(f"  Models:   {', '.join(models) if models else '(none pulled)'}")
    else:
        print(f"  Ollama:   [OFFLINE] NOT reachable at {settings.ollama_url}")
    print(f"  Skills:   {len(registry)} loaded")
    if mem_store:
        print(f"  Memories: {mem_store.count()} active stored")
    print(f"  Config:   chat_main={settings.model_main}  embed={settings.model_embed}")


def main() -> None:
    setup_logging(level=logging.WARNING)   # console: warnings only; file: debug

    memory_store = MemoryStore()

    print(_BANNER)
    _print_status(memory_store)
    print()

    # Load skills
    registry.load_all()
    logger.info("Skills loaded: %s", registry)

    # CLI approval callback for high-risk tools
    def cli_approval(tool_name: str, args: dict) -> bool:
        print(f"\n⚠  HIGH-RISK ACTION REQUESTED")
        print(f"   Tool:   {tool_name}")
        print(f"   Args:   {args}")
        ans = input("   Approve? [y/N] ").strip().lower()
        return ans in ("y", "yes")

    agent = AgentLoop(
        registry=registry,
        approval_cb=cli_approval,
        memory_fn=memory_store.recall_as_prompt,
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # ── Special commands ───────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye.")
                break
            elif cmd == "/help":
                print(_HELP)
            elif cmd == "/reset":
                agent.reset()
                print("Conversation reset.")
            elif cmd == "/history":
                snap = agent.history_snapshot()
                print(f"  {len(snap)} messages, "
                      f"{sum(len(m['content']) for m in snap)} chars total.")
            elif cmd == "/status":
                _print_status(memory_store)
            elif cmd == "/plan":
                plan = agent.active_plan
                if plan is None:
                    print("  No active plan for this turn (query was not complex enough to trigger planning).")
                else:
                    print(plan.format_progress())
            elif cmd == "/memory":
                mems = memory_store.list_all()
                if not mems:
                    print("  No memories recorded yet.")
                else:
                    print(f"  {len(mems)} active memories:")
                    for m in mems[:10]:
                        pinned = " [PINNED]" if m.pinned else ""
                        print(f"    * [{m.kind}]{pinned} {m.text[:80]}")
                    if len(mems) > 10:
                        print(f"   ... and {len(mems) - 10} more.")
            elif user_input.startswith("/remember"):
                fact = user_input[len("/remember"):].strip()
                if not fact:
                    print("Usage: /remember <fact or preference to save>")
                else:
                    mid = memory_store.remember(fact, kind="semantic", source="user_cli")
                    print(f"  [OK] Saved to memory: {fact[:60]} (id: {mid[:8]}...)")
            else:
                print(f"Unknown command: {user_input}. Type /help for commands.")
            continue

        print("Ragnarok: ", end="", flush=True)
        try:
            result = agent.chat(user_input)
            print(result.answer)
            if result.tool_calls_made:
                print(f"  [tools: {', '.join(result.tool_calls_made)} | "
                      f"steps: {result.steps_taken} | "
                      f"{result.latency_ms:.0f}ms]")
            if result.plan and result.plan.steps:
                completed = sum(1 for s in result.plan.steps if s.status == "completed")
                total = len(result.plan.steps)
                print(f"  [plan: {completed}/{total} steps done | status: {result.plan.status}]")
            if result.stopped_early:
                print("  [WARNING] Reached maximum steps -- answer may be incomplete.")

            # Background fact extraction
            try:
                extract_and_save_facts(
                    f"User: {user_input}\nAssistant: {result.answer}",
                    memory_store=memory_store,
                    source="conversation_turn",
                )
            except Exception as extract_err:
                logger.debug("Fact extraction skipped: %s", extract_err)
        except Exception as exc:
            logger.exception("Agent error")
            print(f"\n[ERROR] {exc}")
            print("(Type /reset to start a fresh conversation.)")


if __name__ == "__main__":
    main()

