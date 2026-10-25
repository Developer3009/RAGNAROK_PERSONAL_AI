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
from ragnarok.skills.registry import registry

logger = logging.getLogger(__name__)

_BANNER = r"""
╔══════════════════════════════════════════════════════╗
║          R A G N A R O K   A G E N T  v0.1          ║
║         Local AI  •  Phases 0-1  •  Ollama          ║
╚══════════════════════════════════════════════════════╝
Type your message and press Enter.
Special commands: /help  /reset  /history  /status  /quit
"""

_HELP = """
/help     — show this message
/reset    — start a new conversation (clear history)
/history  — show current conversation turn count and char total
/status   — show Ollama status, loaded models, skill count
/quit     — exit Ragnarok
"""


def _print_status() -> None:
    alive = llm.is_ollama_alive()
    if alive:
        models = llm.list_models()
        print(f"  Ollama:  ✓ online at {settings.ollama_url}")
        print(f"  Models:  {', '.join(models) if models else '(none pulled)'}")
    else:
        print(f"  Ollama:  ✗ NOT reachable at {settings.ollama_url}")
    print(f"  Skills:  {len(registry)} loaded")
    print(f"  Config:  chat_main={settings.model_main}  embed={settings.model_embed}")


def main() -> None:
    setup_logging(level=logging.WARNING)   # console: warnings only; file: debug

    print(_BANNER)
    _print_status()
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

    agent = AgentLoop(registry=registry, approval_cb=cli_approval)

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
                _print_status()
            else:
                print(f"Unknown command: {user_input}. Type /help for commands.")
            continue

        # ── Normal chat turn ───────────────────────────────────────────────────
        print("Ragnarok: ", end="", flush=True)
        try:
            result = agent.chat(user_input)
            print(result.answer)
            if result.tool_calls_made:
                print(f"  [tools used: {', '.join(result.tool_calls_made)} | "
                      f"steps: {result.steps_taken} | "
                      f"{result.latency_ms:.0f}ms]")
            if result.stopped_early:
                print("  ⚠  Reached maximum steps — answer may be incomplete.")
        except Exception as exc:
            logger.exception("Agent error")
            print(f"\n[ERROR] {exc}")
            print("(Type /reset to start a fresh conversation.)")


if __name__ == "__main__":
    main()
