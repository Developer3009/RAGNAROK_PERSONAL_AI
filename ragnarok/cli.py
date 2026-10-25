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

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ragnarok import llm
from ragnarok.agent.loop import AgentLoop
from ragnarok.config import settings
from ragnarok.logging_setup import setup_logging
from ragnarok.memory.recall import MemoryStore
from ragnarok.memory.facts import extract_and_save_facts
from ragnarok.skills.registry import registry
from ragnarok.triggers.scheduler import TriggerScheduler, ScheduledTrigger
from ragnarok.triggers.checkpoint import SessionCheckpoint, CheckpointError
from ragnarok.triggers.watchdog import Watchdog

logger = logging.getLogger(__name__)

_BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  ██████╗  █████╗  ██████╗ ███╗   ██╗ █████╗ ██████╗  ██████╗ ██╗  ██╗              ║
║  ██╔══██╗██╔══██╗██╔════╝ ████╗  ██║██╔══██╗██╔══██╗██╔═══██╗██║ ██╔╝              ║
║  ██████╔╝███████║██║  ███╗██╔██╗ ██║███████║██████╔╝██║   ██║█████╔╝               ║
║  ██╔══██╗██╔══██║██║   ██║██║╚██╗██║██╔══██║██╔══██╗██║   ██║██╔═██╗               ║
║  ██║  ██║██║  ██║╚██████╔╝██║ ╚████║██║  ██║██║  ██║╚██████╔╝██║  ██╗              ║
║  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝              ║
║                                                                                      ║
║          ★  L O C A L   A I   A G E N T   —   v 0 . 8   ★                          ║
║                                                                                      ║
║    ◈ Privacy-First    ◈ Always-On Triggers    ◈ Local Dashboard    ◈ Gemma 3        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
  Type your message and press Enter.
  Commands: /help  /ui  /plan  /memory  /remember  /reset  /history  /status  /quit
"""

_HELP = """
/help              -- show this message
/ui                -- launch or open the Phase 7 local web dashboard
/plan              -- show the current active execution plan and its step status
/memory            -- list active memories stored in SQLite
/remember <fact>   -- manually save a lasting semantic fact to memory
/reset             -- start a new conversation (clear history + active plan)
/history           -- show current conversation turn count and char total
/status            -- show Ollama status, loaded models, memory count, skill count
/triggers          -- list all registered background triggers and their status
/quit              -- exit Ragnarok
"""


_ui_server_thread = None


def _ensure_ui_server_running(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Ensure the FastAPI/uvicorn server is active in a background daemon thread."""
    global _ui_server_thread
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        is_open = (sock.connect_ex((host, port)) == 0)
    finally:
        sock.close()

    if is_open:
        return

    import threading
    import uvicorn
    from ragnarok.ui.app import app

    def _run():
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        server.run()

    _ui_server_thread = threading.Thread(target=_run, daemon=True)
    _ui_server_thread.start()
    import time
    time.sleep(0.6)


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
    # Phase 7 Web Dashboard launch check
    if "--ui" in sys.argv:
        from ragnarok.ui.server import start_server
        port = 8000
        if "--port" in sys.argv:
            try:
                idx = sys.argv.index("--port")
                port = int(sys.argv[idx + 1])
            except (ValueError, IndexError):
                pass
        start_server(port=port)
        return

    setup_logging(level=logging.WARNING)   # console: warnings only; file: debug

    memory_store = MemoryStore()
    checkpoint = SessionCheckpoint()       # Phase 6: crash recovery
    watchdog = Watchdog(                   # Phase 6: hang detection
        timeout_sec=settings.approval_timeout_sec,
        name="ragnarok-turn-watchdog",
    )

    print(_BANNER)
    _print_status(memory_store)

    # ── Phase 6: offer to resume from checkpoint after crash ───────────────
    resumed_history: list[dict] | None = None
    if checkpoint.exists():
        try:
            saved = checkpoint.load()
            print(f"\n[CHECKPOINT DETECTED] Session from {saved.get('saved_at', '?')}")
            print(f"  {len(saved.get('history', []))} turns in checkpoint.")
            ans = input("  Resume previous session? [y/N] ").strip().lower()
            if ans in ("y", "yes"):
                resumed_history = saved.get("history", [])
                print("  Resuming.")
            else:
                checkpoint.clear()
                print("  Starting fresh session.")
        except CheckpointError as exc:
            print(f"  [WARNING] Corrupt checkpoint discarded: {exc}")
            checkpoint.clear()
    print()

    # Load skills
    registry.load_all()
    logger.info("Skills loaded: %s", registry)

    # ── Phase 6: Background trigger scheduler ─────────────────────────────
    def _daily_brief(name: str):
        """Fires at configured daily_brief_time — can be used for reminders."""
        logger.info("Daily brief trigger fired: %s", name)
        # Future: run a morning summary task through the agent

    scheduler = TriggerScheduler()
    daily_time = getattr(settings, "daily_brief_time", "08:00")
    try:
        scheduler.add(ScheduledTrigger("daily_brief", _daily_brief, daily_at=daily_time))
    except Exception:
        pass  # Skip if settings don't have a valid daily_brief_time
    scheduler.start()
    logger.info("Background trigger scheduler started.")

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

    # Restore history after crash-resume
    if resumed_history:
        for msg in resumed_history:
            agent.history.add(msg["role"], msg["content"])

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            scheduler.stop()
            break

        if not user_input:
            continue

        # ── Special commands ───────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye.")
                scheduler.stop()
                break
            elif cmd == "/help":
                print(_HELP)
            elif cmd == "/ui":
                from ragnarok.ui.app import ACTIVE_TOKEN
                print("  Starting local web UI server...")
                _ensure_ui_server_running()
                url = f"http://127.0.0.1:8000/?token={ACTIVE_TOKEN}"
                print(f"  Local Web Dashboard: {url}")
                import webbrowser
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            elif cmd == "/reset":
                agent.reset()
                print("Conversation reset.")
            elif cmd == "/history":
                snap = agent.history_snapshot()
                print(f"  {len(snap)} messages, "
                      f"{sum(len(m['content']) for m in snap)} chars total.")
            elif cmd == "/status":
                _print_status(memory_store)
            elif cmd == "/triggers":
                trigs = scheduler.list_triggers()
                if not trigs:
                    print("  No triggers registered.")
                else:
                    for t in trigs:
                        state = "ON" if t["enabled"] else "OFF"
                        sched_str = t["daily_at"] or f"every {t['interval_sec']}s"
                        print(f"  [{state}] {t['name']} ({sched_str}) last={t['last_fired']}")
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
            # Phase 6: arm watchdog before potentially long LLM call
            watchdog.arm()
            result = agent.chat(user_input)
            watchdog.disarm()
            if watchdog.tripped:
                print("  [WARNING] Turn exceeded watchdog timeout — response may be incomplete.")

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

            # Phase 6: save checkpoint after every successful turn
            try:
                checkpoint.save(
                    history=agent.history_snapshot(),
                    plan_summary=(
                        result.plan.format_progress() if result.plan else None
                    ),
                    metadata={"steps": result.steps_taken, "latency_ms": result.latency_ms},
                )
            except Exception as cp_err:
                logger.debug("Checkpoint save failed (non-fatal): %s", cp_err)

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
            watchdog.disarm()  # Always disarm on exception
            logger.exception("Agent error")
            print(f"\n[ERROR] {exc}")
            print("(Type /reset to start a fresh conversation.)")


if __name__ == "__main__":
    main()

