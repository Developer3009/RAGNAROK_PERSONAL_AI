# Ragnarok — Local AI Agent

A fully local, privacy-first AI agent running on Ollama.
No cloud, no telemetry, no subscription.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running on localhost:11434
- NVIDIA GPU with 4 GB+ VRAM (or CPU-only, slower)

## Quick Start

```bash
# 1. Clone and enter project
cd Solvers_Squad_PS1

# 2. Create venv and install
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama pull gemma3:4b

# 4. Run the agent
python -m ragnarok
```

## Project Structure

```
ragnarok/
  __init__.py          Package root
  config.py            Typed settings loader (reads config.toml)
  llm.py               All Ollama calls: chat, embed, tool-calling
  logging_setup.py     Structured JSON logging → logs/ragnarok.jsonl
  cli.py               Command-line interface
  agent/
    history.py         Message history manager (trim by turns & chars)
    loop.py            Core agent loop (tool-calling, retry, max_steps)
  skills/
    registry.py        Auto-discovers skills, validates manifests
    get_time.py        Returns current date/time
    read_file.py       Reads files (allowlist-enforced)
    add_note.py        Saves notes to data/knowledge/
    search_files.py    Searches files by pattern/content
  memory/
    __init__.py        Package exports
    db.py              SQLite WAL engine with binary float32 embeddings
    recall.py          MemoryStore: vector recall, deduplication, recency boost
    facts.py           Automated user fact extraction from conversation turns
    working.py         Working memory with atomic disk checkpointing
    consolidation.py   Memory consolidation and forgetting with pinned protection
  safety/              [Phase 5] Permissions, sandbox, encryption
  triggers/            [Phase 6] Scheduler, file/email watchers
  ui/                  [Phase 7] FastAPI + web frontend
tests/
  conftest.py
  test_history.py
  test_llm.py
  test_skills.py
  test_skills_extended.py
  test_agent_loop.py
  test_memory.py
  test_phase4_planning.py
config.toml             ALL model names, paths, and limits
requirements.txt
```

## Running Tests

```bash
# Unit tests only (no Ollama required)
pytest

# Include integration tests (requires live Ollama + models pulled)
pytest -m integration
```

## Phase Status

| Phase | Description                        | Status      |
|-------|------------------------------------|-------------|
| 0     | Setup & scoping                    | Done        |
| 1     | Core agent loop with Ollama        | Done        |
| 2     | Persistent memory                  | Done        |
| 3     | Skills & tool calling              | Done        |
| 4     | Planning & self-correction         | Done        |
| 5     | Safety, sandbox, encryption        | Planned     |
| 6     | Always-on triggers & recovery      | Planned     |
| 7     | Local interface & transparency     | Planned     |
| 8     | Evaluation & write-up              | Planned     |

## Target Metrics

| Metric                  | Target   |
|-------------------------|----------|
| Task success rate        | ≥ 80%   |
| Memory recall latency    | < 800ms |
| Embedding latency        | < 200ms |
| Idle CPU usage           | < 30%   |
| Crash recovery time      | < 5 sec |

## License

MIT
