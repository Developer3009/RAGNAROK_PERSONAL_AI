# ⚡ Ragnarok — Privacy-First Local AI Agent

<div align="center">

```
  ██████╗  █████╗  ██████╗ ███╗   ██╗ █████╗ ██████╗  ██████╗ ██╗  ██╗
  ██╔══██╗██╔══██╗██╔════╝ ████╗  ██║██╔══██╗██╔══██╗██╔═══██╗██║ ██╔╝
  ██████╔╝███████║██║  ███╗██╔██╗ ██║███████║██████╔╝██║   ██║█████╔╝ 
  ██╔══██╗██╔══██║██║   ██║██║╚██╗██║██╔══██║██╔══██╗██║   ██║██╔═██╗ 
  ██║  ██║██║  ██║╚██████╔╝██║ ╚████║██║  ██║██║  ██║╚██████╔╝██║  ██╗
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝
```

**An autonomous, privacy-first personal AI assistant that lives entirely on your own machine.**  
*Zero Cloud Calls • Zero Subscriptions • Zero Data Tracking • 100% Offline Capable*

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-33ccff.svg?logo=python&logoColor=white)](https://python.org)
[![Local LLM Ollama](https://img.shields.io/badge/LLM-Ollama%20Local-black.svg?logo=ollama&logoColor=white)](https://ollama.ai)
[![FastAPI Dashboard](https://img.shields.io/badge/Web%20UI-FastAPI%20%2B%20Glassmorphic-009688.svg)](http://localhost:8000)
[![Privacy 100%](https://img.shields.io/badge/Privacy-100%25%20On--Device-brightgreen.svg)](#privacy--security)
[![Tests Passing](https://img.shields.io/badge/Tests-122%2F122%20Passing-success.svg)](tests/)

</div>

---

## 📖 Table of Contents
- [🎯 Problem Statement: Why Did We Build Ragnarok?](#-problem-statement-why-did-we-build-ragnarok)
- [🌟 What is Ragnarok?](#-what-is-ragnarok)
- [🚀 3-Minute Quickstart (Zero Tech Skills Needed)](#-3-minute-quickstart-zero-tech-skills-needed)
- [🧠 The Memory System Explained Simply](#-the-memory-system-explained-simply)
- [🎙️ Voice & Application Launcher Guide](#-voice--application-launcher-guide)
- [💻 Technical User & Developer Setup](#-technical-user--developer-setup)
- [📂 Project Directory Structure](#-project-directory-structure)
- [🛡️ Privacy, Security & Sandboxing](#-privacy-security--sandboxing)
- [❓ Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
- [📜 License](#-license)

---

## 🎯 Problem Statement: Why Did We Build Ragnarok?

> 📄 *Read the full research proposal in [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md), technical deep-dive in [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md), and plain-English breakdown in [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).*

Despite the rapid advancement of modern AI capabilities, there exists a critical dichotomy between **agent autonomy** and **user data sovereignty**. Today's cloud-dependent assistants suffer from three fundamental flaws:

1. **Loss of Data Sovereignty**: Users do not own or control the data used to personalize their assistants. Personal notes, workflow habits, and private context are harvested and transmitted to third-party servers.
2. **Contextual Amnesia**: Existing systems lack persistent memory. They act as reactive, single-turn chatbots that forget previous interactions, user preferences, and evolving project state by the next morning.
3. **Siloed & Ephemeral Execution**: Agents are restricted to isolated dialog boxes, lacking a modular skill system and autonomous orchestration capabilities to navigate real-world desktop and web tasks.

### 🛡️ The Ragnarok Solution:
To solve this, **Ragnarok** was created as a localized, privacy-preserving, "always-on" personal AI companion where **every model call runs locally via Ollama**. It keeps all data on your physical hardware, remembers past interactions across sessions, executes reusable skills in a sandboxed shield-wall, and orchestrates multi-step workflows with zero external API keys and zero telemetry.

---

## 🌟 What is Ragnarok?

Most AI assistants upload your voice, notes, code, and personal queries to third-party cloud servers. **Ragnarok** takes a fundamentally different approach:

1. **100% Local Intelligence**: Powered by open-weights models running directly on your computer via [Ollama](https://ollama.ai) (`llama3.2`, `gemma3:4b`, `nomic-embed-text`).
2. **Persistent Long-Term Memory**: Automatically extracts your preferences, tasks, and key notes, saving them to a fast local vector database on your hard drive.
3. **Voice Input & Natural Speech Output**: Talk to your agent through your microphone and receive natural spoken responses offline.
4. **Autonomous App & Web Launcher**: Ask Ragnarok to search Google, check Gmail, play YouTube videos, find places on Google Maps, look up GitHub repositories, or open Windows desktop apps (Calculator, Notepad, Explorer).
5. **Glassmorphic Cyberpunk Dashboard**: An interactive web interface with glowing status indicators, real-time audio visualizer, execution traces, and instant approval controls.

---

## 🚀 3-Minute Quickstart (Zero Tech Skills Needed)

You do **not** need to be a programmer or know how to code to run Ragnarok. Just follow these 3 simple steps:

### Step 1: Install Ollama (The AI Engine)
1. Download Ollama from: **[https://ollama.com/download](https://ollama.com/download)**
2. Install it like any normal program.
3. Open your computer's terminal (or Command Prompt) and run:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ollama pull gemma3:4b
   ```
   *(This downloads the free local AI models to your computer).*

### Step 2: Download Ragnarok
Click the green **Code** button at the top right of this page and click **Download ZIP**, then extract the folder to your desktop.

### Step 3: Double-Click to Run!
Inside the folder, simply double-click:
- **`start_web_ui.bat`** *(for the full visual Web Browser Interface)*  
  *(It automatically installs needed packages and opens `http://localhost:8000` in your web browser!)*
- **`start_cli.bat`** *(if you prefer an interactive terminal window)*

> **macOS / Linux users**: Open Terminal in the project folder and type:  
> `chmod +x run.sh && ./run.sh`

---

## 🧠 The Memory System Explained Simply

Ragnarok has a human-like memory system that never leaves your computer.

### How it Works:
```
You chat with Ragnarok
       ↓
Automated Fact Extractor detects personal preferences ("I prefer dark mode", "My project deadline is Friday")
       ↓
Saved locally into `db/ragnarok_memory.db` (SQLite + Embeddings)
       ↓
On future conversations, Ragnarok automatically recalls relevant memories in under 20 milliseconds!
```

### Where is My Data Stored?
All your data lives inside two simple folders in the project directory:

| Folder / File | What it Contains | How to Manage It |
|---|---|---|
| `db/ragnarok_memory.db` | The main local SQLite database holding your memories, conversation history, and vector embeddings. | To back up your memories, copy this file. To start with a clean slate, delete it. |
| `data/knowledge/` | Plain text notes, working checkpoints, and documents that Ragnarok writes or reads. | You can open, read, or drop your own `.txt` or `.md` notes here! |
| `data/inbox/` | Incoming files dropped for Ragnarok to inspect and summarize. | Drop any text file here for automated processing. |
| `logs/` | Local audit log (`security_audit.jsonl`) tracking all tool calls. | Tamper-evident record of actions executed by the agent. |

> [!NOTE]
> None of this data is ever sent to the cloud, OpenAI, or external servers. Your conversations stay 100% on your device.

---

## 🎙️ Voice & Application Launcher Guide

Ragnarok can be controlled via typed text or direct voice input through your microphone.

### Common Voice Commands:

| Application | What You Can Say | What Ragnarok Does |
|---|---|---|
| **Google Search** | `"Search on Google latest quantum computing breakthroughs"`<br/>`"Google best coffee shops near me"` | Opens Google search in your default browser |
| **Gmail** | `"Open gmail"` or `"Open gamil"`<br/>`"Search mail for invoice 2026"`<br/>`"Compose new email"` | Opens Gmail, searches your inbox, or opens compose window |
| **YouTube** | `"Open yt and search working of LLMs"`<br/>`"Play bohemian rhapsody on yt"` | Opens YouTube and starts the search query |
| **Google Maps** | `"Open maps"`<br/>`"Directions to Central Park New York"` | Opens Google Maps with direct directions / search |
| **Google Drive & Docs** | `"Open drive"`<br/>`"Search drive for project report"`<br/>`"Open google docs"` / `"Open sheets"` | Opens Google Drive, Docs, or Sheets |
| **Developer Tools** | `"Search github for fastapi template"`<br/>`"Search wikipedia for Alan Turing"` | Opens GitHub or Wikipedia search |
| **Productivity Apps** | `"Open chatgpt"`<br/>`"Open claude"`<br/>`"Open whatsapp"`<br/>`"Search spotify for lofi study beats"` | Launches web apps and searches Spotify |
| **Windows Desktop Apps** | `"Open calculator"` / `"Open calc"`<br/>`"Open notepad"`<br/>`"Open file explorer"` | Launches native desktop programs on Windows |

---

## 💻 Technical User & Developer Setup

If you are a developer and want to run, modify, or extend Ragnarok manually:

### 1. Manual Installation
```bash
# Clone the repository
git clone https://github.com/Developer3009/Solvers_Squad_PS1.git
cd Solvers_Squad_PS1

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Agent
- **Web UI & Dashboard**:
  ```bash
  python -m uvicorn ragnarok.ui.app:app --host 127.0.0.1 --port 8000
  ```
  Visit: `http://localhost:8000`
- **Interactive Terminal CLI**:
  ```bash
  python -m ragnarok
  ```

### 3. Running the Test Suite
Ragnarok includes 122 automated test suites covering end-to-end user flows, memory lifecycle, safety sandboxing, and voice integration:
```bash
# Run all unit and mock integration tests
python -m pytest tests/ -v

# Run 3x end-to-end user flow verification
python -m pytest tests/test_e2e_user_flows_3x.py -v
```

---

## 📂 Project Directory Structure

```
Solvers_Squad_PS1/
├── ragnarok/                     # Core Agent Package
│   ├── agent/                   # Agent loop, planning & history management
│   │   ├── history.py           # Sliding context window
│   │   ├── loop.py              # Tool-calling loop & direct voice routing
│   │   └── planner.py           # Autonomous goal decomposition
│   ├── memory/                  # Long-term memory & vector store
│   │   ├── db.py                # SQLite WAL + binary float32 embeddings
│   │   ├── recall.py            # Cosine similarity vector search
│   │   ├── facts.py             # Automatic fact extraction engine
│   │   └── consolidation.py     # Memory forgetting & pinned fact protection
│   ├── safety/                  # Zero-trust security framework
│   │   ├── csrf_guard.py        # Localhost-only HTTPConnection CSRF protection
│   │   ├── sandbox.py           # PathSandbox path-traversal prevention
│   │   ├── encryption.py        # AES-128 Fernet encrypted secrets
│   │   └── audit_integrity.py   # HMAC-signed tamper-evident audit log
│   ├── skills/                  # Extensible skill registry
│   │   ├── open_browser.py      # Multi-app web & desktop application launcher
│   │   ├── voice_listen.py      # Real-time microphone capture & speech-to-text
│   │   ├── voice_speak.py       # Offline speech synthesis (TTS)
│   │   ├── read_file.py         # Sandboxed file reader
│   │   ├── write_file.py        # Sandboxed file writer
│   │   └── add_note.py          # Knowledge note creator
│   ├── ui/                      # FastAPI Web Application
│   │   ├── app.py               # REST API endpoints & WebSocket streams
│   │   └── static/index.html    # Single-page glassmorphic UI dashboard
│   ├── cli.py                   # Terminal interface with ASCII art & audio I/O
│   ├── config.py                # Pydantic configuration loader
│   └── llm.py                   # Ollama client with structured JSON outputs
├── data/                        # User knowledge files & dropped notes
│   ├── inbox/                   # Drop-in documents for automatic summarization
│   └── knowledge/               # Pinned notes and working memory checkpoints
├── db/                          # Local memory database directory
│   └── ragnarok_memory.db       # SQLite local database (created automatically)
├── logs/                        # System logs & HMAC security audit trail
├── tests/                       # 122 automated pytest test suites
├── start_web_ui.bat             # 1-Click launcher for Web UI on Windows
├── start_cli.bat                # 1-Click launcher for Terminal on Windows
├── run.sh                       # 1-Click launcher for macOS / Linux
├── config.toml                  # Model names, latency limits, and paths
└── requirements.txt             # Python package dependencies
```

---

## 🛡️ Privacy, Security & Sandboxing

Ragnarok was engineered with **zero-trust security principles**:

1. **Strict Localhost Origin Guard**: The Web UI rejects all non-local connections (`require_local_origin`), eliminating CSRF and DNS-rebinding attacks.
2. **Path Traversal Sandboxing**: File operations are strictly confined within the project directory. Any attempt to access system files (e.g. `C:\Windows\System32` or `/etc/passwd`) is blocked and flagged in the audit log.
3. **HMAC-Signed Audit Log**: Every tool execution is recorded in `logs/security_audit.jsonl` with an HMAC signature to guarantee tamper resistance.
4. **Encrypted Secrets**: Sensitive credentials and tokens are encrypted at rest using AES-128 Fernet encryption (`SecretStore`).

---

## ⚡ Next-Generation Innovations (The 7 Pillars)

Ragnarok includes 7 state-of-the-art innovations that elevate it from a standard terminal assistant into a self-evolving, hardware-secured, multimodal operating agent:

### 1. 👁️ Offline VLM GUI Grounding (`gui_action.py`)
- **What it does**: Enables true local "Computer Use" without Anthropic cloud APIs. Uses local Ollama vision models (e.g. `llava`, `llama3.2-vision`) to inspect screen state, locate visual targets with normalized coordinates, and execute mouse/keyboard automation via `pyautogui` + `mss`.
- **Safety**: Built-in failsafe triggers, bounded display coordinates, and high-risk permission gate.

### 2. 🌲 Temporal Knowledge Graph Memory ("Yggdrasil") (`knowledge_graph.py`)
- **What it does**: Augments vector similarity search with structured relational graph memory. Tracks multi-hop relationships (`User` → `works_on` → `Project X` → `written_in` → `Rust`) and enforces **temporal validity**: updating a fact automatically timestamps and deprecates the old fact rather than overwriting it, giving Ragnarok perfect time-aware recall.

### 3. 🛡️ P2P Local LAN Mesh Swarm ("Valhalla Cluster") (`mesh_swarm.py`)
- **What it does**: Discovers other Ragnarok nodes across the local home or office network via mDNS/Zeroconf. Enables secure task offloading (e.g. sending GPU-heavy embedding or LLM inference to a desktop workstation while commanding from a lightweight laptop).
- **Security**: AES-128 Fernet encrypted TCP transport with private-IP RFC1918 enforcement — strictly refuses any public internet routing.

### 4. 🎙️ Sub-300ms Real-Time Voice Duplex (`streaming_voice.py`)
- **What it does**: Replaces high-latency voice loops with a pipelined audio architecture: Silero-VAD speech detection → local `faster-whisper` transcription → Ollama token streaming → neural TTS chunk playback.
- **Barge-in Support**: Thread-safe interrupt flag enables users to speak over Ragnarok mid-sentence, immediately cutting audio playback just like a human conversation.

### 5. 🔌 Native MCP Client Hub & Auto-Forge (`mcp_hub.py`)
- **What it does**: 
  - **MCP Client**: Seamlessly connects to community Model Context Protocol (MCP) servers (GitHub, Docker, PostgreSQL, Brave Search, Filesystem) over stdio JSON-RPC, dynamically exposing external tools as native Ragnarok skills.
  - **Auto-Forge**: When encountering an unmet need, the agent autonomously drafts, sandbox-validates, and permanently registers a new Python skill module adhering to security AST policies.

### 6. 🌙 Autonomous Shadow-Worker (`shadow_worker.py`)
- **What it does**: Proactive daemon background engine running two silent services:
  - **Morning Standup Briefing**: Automatically synthesizes recent git activity, pending pinned tasks, and notes into an executive spoken/displayed briefing at configured morning hours.
  - **Silent Error Sentinel**: Continuously monitors project logs and test outputs. When a Python traceback or test failure occurs, it diagnoses root cause and pre-computes a proposed fix in pinned memory.

### 7. 🔐 Hardware-Bound Vault & Merkle Audit Tree (`vault.py`)
- **What it does**:
  - **Hardware Binding**: Derives encryption keys bound to physical machine hardware identifiers via Windows DPAPI and machine fingerprinting — secrets copied to another PC cannot be decrypted.
  - **Cryptographic Merkle Audit**: Links every audit event to previous entries via SHA-256 hash pointers (`entry_n.hash = SHA256(data + prev_hash)`). Any manual tampering with historical logs instantly breaks the chain and is flagged in health checks.

---

## ❓ Frequently Asked Questions (FAQ)

### Q: Does Ragnarok require an internet connection?
**A:** No! Ragnarok and Ollama run completely offline. Internet access is only used if you explicitly ask Ragnarok to open a web page (like YouTube or Google) in your browser.

### Q: What hardware do I need?
**A:** 
- **Recommended**: An NVIDIA GPU with 4GB+ VRAM or Apple Silicon (M1/M2/M3/M4) for sub-second responses.
- **Minimum**: Any modern 4-core CPU with 8GB RAM (Ragnarok will run in CPU-mode smoothly with models like `llama3.2:3b`).

### Q: How do I change the AI model?
Open `config.toml` in any text editor and change `default_model` to any model installed in your Ollama (e.g. `mistral`, `phi3`, `qwen2.5`, etc.).

### Q: How do I delete my memories or reset?
Simply delete the file `db/ragnarok_memory.db`. Ragnarok will automatically create a fresh, clean database on its next run.

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it for personal and commercial applications.
