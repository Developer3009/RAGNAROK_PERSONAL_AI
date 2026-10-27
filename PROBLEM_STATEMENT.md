# 📜 Ragnarok: Problem Statement & Research Foundation

> **A Sovereign, Always-On Personal AI Agent: Architecting Privacy-Preserving Autonomous Workflows with Persistent Memory on a Fully Local, Ollama-Based Stack**

---

## 1. Context and Background

The proliferation of Large Language Model (LLM) based assistants has revolutionized human-computer interaction. However, current commercial AI assistants operate on a centralized, cloud-dependent paradigm — a distant Asgard whose halls the user may never enter. This architecture inherently compromises user privacy, as personal data, conversational contexts, and workflow habits are continuously harvested by third-party providers, carried off like tribute to a foreign hoard.

Furthermore, existing assistants are predominantly stateless or rely on limited context windows, rendering them incapable of true long-term personalization. Like a skald who forgets the saga by morning, they function as reactive chatbots rather than proactive, autonomous agents capable of executing complex, multi-step tasks across a user's diverse daily workflows.

Today, small open models can run on an ordinary laptop through **Ollama**, which serves chat, embeddings, and tool calling over a local address (`127.0.0.1:11434`). The forge can now be raised inside the user's own hall, with nothing leaving the machine.

---

## 2. Problem Statement: Why Did We Build Ragnarok?

Despite the rapid advancement of AI capabilities, there exists a critical dichotomy between agent autonomy and user data sovereignty. Current AI assistants fail to provide a unified, always-on operational framework that respects user privacy while delivering deep, context-aware workflow automation.

Specifically, the problem is threefold:

```
                  ┌──────────────────────────────────────────────┐
                  │          THE CORE TRIPLE CHALLENGE           │
                  └──────────────────────┬───────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
 1. Data Sovereignty Loss       2. Contextual Amnesia          3. Siloed Execution
 ────────────────────────       ────────────────────           ───────────────────
 Cloud providers own your       Sessions reset; models         Isolated turn-by-turn;
 data, prompts, voice, &        forget who you are, your       cannot orchestrate apps,
 habits. Privacy breach risk.   preferences, and past tasks.   tools, or workflows offline.
```

### 1. Loss of Data Sovereignty
Users do not own or control the data used to personalize their assistants, leading to privacy vulnerabilities, corporate surveillance, and a lack of trust — their hoard plundered by distant jarls. Users are forced to trade private personal and corporate information for intelligence.

### 2. Contextual Amnesia
Existing systems lack robust, persistent memory architectures — like Mimir's well run dry. They cannot reliably retain long-term user preferences, historical context, or evolving project states, forcing users to repeatedly re-establish context and re-explain who they are in every single conversation.

### 3. Siloed and Ephemeral Execution
Current agents are restricted to isolated, single-turn interactions — each of the Nine Realms severed from the rest. They lack a modular, reusable skill framework and the orchestration capabilities required to autonomously navigate and execute tasks across heterogeneous, real-world daily workflows (e.g., transitioning from email triage to calendar scheduling to code generation and application launching).

---

## 🎯 Research Objective

To design, build, and evaluate a localized, privacy-preserving, **"always-on"** personal AI agent — a hearth-bound guardian standing watch like Heimdall, where **every model call goes to a local Ollama instance**. 

The system must:
- Keep all data on the user's machine (zero telemetry, zero cloud APIs, zero subscriptions).
- Remember past interactions across time with episodic & semantic memory.
- Use modular, reusable skills and tools with sandboxed execution.
- Carry out multi-step tasks autonomously on its own, with no cloud AI service and no API key anywhere.

---

## 3. System Requirements

To address the aforementioned problem, Ragnarok implements the following functional and non-functional requirements — laws carved in runestone:

### 3.1 Functional Requirements (FR)

| Code | Category | Requirement Description | Norse Allegory |
|:---|:---|:---|:---|
| **FR1.1** | Persistent Memory | **Episodic Memory**: Maintain a chronological log of user interactions, completed tasks, and contextual events. | A saga-log kept by Muninn, allowing recall of specific past occurrences. |
| **FR1.2** | Persistent Memory | **Semantic Memory**: Extract, store, and retrieve generalized knowledge, user preferences, and factual data in vector embeddings. | Wisdom drawn from Mimir's well. |
| **FR1.3** | Persistent Memory | **Memory Consolidation & Forgetting**: Autonomously consolidate working memory into long-term storage and prune obsolete or irrelevant data. | As the Norns weave what endures and cut what fades. |
| **FR2.1** | Modular Skill Execution | **Skill Registration**: Define, register, and manage reusable skills (Python scripts, local CLI commands, API wrappers) via a standardized interface. | Each skill a rune inscribed in the dwarves' forge. |
| **FR2.2** | Modular Skill Execution | **Dynamic Tool Selection**: Analyze intent, select appropriate registered skills, and generate execution parameters automatically. | Choosing the right weapon for the battle. |
| **FR2.3** | Modular Skill Execution | **Sandboxed Execution**: Execute skills in isolated, sandboxed environments with path-traversal prevention. | Behind a shield-wall, preventing malicious code from compromising the host. |
| **FR3.1** | Autonomous Orchestration | **Multi-step Planning**: Decompose high-level, ambiguous user goals into a sequence of actionable logical sub-tasks. | Charting the voyage across the Nine Realms. |
| **FR3.2** | Autonomous Orchestration | **Cross-Application Integration**: Seamlessly interact with external, user-approved applications (browser, email, desktop apps, IDEs). | Crossing a Bifrost bridge to carry out sub-tasks. |
| **FR3.3** | Autonomous Orchestration | **Self-Correction & Reflection**: Monitor skill output, evaluate criteria, and autonomously retry or adjust plan upon failure. | As Odin sought wisdom again and again at the well. |
| **FR4.1** | Proactive Interaction | **Always-On Background Processing**: Continuously monitor user-defined triggers (file changes, background tasks, time-based cron jobs). | A sleepless sentinel like Heimdall, initiating tasks without prompting. |
| **FR4.2** | Proactive Interaction | **Contextual Proactivity**: Proactively suggest actions or provide relevant information based on persistent memory and current context. | As Odin's ravens return each dawn with news. |

---

### 3.2 Non-Functional Requirements (NFR)

| Code | Category | Requirement Specification |
|:---|:---|:---|
| **NFR1.1** | Privacy & Security | **Local-First Architecture**: All raw user data, memory stores, and execution logs must be stored locally on the user's hardware. Zero raw data transmitted to external servers. |
| **NFR1.2** | Privacy & Security | **Encrypted Memory at Rest**: Persistent memory database and sensitive tokens encrypted at rest with user-held cryptographic keys (AES-128 Fernet). |
| **NFR1.3** | Privacy & Security | **Fully Local LLM Inference**: All reasoning and embedding calls served by local Ollama models. No external LLM API used; no PII redaction pipeline needed because no prompt leaves the machine. |
| **NFR2.1** | Performance & Efficiency | **Responsiveness**: Direct user queries processed with sub-800ms latency for memory retrieval and intent classification (*as swift as Sleipnir*). |
| **NFR2.2** | Performance & Efficiency | **Resource Footprint**: Background monitoring and local inference consume $\le 30\%$ baseline host resources, keeping primary workflows uninhibited. |
| **NFR3.1** | Fault Tolerance | **Graceful Degradation**: If an external tool fails, halt that sub-task gracefully, inform the user, and continue independent tasks (*a single bough of Yggdrasil breaking while the tree still stands*). |
| **NFR3.2** | Fault Tolerance | **State Recovery**: System restores operational state, incomplete task graphs, and working memory within 5 seconds of restart (*rising again from Helheim*). |
| **NFR4.1** | Usability & Transparency | **Explainability**: Transparent "chain of thought" and execution logs (*a seer's telling*), showing why each tool was chosen. |
| **NFR4.2** | Usability & Transparency | **User Control over Memory**: User-friendly dashboard to view, inspect, export, or permanently purge memories (*as a jarl commands his own hall*). |
| **NFR4.3** | Usability & Transparency | **Permission Granularity**: Granular Role-Based Access Control (RBAC) and explicit user approval before executing sensitive or high-risk actions (*no gate opened until the gatekeeper's horn is sounded*). |

---

## 4. How Ragnarok Solves the Problem in Code

| Problem Dimension | Ragnarok Technical Solution in Codebase |
|:---|:---|
| **Loss of Data Sovereignty** | `ragnarok/llm.py` connects strictly to `http://127.0.0.1:11434`. `ragnarok/ui/routes.py` enforces local origin checks. No cloud keys exist or are required. |
| **Contextual Amnesia** | `ragnarok/memory/db.py`, `extractor.py`, and `embeddings.py` extract facts automatically via local Ollama embeddings into SQLite vector tables, recalling relevant facts in under 20ms. |
| **Siloed & Ephemeral Execution** | `ragnarok/agent/loop.py` and `planner.py` break ambiguous tasks into sub-plans, executing local skills (`open_browser`, `run_command`, `file_ops`) with reflection and retries. |
| **Security & Shield-Wall** | `ragnarok/safety/input_guard.py`, `sandbox.py`, and `audit.py` enforce strict path confinement, command whitelisting, and HMAC-signed audit logs. |

---

## 5. Conclusion

The realization of this system requires a paradigm shift from centralized, stateless AI chatbots to decentralized, stateful, and sovereign agents — **a new world rising green from the sea after Ragnarok**. 

By running every model, memory store, and skill on the user's own machine through Ollama, and by strictly adhering to the functional and non-functional requirements outlined above, Ragnarok establishes a foundational architecture for an always-on personal assistant that amplifies human productivity through persistent memory and autonomous workflow execution, without sacrificing the fundamental right to data privacy and user control.
