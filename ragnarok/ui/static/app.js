/**
 * Ragnarok AI — Frontend Client Logic (Phase 7)
 * Controls View Switching, Chat, Memory CRUD, Approvals, Execution Timeline, and Settings.
 */

// ── Authentication Token ───────────────────────────────────────────────────────
const tokenMeta = document.querySelector('meta[name="ragnarok-token"]');
const urlParams = new URLSearchParams(window.location.search);
const RAGNAROK_TOKEN = urlParams.get("token") || (tokenMeta ? tokenMeta.getAttribute("content") : "");

async function apiFetch(endpoint, options = {}) {
    const headers = options.headers || {};
    headers["X-Ragnarok-Token"] = RAGNAROK_TOKEN;
    if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
        headers["Content-Type"] = "application/json";
    }
    const res = await fetch(endpoint, { ...options, headers });
    if (res.status === 401) {
        console.error("Unauthorized: invalid Ragnarok security token.");
    }
    return res;
}

// ── State ─────────────────────────────────────────────────────────────────────
let currentView = "chat-view";
let pendingApprovalsCache = [];

// ── DOM Elements ──────────────────────────────────────────────────────────────
const navButtons = document.querySelectorAll(".nav-item");
const viewPanels = document.querySelectorAll(".view-panel");
const viewTitle = document.getElementById("view-title");
const viewDesc = document.getElementById("view-description");

// Chat
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatMessagesContainer = document.getElementById("chat-messages-container");
const chatSubmitBtn = document.getElementById("chat-submit-btn");

// Memory
const memoryGrid = document.getElementById("memory-grid-container");
const memorySearchInput = document.getElementById("memory-search-input");
const memorySearchBtn = document.getElementById("memory-search-btn");
const memoryClearBtn = document.getElementById("memory-clear-btn");
const memoryAddModalBtn = document.getElementById("memory-add-modal-btn");
const memoryExportBtn = document.getElementById("memory-export-btn");
const memoryCounterBadge = document.getElementById("memory-counter-badge");

// Modals
const addMemoryModal = document.getElementById("add-memory-modal");
const addMemoryForm = document.getElementById("add-memory-form");
const closeAddModalBtn = document.getElementById("close-add-modal-btn");
const cancelAddModalBtn = document.getElementById("cancel-add-modal-btn");
const newMemoryImportance = document.getElementById("new-memory-importance");
const importanceValLabel = document.getElementById("importance-val-label");

const editMemoryModal = document.getElementById("edit-memory-modal");
const editMemoryForm = document.getElementById("edit-memory-form");
const closeEditModalBtn = document.getElementById("close-edit-modal-btn");
const cancelEditModalBtn = document.getElementById("cancel-edit-modal-btn");
const editMemoryImportance = document.getElementById("edit-memory-importance");
const editImportanceValLabel = document.getElementById("edit-importance-val-label");

const deleteConfirmModal = document.getElementById("delete-confirm-modal");
const closeDeleteModalBtn = document.getElementById("close-delete-modal-btn");
const cancelDeleteModalBtn = document.getElementById("cancel-delete-modal-btn");
const confirmDeleteActionBtn = document.getElementById("confirm-delete-action-btn");
const deleteMemoryPreview = document.getElementById("delete-memory-preview");
const deleteTargetId = document.getElementById("delete-target-id");

// Approvals
const pendingApprovalsList = document.getElementById("pending-approvals-list");
const approvalsHistoryBody = document.getElementById("approvals-history-body");
const approvalsBadge = document.getElementById("approvals-badge");
const editApprovalModal = document.getElementById("edit-approval-modal");
const editApprovalForm = document.getElementById("edit-approval-form");
const closeApprModalBtn = document.getElementById("close-appr-modal-btn");
const cancelApprModalBtn = document.getElementById("cancel-appr-modal-btn");
const notifyPermBtn = document.getElementById("notify-perm-btn");

// Logs & Timeline
const executionTimelineContainer = document.getElementById("execution-timeline-container");
const rawEventsContent = document.getElementById("raw-events-content");

// Settings & Telemetry
const metricCpu = document.getElementById("metric-cpu");
const metricRam = document.getElementById("metric-ram");
const metricModel = document.getElementById("metric-model");
const roleDisplayPill = document.getElementById("role-display-pill");
const settingRoleSelect = document.getElementById("setting-role-select");
const skillsListContainer = document.getElementById("skills-list-container");


// ── View Switching ────────────────────────────────────────────────────────────
const VIEW_METADATA = {
    "chat-view": {
        title: "Chat & Agent",
        desc: "Communicate with Ragnarok running locally on your hardware",
    },
    "memory-view": {
        title: "Memory Manager",
        desc: "Inspect, search, edit, pin, and permanently manage stored agent memories",
    },
    "approvals-view": {
        title: "Security Approvals",
        desc: "Review and gate high-risk actions before tool execution",
    },
    "logs-view": {
        title: "Execution Log & Explainability",
        desc: "Inspect step-by-step reasoning, tool choices, arguments, and latencies",
    },
    "settings-view": {
        title: "Settings & System Status",
        desc: "Inspect active models, security sandbox folders, roles, and scheduled triggers",
    },
};

function switchView(viewId) {
    currentView = viewId;
    navButtons.forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-view") === viewId);
    });
    viewPanels.forEach(panel => {
        panel.classList.toggle("active", panel.id === viewId);
    });

    const meta = VIEW_METADATA[viewId] || { title: "Ragnarok", desc: "" };
    viewTitle.textContent = meta.title;
    viewDesc.textContent = meta.desc;

    // Refresh view-specific data
    if (viewId === "memory-view") loadMemories();
    if (viewId === "approvals-view") loadApprovals();
    if (viewId === "logs-view") loadLogs();
    if (viewId === "settings-view") loadSettings();
}

navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        const view = btn.getAttribute("data-view");
        if (view) switchView(view);
    });
});


// ── Telemetry & System Status Polling ─────────────────────────────────────────
async function pollStatus() {
    try {
        const res = await apiFetch("/api/status");
        if (!res.ok) return;
        const data = await res.json();

        metricCpu.textContent = `${data.cpu_percent}%`;
        metricRam.textContent = `${data.ram_used_mb} MB`;
        metricModel.textContent = data.models.main || "llama3.2:3b";
        roleDisplayPill.textContent = `Role: ${data.active_role}`;
        memoryCounterBadge.textContent = data.memory_count;

        // Approvals badge & desktop notification
        const pendingCount = data.pending_approvals_count || 0;
        if (pendingCount > 0) {
            approvalsBadge.style.display = "inline-block";
            approvalsBadge.textContent = pendingCount;
            if (pendingCount > pendingApprovalsCache.length) {
                showDesktopNotification("High-Risk Action Approval", `${pendingCount} action(s) require operator approval.`);
            }
        } else {
            approvalsBadge.style.display = "none";
        }
    } catch (err) {
        console.warn("Status poll error:", err);
    }
}

setInterval(pollStatus, 3000);
pollStatus();


// ── Desktop Notifications ─────────────────────────────────────────────────────
notifyPermBtn.addEventListener("click", () => {
    if (!("Notification" in window)) {
        alert("This browser does not support desktop notifications.");
        return;
    }
    Notification.requestPermission().then(permission => {
        if (permission === "granted") {
            notifyPermBtn.textContent = "🔔 Alerts Enabled";
            notifyPermBtn.classList.remove("btn-outline");
            notifyPermBtn.classList.add("btn-success");
            new Notification("Ragnarok Alerts", { body: "Desktop notifications are enabled." });
        }
    });
});

function showDesktopNotification(title, body) {
    if ("Notification" in window && Notification.permission === "granted") {
        new Notification(title, { body, icon: "/static/favicon.ico" });
    }
}


// ── Screen 1: Chat Flow ───────────────────────────────────────────────────────
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    appendMessage("user", query);
    chatInput.value = "";
    chatSubmitBtn.disabled = true;

    // Show temporary typing indicator
    const typingId = appendTypingIndicator();

    try {
        const res = await apiFetch("/api/chat", {
            method: "POST",
            body: JSON.stringify({ message: query }),
        });
        removeTypingIndicator(typingId);

        if (!res.ok) {
            const err = await res.json();
            appendMessage("assistant", `⚠️ Error: ${err.detail || "Failed to process chat."}`);
            return;
        }

        const data = await res.json();
        appendMessage("assistant", data.answer, data.tool_calls, data.latency_ms, data.steps_taken);
    } catch (err) {
        removeTypingIndicator(typingId);
        appendMessage("assistant", `⚠️ Network or server error: ${err.message}`);
    } finally {
        chatSubmitBtn.disabled = false;
        pollStatus();
    }
});

function appendMessage(role, text, toolCalls = [], latencyMs = null, steps = null) {
    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "👤" : (role === "assistant" ? "⚡" : "⚙️");

    const content = document.createElement("div");
    content.className = "bubble-content";

    const p = document.createElement("p");
    p.textContent = text;
    content.appendChild(p);

    if (toolCalls && toolCalls.length > 0) {
        const toolsDiv = document.createElement("div");
        toolsDiv.className = "tools-used-row";
        toolCalls.forEach(t => {
            const tag = document.createElement("span");
            tag.className = "tool-tag";
            tag.textContent = `🛠️ ${t}`;
            toolsDiv.appendChild(tag);
        });
        content.appendChild(toolsDiv);
    }

    const ts = document.createElement("span");
    ts.className = "timestamp";
    const nowStr = new Date().toLocaleTimeString();
    if (latencyMs) {
        ts.textContent = `${nowStr} • ${latencyMs.toFixed(0)}ms • ${steps} step(s)`;
    } else {
        ts.textContent = nowStr;
    }
    content.appendChild(ts);

    bubble.appendChild(avatar);
    bubble.appendChild(content);
    chatMessagesContainer.appendChild(bubble);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
}

function appendTypingIndicator() {
    const id = `typing-${Date.now()}`;
    const bubble = document.createElement("div");
    bubble.id = id;
    bubble.className = "message-bubble assistant";
    bubble.innerHTML = `
        <div class="avatar">⚡</div>
        <div class="bubble-content">
            <p><em>Thinking and executing...</em></p>
        </div>
    `;
    chatMessagesContainer.appendChild(bubble);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}


// ── Screen 2: Memory Manager Flow ─────────────────────────────────────────────
async function loadMemories(searchQuery = "") {
    memoryGrid.innerHTML = '<div class="empty-state">Loading memories...</div>';
    try {
        const url = searchQuery ? `/api/memory?query=${encodeURIComponent(searchQuery)}` : "/api/memory";
        const res = await apiFetch(url);
        if (!res.ok) throw new Error("Failed to fetch memories");
        const memories = await res.json();

        if (memories.length === 0) {
            memoryGrid.innerHTML = '<div class="empty-state">No memories found. Click "+ Add Memory" to create one.</div>';
            return;
        }

        memoryGrid.innerHTML = "";
        memories.forEach(m => {
            const card = document.createElement("div");
            card.className = "memory-card";
            card.id = `mem-card-${m.id}`;

            const kindClass = m.kind.toLowerCase();
            const dateStr = new Date(m.created_at * 1000).toLocaleDateString();

            card.innerHTML = `
                <div class="memory-card-header">
                    <span class="kind-badge ${kindClass}">${m.kind}</span>
                    <div class="memory-actions">
                        <button class="action-icon-btn ${m.pinned ? 'pinned' : ''}" title="${m.pinned ? 'Unpin' : 'Pin'}" onclick="togglePinMemory('${m.id}', ${!m.pinned})">
                            📌
                        </button>
                        <button class="action-icon-btn" title="Edit" onclick="openEditMemoryModal('${m.id}', ${JSON.stringify(m.text).replace(/"/g, '&quot;')}, ${m.importance}, ${m.pinned})">
                            ✏️
                        </button>
                        <button class="action-icon-btn" title="Delete" onclick="openDeleteMemoryModal('${m.id}', ${JSON.stringify(m.text).replace(/"/g, '&quot;')})">
                            🗑️
                        </button>
                    </div>
                </div>
                <div class="memory-text">${escapeHtml(m.text)}</div>
                <div class="memory-meta">
                    <span>Imp: ${(m.importance || 0.5).toFixed(1)}</span>
                    <span>Used: ${m.use_count || 0}x</span>
                    <span>${dateStr}</span>
                </div>
            `;
            memoryGrid.appendChild(card);
        });
    } catch (err) {
        memoryGrid.innerHTML = `<div class="empty-state text-danger">Error loading memories: ${err.message}</div>`;
    }
}

memorySearchBtn.addEventListener("click", () => {
    loadMemories(memorySearchInput.value.trim());
});
memorySearchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") loadMemories(memorySearchInput.value.trim());
});
memoryClearBtn.addEventListener("click", () => {
    memorySearchInput.value = "";
    loadMemories();
});

memoryExportBtn.addEventListener("click", () => {
    window.location.href = `/api/memory/export?token=${RAGNAROK_TOKEN}`;
});

// Add Memory Modal
memoryAddModalBtn.addEventListener("click", () => {
    addMemoryForm.reset();
    importanceValLabel.textContent = "0.5";
    addMemoryModal.style.display = "flex";
});
closeAddModalBtn.addEventListener("click", () => addMemoryModal.style.display = "none");
cancelAddModalBtn.addEventListener("click", () => addMemoryModal.style.display = "none");
newMemoryImportance.addEventListener("input", (e) => importanceValLabel.textContent = e.target.value);

addMemoryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = document.getElementById("new-memory-text").value.trim();
    const kind = document.getElementById("new-memory-kind").value;
    const importance = parseFloat(newMemoryImportance.value);
    const pinned = document.getElementById("new-memory-pinned").checked;

    try {
        const res = await apiFetch("/api/memory", {
            method: "POST",
            body: JSON.stringify({ text, kind, importance, pinned }),
        });
        if (res.ok) {
            addMemoryModal.style.display = "none";
            loadMemories();
            pollStatus();
        } else {
            alert("Failed to save memory.");
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
});

// Edit Memory Modal
window.openEditMemoryModal = function(id, text, importance, pinned) {
    document.getElementById("edit-memory-id").value = id;
    document.getElementById("edit-memory-text").value = text;
    editMemoryImportance.value = importance || 0.5;
    editImportanceValLabel.textContent = importance || 0.5;
    document.getElementById("edit-memory-pinned").checked = Boolean(pinned);
    editMemoryModal.style.display = "flex";
};
closeEditModalBtn.addEventListener("click", () => editMemoryModal.style.display = "none");
cancelEditModalBtn.addEventListener("click", () => editMemoryModal.style.display = "none");
editMemoryImportance.addEventListener("input", (e) => editImportanceValLabel.textContent = e.target.value);

editMemoryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = document.getElementById("edit-memory-id").value;
    const text = document.getElementById("edit-memory-text").value.trim();
    const importance = parseFloat(editMemoryImportance.value);
    const pinned = document.getElementById("edit-memory-pinned").checked;

    try {
        const res = await apiFetch(`/api/memory/${id}`, {
            method: "PUT",
            body: JSON.stringify({ text, importance, pinned }),
        });
        if (res.ok) {
            editMemoryModal.style.display = "none";
            loadMemories();
        } else {
            alert("Failed to update memory.");
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
});

// Toggle Pin
window.togglePinMemory = async function(id, pinned) {
    try {
        await apiFetch(`/api/memory/${id}/pin?pinned=${pinned}`, { method: "POST" });
        loadMemories();
    } catch (err) {
        alert(`Error pinning memory: ${err.message}`);
    }
};

// Delete Memory with Confirmation
window.openDeleteMemoryModal = function(id, text) {
    deleteTargetId.value = id;
    deleteMemoryPreview.textContent = text.length > 120 ? text.slice(0, 120) + "..." : text;
    deleteConfirmModal.style.display = "flex";
};
closeDeleteModalBtn.addEventListener("click", () => deleteConfirmModal.style.display = "none");
cancelDeleteModalBtn.addEventListener("click", () => deleteConfirmModal.style.display = "none");

confirmDeleteActionBtn.addEventListener("click", async () => {
    const id = deleteTargetId.value;
    try {
        const res = await apiFetch(`/api/memory/${id}?confirm=true`, { method: "DELETE" });
        if (res.ok) {
            deleteConfirmModal.style.display = "none";
            loadMemories();
            pollStatus();
        } else {
            alert("Failed to delete memory.");
        }
    } catch (err) {
        alert(`Error deleting memory: ${err.message}`);
    }
});


// ── Screen 3: Approvals Flow ──────────────────────────────────────────────────
async function loadApprovals() {
    try {
        const res = await apiFetch("/api/approvals");
        if (!res.ok) return;
        const data = await res.json();
        pendingApprovalsCache = data.pending || [];

        // Render Pending
        if (pendingApprovalsCache.length === 0) {
            pendingApprovalsList.innerHTML = '<div class="empty-state">No pending approvals. System is running smoothly.</div>';
        } else {
            pendingApprovalsList.innerHTML = "";
            pendingApprovalsCache.forEach(appr => {
                const card = document.createElement("div");
                card.className = "approval-card";
                card.id = `appr-card-${appr.request_id}`;
                card.innerHTML = `
                    <div class="approval-card-header">
                        <span class="tool-name">⚙️ ${appr.tool_name}</span>
                        <span class="risk-badge ${appr.risk_level.toLowerCase()}">${appr.risk_level} RISK</span>
                    </div>
                    <p class="approval-reason">${escapeHtml(appr.reason)}</p>
                    <div class="approval-args">
                        <pre>${escapeHtml(JSON.stringify(appr.args, null, 2))}</pre>
                    </div>
                    <div class="approval-actions">
                        <button class="btn btn-secondary" onclick="openEditApprovalModal('${appr.request_id}', '${appr.tool_name}', ${JSON.stringify(appr.args).replace(/"/g, '&quot;')})">
                            ✏️ Edit Parameters
                        </button>
                        <button class="btn btn-danger" onclick="respondApproval('${appr.request_id}', 'denied')">
                            ❌ Deny
                        </button>
                        <button class="btn btn-success" onclick="respondApproval('${appr.request_id}', 'approved')">
                            ✅ Approve
                        </button>
                    </div>
                `;
                pendingApprovalsList.appendChild(card);
            });
        }

        // Render History
        const history = data.history || [];
        if (history.length === 0) {
            approvalsHistoryBody.innerHTML = '<tr><td colspan="4" class="text-muted">No history yet.</td></tr>';
        } else {
            approvalsHistoryBody.innerHTML = "";
            history.forEach(h => {
                const tr = document.createElement("tr");
                const statusColor = h.status === "approved" || h.status === "edited" ? "var(--success)" : "var(--danger)";
                tr.innerHTML = `
                    <td style="color: ${statusColor}; font-weight: 700;">${h.status.toUpperCase()}</td>
                    <td><code>${h.tool_name}</code></td>
                    <td><code>${escapeHtml(JSON.stringify(h.args))}</code></td>
                    <td>${escapeHtml(h.comment || "-")}</td>
                `;
                approvalsHistoryBody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Error loading approvals:", err);
    }
}

window.respondApproval = async function(requestId, status, editedArgs = null) {
    try {
        const res = await apiFetch(`/api/approvals/${requestId}/respond`, {
            method: "POST",
            body: JSON.stringify({ status, edited_args: editedArgs }),
        });
        if (res.ok) {
            loadApprovals();
            pollStatus();
        } else {
            alert("Failed to respond to approval.");
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    }
};

window.openEditApprovalModal = function(id, toolName, args) {
    document.getElementById("edit-approval-id").value = id;
    document.getElementById("edit-appr-tool-name").textContent = toolName;
    document.getElementById("edit-appr-args-json").value = JSON.stringify(args, null, 2);
    editApprovalModal.style.display = "flex";
};
closeApprModalBtn.addEventListener("click", () => editApprovalModal.style.display = "none");
cancelApprModalBtn.addEventListener("click", () => editApprovalModal.style.display = "none");

editApprovalForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = document.getElementById("edit-approval-id").value;
    let parsedArgs;
    try {
        parsedArgs = JSON.parse(document.getElementById("edit-appr-args-json").value);
    } catch (err) {
        alert("Invalid JSON format in parameters.");
        return;
    }
    editApprovalModal.style.display = "none";
    await respondApproval(id, "edited", parsedArgs);
});


// ── Screen 4: Execution Log Timeline ──────────────────────────────────────────
async function loadLogs() {
    try {
        const res = await apiFetch("/api/logs");
        if (!res.ok) return;
        const data = await res.json();

        // Execution Traces
        const traces = data.execution_traces || [];
        if (traces.length === 0) {
            executionTimelineContainer.innerHTML = '<div class="empty-state">No execution traces yet.</div>';
        } else {
            executionTimelineContainer.innerHTML = "";
            traces.forEach(t => {
                const item = document.createElement("div");
                item.className = "timeline-item";

                let stepsHtml = "";
                if (t.plan && t.plan.steps) {
                    stepsHtml = `<div class="timeline-steps"><strong>Plan Execution:</strong>`;
                    t.plan.steps.forEach(s => {
                        stepsHtml += `<div class="timeline-step">Step ${s.step_num}: ${s.description} [${s.status}] -> Tool: ${s.tool || "none"}</div>`;
                    });
                    stepsHtml += `</div>`;
                }

                item.innerHTML = `
                    <div class="timeline-header">
                        <span class="timeline-query">User: "${escapeHtml(t.user_query)}"</span>
                        <span class="timeline-metrics">${t.latency_ms}ms • ${t.steps_taken} steps</span>
                    </div>
                    <p><strong>Answer:</strong> ${escapeHtml(t.answer)}</p>
                    ${stepsHtml}
                    <div><strong>Tools Invoked:</strong> ${t.tool_calls.length ? t.tool_calls.join(", ") : "None"}</div>
                `;
                executionTimelineContainer.appendChild(item);
            });
        }

        // Raw Tool Events
        const rawEvents = data.recent_tool_events || [];
        rawEventsContent.textContent = rawEvents.length
            ? rawEvents.map(e => JSON.stringify(e)).join("\n")
            : "No structured tool events recorded in logs/ragnarok.jsonl yet.";
    } catch (err) {
        console.error("Error loading logs:", err);
    }
}


// ── Screen 5: Settings Flow ───────────────────────────────────────────────────
async function loadSettings() {
    try {
        const res = await apiFetch("/api/settings");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("setting-model-main").textContent = data.models.main;
        document.getElementById("setting-model-small").textContent = data.models.small;
        document.getElementById("setting-model-embed").textContent = data.models.embed;
        document.getElementById("setting-timeout").textContent = `${data.safety.approval_timeout_sec}s`;
        document.getElementById("setting-sandbox-path").textContent = data.paths.allowed_folders.join("; ");
        document.getElementById("setting-daily-brief").textContent = data.scheduler.daily_brief_time;
        settingRoleSelect.value = data.permissions.active_role;

        // Skills grid
        skillsListContainer.innerHTML = "";
        data.skills.forEach(s => {
            const card = document.createElement("div");
            card.className = "skill-card-item";
            card.innerHTML = `
                <div class="skill-card-top">
                    <span class="skill-name">🛠️ ${s.name}</span>
                    <span class="risk-badge ${s.risk_level}">${s.risk_level}</span>
                </div>
                <p class="skill-desc">${escapeHtml(s.description)}</p>
            `;
            skillsListContainer.appendChild(card);
        });
    } catch (err) {
        console.error("Error loading settings:", err);
    }
}

settingRoleSelect.addEventListener("change", async (e) => {
    const newRole = e.target.value;
    try {
        await apiFetch("/api/settings", {
            method: "POST",
            body: JSON.stringify({ active_role: newRole }),
        });
        pollStatus();
    } catch (err) {
        alert(`Failed to change role: ${err.message}`);
    }
});


// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
