import re
from pathlib import Path

path = Path("ragnarok/ui/static/index.html")
text = path.read_text(encoding="utf-8")

# 1. Insert Telemetry & Orchestrator sections after line 1979 (the end of view-agenda-calendar)
target_section_end = '</section>\n\n<!-- ═══════════════════════════════════════════════════════════════════════\n     CREATE EVENT MODAL'
if target_section_end not in text:
    # Try normalized newlines
    target_section_end = '</section>\r\n\r\n<!-- ═══════════════════════════════════════════════════════════════════════\r\n     CREATE EVENT MODAL'

telemetry_and_orchestrator_html = """</section>

<!-- ═══════════════════════════════════════════════════════════════════════
     TELEMETRY & LOGS VIEW (data-path="telemetry-benchmarks")
     ═══════════════════════════════════════════════════════════════════════ -->
<section id="view-telemetry-benchmarks" class="hidden w-full px-space-md lg:px-margin py-space-lg flex flex-col gap-space-lg">
  <!-- Top Banner -->
  <div class="flex flex-wrap items-center justify-between gap-space-md bg-surface-container-lowest/90 backdrop-blur-xl px-space-md py-space-sm rounded-xl shadow-lg border border-outline-variant/30">
    <div class="flex items-center gap-space-md">
      <div class="flex items-center gap-space-xs">
        <span class="w-2.5 h-2.5 rounded-full bg-primary-container animate-pulse shadow-[0_0_10px_rgba(0,240,255,0.8)]"></span>
        <span class="font-label-md text-label-md text-primary tracking-widest uppercase">HARDWARE TELEMETRY &amp; TOOL LOGS</span>
      </div>
      <div class="h-4 w-px bg-outline-variant hidden sm:block"></div>
      <div class="flex items-center gap-1.5 bg-surface-container-high px-space-sm py-1 rounded-full border border-outline-variant/40">
        <span class="w-2 h-2 rounded-full bg-green-400"></span>
        <span id="telemetry-status-text" class="font-label-sm text-label-sm text-green-300 font-semibold uppercase">TELEMETRY LIVE</span>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <button onclick="refreshTelemetryAndLogs()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[12px] font-semibold bg-surface-container hover:bg-surface-container-high text-on-surface border border-outline-variant/30 transition-all">
        <span class="material-symbols-outlined text-[16px]">refresh</span> Refresh Metrics
      </button>
    </div>
  </div>

  <!-- Hardware Metrics Grid -->
  <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-space-md">
    <!-- CPU Card -->
    <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col justify-between">
      <div class="flex items-center justify-between text-on-surface-variant font-label-sm">
        <span class="flex items-center gap-1.5"><span class="material-symbols-outlined text-[18px] text-primary">memory</span> CPU LOAD</span>
        <span id="telemetry-cpu-pct" class="text-primary font-mono text-[14px] font-bold">--%</span>
      </div>
      <div class="my-3 w-full bg-surface-container-highest rounded-full h-2 overflow-hidden">
        <div id="telemetry-cpu-bar" class="bg-primary-container h-full transition-all duration-500" style="width: 0%"></div>
      </div>
      <div class="text-[11px] text-on-surface-variant/80 font-mono flex justify-between">
        <span>Cores: Native</span>
        <span id="telemetry-cpu-status">Nominal</span>
      </div>
    </div>

    <!-- RAM Card -->
    <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col justify-between">
      <div class="flex items-center justify-between text-on-surface-variant font-label-sm">
        <span class="flex items-center gap-1.5"><span class="material-symbols-outlined text-[18px] text-secondary">storage</span> SYSTEM RAM</span>
        <span id="telemetry-ram-text" class="text-secondary font-mono text-[13px] font-bold">-- / -- MB</span>
      </div>
      <div class="my-3 w-full bg-surface-container-highest rounded-full h-2 overflow-hidden">
        <div id="telemetry-ram-bar" class="bg-secondary-container h-full transition-all duration-500" style="width: 0%"></div>
      </div>
      <div class="text-[11px] text-on-surface-variant/80 font-mono flex justify-between">
        <span>RAM Usage</span>
        <span id="telemetry-ram-pct">--%</span>
      </div>
    </div>

    <!-- Active Models Card -->
    <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col justify-between">
      <div class="flex items-center justify-between text-on-surface-variant font-label-sm">
        <span class="flex items-center gap-1.5"><span class="material-symbols-outlined text-[18px] text-primary-fixed-dim">psychology</span> LOCAL MODELS</span>
        <span class="text-primary-fixed-dim font-mono text-[12px] font-semibold">OLLAMA</span>
      </div>
      <div class="my-2 flex flex-col gap-1 font-mono text-[11px]">
        <div class="truncate text-on-surface">Main: <span id="telemetry-model-main" class="text-primary">--</span></div>
        <div class="truncate text-on-surface-variant">Fast: <span id="telemetry-model-small" class="text-primary-fixed-dim">--</span></div>
      </div>
      <div class="text-[10px] text-on-surface-variant/80 font-mono truncate">
        URL: <span id="telemetry-ollama-url" class="text-on-surface">http://localhost:11434</span>
      </div>
    </div>

    <!-- Uptime & System Health Card -->
    <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col justify-between">
      <div class="flex items-center justify-between text-on-surface-variant font-label-sm">
        <span class="flex items-center gap-1.5"><span class="material-symbols-outlined text-[18px] text-green-400">timer</span> SYSTEM UPTIME</span>
        <span id="telemetry-uptime" class="text-green-300 font-mono text-[13px] font-bold">0s</span>
      </div>
      <div class="my-2 flex flex-col gap-1 font-mono text-[11px]">
        <div class="text-on-surface">Role: <span id="telemetry-active-role" class="text-primary uppercase font-bold">DEFAULT</span></div>
        <div class="text-on-surface-variant">Memories: <span id="telemetry-memory-count" class="text-on-surface font-semibold">0</span> items</div>
      </div>
      <div class="text-[10px] text-on-surface-variant/80 font-mono flex justify-between">
        <span>Sandbox: Airgapped</span>
        <span class="text-primary-container font-semibold">SECURE</span>
      </div>
    </div>
  </div>

  <!-- Real-time Execution Traces & Structured Tool Logs -->
  <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col gap-space-md">
    <div class="flex flex-wrap items-center justify-between gap-space-sm border-b border-outline-variant/30 pb-3">
      <div class="flex items-center gap-2">
        <span class="material-symbols-outlined text-primary text-[20px]">terminal</span>
        <h3 class="font-headline-sm text-headline-sm text-on-surface">Visual Execution Timeline &amp; Tool Logs</h3>
      </div>
      <div class="flex items-center gap-2">
        <!-- Log Filter Buttons -->
        <div class="flex items-center bg-surface-container rounded-lg p-0.5 border border-outline-variant/30 text-[11px] font-mono">
          <button onclick="setLogFilter('all')" id="log-filter-all" class="px-2.5 py-1 rounded bg-primary-container/20 text-primary font-bold">ALL</button>
          <button onclick="setLogFilter('tools')" id="log-filter-tools" class="px-2.5 py-1 rounded text-on-surface-variant hover:text-on-surface">TOOLS</button>
          <button onclick="setLogFilter('traces')" id="log-filter-traces" class="px-2.5 py-1 rounded text-on-surface-variant hover:text-on-surface">TRACES</button>
        </div>
        <button onclick="clearUiLogs()" class="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface text-[12px] border border-outline-variant/30" title="Clear logs view">
          <span class="material-symbols-outlined text-[16px]">clear_all</span>
        </button>
      </div>
    </div>

    <!-- Logs Container -->
    <div id="telemetry-logs-stream" class="bg-surface-container-lowest rounded-xl p-4 font-mono text-[11px] leading-relaxed max-h-[460px] overflow-y-auto flex flex-col gap-2 border border-outline-variant/20 shadow-inner">
      <div class="text-on-surface-variant italic">Polling execution logs &amp; system telemetry...</div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════════════════
     NEURAL AGENT ORCHESTRATOR VIEW (data-path="agent-orchestrator")
     ═══════════════════════════════════════════════════════════════════════ -->
<section id="view-agent-orchestrator" class="hidden w-full px-space-md lg:px-margin py-space-lg flex flex-col gap-space-lg">
  <!-- Top Banner -->
  <div class="flex flex-wrap items-center justify-between gap-space-md bg-surface-container-lowest/90 backdrop-blur-xl px-space-md py-space-sm rounded-xl shadow-lg border border-outline-variant/30">
    <div class="flex items-center gap-space-md">
      <div class="flex items-center gap-space-xs">
        <span class="w-2.5 h-2.5 rounded-full bg-secondary-fixed-dim animate-pulse shadow-[0_0_10px_rgba(221,183,255,0.8)]"></span>
        <span class="font-label-md text-label-md text-secondary tracking-widest uppercase">NEURAL AGENT SWARM &amp; TOOL GOVERNANCE</span>
      </div>
      <div class="h-4 w-px bg-outline-variant hidden sm:block"></div>
      <div class="flex items-center gap-1.5 bg-surface-container-high px-space-sm py-1 rounded-full border border-outline-variant/40">
        <span class="w-2 h-2 rounded-full bg-primary-container"></span>
        <span class="font-label-sm text-label-sm text-primary font-semibold uppercase">MULTI-WORKER ENGINE ACTIVE</span>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <button onclick="refreshAgentOrchestrator()" class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[12px] font-semibold bg-surface-container hover:bg-surface-container-high text-on-surface border border-outline-variant/30 transition-all">
        <span class="material-symbols-outlined text-[16px]">refresh</span> Refresh Swarm State
      </button>
    </div>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-space-lg">
    <!-- Left Column: Pending Approvals & Security Gate (7 cols) -->
    <div class="lg:col-span-7 flex flex-col gap-space-lg">
      <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col gap-space-md">
        <div class="flex items-center justify-between border-b border-outline-variant/30 pb-3">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-amber-400 text-[22px]">shield</span>
            <h3 class="font-headline-sm text-headline-sm text-on-surface">High-Risk Tool Approvals</h3>
          </div>
          <span id="orchestrator-pending-badge" class="font-mono text-[11px] px-2 py-0.5 rounded-full bg-surface-container text-on-surface-variant">0 Pending</span>
        </div>
        
        <!-- Pending Approvals Queue -->
        <div id="orchestrator-approvals-queue" class="flex flex-col gap-3 min-h-[140px]">
          <div class="flex flex-col items-center justify-center p-8 text-on-surface-variant font-mono text-[12px] border border-dashed border-outline-variant/30 rounded-xl">
            <span class="material-symbols-outlined text-[28px] text-green-400 mb-1">verified_user</span>
            <span>No pending approval requests. System running safely within security thresholds.</span>
          </div>
        </div>

        <!-- Recent Resolved Approvals History -->
        <div class="mt-2 border-t border-outline-variant/30 pt-3">
          <div class="text-[12px] font-semibold text-on-surface-variant mb-2 flex items-center gap-1.5">
            <span class="material-symbols-outlined text-[16px]">history</span> Recent Approval Decisions
          </div>
          <div id="orchestrator-approvals-history" class="flex flex-col gap-2 max-h-48 overflow-y-auto font-mono text-[11px]">
            <div class="text-on-surface-variant italic">No recent approval history.</div>
          </div>
        </div>
      </div>

      <!-- Autonomous Workers & Shadow Execution Status -->
      <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col gap-space-sm">
        <div class="flex items-center gap-2 border-b border-outline-variant/30 pb-2">
          <span class="material-symbols-outlined text-primary text-[20px]">smart_toy</span>
          <h3 class="font-headline-sm text-headline-sm text-on-surface">Autonomous Background Workers</h3>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-1">
          <div class="bg-surface-container rounded-xl p-3 border border-outline-variant/30 flex flex-col gap-1.5">
            <div class="flex items-center justify-between">
              <span class="font-semibold text-[13px] text-on-surface flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-green-400"></span> Shadow Worker</span>
              <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-green-500/20 text-green-300">STANDBY</span>
            </div>
            <p class="text-[11px] text-on-surface-variant">Continuous background goal execution &amp; autonomous self-healing engine.</p>
          </div>
          <div class="bg-surface-container rounded-xl p-3 border border-outline-variant/30 flex flex-col gap-1.5">
            <div class="flex items-center justify-between">
              <span class="font-semibold text-[13px] text-on-surface flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-primary-container"></span> Ambient Context</span>
              <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary-container/20 text-primary">LISTENING</span>
            </div>
            <p class="text-[11px] text-on-surface-variant">Perception buffer capturing desktop screen changes &amp; voice triggers.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column: Loaded Skills Gallery & Risk Manifest (5 cols) -->
    <div class="lg:col-span-5 flex flex-col gap-space-md">
      <div class="bg-surface-container-low rounded-2xl p-space-md border border-outline-variant/30 flex flex-col gap-space-md">
        <div class="flex items-center justify-between border-b border-outline-variant/30 pb-3">
          <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-primary-fixed-dim text-[22px]">extension</span>
            <h3 class="font-headline-sm text-headline-sm text-on-surface">Registered Skills Gallery</h3>
          </div>
          <span id="orchestrator-skill-count" class="font-mono text-[11px] px-2 py-0.5 rounded-full bg-surface-container text-primary">0 Skills</span>
        </div>

        <div id="orchestrator-skills-list" class="flex flex-col gap-2.5 max-h-[580px] overflow-y-auto pr-1">
          <div class="text-on-surface-variant text-[12px] italic">Loading registered skill manifests...</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════════════════════════════════
     CREATE EVENT MODAL"""

if "id=\"view-telemetry-benchmarks\"" not in text:
    text = text.replace(target_section_end, telemetry_and_orchestrator_html)

# 2. Update Nav Tab Integration script to include all 5 views
old_nav_code = """          // Hide all view sections
          const views = [
            document.getElementById('view-voice-interaction-studio'),
            document.getElementById('view-multimodal-vision'),
            document.getElementById('view-agenda-calendar'),
          ];
          views.forEach(v => { if (v) v.classList.add('hidden'); });

          // Update active link styles
          const activeClasses = (link.closest('nav[data-active-classes]')?.getAttribute('data-active-classes')||'').split(' ');
          navLinks.forEach(function(l) {
            l.removeAttribute('aria-current');
            activeClasses.forEach(c => { if(c) l.classList.remove(c); });
            l.classList.add('font-label-md','text-label-md','text-on-surface-variant');
          });
          link.setAttribute('aria-current','page');
          activeClasses.forEach(c => { if(c) link.classList.add(c); });
          link.classList.remove('text-on-surface-variant');

          if (path === 'voice-interaction-studio') {
            const el = document.getElementById('view-voice-interaction-studio');
            if (el) el.classList.remove('hidden');
          } else if (path === 'multimodal-vision') {
            const el = document.getElementById('view-multimodal-vision');
            if (el) el.classList.remove('hidden');
          } else if (path === 'agenda-calendar') {
            const el = document.getElementById('view-agenda-calendar');
            if (el) el.classList.remove('hidden');
            agendaRenderCalendar();
            agendaLoadTodos('all');
          } else {
            const el = document.getElementById('view-voice-interaction-studio');
            if (el) el.classList.remove('hidden');
          }"""

new_nav_code = """          // Hide all view sections
          const views = [
            document.getElementById('view-voice-interaction-studio'),
            document.getElementById('view-multimodal-vision'),
            document.getElementById('view-agenda-calendar'),
            document.getElementById('view-telemetry-benchmarks'),
            document.getElementById('view-agent-orchestrator')
          ];
          views.forEach(v => { if (v) v.classList.add('hidden'); });

          // Update active link styles
          const activeClasses = (link.closest('nav[data-active-classes]')?.getAttribute('data-active-classes')||'').split(' ');
          navLinks.forEach(function(l) {
            l.removeAttribute('aria-current');
            activeClasses.forEach(c => { if(c) l.classList.remove(c); });
            l.classList.add('font-label-md','text-label-md','text-on-surface-variant');
          });
          link.setAttribute('aria-current','page');
          activeClasses.forEach(c => { if(c) link.classList.add(c); });
          link.classList.remove('text-on-surface-variant');

          if (path === 'voice-interaction-studio') {
            const el = document.getElementById('view-voice-interaction-studio');
            if (el) el.classList.remove('hidden');
          } else if (path === 'multimodal-vision') {
            const el = document.getElementById('view-multimodal-vision');
            if (el) el.classList.remove('hidden');
          } else if (path === 'telemetry-benchmarks') {
            const el = document.getElementById('view-telemetry-benchmarks');
            if (el) el.classList.remove('hidden');
            if (window.refreshTelemetryAndLogs) window.refreshTelemetryAndLogs();
          } else if (path === 'agent-orchestrator') {
            const el = document.getElementById('view-agent-orchestrator');
            if (el) el.classList.remove('hidden');
            if (window.refreshAgentOrchestrator) window.refreshAgentOrchestrator();
          } else if (path === 'agenda-calendar') {
            const el = document.getElementById('view-agenda-calendar');
            if (el) el.classList.remove('hidden');
            if (window.agendaRenderCalendar) window.agendaRenderCalendar();
            if (window.agendaLoadTodos) window.agendaLoadTodos('all');
          } else {
            const el = document.getElementById('view-voice-interaction-studio');
            if (el) el.classList.remove('hidden');
          }"""

text = text.replace(old_nav_code, new_nav_code)

# 3. Add Telemetry & Orchestrator JS Module before the screen sharing module or after it
telemetry_and_orchestrator_js = """
// ═══════════════════════════════════════════════════════════════════════════
//  TELEMETRY, LOGS & NEURAL ORCHESTRATOR MODULE
// ═══════════════════════════════════════════════════════════════════════════
(function() {
  let activeLogFilter = 'all';
  let cachedLogs = { execution_traces: [], recent_tool_events: [] };

  window.setLogFilter = function(f) {
    activeLogFilter = f;
    ['all', 'tools', 'traces'].forEach(id => {
      const btn = document.getElementById('log-filter-' + id);
      if (btn) {
        if (id === f) {
          btn.className = 'px-2.5 py-1 rounded bg-primary-container/20 text-primary font-bold';
        } else {
          btn.className = 'px-2.5 py-1 rounded text-on-surface-variant hover:text-on-surface';
        }
      }
    });
    renderLogs(cachedLogs);
  };

  window.clearUiLogs = function() {
    cachedLogs = { execution_traces: [], recent_tool_events: [] };
    const container = document.getElementById('telemetry-logs-stream');
    if (container) container.innerHTML = '<div class="text-on-surface-variant italic">Logs cleared.</div>';
  };

  function renderLogs(data) {
    const container = document.getElementById('telemetry-logs-stream');
    if (!container) return;
    const items = [];

    if (activeLogFilter === 'all' || activeLogFilter === 'traces') {
      const traces = data.execution_traces || [];
      traces.forEach(t => {
        items.push({
          type: 'trace',
          time: t.timestamp || '',
          label: t.action || 'Trace Event',
          desc: t.details || JSON.stringify(t),
          color: 'text-primary'
        });
      });
    }

    if (activeLogFilter === 'all' || activeLogFilter === 'tools') {
      const toolEvents = data.recent_tool_events || [];
      toolEvents.forEach(e => {
        items.push({
          type: 'tool',
          time: e.timestamp || e.asctime || '',
          label: e.tool || e.event || 'Tool Call',
          desc: e.message || JSON.stringify(e),
          color: 'text-secondary'
        });
      });
    }

    if (items.length === 0) {
      container.innerHTML = '<div class="text-on-surface-variant italic">No execution events recorded yet. Run a prompt in the Studio to generate traces.</div>';
      return;
    }

    container.innerHTML = items.slice(-50).map(item => `
      <div class="flex items-start gap-2 border-b border-outline-variant/10 pb-1.5">
        <span class="text-on-surface-variant/60 flex-shrink-0">[${(item.time.split('T')[1] || item.time).slice(0, 8)}]</span>
        <span class="${item.color} font-semibold flex-shrink-0 uppercase text-[10px] px-1.5 py-0.2 rounded bg-surface-container">${item.type}: ${item.label}</span>
        <span class="text-on-surface truncate">${item.desc}</span>
      </div>
    `).join('');
    container.scrollTop = container.scrollHeight;
  }

  window.refreshTelemetryAndLogs = async function() {
    const token = document.querySelector('meta[name="ragnarok-token"]')?.content || '';
    try {
      const rStat = await fetch('/api/status', { credentials: 'omit', headers: { 'X-Ragnarok-Token': token } });
      if (rStat.ok) {
        const d = await rStat.json();
        const cpuPct = d.cpu_percent || 0;
        const cpuEl = document.getElementById('telemetry-cpu-pct');
        const cpuBar = document.getElementById('telemetry-cpu-bar');
        if (cpuEl) cpuEl.textContent = `${cpuPct}%`;
        if (cpuBar) cpuBar.style.width = `${Math.min(100, Math.max(0, cpuPct))}%`;

        const ramUsed = d.ram_used_mb || 0;
        const ramTotal = d.ram_total_mb || 1;
        const ramPct = Math.round((ramUsed / ramTotal) * 100);
        const ramText = document.getElementById('telemetry-ram-text');
        const ramBar = document.getElementById('telemetry-ram-bar');
        const ramPctEl = document.getElementById('telemetry-ram-pct');
        if (ramText) ramText.textContent = `${ramUsed} / ${ramTotal} MB`;
        if (ramBar) ramBar.style.width = `${ramPct}%`;
        if (ramPctEl) ramPctEl.textContent = `${ramPct}%`;

        const mMain = document.getElementById('telemetry-model-main');
        const mSmall = document.getElementById('telemetry-model-small');
        const mUrl = document.getElementById('telemetry-ollama-url');
        if (mMain) mMain.textContent = d.models?.main || 'default';
        if (mSmall) mSmall.textContent = d.models?.small || 'fast';
        if (mUrl) mUrl.textContent = d.ollama_url || 'http://localhost:11434';

        const upEl = document.getElementById('telemetry-uptime');
        const roleEl = document.getElementById('telemetry-active-role');
        const memEl = document.getElementById('telemetry-memory-count');
        if (upEl) {
          const s = Math.round(d.uptime_seconds || 0);
          upEl.textContent = s > 60 ? `${Math.floor(s/60)}m ${s%60}s` : `${s}s`;
        }
        if (roleEl) roleEl.textContent = d.active_role || 'DEFAULT';
        if (memEl) memEl.textContent = d.memory_count || '0';
      }

      const rLogs = await fetch('/api/logs?limit=40', { credentials: 'omit', headers: { 'X-Ragnarok-Token': token } });
      if (rLogs.ok) {
        cachedLogs = await rLogs.json();
        renderLogs(cachedLogs);
      }
    } catch (e) {
      console.warn('Telemetry refresh failed:', e);
    }
  };

  window.refreshAgentOrchestrator = async function() {
    const token = document.querySelector('meta[name="ragnarok-token"]')?.content || '';
    try {
      // 1. Load Approvals
      const rAppr = await fetch('/api/approvals', { credentials: 'omit', headers: { 'X-Ragnarok-Token': token } });
      if (rAppr.ok) {
        const d = await rAppr.json();
        const pendingBadge = document.getElementById('orchestrator-pending-badge');
        const queueContainer = document.getElementById('orchestrator-approvals-queue');
        const historyContainer = document.getElementById('orchestrator-approvals-history');

        const pending = d.pending || [];
        if (pendingBadge) {
          pendingBadge.textContent = `${pending.length} Pending`;
          pendingBadge.className = pending.length > 0
            ? 'font-mono text-[11px] px-2 py-0.5 rounded-full bg-error-container text-error font-bold animate-pulse'
            : 'font-mono text-[11px] px-2 py-0.5 rounded-full bg-surface-container text-on-surface-variant';
        }

        if (queueContainer) {
          if (pending.length === 0) {
            queueContainer.innerHTML = `
              <div class="flex flex-col items-center justify-center p-8 text-on-surface-variant font-mono text-[12px] border border-dashed border-outline-variant/30 rounded-xl">
                <span class="material-symbols-outlined text-[28px] text-green-400 mb-1">verified_user</span>
                <span>No pending approval requests. System running safely within security thresholds.</span>
              </div>
            `;
          } else {
            queueContainer.innerHTML = pending.map(req => `
              <div class="bg-surface-container rounded-xl p-3.5 border border-outline-variant/40 flex flex-col gap-2 shadow-sm">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-amber-400 text-[18px]">warning</span>
                    <span class="font-bold text-[13px] text-on-surface">${req.tool_name || 'High-Risk Tool'}</span>
                    <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-error/20 text-error font-semibold uppercase">${req.risk_tier || 'HIGH'}</span>
                  </div>
                  <span class="text-[10px] font-mono text-on-surface-variant">${(req.created_at || '').slice(11,19)}</span>
                </div>
                <div class="text-[11px] font-mono bg-surface-container-high rounded p-2 text-on-surface whitespace-pre-wrap">${JSON.stringify(req.args || {}, null, 2)}</div>
                <div class="flex items-center gap-2 justify-end mt-1">
                  <button onclick="respondApprovalAction('${req.request_id}', 'denied')" class="px-3 py-1.5 rounded-lg bg-surface-container-high hover:bg-error/30 text-error font-medium text-[12px] border border-outline-variant/30 transition-colors">Deny</button>
                  <button onclick="respondApprovalAction('${req.request_id}', 'approved')" class="px-3 py-1.5 rounded-lg bg-primary-container text-on-primary font-semibold text-[12px] transition-colors shadow-[0_0_10px_rgba(0,240,255,0.2)]">Approve &amp; Run</button>
                </div>
              </div>
            `).join('');
          }
        }

        if (historyContainer) {
          const hist = d.history || [];
          if (hist.length === 0) {
            historyContainer.innerHTML = '<div class="text-on-surface-variant italic">No recent approval history.</div>';
          } else {
            historyContainer.innerHTML = hist.slice(-10).reverse().map(h => `
              <div class="flex items-center justify-between p-2 rounded bg-surface-container border border-outline-variant/20">
                <div class="flex items-center gap-2 truncate">
                  <span class="w-1.5 h-1.5 rounded-full ${h.status === 'approved' ? 'bg-green-400' : 'bg-red-400'}"></span>
                  <span class="font-semibold text-on-surface truncate">${h.tool_name || 'Tool'}</span>
                  <span class="text-[10px] uppercase font-bold ${h.status === 'approved' ? 'text-green-300' : 'text-error'}">${h.status}</span>
                </div>
                <span class="text-[10px] text-on-surface-variant/70 font-mono">${(h.resolved_at || '').slice(11,19)}</span>
              </div>
            `).join('');
          }
        }
      }

      // 2. Load Skills Gallery
      const rSet = await fetch('/api/settings', { credentials: 'omit', headers: { 'X-Ragnarok-Token': token } });
      if (rSet.ok) {
        const d = await rSet.json();
        const skills = d.skills || [];
        const countBadge = document.getElementById('orchestrator-skill-count');
        const listContainer = document.getElementById('orchestrator-skills-list');
        if (countBadge) countBadge.textContent = `${skills.length} Skills`;
        if (listContainer) {
          if (skills.length === 0) {
            listContainer.innerHTML = '<div class="text-on-surface-variant italic">No skills registered.</div>';
          } else {
            const riskColors = {
              low: 'bg-green-500/20 text-green-300 border-green-500/30',
              medium: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
              high: 'bg-red-500/20 text-red-300 border-red-500/30',
              critical: 'bg-purple-500/20 text-purple-300 border-purple-500/30'
            };
            listContainer.innerHTML = skills.map(s => `
              <div class="p-2.5 rounded-xl bg-surface-container border border-outline-variant/30 flex flex-col gap-1 hover:border-primary/40 transition-colors">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-1.5">
                    <span class="material-symbols-outlined text-[16px] text-primary">bolt</span>
                    <span class="font-semibold text-[12px] text-on-surface font-mono">${s.name}</span>
                  </div>
                  <span class="text-[9px] uppercase font-mono px-2 py-0.5 rounded-full border ${riskColors[s.risk_level] || riskColors.low}">${s.risk_level || 'low'}</span>
                </div>
                <div class="text-[11px] text-on-surface-variant/80 line-clamp-2">${s.description || 'No description provided.'}</div>
              </div>
            `).join('');
          }
        }
      }
    } catch (e) {
      console.warn('Orchestrator refresh failed:', e);
    }
  };

  window.respondApprovalAction = async function(requestId, action) {
    const token = document.querySelector('meta[name="ragnarok-token"]')?.content || '';
    try {
      const r = await fetch(`/api/approvals/${requestId}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Ragnarok-Token': token },
        body: JSON.stringify({ status: action })
      });
      if (r.ok) {
        refreshAgentOrchestrator();
      } else {
        const err = await r.json();
        alert('Action failed: ' + (err.detail || 'Unknown error'));
      }
    } catch (e) {
      alert('Network error: ' + e.message);
    }
  };
})();
"""

if "window.refreshTelemetryAndLogs" not in text:
    target_js_insert = "// ═══════════════════════════════════════════════════════════════════════════\n//  SCREEN SHARING & MULTIMODAL VISION MODULE"
    if target_js_insert in text:
        text = text.replace(target_js_insert, telemetry_and_orchestrator_js + "\n" + target_js_insert)
    else:
        # Fallback to appending before the last </script>
        text = text.replace("})();\n\n</script>", "})();\n" + telemetry_and_orchestrator_js + "\n</script>")

# 4. Fix Footer and closing tags
target_footer = re.compile(r'</div></main><footer.*$', re.DOTALL)
proper_footer = """</div><!-- /flex flex-col w-full -->
</main>

<footer class="w-full bg-surface-container-lowest border-t border-outline-variant/30 py-space-md mt-auto">
  <div class="w-full px-space-md lg:px-margin flex flex-col sm:flex-row items-center justify-between gap-space-sm">
    <div class="flex items-center gap-space-sm text-on-surface-variant font-label-sm text-label-sm">
      <span class="w-2 h-2 rounded-full bg-primary-container animate-pulse"></span>
      <span>© 2026 RAGNAROK AI · LOCAL MULTIMODAL AGENT CORE</span>
    </div>
    <div class="flex items-center gap-space-md font-label-sm text-label-sm text-on-surface-variant">
      <span class="flex items-center gap-space-xs"><span class="w-1.5 h-1.5 rounded-full bg-primary-container"></span>WebRTC v2 Active</span>
      <span class="flex items-center gap-space-xs"><span class="w-1.5 h-1.5 rounded-full bg-primary-fixed-dim"></span>48.0 kHz Lossless</span>
      <span class="flex items-center gap-space-xs"><span class="w-1.5 h-1.5 rounded-full bg-green-400"></span>100% Offline Core</span>
    </div>
  </div>
</footer>

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.31/dist/bundle.min.js" defer></script>
</body>
</html>
"""

text = target_footer.sub(proper_footer, text)

# Write to ragnarok/ui/static/index.html
path.write_text(text, encoding="utf-8")
print("Updated ragnarok/ui/static/index.html successfully!")

# Copy to root index.html so both are 100% synced!
root_path = Path("index.html")
root_path.write_text(text, encoding="utf-8")
print("Synced root index.html successfully!")
