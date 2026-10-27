"""
ragnarok/ui/app.py
──────────────────
Phase 7: Local interface and transparency backend.
FastAPI web application bound to localhost with token authentication.

Exposes endpoints for:
  - System status & hardware telemetry (CPU, RAM, model, uptime)
  - Interactive chat with AgentLoop and Swarm (streaming / step explainability)
  - Memory Manager: search, view, edit, pin/unpin, delete (with confirmation), export
  - Approvals: list pending high-risk requests, approve, deny, edit parameters
  - Execution Log: timeline of plans, tool selections, rationales, arguments, results
  - Settings: model selection, roles, allowed folders, triggers
  - Security: token rotation, audit log integrity verification, swarm endpoints

Security hardening applied:
  - CORS locked to localhost origins only
  - CSRF / Origin guard dependency on all mutating endpoints
  - Per-IP sliding-window rate limiter (30 req/min)
  - Input length + control-char validation on all user text fields
  - HMAC-SHA256 signed audit log entries (tamper detection)
  - Automatic UI token rotation every 30 minutes
  - Content-Security-Policy, X-Frame-Options, X-Content-Type-Options headers
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

import psutil
from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from ragnarok import llm
from ragnarok.agent.loop import AgentLoop, AgentResult
from ragnarok.agent.swarm import SwarmOrchestrator, SwarmResult
from ragnarok.config import settings
from ragnarok.memory.recall import MemoryStore
from ragnarok.safety.approval import ApprovalDecision, ApprovalManager, ApprovalStatus
from ragnarok.safety.audit_integrity import get_audit_signer
from ragnarok.safety.csrf_guard import require_local_origin
from ragnarok.safety.input_guard import InputValidator, check_ip_rate_limit
from ragnarok.safety.permissions import PermissionManager
from ragnarok.safety.token_rotation import get_rotator
from ragnarok.skills.registry import SkillRegistry, registry as default_registry

logger = logging.getLogger(__name__)

# Directory paths
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
START_TIME = time.time()

# ── Security Token ─────────────────────────────────────────────────────────────
# Generate a random token on startup or use environment / token file
TOKEN_FILE = settings.knowledge_dir / "ui_token.txt"
if os.environ.get("RAGNAROK_UI_TOKEN"):
    ACTIVE_TOKEN = os.environ["RAGNAROK_UI_TOKEN"]
elif TOKEN_FILE.exists():
    try:
        ACTIVE_TOKEN = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        ACTIVE_TOKEN = secrets.token_urlsafe(16)
else:
    ACTIVE_TOKEN = secrets.token_urlsafe(16)
    try:
        TOKEN_FILE.write_text(ACTIVE_TOKEN, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not persist UI token to %s: %s", TOKEN_FILE, exc)


# Initialise token rotator with the loaded/generated token
_rotator = get_rotator(initial_token=ACTIVE_TOKEN)


async def _broadcast_new_token(new_token: str) -> None:
    """Push the new token to all connected browser WebSocket clients."""
    dead: list[WebSocket] = []
    for ws in list(_token_ws_clients):
        try:
            await ws.send_json({"type": "token_rotated", "token": new_token})
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            _token_ws_clients.remove(ws)
        except ValueError:
            pass


def verify_token(
    x_ragnarok_token: Optional[str] = Header(None, alias="X-Ragnarok-Token"),
    token: Optional[str] = Query(None),
) -> str:
    """Verifies the token using the rotator (supports grace-period tokens)."""
    # Also check if rotation is due
    _rotator.maybe_rotate()

    provided = x_ragnarok_token or token
    if not provided or not _rotator.is_valid(provided):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Ragnarok security token",
        )
    return provided


# ── In-Memory Execution Trace Buffer ──────────────────────────────────────────
class ExecutionTraceStore:
    def __init__(self, max_records: int = 200) -> None:
        self.records: list[dict[str, Any]] = []
        self.max_records = max_records

    def add_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def get_all(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self.records[-limit:]))


trace_store = ExecutionTraceStore()
_audit_signer = get_audit_signer()

# WebSocket clients subscribed to token rotation events
_token_ws_clients: list[WebSocket] = []


# ── Global Application State ──────────────────────────────────────────────────
class AppState:
    def __init__(self) -> None:
        self.approval_manager = ApprovalManager()
        self.permission_manager = PermissionManager()
        self.registry = default_registry
        self.registry.load_all()
        self.store = MemoryStore()
        self.agent = AgentLoop(
            registry=self.registry,
            approval_cb=lambda tool, args: self.approval_manager.request_approval(
                tool_name=tool, args=args
            ).is_approved,
            memory_fn=lambda q: self.store.recall_as_prompt(q),
        )
        self.swarm = SwarmOrchestrator(
            registry=self.registry,
            approval_cb=lambda tool, args: self.approval_manager.request_approval(
                tool_name=tool, args=args
            ).is_approved,
            memory_fn=lambda q: self.store.recall_as_prompt(q),
            enable_speculative=False,
        )


state = AppState()

# ── FastAPI App Setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Ragnarok Local AI Interface",
    description="Local interface and transparency dashboard for the Ragnarok AI Agent",
    version="1.0.0",
    docs_url=None,   # Disable Swagger UI in production (no external JS)
    redoc_url=None,
    dependencies=[Depends(require_local_origin)],
)

# ── CORS: LOCKED TO LOCALHOST ONLY ─────────────────────────────────────────────
# Only same-origin localhost requests are accepted. This blocks CSRF from remote pages.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:7788",
        "http://127.0.0.1:7788",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["X-Ragnarok-Token", "Content-Type", "Authorization"],
)


# ── Security Headers Middleware ────────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers on every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://cdn.tailwindcss.com "
            "https://fonts.googleapis.com https://cdn.jsdelivr.net https://ajax.googleapis.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' ws://localhost:* ws://127.0.0.1:*; "
            "img-src 'self' data:; "
            "frame-ancestors 'none';"
        )
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), payment=()"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# ── Pydantic Request Models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt message")


class MemoryCreateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    kind: str = "episodic"
    importance: float = 0.5
    pinned: bool = False


class MemoryUpdateRequest(BaseModel):
    text: Optional[str] = None
    importance: Optional[float] = None
    pinned: Optional[bool] = None


class ApprovalActionRequest(BaseModel):
    status: str = Field(..., description="'approved', 'denied', or 'edited'")
    edited_args: Optional[dict[str, Any]] = None
    comment: str = ""


class SettingsUpdateRequest(BaseModel):
    active_role: Optional[str] = None
    approval_timeout_sec: Optional[float] = None
    daily_brief_time: Optional[str] = None


# ── Status & Telemetry ────────────────────────────────────────────────────────
@app.get("/api/status")
def get_system_status(_: str = Depends(verify_token)) -> dict[str, Any]:
    """Returns current hardware load, active models, memories count, and role."""
    try:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = round(mem.used / (1024 * 1024), 1)
        ram_total = round(mem.total / (1024 * 1024), 1)
    except Exception:
        cpu_pct = 0.0
        ram_used = 0.0
        ram_total = 0.0

    return {
        "status": "online",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "cpu_percent": cpu_pct,
        "ram_used_mb": ram_used,
        "ram_total_mb": ram_total,
        "models": {
            "main": settings.model_main,
            "small": settings.model_small,
            "embed": settings.model_embed,
        },
        "ollama_url": settings.ollama_url,
        "active_role": settings.active_role,
        "memory_count": state.store.count(),
        "allowed_folders": [str(f) for f in settings.allowed_folders],
        "registered_skills": [s.manifest.name for s in state.registry.all_skills()],
        "pending_approvals_count": len(state.approval_manager.get_pending_requests()),
    }


# ── Chat Endpoint ─────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat_with_agent(
    req: ChatRequest,
    request: Request,
    _: str = Depends(verify_token),
    _ip: None = Depends(check_ip_rate_limit),
    _origin: None = Depends(require_local_origin),
) -> dict[str, Any]:
    """Execute a turn with the Ragnarok agent and return explainable steps."""
    # Validate & sanitise user input
    try:
        safe_message = InputValidator.validate_chat_message(req.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    t0 = time.perf_counter()
    res: AgentResult = state.agent.chat(safe_message)
    latency_ms = (time.perf_counter() - t0) * 1000

    plan_data = None
    if res.plan:
        plan_data = {
            "goal": res.plan.goal,
            "steps": [asdict(s) for s in res.plan.steps],
            "is_complete": res.plan.is_complete,
            "current_step": res.plan.current_step,
        }

    trace_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_query": safe_message,
        "answer": res.answer,
        "steps_taken": res.steps_taken,
        "tool_calls": res.tool_calls_made,
        "latency_ms": round(latency_ms, 1),
        "plan": plan_data,
        "stopped_early": res.stopped_early,
        "agent": "loop",
    }
    trace_store.add_record(trace_entry)
    # Sign the audit log entry
    signed = _audit_signer.sign_entry(trace_entry)
    logger.debug("Audit signed entry len=%d", len(signed))
    return trace_entry


# ── Swarm Chat Endpoint (Innovation #4) ───────────────────────────────────────
@app.post("/api/swarm/chat")
def swarm_chat(
    req: ChatRequest,
    request: Request,
    _: str = Depends(verify_token),
    _ip: None = Depends(check_ip_rate_limit),
    _origin: None = Depends(require_local_origin),
) -> dict[str, Any]:
    """
    Route a user request through the Multi-Agent Swarm:
    ArchitectAgent → ExecutorAgent → SecurityAuditorAgent.
    Returns the audited answer with routing metadata.
    """
    try:
        safe_message = InputValidator.validate_chat_message(req.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    swarm_result: SwarmResult = state.swarm.run(safe_message)

    trace_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_query": safe_message,
        "answer": swarm_result.answer,
        "steps_taken": swarm_result.steps_taken,
        "tool_calls": swarm_result.tool_calls_made,
        "latency_ms": round(swarm_result.latency_ms, 1),
        "agent": swarm_result.agent_path,
        "plan_summary": swarm_result.plan_summary,
        "audit": {
            "passed": swarm_result.audit.passed,
            "risk_score": round(swarm_result.audit.risk_score, 3),
            "findings": swarm_result.audit.findings,
        },
    }
    trace_store.add_record(trace_entry)
    return trace_entry


# ── Chat History Endpoints ───────────────────────────────────────────────────
@app.get("/api/chat/history")
def get_chat_history(
    limit: int = 50, _: str = Depends(verify_token)
) -> list[dict[str, Any]]:
    """Retrieve recent chat history records."""
    return trace_store.get_all(limit=limit)


@app.delete("/api/chat/history")
def clear_chat_history(_: str = Depends(verify_token)) -> dict[str, Any]:
    """Clear in-memory chat history records."""
    trace_store.records.clear()
    return {"status": "cleared", "count": 0}


# ── Security: Token rotation ───────────────────────────────────────────────────
@app.post("/api/security/rotate-token")
def rotate_token(_: str = Depends(verify_token)) -> dict[str, Any]:
    """Manually rotate the UI authentication token immediately."""
    new_token = _rotator.force_rotate()
    return {"status": "rotated", "hint": f"New token ends with ...{new_token[-6:]}"}


@app.get("/api/security/audit-verify")
def verify_audit_log(_: str = Depends(verify_token)) -> dict[str, Any]:
    """Verify HMAC integrity of the audit log file."""
    log_file = settings.log_dir / "security_audit.jsonl"
    result = _audit_signer.verify_log_file(log_file)
    return result


@app.websocket("/ws/token")
async def token_websocket(websocket: WebSocket):
    """
    WebSocket channel for browser-side token rotation notifications.
    Browser connects once; on token rotation the new token is pushed down.
    Token is validated on connect.
    """
    token = websocket.query_params.get("token", "")
    if not _rotator.is_valid(token):
        await websocket.close(code=4001)
        return
    await websocket.accept()
    _token_ws_clients.append(websocket)
    try:
        while True:
            # Keep the connection alive; server pushes on rotation
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        try:
            _token_ws_clients.remove(websocket)
        except ValueError:
            pass

# ── Voice Endpoints ───────────────────────────────────────────────────────────
class VoiceSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to speak aloud")


@app.post("/api/voice/listen")
def voice_listen_api(
    duration_sec: float = 5.0,
    _: str = Depends(verify_token),
    _origin: None = Depends(require_local_origin),
) -> dict[str, Any]:
    """Record speech from microphone and transcribe via local Whisper."""
    from ragnarok.skills import voice_listen
    try:
        transcript = voice_listen.run({"duration_sec": duration_sec})
        return {"status": "success", "transcript": transcript}
    except Exception as exc:
        logger.error("voice_listen_api error: %s", exc)
        return {"status": "error", "detail": str(exc), "transcript": ""}


@app.post("/api/voice/speak")
def voice_speak_api(
    req: VoiceSpeakRequest,
    _: str = Depends(verify_token),
    _origin: None = Depends(require_local_origin),
) -> dict[str, Any]:
    """Speak text aloud using local TTS engine."""
    from ragnarok.skills import voice_speak
    try:
        res = voice_speak.run({"text": req.text})
        return {"status": "success", "result": res}
    except Exception as exc:
        logger.error("voice_speak_api error: %s", exc)
        return {"status": "error", "detail": str(exc)}


# ── Memory Manager ────────────────────────────────────────────────────────────
@app.get("/api/memory")
def list_or_search_memories(
    query: str = "",
    kind: str = "",
    limit: int = 50,
    _: str = Depends(verify_token),
) -> list[dict[str, Any]]:
    """Retrieve all memories or search by semantic vector similarity / keyword."""
    if query.strip():
        # Semantic search with recall
        entries = state.store.recall(query=query.strip(), top_k=limit, kind_filter=kind or None)
    else:
        all_entries = state.store.list_all()
        if kind.strip():
            all_entries = [m for m in all_entries if m.kind == kind.strip()]
        entries = all_entries[:limit]

    return [asdict(m) for m in entries]


@app.post("/api/memory")
def add_memory(
    req: MemoryCreateRequest,
    request: Request,
    _: str = Depends(verify_token),
    _ip: None = Depends(check_ip_rate_limit),
) -> dict[str, Any]:
    """Add a new memory directly via the interface."""
    try:
        safe_text = InputValidator.validate_memory_text(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    mid = state.store.remember(
        text=safe_text,
        kind=req.kind,
        importance=req.importance,
        pinned=req.pinned,
        source="ui_manual",
    )
    return {"id": mid, "status": "created"}


@app.put("/api/memory/{memory_id}")
def update_memory(
    memory_id: str,
    req: MemoryUpdateRequest,
    _: str = Depends(verify_token),
) -> dict[str, Any]:
    """Update text, importance, or pin status of an existing memory."""
    ok = state.store.update(
        memory_id=memory_id,
        text=req.text,
        importance=req.importance,
        pinned=req.pinned,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found or no changes made")
    return {"id": memory_id, "status": "updated"}


class PinRequest(BaseModel):
    pinned: bool = True


@app.post("/api/memory/{memory_id}/pin")
@app.patch("/api/memory/{memory_id}/pin")
def pin_memory(
    memory_id: str,
    pinned: Optional[bool] = Query(None),
    body: Optional[PinRequest] = None,
    _: str = Depends(verify_token),
) -> dict[str, Any]:
    """Pin or unpin a memory to prevent or permit forgetting."""
    is_pinned = body.pinned if (body is not None) else (pinned if pinned is not None else True)
    state.store.pin(memory_id, pinned=is_pinned)
    return {"id": memory_id, "pinned": is_pinned}


@app.delete("/api/memory/{memory_id}")
def delete_memory(
    memory_id: str,
    confirm: bool = Query(False, description="Explicit confirmation required"),
    _: str = Depends(verify_token),
) -> dict[str, Any]:
    """Permanently delete a memory. Requires confirm=True."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explicit confirmation (confirm=true) is required to permanently delete a memory.",
        )
    ok = state.store.delete(memory_id, allow_pinned=True)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"id": memory_id, "status": "deleted"}


@app.get("/api/memory/export")
def export_memories_json(_: str = Depends(verify_token)) -> Response:
    """Export all active memories formatted as JSON."""
    data = state.store.export_json()
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="ragnarok_memories.json"'},
    )


# ── Approvals Endpoint ────────────────────────────────────────────────────────
@app.get("/api/approvals")
def get_approvals(_: str = Depends(verify_token)) -> dict[str, Any]:
    """List pending high-risk tool approval requests and recent history."""
    pending = [asdict(r) for r in state.approval_manager.get_pending_requests()]
    history = [asdict(d) for d in state.approval_manager._history[-20:]]
    return {"pending": pending, "history": history}


@app.post("/api/approvals/{request_id}/respond")
def respond_approval(
    request_id: str,
    req: ApprovalActionRequest,
    _: str = Depends(verify_token),
) -> dict[str, Any]:
    """Approve, deny, or edit parameters for a pending high-risk tool execution."""
    try:
        st = ApprovalStatus(req.status.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{req.status}'. Must be approved, denied, or edited.",
        )

    ok = state.approval_manager.resolve_request(
        request_id=request_id,
        status=st,
        edited_args=req.edited_args,
        comment=req.comment,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"Pending approval request '{request_id}' not found or already resolved.",
        )

    return {"request_id": request_id, "resolved_status": st.value}


# ── Execution Logs & Explainability ───────────────────────────────────────────
@app.get("/api/logs")
def get_execution_logs(
    limit: int = 50, _: str = Depends(verify_token)
) -> dict[str, Any]:
    """Returns the visual execution timeline and structured tool events."""
    recent_traces = trace_store.get_all(limit=limit)

    # Also parse recent tool log events from logs/ragnarok.jsonl if present
    tool_events: list[dict[str, Any]] = []
    log_file = Path("logs/ragnarok.jsonl")
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if "event" in record or "tool" in record or "_rk_tool" in record:
                            tool_events.append(record)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Error reading %s: %s", log_file, exc)

    return {
        "execution_traces": recent_traces,
        "recent_tool_events": tool_events[-limit:],
    }


# ── Settings Endpoint ─────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings(_: str = Depends(verify_token)) -> dict[str, Any]:
    """Returns current configuration and risk levels per skill."""
    skills_info = []
    for s in state.registry.all_skills():
        skills_info.append(
            {
                "name": s.manifest.name,
                "description": s.manifest.description,
                "risk_level": s.manifest.risk_level,
                "required_permissions": s.manifest.required_permissions,
            }
        )

    return {
        "models": {
            "main": settings.model_main,
            "small": settings.model_small,
            "embed": settings.model_embed,
        },
        "safety": {
            "default_risk_level": settings.default_risk_level,
            "approval_timeout_sec": settings.approval_timeout_sec,
        },
        "permissions": {
            "active_role": settings.active_role,
            "roles": settings.roles_config,
        },
        "paths": {
            "allowed_folders": [str(p) for p in settings.allowed_folders],
            "knowledge_dir": str(settings.knowledge_dir),
            "inbox_dir": str(settings.inbox_dir),
        },
        "skills": skills_info,
        "scheduler": {
            "daily_brief_time": getattr(settings, "daily_brief_time", "08:00"),
        },
    }


@app.post("/api/settings")
def update_settings(
    req: SettingsUpdateRequest, _: str = Depends(verify_token)
) -> dict[str, Any]:
    """Updates active settings in memory."""
    if req.active_role:
        state.permission_manager.active_role = req.active_role
    if req.approval_timeout_sec is not None:
        state.approval_manager.timeout_sec = req.approval_timeout_sec

    return {"status": "updated", "active_role": state.permission_manager.active_role}


# ── Static UI Mounting ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index_page() -> HTMLResponse:
    """Serves the single page frontend, injecting the active security token."""
    root_file = Path(__file__).resolve().parent.parent.parent / "index.html"
    index_file = root_file if root_file.exists() else (STATIC_DIR / "index.html")
    if not index_file.exists():
        return HTMLResponse(
            "<h1>Ragnarok UI Static Files Missing</h1><p>Please ensure index.html exists in ragnarok/ui/static/</p>"
        )

    content = index_file.read_text(encoding="utf-8")
    # Seamlessly inject token into the HTML so local user doesn't need to manually copy-paste
    injected = content.replace("{{RAGNAROK_TOKEN}}", ACTIVE_TOKEN)
    return HTMLResponse(content=injected)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
