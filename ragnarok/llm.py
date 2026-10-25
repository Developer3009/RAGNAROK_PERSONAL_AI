"""
ragnarok/llm.py
───────────────
Central LLM interface for Ragnarok.  All Ollama calls go through here.

Public API:
    chat(messages, model, temperature) → str
    embed(text, model) → list[float]
    chat_with_tools(messages, tools, model) → ChatToolResult
    stream_chat(messages, model) → Generator[str, None, None]

Features:
    - Retry with exponential back-off on transient errors
    - Timeout enforcement (from config.toml)
    - Per-call structured logging (latency, model, token count)
    - Tool-call JSON validation before returning
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx

from ragnarok.config import settings

logger = logging.getLogger(__name__)


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class ChatResult:
    content: str
    model: str
    latency_ms: float
    token_count: int


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatToolResult:
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_content: str | None = None
    model: str = ""
    latency_ms: float = 0.0
    token_count: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_client() -> httpx.Client:
    return httpx.Client(
        base_url=settings.ollama_url,
        timeout=settings.ollama_timeout,
    )


def _retry(fn, *, attempts: int, delay: float):
    """Call fn(), retrying up to `attempts` times on httpx errors."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except (httpx.HTTPError, httpx.ConnectError) as exc:
            last_err = exc
            if i < attempts - 1:
                wait = delay * (2 ** i)
                logger.warning("Ollama call failed (%s); retry %d in %.1fs", exc, i + 1, wait)
                time.sleep(wait)
    raise RuntimeError(f"Ollama unreachable after {attempts} attempts") from last_err


# ── Public API ────────────────────────────────────────────────────────────────

def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    system: str | None = None,
) -> ChatResult:
    """
    Send a multi-turn conversation to Ollama and return the assistant reply.

    Args:
        messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
        model:    Override model from config (defaults to chat_main).
        temperature: Sampling temperature.
        system:   Optional system prompt prepended to messages.

    Returns:
        ChatResult with content, model, latency_ms, token_count.
    """
    _model = model or settings.model_main
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload = {
        "model": _model,
        "messages": full_messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    t0 = time.perf_counter()

    def _call():
        with _make_client() as client:
            r = client.post("/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    data = _retry(_call, attempts=settings.ollama_retries, delay=settings.ollama_retry_delay)
    latency_ms = (time.perf_counter() - t0) * 1000

    result = ChatResult(
        content=data["message"]["content"],
        model=data.get("model", _model),
        latency_ms=latency_ms,
        token_count=data.get("eval_count", 0),
    )
    logger.info(
        "chat model=%s tokens=%d latency_ms=%.1f",
        result.model, result.token_count, result.latency_ms,
    )
    return result


_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}

def embed(text: str, model: str | None = None) -> list[float]:
    """
    Return an embedding vector for `text`.

    Args:
        text:  Input text to embed.
        model: Override embedding model (defaults to config embed model).

    Returns:
        List[float] embedding vector.
    """
    _model = model or settings.model_embed
    cache_key = (_model, text)
    if cache_key in _EMBED_CACHE:
        return _EMBED_CACHE[cache_key]

    t0 = time.perf_counter()

    def _call():
        with _make_client() as client:
            r = client.post("/api/embed", json={"model": _model, "input": text})
            r.raise_for_status()
            return r.json()

    data = _retry(_call, attempts=settings.ollama_retries, delay=settings.ollama_retry_delay)
    latency_ms = (time.perf_counter() - t0) * 1000

    vector: list[float] = data["embeddings"][0]
    _EMBED_CACHE[cache_key] = vector
    logger.info(
        "embed model=%s dims=%d latency_ms=%.1f",
        _model, len(vector), latency_ms,
    )
    return vector



def chat_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.0,
    system: str | None = None,
) -> ChatToolResult:
    """
    Chat with tool definitions.  Parses the model's tool_call response.

    Args:
        messages: Conversation history.
        tools:    List of tool schema dicts (OpenAI function-call format).
        model:    Model override.
        temperature: Lower = more deterministic tool selection.
        system:   Optional system prompt.

    Returns:
        ChatToolResult — check .has_tool_calls to decide next step.
    """
    _model = model or settings.model_main
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload = {
        "model": _model,
        "messages": full_messages,
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature},
    }

    t0 = time.perf_counter()

    def _call():
        with _make_client() as client:
            r = client.post("/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    data = _retry(_call, attempts=settings.ollama_retries, delay=settings.ollama_retry_delay)
    latency_ms = (time.perf_counter() - t0) * 1000

    msg = data.get("message", {})
    tool_calls_raw = msg.get("tool_calls", [])
    final_content = msg.get("content") or None

    parsed_calls: list[ToolCall] = []
    for tc in tool_calls_raw:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", {})
        # arguments may be a string (JSON) or already a dict
        if isinstance(args_raw, str):
            try:
                args_raw = json.loads(args_raw)
            except json.JSONDecodeError:
                args_raw = {}
        parsed_calls.append(ToolCall(name=name, arguments=args_raw))

    result = ChatToolResult(
        tool_calls=parsed_calls,
        final_content=final_content,
        model=data.get("model", _model),
        latency_ms=latency_ms,
        token_count=data.get("eval_count", 0),
    )
    logger.info(
        "chat_with_tools model=%s tool_calls=%d latency_ms=%.1f",
        result.model, len(result.tool_calls), result.latency_ms,
    )
    return result


def stream_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    system: str | None = None,
) -> Generator[str, None, None]:
    """
    Yield response tokens as they arrive (streaming mode).

    Args:
        messages:    Conversation history.
        model:       Model override.
        temperature: Sampling temperature.
        system:      Optional system prompt.

    Yields:
        str — individual token strings.
    """
    _model = model or settings.model_main
    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    payload = {
        "model": _model,
        "messages": full_messages,
        "stream": True,
        "options": {"temperature": temperature},
    }

    with httpx.Client(base_url=settings.ollama_url, timeout=settings.ollama_timeout) as client:
        with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if not chunk.get("done"):
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token


def is_ollama_alive() -> bool:
    """Quick health-check for Ollama server."""
    try:
        with _make_client() as client:
            r = client.get("/")
            return r.status_code == 200
    except httpx.ConnectError:
        return False


def list_models() -> list[str]:
    """Return the names of all locally-pulled models."""
    with _make_client() as client:
        r = client.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
