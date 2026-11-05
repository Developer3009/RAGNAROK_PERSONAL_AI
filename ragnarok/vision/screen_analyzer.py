"""
ragnarok/vision/screen_analyzer.py
───────────────────────────────────
Multimodal vision processing for laptop screen sharing and desktop inspection.

Provides:
  - analyze_screen_frame(image_bytes, question, vision_model) -> dict[str, Any]
  - get_screen_metadata(image_bytes) -> dict[str, Any]
  - Fallback diagnostic when a local vision model is not yet pulled into Ollama.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

import httpx

from ragnarok.config import settings

logger = logging.getLogger(__name__)

# Preferred vision models in priority order
RECOMMENDED_VISION_MODELS = ["llava", "llama3.2-vision", "minicpm-v", "llava-phi3"]


def get_screen_metadata(image_bytes: bytes) -> dict[str, Any]:
    """Extract lightweight image metrics (dimensions, format, size) using PIL."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format or "PNG",
                "mode": img.mode,
                "size_bytes": len(image_bytes),
            }
    except Exception as exc:
        logger.debug("PIL inspection error: %s", exc)
        return {
            "width": 0,
            "height": 0,
            "format": "UNKNOWN",
            "mode": "RGB",
            "size_bytes": len(image_bytes),
        }


def analyze_screen_frame(
    image_bytes: bytes,
    question: str = "",
    vision_model: str = "llava",
) -> dict[str, Any]:
    """
    Send a screen frame (PNG/JPEG bytes) to Ollama's multimodal vision API.

    Returns:
        {
            "answer": str,
            "model": str,
            "latency_ms": float,
            "metadata": dict,
            "status": "success" | "model_missing" | "error"
        }
    """
    t0 = time.perf_counter()
    prompt = question.strip() or "Describe what you see on this shared laptop screen in detail. If there is code, text, or errors, summarize them clearly."
    model = vision_model.strip() or "llava"

    meta = get_screen_metadata(image_bytes)
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        with httpx.Client(base_url=settings.ollama_url, timeout=90.0) as client:
            r = client.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            answer = data.get("message", {}).get("content", "").strip()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "answer": answer,
                "model": data.get("model", model),
                "latency_ms": round(latency_ms, 1),
                "metadata": meta,
                "status": "success",
            }
    except httpx.HTTPStatusError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        err_text = exc.response.text.lower()
        if exc.response.status_code == 400 and ("not found" in err_text or "does not exist" in err_text or "try pulling" in err_text):
            # Model not pulled in Ollama
            missing_msg = (
                f"Vision model '{model}' is not currently pulled in local Ollama.\n\n"
                f"To enable full AI screen vision, open your terminal and run:\n"
                f"  ollama pull {model}\n\n"
                f"Screen Frame Captured: {meta.get('width')}x{meta.get('height')} px ({meta.get('size_bytes', 0):,} bytes)."
            )
            return {
                "answer": missing_msg,
                "model": model,
                "latency_ms": round(latency_ms, 1),
                "metadata": meta,
                "status": "model_missing",
            }
        logger.error("Ollama vision call failed with HTTP %d: %s", exc.response.status_code, exc.response.text)
        return {
            "answer": f"Vision analysis failed (HTTP {exc.response.status_code}): {exc.response.text}",
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "metadata": meta,
            "status": "error",
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Vision call exception: %s", exc)
        return {
            "answer": f"Unable to reach local Ollama vision endpoint: {exc}",
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "metadata": meta,
            "status": "error",
        }
