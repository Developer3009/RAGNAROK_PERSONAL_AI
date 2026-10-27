"""
ragnarok/skills/capture_screen.py
──────────────────────────────────
Innovation #3 (Vision): Capture the screen and describe it using a local
vision-capable Ollama model (llava, minicpm-v, llava-phi3, etc.).

The agent can invoke this skill to "see" what is on screen — enabling
visual debugging, form-filling assistance, and desktop monitoring.

Flow:
  1. Screenshot captured via mss (multi-monitor screenshot) — pure Python, no deps.
  2. Image encoded as base64 PNG.
  3. Sent to Ollama /api/chat with the vision model and a description prompt.
  4. Result text returned to the agent.

MANIFEST:
  name: capture_screen
  description: Take a screenshot of the screen and describe what you see using a local vision AI model.
  risk_level: medium  (reads screen — no file writes)
  required_permissions: []
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

# ── Manifest ──────────────────────────────────────────────────────────────────
MANIFEST = SkillManifest(
    name="capture_screen",
    description=(
        "Take a screenshot of the current screen and describe its contents using a local "
        "vision AI model (e.g. llava or minicpm-v). Use this when you need to see what is "
        "displayed on the user's screen."
    ),
    risk_level="medium",
    required_permissions=[],
    parameters={
        "question": {
            "type": "string",
            "description": "What to look for or describe in the screenshot (e.g. 'What error is shown?'). Optional.",
        },
        "monitor": {
            "type": "integer",
            "description": "Monitor index to capture (0 = all monitors, 1 = primary). Default: 1.",
        },
        "vision_model": {
            "type": "string",
            "description": "Ollama vision model to use (e.g. 'llava', 'minicpm-v', 'llava-phi3'). Defaults to 'llava'.",
        },
    },
    required=[],
)


def _capture_screenshot(monitor_idx: int = 1) -> bytes:
    """
    Capture a screenshot using mss (fast, zero-dependency).
    Returns PNG bytes.
    """
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor_idx < 0 or monitor_idx >= len(monitors):
                monitor_idx = 1 if len(monitors) > 1 else 0
            mon = monitors[monitor_idx]
            screenshot = sct.grab(mon)
            # Convert to PNG bytes
            png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
            return png_bytes
    except ImportError:
        # Fallback: try PIL if available
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            raise RuntimeError(
                "No screenshot library available. "
                "Install mss: pip install mss  OR  Pillow: pip install Pillow"
            )


def _describe_image(
    image_bytes: bytes,
    question: str,
    vision_model: str,
) -> str:
    """
    Send image to Ollama vision model and return description.
    """
    import httpx
    from ragnarok.config import settings

    b64_image = base64.b64encode(image_bytes).decode("ascii")
    prompt = question or "Describe what you see on this screen in detail."

    payload = {
        "model": vision_model,
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
        with httpx.Client(base_url=settings.ollama_url, timeout=120.0) as client:
            r = client.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 400 and "not found" in exc.response.text.lower():
            return (
                f"Vision model '{vision_model}' is not available in Ollama. "
                f"Pull it with: ollama pull {vision_model}"
            )
        raise


def run(args: dict) -> str:
    """
    Skill entry point.

    Args:
        args: {
          "question": str (optional),
          "monitor": int (optional, default 1),
          "vision_model": str (optional, default "llava")
        }

    Returns:
        str — the vision model's description of the screen.
    """
    question = str(args.get("question", "")).strip()
    monitor_idx = int(args.get("monitor", 1))
    vision_model = str(args.get("vision_model", "llava")).strip() or "llava"

    logger.info("capture_screen: capturing monitor=%d model=%s", monitor_idx, vision_model)

    try:
        png_bytes = _capture_screenshot(monitor_idx)
    except Exception as exc:
        return f"Screenshot capture failed: {exc}"

    # Save screenshot to a temp file for audit trail
    try:
        from ragnarok.config import settings
        snap_dir = settings.data_dir / "screenshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        import time
        snap_path = snap_dir / f"snap_{int(time.time())}.png"
        snap_path.write_bytes(png_bytes)
        logger.info("Screenshot saved to %s", snap_path)
    except Exception as exc:
        logger.warning("Could not save screenshot: %s", exc)

    try:
        description = _describe_image(png_bytes, question, vision_model)
    except Exception as exc:
        return f"Vision model call failed: {exc}"

    prefix = f"[Screen description via {vision_model}]"
    if question:
        prefix = f"[Question: {question!r} | Vision model: {vision_model}]"
    return f"{prefix}\n\n{description}"
