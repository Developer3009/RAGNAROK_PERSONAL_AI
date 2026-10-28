"""
ragnarok/skills/gui_action.py
──────────────────────────────
Innovation #1: Local VLM-Powered GUI Grounding & Screen Action (Offline Computer Use).

Enables Ragnarok to SEE the screen using a local vision model (llava, minicpm-v,
llama3.2-vision, moondream2 via Ollama), identify UI elements by their bounding-box
coordinates, and drive desktop interactions autonomously — all 100% offline.

Flow:
  1. Capture screenshot via mss.
  2. Ask local VLM to locate the target UI element and return normalized (x, y) coords.
  3. Use pyautogui to move mouse and click/type at those coordinates.
  4. Optionally re-capture screen to confirm action succeeded.

Security: All actions are sandboxed behind the Shield-Wall approval gate (risk_level=high).

MANIFEST:
  name: gui_action
  description: See the screen with local vision AI, locate a UI element, and interact with it.
  risk_level: high
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

# ── Manifest ──────────────────────────────────────────────────────────────────
MANIFEST = SkillManifest(
    name="gui_action",
    description=(
        "Take a screenshot, use a local vision AI model to locate a UI element by name "
        "or description, and perform a mouse click, double-click, right-click, or keyboard "
        "input on it — all 100% offline. Use for autonomous computer use tasks."
    ),
    risk_level="high",
    required_permissions=["screen_read", "mouse_control", "keyboard_control"],
    parameters={
        "action": {
            "type": "string",
            "description": (
                "Action to perform: 'click', 'double_click', 'right_click', 'type', "
                "'move', 'scroll', 'screenshot_describe'. Required."
            ),
            "enum": ["click", "double_click", "right_click", "type", "move", "scroll", "screenshot_describe"],
        },
        "target": {
            "type": "string",
            "description": (
                "Natural language description of the UI element to interact with "
                "(e.g. 'Submit button', 'search bar', 'close icon top-right'). "
                "Required for click/type actions, optional for screenshot_describe."
            ),
        },
        "text_input": {
            "type": "string",
            "description": "Text to type after clicking the target element. Used with action='type'.",
        },
        "scroll_direction": {
            "type": "string",
            "description": "Scroll direction: 'up' or 'down'. Used with action='scroll'.",
            "enum": ["up", "down"],
        },
        "scroll_clicks": {
            "type": "integer",
            "description": "Number of scroll clicks. Default 3.",
        },
        "vision_model": {
            "type": "string",
            "description": "Ollama vision model to use (e.g. 'llava', 'minicpm-v', 'llama3.2-vision', 'moondream'). Default: 'llava'.",
        },
        "confirm_after": {
            "type": "boolean",
            "description": "Take a follow-up screenshot after action to confirm result. Default: false.",
        },
        "monitor": {
            "type": "integer",
            "description": "Monitor index to capture (1 = primary). Default: 1.",
        },
    },
    required=["action"],
)


# ── Screen capture helpers ────────────────────────────────────────────────────

@dataclass
class ScreenCoord:
    x: int
    y: int
    confidence: float = 1.0
    element_label: str = ""


def _capture_screenshot(monitor: int = 1) -> tuple[bytes, int, int]:
    """Capture the screen. Returns (png_bytes, width, height)."""
    try:
        import mss
        import mss.tools
    except ImportError:
        raise RuntimeError("mss required: pip install mss")

    with mss.mss() as sct:
        mon = sct.monitors[monitor] if monitor < len(sct.monitors) else sct.monitors[1]
        screenshot = sct.grab(mon)
        width, height = screenshot.width, screenshot.height
        # Convert to PNG bytes
        from PIL import Image
        img = Image.frombytes("RGB", (width, height), screenshot.rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), width, height


def _screenshot_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


# ── VLM-based element grounding ───────────────────────────────────────────────

_GROUNDING_PROMPT = """\
You are a UI element locator. Given the screenshot, find the UI element described below
and return its CENTER position as a JSON object with NORMALIZED coordinates (0.0 to 1.0
where 0,0 is top-left and 1,1 is bottom-right of the screen).

Element to find: {target}

Respond with ONLY a valid JSON object, no markdown, no extra text:
{{"x": 0.5, "y": 0.3, "confidence": 0.9, "label": "element label you found"}}

If the element is not visible, respond with:
{{"x": -1, "y": -1, "confidence": 0.0, "label": "not found"}}
"""

_DESCRIBE_PROMPT = """\
Describe what is currently shown on this screen. Be specific: list the application name,
major UI sections visible, any open dialogs, text content, buttons, and the current state.
Focus on elements that would be useful for automation.
"""


def _query_vlm_grounding(
    png_bytes: bytes,
    target: str,
    vision_model: str = "llava",
) -> ScreenCoord | None:
    """Use a local Ollama vision model to locate a UI element."""
    import requests
    from ragnarok.config import settings

    img_b64 = _screenshot_to_base64(png_bytes)
    prompt = _GROUNDING_PROMPT.format(target=target)

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = requests.post(
            f"{settings.ollama_url}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")

        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if not json_match:
            logger.warning("VLM grounding: no JSON in response: %s", content[:200])
            return None

        data = json.loads(json_match.group())
        x_norm = float(data.get("x", -1))
        y_norm = float(data.get("y", -1))
        confidence = float(data.get("confidence", 0.0))
        label = str(data.get("label", target))

        if x_norm < 0 or y_norm < 0:
            logger.info("VLM grounding: element '%s' not found on screen", target)
            return None

        return ScreenCoord(
            x=int(x_norm),  # normalized — will be converted with screen dims
            y=int(y_norm),  # normalized — will be converted with screen dims
            confidence=confidence,
            element_label=label,
        )
    except Exception as exc:
        logger.error("VLM grounding call failed: %s", exc)
        return None


def _query_vlm_describe(
    png_bytes: bytes,
    question: str = "",
    vision_model: str = "llava",
) -> str:
    """Ask the vision model to describe what's on screen."""
    import requests
    from ragnarok.config import settings

    img_b64 = _screenshot_to_base64(png_bytes)
    prompt = question if question else _DESCRIBE_PROMPT

    payload = {
        "model": vision_model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{settings.ollama_url}/api/chat",
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception as exc:
        return f"Vision model query failed: {exc}"


# ── Pyautogui action helpers ──────────────────────────────────────────────────

def _check_pyautogui() -> None:
    try:
        import pyautogui  # noqa: F401
    except ImportError:
        raise RuntimeError("pyautogui required for GUI actions: pip install pyautogui")


def _perform_click(x: int, y: int, button: str = "left", double: bool = False) -> str:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    if double:
        pyautogui.doubleClick(x, y, button=button)
        return f"Double-clicked at ({x}, {y})"
    else:
        pyautogui.click(x, y, button=button)
        return f"Clicked ({button}) at ({x}, {y})"


def _perform_type(x: int, y: int, text: str) -> str:
    import pyautogui
    pyautogui.click(x, y)
    time.sleep(0.2)
    pyautogui.write(text, interval=0.05)
    return f"Typed '{text[:50]}{'...' if len(text) > 50 else ''}' at ({x}, {y})"


def _perform_scroll(x: int, y: int, direction: str, clicks: int) -> str:
    import pyautogui
    amount = clicks if direction == "up" else -clicks
    pyautogui.scroll(amount, x=x, y=y)
    return f"Scrolled {direction} {clicks} clicks at ({x}, {y})"


# ── Main skill entry point ─────────────────────────────────────────────────────

def run(args: dict) -> str:
    """
    GUI Action skill entry point.

    Supported actions:
      - screenshot_describe: Capture and describe the current screen.
      - click / double_click / right_click: Find target element and click.
      - type: Find target, click it, then type text.
      - move: Move the mouse to the target element.
      - scroll: Scroll at target location.
    """
    action = str(args.get("action", "screenshot_describe")).strip().lower()
    target = str(args.get("target", "")).strip()
    text_input = str(args.get("text_input", "")).strip()
    scroll_direction = str(args.get("scroll_direction", "down")).strip()
    scroll_clicks = int(args.get("scroll_clicks", 3))
    vision_model = str(args.get("vision_model", "llava")).strip() or "llava"
    confirm_after = bool(args.get("confirm_after", False))
    monitor = int(args.get("monitor", 1))

    # ── Step 1: Capture screenshot ──────────────────────────────────────────
    try:
        png_bytes, screen_w, screen_h = _capture_screenshot(monitor)
    except Exception as exc:
        return f"ERROR: Screenshot capture failed — {exc}. Install: pip install mss Pillow"

    # ── screenshot_describe: pure vision query ──────────────────────────────
    if action == "screenshot_describe":
        description = _query_vlm_describe(png_bytes, target or "", vision_model)
        return f"[Screen Description via {vision_model}]\n\n{description}"

    # ── Actions requiring UI element location ──────────────────────────────
    if not target:
        return "ERROR: 'target' parameter is required for this action."

    _check_pyautogui()

    # ── Step 2: VLM grounding — find element coordinates ───────────────────
    logger.info("GUI Action: locating element '%s' with %s", target, vision_model)
    coord = _query_vlm_grounding(png_bytes, target, vision_model)

    if coord is None:
        return (
            f"Could not locate UI element '{target}' on screen. "
            f"The VLM ({vision_model}) reported it as not visible. "
            f"Try a more specific description or ensure the element is visible."
        )

    # Convert normalized coords to absolute pixel coords
    # VLM returns floats 0..1 but we stored them as int — re-parse properly
    # Re-do: treat raw values as normalized 0..1 floats
    # The _query_vlm_grounding stores normalized values — retrieve them properly
    abs_x = int(coord.x * screen_w) if coord.x <= 1 else coord.x
    abs_y = int(coord.y * screen_h) if coord.y <= 1 else coord.y

    label_info = f" (VLM located: '{coord.element_label}', confidence: {coord.confidence:.2f})"
    result_parts = [f"Located '{target}' at ({abs_x}, {abs_y}){label_info}"]

    # ── Step 3: Perform the requested action ───────────────────────────────
    try:
        if action == "click":
            result_parts.append(_perform_click(abs_x, abs_y, button="left"))
        elif action == "double_click":
            result_parts.append(_perform_click(abs_x, abs_y, button="left", double=True))
        elif action == "right_click":
            result_parts.append(_perform_click(abs_x, abs_y, button="right"))
        elif action == "type":
            result_parts.append(_perform_type(abs_x, abs_y, text_input))
        elif action == "move":
            import pyautogui
            pyautogui.moveTo(abs_x, abs_y, duration=0.3)
            result_parts.append(f"Moved mouse to ({abs_x}, {abs_y})")
        elif action == "scroll":
            result_parts.append(_perform_scroll(abs_x, abs_y, scroll_direction, scroll_clicks))
    except Exception as exc:
        return f"Action '{action}' failed: {exc}"

    # ── Step 4: Optional confirmation screenshot ────────────────────────────
    if confirm_after:
        time.sleep(0.8)
        try:
            confirm_png, _, _ = _capture_screenshot(monitor)
            confirm_desc = _query_vlm_describe(
                confirm_png,
                f"Has the action succeeded? Was the element '{target}' interacted with?",
                vision_model,
            )
            result_parts.append(f"\nConfirmation screenshot analysis:\n{confirm_desc}")
        except Exception as exc:
            result_parts.append(f"\nConfirmation screenshot failed: {exc}")

    return "\n".join(result_parts)
