"""
ragnarok/skills/voice_speak.py
───────────────────────────────
Innovation #3 (Audio Out): Convert text to speech using Piper TTS — a
fast, offline, neural text-to-speech system that runs entirely on CPU.

Piper is developed by Rhasspy and produces high-quality, natural-sounding
voices from ONNX models. No internet connection required.

Flow:
  1. Call `piper` CLI subprocess (must be installed and on PATH).
  2. Pass text via stdin, receive WAV audio via stdout.
  3. Play the WAV through the default audio output device using sounddevice.

Alternatively, if `piper` CLI is not available, falls back to
`pyttsx3` (Windows/Linux offline TTS) or raises a clear error.

MANIFEST:
  name: voice_speak
  description: Speak text aloud using a local offline text-to-speech engine (Piper TTS or pyttsx3).
  risk_level: low
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

MANIFEST = SkillManifest(
    name="voice_speak",
    description=(
        "Convert text to speech and play it aloud using a local offline TTS engine "
        "(Piper TTS preferred, pyttsx3 fallback). No internet connection is required."
    ),
    risk_level="low",
    required_permissions=[],
    parameters={
        "text": {
            "type": "string",
            "description": "The text to speak aloud.",
        },
        "voice_model": {
            "type": "string",
            "description": "Path to a Piper ONNX voice model file (.onnx). Leave empty to use pyttsx3 fallback.",
        },
        "rate": {
            "type": "number",
            "description": "Speech rate multiplier for pyttsx3 (default: 1.0). Range: 0.5–2.0.",
        },
    },
    required=["text"],
)

# Max characters to prevent runaway TTS
_MAX_TEXT_LEN = 2000


def _speak_piper(text: str, voice_model: str) -> str:
    """Synthesize speech via Piper TTS CLI and play it."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise RuntimeError("sounddevice and soundfile required: pip install sounddevice soundfile")

    # Write to a temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = ["piper", "--model", voice_model, "--output_file", tmp_path]
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Piper exited {result.returncode}: {err}")

        # Play audio
        data, sr = sf.read(tmp_path, dtype="float32")
        sd.play(data, samplerate=sr)
        sd.wait()
        return f"[Spoken via Piper TTS model: {Path(voice_model).name}]"
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _speak_pyttsx3(text: str, rate_mult: float) -> str:
    """Fallback: speak via pyttsx3 (Windows SAPI / eSpeak on Linux)."""
    try:
        import pyttsx3
    except ImportError:
        raise RuntimeError("pyttsx3 not installed: pip install pyttsx3")

    engine = pyttsx3.init()
    default_rate = engine.getProperty("rate")
    engine.setProperty("rate", int(default_rate * rate_mult))
    engine.say(text)
    engine.runAndWait()
    return "[Spoken via pyttsx3 offline TTS]"


def _check_piper_available(voice_model: str) -> bool:
    """Return True if `piper` is on PATH and the model file exists."""
    import shutil
    if not shutil.which("piper"):
        return False
    return bool(voice_model) and Path(voice_model).exists()


def run(args: dict) -> str:
    """
    Skill entry point.

    Args:
        args: {
          "text": str (required),
          "voice_model": str (path to .onnx, optional),
          "rate": float (1.0 default, pyttsx3 only)
        }

    Returns:
        Status string confirming speech was played.
    """
    text = str(args.get("text", "")).strip()
    if not text:
        return "ERROR: 'text' parameter is required."
    if len(text) > _MAX_TEXT_LEN:
        text = text[:_MAX_TEXT_LEN] + "..."
        logger.warning("voice_speak: text truncated to %d chars", _MAX_TEXT_LEN)

    voice_model = str(args.get("voice_model", "")).strip()
    rate_mult = float(args.get("rate", 1.0))
    rate_mult = max(0.5, min(2.0, rate_mult))

    # Try Piper if model path provided
    if _check_piper_available(voice_model):
        try:
            return _speak_piper(text, voice_model)
        except Exception as exc:
            logger.warning("Piper TTS failed (%s); falling back to pyttsx3", exc)

    # Fallback to pyttsx3
    try:
        return _speak_pyttsx3(text, rate_mult)
    except RuntimeError as exc:
        return (
            f"TTS unavailable: {exc}\n\n"
            "To enable voice output, install ONE of:\n"
            "  1. Piper TTS: https://github.com/rhasspy/piper  +  pip install sounddevice soundfile\n"
            "  2. pyttsx3: pip install pyttsx3"
        )
