"""
ragnarok/skills/voice_listen.py
────────────────────────────────
Innovation #3 (Audio In): Transcribe live microphone input using OpenAI Whisper
running entirely locally.  No cloud API calls — all inference happens on device.

Flow:
  1. Record audio for `duration_sec` seconds via sounddevice.
  2. Pass audio buffer to faster-whisper (CTranslate2 backend) for transcription.
  3. Return the transcript string to the agent.

Fallback: if faster-whisper is not installed, attempts standard `whisper` package.

MANIFEST:
  name: voice_listen
  description: Record speech from the microphone and transcribe it to text using local Whisper.
  risk_level: medium  (accesses microphone)
"""

from __future__ import annotations

import logging

from ragnarok.skills.registry import SkillManifest

logger = logging.getLogger(__name__)

MANIFEST = SkillManifest(
    name="voice_listen",
    description=(
        "Record speech from the microphone for a specified duration and transcribe it to text "
        "using the local Whisper speech recognition model. Returns the spoken text."
    ),
    risk_level="medium",
    required_permissions=[],
    parameters={
        "duration_sec": {
            "type": "number",
            "description": "How many seconds of audio to record (default: 5, max: 60).",
        },
        "model_size": {
            "type": "string",
            "description": "Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'. Default: 'base'.",
        },
        "language": {
            "type": "string",
            "description": "Expected language code, e.g. 'en', 'hi', 'fr'. Empty = auto-detect.",
        },
    },
    required=[],
)


def _record_audio(duration_sec: float, sample_rate: int = 16000) -> "np.ndarray":
    """Record microphone audio for `duration_sec` seconds."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "Microphone input requires sounddevice and numpy: pip install sounddevice numpy"
        )

    logger.info("Recording audio for %.1fs at %dHz...", duration_sec, sample_rate)
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    import numpy as np
    return audio.flatten()


def _transcribe_faster_whisper(
    audio: "np.ndarray",
    sample_rate: int,
    model_size: str,
    language: str | None,
) -> str:
    """Transcribe using faster-whisper (preferred — CTranslate2, CPU-friendly)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    import numpy as np
    segments, info = model.transcribe(
        audio.astype(np.float32),
        language=language or None,
        beam_size=5,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    logger.info("faster-whisper transcript: %r (lang=%s)", text[:80], info.language)
    return text


def _transcribe_openai_whisper(
    audio: "np.ndarray",
    sample_rate: int,
    model_size: str,
    language: str | None,
) -> str:
    """Transcribe using openai-whisper (fallback)."""
    import tempfile
    import numpy as np
    import whisper
    import soundfile as sf

    model = whisper.load_model(model_size)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio.astype(np.float32), sample_rate)
        tmp_path = f.name

    opts: dict = {}
    if language:
        opts["language"] = language

    result = model.transcribe(tmp_path, **opts)
    import os
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return result.get("text", "").strip()


def run(args: dict) -> str:
    """
    Skill entry point.

    Args:
        args: {
          "duration_sec": float (default 5, max 60),
          "model_size": str (default "base"),
          "language": str (default "" = auto)
        }

    Returns:
        Transcript string.
    """
    duration_sec = min(float(args.get("duration_sec", 5)), 60.0)
    model_size = str(args.get("model_size", "base")).strip() or "base"
    language = str(args.get("language", "")).strip() or None
    sample_rate = 16_000

    try:
        audio = _record_audio(duration_sec, sample_rate)
    except Exception as exc:
        return f"Audio recording failed: {exc}"

    # Try faster-whisper first, fall back to openai-whisper
    try:
        return _transcribe_faster_whisper(audio, sample_rate, model_size, language)
    except ImportError:
        logger.info("faster-whisper not installed, trying openai-whisper")
    except Exception as exc:
        logger.warning("faster-whisper failed: %s — trying openai-whisper", exc)

    try:
        return _transcribe_openai_whisper(audio, sample_rate, model_size, language)
    except ImportError:
        return (
            "No Whisper backend found. Install one of:\n"
            "  pip install faster-whisper\n"
            "  pip install openai-whisper soundfile\n"
            "  pip install sounddevice"
        )
    except Exception as exc:
        return f"Transcription failed: {exc}"
