"""
ragnarok/config.py
──────────────────
Loads config.toml from the project root and exposes a typed Settings object.
All model names, paths, and limits are read from there — never hardcoded.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from functools import lru_cache
from typing import Any

# Resolve config.toml relative to this file's grandparent (project root)
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    """Read and parse config.toml once, then cache it."""
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


class Settings:
    """Typed access to config.toml sections."""

    def __init__(self) -> None:
        self._raw = _load_raw()

    # ── Ollama ─────────────────────────────────────────────────────────────────
    @property
    def ollama_url(self) -> str:
        return self._raw["ollama"]["base_url"]

    @property
    def ollama_timeout(self) -> float:
        return float(self._raw["ollama"]["timeout_seconds"])

    @property
    def ollama_retries(self) -> int:
        return int(self._raw["ollama"]["retry_attempts"])

    @property
    def ollama_retry_delay(self) -> float:
        return float(self._raw["ollama"]["retry_delay_sec"])

    # ── Models ─────────────────────────────────────────────────────────────────
    @property
    def model_small(self) -> str:
        return self._raw["models"]["chat_small"]

    @property
    def model_main(self) -> str:
        return self._raw["models"]["chat_main"]

    @property
    def model_embed(self) -> str:
        return self._raw["models"]["embed"]

    # ── Agent ──────────────────────────────────────────────────────────────────
    @property
    def max_steps(self) -> int:
        return int(self._raw["agent"]["max_steps"])

    @property
    def history_turns(self) -> int:
        return int(self._raw["agent"]["history_turns"])

    @property
    def memory_budget_chars(self) -> int:
        return int(self._raw["agent"]["memory_budget_chars"])

    @property
    def log_dir(self) -> Path:
        return _PROJECT_ROOT / self._raw["agent"]["log_dir"]

    # ── Paths ──────────────────────────────────────────────────────────────────
    @property
    def db_dir(self) -> Path:
        return _PROJECT_ROOT / self._raw["paths"]["db_dir"]

    @property
    def inbox_dir(self) -> Path:
        return _PROJECT_ROOT / self._raw["paths"]["data_inbox"]

    @property
    def processed_dir(self) -> Path:
        return _PROJECT_ROOT / self._raw["paths"]["data_processed"]

    @property
    def knowledge_dir(self) -> Path:
        return _PROJECT_ROOT / self._raw["paths"]["data_knowledge"]

    @property
    def allowed_folders(self) -> list[Path]:
        return [
            _PROJECT_ROOT / f
            for f in self._raw["paths"]["allowed_folders"]
        ]

    # ── Memory ─────────────────────────────────────────────────────────────────
    @property
    def recall_top_k(self) -> int:
        return int(self._raw["memory"]["recall_top_k"])

    @property
    def recency_boost_weight(self) -> float:
        return float(self._raw["memory"]["recency_boost_weight"])

    @property
    def forget_threshold(self) -> float:
        return float(self._raw["memory"]["forget_threshold"])

    @property
    def consolidation_interval_sec(self) -> int:
        return int(self._raw["memory"]["consolidation_interval_sec"])

    # ── Metrics ────────────────────────────────────────────────────────────────
    @property
    def max_recall_latency_ms(self) -> float:
        return float(self._raw["metrics"]["max_recall_latency_ms"])


# Convenience singleton
settings = Settings()
