"""
ragnarok/memory/consolidation.py
──────────────────────────────────
Memory consolidation: groups related episodic memories, summarises them
into one semantic memory, then archives the originals.

Runs as a scheduled job (Phase 6). Can also be triggered manually.

Forgetting: scores each non-pinned memory by recency, frequency, and
importance; archives/deletes low-scorers below the config threshold.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ragnarok import llm
from ragnarok.config import settings

if TYPE_CHECKING:
    from ragnarok.memory.db import MemoryDB
    from ragnarok.memory.recall import MemoryStore

logger = logging.getLogger(__name__)

# Minimum number of episodic memories before consolidation runs
_MIN_EPISODICS_TO_CONSOLIDATE = 5
# Consolidation batch size
_BATCH_SIZE = 10


# ── Scoring ────────────────────────────────────────────────────────────────────

def _forget_score(row, now: float) -> float:
    """
    Compute an importance-weighted score for forgetting.

    Score = 0.4 * recency + 0.4 * importance + 0.2 * frequency
    All components are normalised to [0, 1].
    Low score → candidate for forgetting.
    """
    age_days = (now - float(row["created_at"])) / 86_400
    recency = max(0.0, 1.0 - age_days / 30.0)      # zero at 30 days old

    importance = min(1.0, max(0.0, float(row["importance"])))

    # Frequency: use_count relative to a soft cap of 20
    freq = min(1.0, int(row["use_count"]) / 20.0)

    return 0.4 * recency + 0.4 * importance + 0.2 * freq


# ── Consolidation ─────────────────────────────────────────────────────────────

def consolidate(memory_store: "MemoryStore") -> int:
    """
    Group batches of episodic memories and summarise each into one semantic
    memory. Marks originals as superseded by the new summary.

    Returns:
        Number of consolidations performed.
    """
    db = memory_store._db
    rows = db.get_episodic_for_consolidation()

    if len(rows) < _MIN_EPISODICS_TO_CONSOLIDATE:
        logger.info(
            "Consolidation skipped: only %d episodic memories (need %d)",
            len(rows), _MIN_EPISODICS_TO_CONSOLIDATE,
        )
        return 0

    count = 0
    # Process in batches
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i : i + _BATCH_SIZE]
        if len(batch) < 3:
            break  # too small to consolidate

        texts = "\n".join(f"- {r['text']}" for r in batch)
        prompt = (
            f"Summarise these {len(batch)} related memories into ONE concise fact "
            f"(one sentence, max 120 chars):\n{texts}\n\nSummary:"
        )

        try:
            result = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=settings.model_small,
                temperature=0.1,
            )
            summary = result.content.strip()
        except Exception as exc:
            logger.warning("Consolidation LLM call failed: %s", exc)
            continue

        if not summary or len(summary) > 500:
            continue

        # Save the consolidated semantic memory
        try:
            new_id = memory_store.remember(
                text=summary,
                kind="semantic",
                source=f"consolidation_of_{len(batch)}_episodics",
                importance=0.7,
            )
        except Exception as exc:
            logger.warning("Could not save consolidated memory: %s", exc)
            continue

        # Mark originals as superseded
        for row in batch:
            db.mark_superseded(row["id"], new_id)

        count += 1
        logger.info(
            "Consolidated %d episodic memories → semantic %s: %r",
            len(batch), new_id, summary[:60],
        )

    return count


# ── Forgetting ────────────────────────────────────────────────────────────────

def forget(memory_store: "MemoryStore", dry_run: bool = False) -> list[str]:
    """
    Score all non-pinned, non-superseded memories and delete those below
    the forget_threshold from config.

    Pinned memories are NEVER deleted.

    Args:
        dry_run: If True, return candidates without deleting.

    Returns:
        List of deleted (or candidate) memory IDs.
    """
    db = memory_store._db
    threshold = settings.forget_threshold
    now = time.time()

    candidates = db.get_candidates_for_forgetting(threshold)
    to_delete: list[str] = []

    for row in candidates:
        score = _forget_score(row, now)
        if score < threshold:
            to_delete.append(row["id"])
            logger.info(
                "Forget candidate: %s score=%.3f text=%r",
                row["id"], score, row["text"][:60],
            )

    if not dry_run and to_delete:
        db.delete_many(to_delete)
        logger.info("Forgot %d memories (score < %.2f)", len(to_delete), threshold)

    return to_delete
