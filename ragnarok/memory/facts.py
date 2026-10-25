"""
ragnarok/memory/facts.py
─────────────────────────
Fact extraction: after each conversation turn, ask the local model to
identify lasting facts and preferences as a JSON list, then save them
as semantic memories.

Only facts that the USER explicitly stated are saved.
Every extracted fact records the source message for auditability.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ragnarok import llm
from ragnarok.config import settings

if TYPE_CHECKING:
    from ragnarok.memory.recall import MemoryStore

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
You are a fact extractor. Analyze the conversation and identify any lasting \
facts, preferences, or personal information that the USER explicitly stated \
and that would be useful to remember across sessions.

Return ONLY a valid JSON array of strings. Each string is one self-contained fact.
If there are no lasting facts, return an empty array: []

Rules:
- Only include facts the USER directly stated (not implied or inferred).
- Do NOT include temporary context (e.g., "the user asked about the weather").
- Do NOT include anything the ASSISTANT said.
- Be specific and concise (e.g., "User prefers dark mode" not "user likes things dark").
- Maximum 5 facts per turn.

Conversation excerpt:
{conversation}

JSON array of facts (no other text):"""


def extract_and_save_facts(
    conversation_text: str,
    memory_store: "MemoryStore",
    source: str = "",
) -> list[str]:
    """
    Ask the small/fast model to extract lasting facts from `conversation_text`,
    then save each as a semantic memory.

    Args:
        conversation_text: Recent conversation text (last 2-4 turns).
        memory_store:      Where to save the extracted facts.
        source:            Label for the source (e.g., "session_2026-10-25").

    Returns:
        List of saved memory IDs.
    """
    prompt = _EXTRACTION_PROMPT.format(conversation=conversation_text[:3000])

    try:
        result = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.model_small,
            temperature=0.1,
        )
        raw = result.content.strip()
    except Exception as exc:
        logger.warning("Fact extraction LLM call failed: %s", exc)
        return []

    # ── Parse JSON array ───────────────────────────────────────────────────────
    # The model may wrap the array in ```json ... ``` — strip that.
    if "```" in raw:
        lines = raw.split("\n")
        raw = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )

    try:
        facts: list = json.loads(raw)
        if not isinstance(facts, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not parse fact extraction output: %s | raw=%r", exc, raw[:300])
        return []

    # ── Save valid string facts ───────────────────────────────────────────────
    saved_ids: list[str] = []
    for fact in facts[:5]:  # hard cap at 5
        if not isinstance(fact, str) or not fact.strip():
            continue
        try:
            mid = memory_store.remember(
                text=fact.strip(),
                kind="semantic",
                source=source or "fact_extraction",
                importance=0.6,
            )
            saved_ids.append(mid)
            logger.info("Extracted fact: %r → memory_id=%s", fact[:80], mid)
        except Exception as exc:
            logger.warning("Failed to save fact %r: %s", fact[:80], exc)

    return saved_ids
