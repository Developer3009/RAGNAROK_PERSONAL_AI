"""
ragnarok/agent/history.py
─────────────────────────
Message history manager for the agent loop.

Keeps the last N conversation turns in memory and trims older ones to
stay inside the model's effective context window (estimated by char count).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ragnarok.config import settings

Role = Literal["system", "user", "assistant", "tool"]

# Rough chars-per-token estimate — conservative for mixed EN/code text
_CHARS_PER_TOKEN: int = 4
# Leave this many tokens headroom for the model's reply
_REPLY_HEADROOM_TOKENS: int = 512


@dataclass
class Message:
    role: Role
    content: str
    tool_name: str | None = None      # set when role == "tool"
    tool_call_id: str | None = None   # for OpenAI-compatible formats

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_name:
            d["name"] = self.tool_name
        return d


class MessageHistory:
    """
    Maintains an ordered list of messages and trims to fit context.

    Args:
        max_turns:      Maximum number of user/assistant turn-pairs to keep.
        max_context_chars: Hard cap on total character count before trimming.
    """

    def __init__(
        self,
        max_turns: int | None = None,
        max_context_chars: int = 12_000,
    ) -> None:
        self._max_turns = max_turns or settings.history_turns
        self._max_context_chars = max_context_chars
        self._messages: list[Message] = []

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add(self, role: Role, content: str, **kwargs) -> None:
        """Append a message and trim if necessary."""
        self._messages.append(Message(role=role, content=content, **kwargs))
        self._trim()

    def add_user(self, text: str) -> None:
        self.add("user", text)

    def add_assistant(self, text: str) -> None:
        self.add("assistant", text)

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """Add a tool result message (role=tool)."""
        self._messages.append(
            Message(role="tool", content=result, tool_name=tool_name)
        )
        self._trim()

    def clear(self) -> None:
        self._messages.clear()

    # ── Query ──────────────────────────────────────────────────────────────────

    def as_dicts(self) -> list[dict]:
        """Return all messages as plain dicts for the Ollama API."""
        return [m.to_dict() for m in self._messages]

    def total_chars(self) -> int:
        return sum(len(m.content) for m in self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"MessageHistory(messages={len(self._messages)}, "
            f"chars={self.total_chars()})"
        )

    # ── Trimming ───────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        """
        Remove the oldest non-system messages until both constraints are met:
          1. ≤ max_turns user/assistant pairs
          2. total chars ≤ max_context_chars
        """
        # Count user/assistant turn-pairs (one user + one assistant = 1 turn)
        while True:
            user_msgs = [m for m in self._messages if m.role == "user"]
            over_turns = len(user_msgs) > self._max_turns
            over_chars = self.total_chars() > self._max_context_chars

            if not (over_turns or over_chars):
                break

            # Drop the oldest non-system message
            for i, msg in enumerate(self._messages):
                if msg.role != "system":
                    self._messages.pop(i)
                    break
            else:
                # Only system messages left — can't trim further
                break

    # ── Snapshot / restore ────────────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        """Serialise for working-memory checkpointing."""
        return [
            {
                "role": m.role,
                "content": m.content,
                "tool_name": m.tool_name,
                "tool_call_id": m.tool_call_id,
            }
            for m in self._messages
        ]

    @classmethod
    def from_snapshot(cls, data: list[dict], **kwargs) -> "MessageHistory":
        h = cls(**kwargs)
        for d in data:
            h._messages.append(
                Message(
                    role=d["role"],
                    content=d["content"],
                    tool_name=d.get("tool_name"),
                    tool_call_id=d.get("tool_call_id"),
                )
            )
        return h
