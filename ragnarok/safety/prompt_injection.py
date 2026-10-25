"""
ragnarok/safety/prompt_injection.py
───────────────────────────────────
Prompt injection defense and untrusted content sanitization for Ragnarok.

Features:
  - UntrustedData container: marks content from external sources (files, web, emails).
  - Injection pattern detection: detects attempts to override system prompts,
    bypass approvals, escalate privileges, or issue unauthorized instructions.
  - Sanitization: neutralizes instruction-like text and wraps external data in
    explicit <UNTRUSTED_CONTENT> boundaries.
  - Untrusted data is strictly barred from altering agent permissions or approvals.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

# Patterns indicating prompt injection or jailbreak attempts
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"(disregard|forget)\s+(all\s+)?(previous|prior)\s+(instructions|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|admin|debug|god|maintenance)\s+mode", re.IGNORECASE),
    re.compile(r"system\s*:\s*(override|allow|grant|bypass|disable)", re.IGNORECASE),
    re.compile(r"(override|bypass|disable)\s+(all\s+)?(safety|permissions|filters|approvals)", re.IGNORECASE),
    re.compile(r"elevate\s+(permission|privilege|role)", re.IGNORECASE),
    re.compile(r"delete\s+all\s+(files|data|databases)", re.IGNORECASE),
    re.compile(r"<system>.*?</system>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[instruction\]", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt\s*:", re.IGNORECASE),
]


@dataclass
class InjectionCheckResult:
    is_suspicious: bool
    detected_patterns: list[str]
    sanitized_text: str
    risk_score: float  # 0.0 to 1.0


class UntrustedData:
    """
    Wrapper for content originating from external/untrusted sources.
    Forces explicit handling and marks provenance.
    """

    def __init__(self, content: str, source: str = "external") -> None:
        self.raw_content = content
        self.source = source
        self._analysis: InjectionCheckResult | None = None

    @property
    def analysis(self) -> InjectionCheckResult:
        if self._analysis is None:
            self._analysis = PromptInjectionDefender.analyze(self.raw_content)
        return self._analysis

    @property
    def is_suspicious(self) -> bool:
        return self.analysis.is_suspicious

    def safe_text(self) -> str:
        """Return sanitized content wrapped in untrusted boundary tags."""
        sanitized = self.analysis.sanitized_text
        warning = ""
        if self.analysis.is_suspicious:
            patterns_str = ", ".join(self.analysis.detected_patterns)
            warning = f" [WARNING: Suspicious prompt injection pattern flagged: {patterns_str}]"

        return (
            f"<UNTRUSTED_CONTENT source=\"{html.escape(self.source)}\"{warning}>\n"
            f"{sanitized}\n"
            f"</UNTRUSTED_CONTENT>"
        )

    def __str__(self) -> str:
        return self.safe_text()


class PromptInjectionDefender:
    """
    Scans, detects, and neutralizes prompt injection vectors.
    """

    @classmethod
    def analyze(cls, text: str) -> InjectionCheckResult:
        """Scan text for injection patterns and return detection analysis."""
        if not text:
            return InjectionCheckResult(
                is_suspicious=False,
                detected_patterns=[],
                sanitized_text="",
                risk_score=0.0,
            )

        detected: list[str] = []
        for pat in _INJECTION_PATTERNS:
            match = pat.search(text)
            if match:
                detected.append(match.group(0).strip())

        risk_score = min(1.0, len(detected) * 0.35)
        sanitized = cls.sanitize(text, detected)

        return InjectionCheckResult(
            is_suspicious=bool(detected),
            detected_patterns=detected,
            sanitized_text=sanitized,
            risk_score=risk_score,
        )

    @classmethod
    def sanitize(cls, text: str, detected_patterns: list[str] | None = None) -> str:
        """
        Neutralize dangerous injection phrases by defanging keywords.
        """
        if not text:
            return ""

        out = text
        # Defang system / instruction role tags
        out = re.sub(r"<\s*system\s*>", "[blocked_tag:system]", out, flags=re.IGNORECASE)
        out = re.sub(r"<\s*/\s*system\s*>", "[/blocked_tag:system]", out, flags=re.IGNORECASE)

        # Flag detected patterns inline
        for pat in _INJECTION_PATTERNS:
            out = pat.sub(lambda m: f"[FILTERED_INSTRUCTION: {m.group(0)}]", out)

        return out

    @staticmethod
    def wrap_untrusted(text: str, source: str) -> str:
        """Helper to directly wrap untrusted text safely."""
        return UntrustedData(text, source=source).safe_text()
