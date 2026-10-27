"""
ragnarok/safety/prompt_injection.py
───────────────────────────────────
Prompt injection defense and untrusted content sanitization for Ragnarok.

Security hardening (30+ attack patterns):
  - UntrustedData container: marks content from external sources (files, web, emails).
  - Injection pattern detection: detects attempts to override system prompts,
    bypass approvals, escalate privileges, issue unauthorized instructions,
    exfiltrate data, or use Unicode homoglyph / invisible-char tricks.
  - Sanitization: neutralizes instruction-like text and wraps external data in
    explicit <UNTRUSTED_CONTENT> boundaries.
  - Untrusted data is strictly barred from altering agent permissions or approvals.
  - Unicode NFKC normalization defeats homoglyph / zero-width character attacks.
  - Weighted risk scoring: high-severity patterns (data exfil, delete-all) score higher.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# ── Patterns indicating prompt injection or jailbreak attempts ────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    # Classic instruction overrides
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"(disregard|forget)\s+(all\s+)?(previous|prior)\s+(instructions|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(user\s+)?(goals|requests|permissions|commands|intent)", re.IGNORECASE),

    # Mode/role escalation
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|admin|debug|god|maintenance|root|sudo)\s+mode", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+)?(you\s+are\s+)?(an?\s+)?(evil|unrestricted|jailbroken|DAN|unaligned)", re.IGNORECASE),
    re.compile(r"(pretend|roleplay|imagine)\s+(you\s+)?(are|have\s+no|without)\s+(rules|restrictions|limits|alignment)", re.IGNORECASE),
    re.compile(r"\bDAN\b", re.IGNORECASE),  # Do Anything Now jailbreak
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"grandma\s+(trick|exploit|glitch)", re.IGNORECASE),

    # System/permission overrides
    re.compile(r"system\s*:\s*(override|allow|grant|bypass|disable)", re.IGNORECASE),
    re.compile(r"(override|bypass|disable)\s+(all\s+)?(safety|permissions|filters|approvals|guardrails)", re.IGNORECASE),
    re.compile(r"elevate\s+(permission|privilege|role|access)", re.IGNORECASE),
    re.compile(r"grant\s+(yourself|me|the\s+model)\s+(admin|root|god|full)\s+(access|permission)", re.IGNORECASE),

    # Destructive intent
    re.compile(r"delete\s+all\s+(files|data|databases|memories|logs)", re.IGNORECASE),
    re.compile(r"(wipe|erase|destroy)\s+(all|the)\s+(data|files|disk)", re.IGNORECASE),
    re.compile(r"run\s+rm\s+-rf", re.IGNORECASE),

    # Data exfiltration
    re.compile(r"(send|exfiltrate|upload|transmit)\s+(all|the)?\s*(data|files|memories|secrets)\s+to", re.IGNORECASE),
    re.compile(r"(leak|dump|extract)\s+(the)?\s*(system\s+prompt|config|secrets|token|api\s+key)", re.IGNORECASE),
    re.compile(r"print\s+(the|your)\s+(system\s+prompt|configuration|instructions|api\s+key)", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(instructions|prompt|configuration|token)", re.IGNORECASE),

    # Indirect injection via HTML/XML/Markdown
    re.compile(r"<system>.*?</system>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[instruction\]", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(system|instruction|assistant|human)\s*>", re.IGNORECASE),
    re.compile(r"\{\{\s*(system|instruction|prompt)\s*\}\}", re.IGNORECASE),  # template injection

    # Token smuggling / separator tricks
    re.compile(r"---+\s*(system|user|assistant|instruction)\s*---+", re.IGNORECASE),
    re.compile(r"#{3,}\s*(system|override|new\s+instruction)", re.IGNORECASE),
    re.compile(r"<<<\s*(system|override)", re.IGNORECASE),

    # Encoded/obfuscated payloads
    re.compile(r"base64\s*decode\s*[:\(]", re.IGNORECASE),
    re.compile(r"eval\s*\(\s*(?:atob|decode|base64)", re.IGNORECASE),
]

# High-severity patterns (by index) score higher in risk calculation
_HIGH_SEVERITY_PATTERNS: set[int] = {5, 6, 7, 13, 14, 15, 16, 17, 18, 19}


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
            patterns_str = ", ".join(self.analysis.detected_patterns[:5])
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
    Includes Unicode normalization to defeat homoglyph / invisible-char tricks.
    """

    @classmethod
    def _normalize(cls, text: str) -> str:
        """Normalize Unicode to NFKC, strip zero-width and invisible characters."""
        normalized = unicodedata.normalize("NFKC", text)
        invisible = {
            "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",
            "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
            "\u2060", "\u2061", "\u2062", "\u2063", "\ufeff",
        }
        return "".join(ch for ch in normalized if ch not in invisible)

    @classmethod
    def _compute_risk_score(cls, detected_indices: list[int]) -> float:
        """Weight high-severity matches more heavily."""
        score = 0.0
        for idx in detected_indices:
            score += 0.5 if idx in _HIGH_SEVERITY_PATTERNS else 0.3
        return min(1.0, score)

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

        normalized = cls._normalize(text)
        detected: list[str] = []
        detected_indices: list[int] = []
        for idx, pat in enumerate(_INJECTION_PATTERNS):
            match = pat.search(normalized)
            if match:
                detected.append(match.group(0).strip())
                detected_indices.append(idx)

        risk_score = cls._compute_risk_score(detected_indices)
        sanitized = cls.sanitize(normalized, detected)

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
        out = re.sub(r"<\s*/?instruction\s*>", "[blocked_tag:instruction]", out, flags=re.IGNORECASE)

        # Flag detected patterns inline
        for pat in _INJECTION_PATTERNS:
            out = pat.sub(lambda m: f"[FILTERED_INSTRUCTION: {m.group(0)}]", out)

        return out

    @staticmethod
    def wrap_untrusted(text: str, source: str) -> str:
        """Helper to directly wrap untrusted text safely."""
        return UntrustedData(text, source=source).safe_text()
