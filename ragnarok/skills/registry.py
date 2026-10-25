"""
ragnarok/skills/registry.py
───────────────────────────
Skill registry: loads skills from this package, validates their manifests,
and exposes them as Ollama-compatible tool schemas.

A skill is a Python module in ragnarok/skills/ that exposes:
    MANIFEST : SkillManifest  — metadata + JSON schema
    run(args: dict) -> str    — executes the skill, returns a string result

Adding a new skill = dropping one .py file in ragnarok/skills/.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

# ── Manifest schema ───────────────────────────────────────────────────────────

class SkillParam(BaseModel):
    type: str
    description: str
    enum: list[str] | None = None


class SkillManifest(BaseModel):
    name: str
    description: str
    risk_level: str = "low"          # low | medium | high
    required_permissions: list[str] = Field(default_factory=list)
    parameters: dict[str, SkillParam] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _normalize_manifest(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        params = d.get("parameters")
        if isinstance(params, dict) and "properties" in params:
            # Standard JSON Schema / OpenAI format: {"type": "object", "properties": {...}, "required": [...]}
            if not d.get("required") and "required" in params:
                d["required"] = params.get("required", [])
            d["parameters"] = params.get("properties", {})
        return d


@dataclass
class Skill:
    manifest: SkillManifest
    run: Callable[[dict[str, Any]], str]
    module_name: str

    def to_ollama_tool(self) -> dict[str, Any]:
        """Return an Ollama/OpenAI-compatible tool dict for this skill."""
        props = {
            k: {"type": v.type, "description": v.description}
            | ({"enum": v.enum} if v.enum else {})
            for k, v in self.manifest.parameters.items()
        }
        return {
            "type": "function",
            "function": {
                "name": self.manifest.name,
                "description": self.manifest.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": self.manifest.required,
                },
            },
        }


# ── Registry ──────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Auto-discovers and validates skills in the ragnarok/skills/ package.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def load_all(self) -> None:
        """Import every .py file in ragnarok/skills/ that exposes MANIFEST."""
        skills_dir = Path(__file__).parent
        for path in sorted(skills_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue  # skip __init__, registry, etc.
            module_name = f"ragnarok.skills.{path.stem}"
            try:
                mod = importlib.import_module(module_name)
            except Exception as exc:
                logger.warning("Failed to import skill module %s: %s", module_name, exc)
                continue

            if not hasattr(mod, "MANIFEST") or not hasattr(mod, "run"):
                logger.debug("Skipping %s — no MANIFEST or run()", module_name)
                continue

            try:
                manifest = SkillManifest.model_validate(
                    mod.MANIFEST if isinstance(mod.MANIFEST, dict) else mod.MANIFEST.model_dump()
                )
            except ValidationError as exc:
                logger.warning("Invalid manifest in %s: %s", module_name, exc)
                continue

            skill = Skill(manifest=manifest, run=mod.run, module_name=module_name)
            self._skills[manifest.name] = skill
            logger.info("Loaded skill: %s (risk=%s)", manifest.name, manifest.risk_level)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def as_ollama_tools(self, names: list[str] | None = None) -> list[dict]:
        """
        Return Ollama-compatible tool list.
        If `names` is provided, only return those skills (dynamic selection).
        """
        skills = (
            [self._skills[n] for n in names if n in self._skills]
            if names
            else self.all_skills()
        )
        return [s.to_ollama_tool() for s in skills]

    def validate_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Validate a tool call against its schema before execution.

        Returns:
            (True, "") if valid.
            (False, error_message) if invalid — send error back to model.
        """
        skill = self.get(name)
        if skill is None:
            return False, f"Unknown skill '{name}'. Available: {list(self._skills)}"

        manifest = skill.manifest
        # Check required fields
        for req in manifest.required:
            if req not in arguments:
                return False, f"Missing required parameter '{req}' for skill '{name}'"

        # Check types
        for param_name, value in arguments.items():
            if param_name not in manifest.parameters:
                continue  # extra params are tolerated
            expected_type = manifest.parameters[param_name].type
            py_type = {"string": str, "integer": int, "number": (int, float), "boolean": bool}.get(expected_type)
            if py_type and not isinstance(value, py_type):
                # Local models (Gemma, Llama) frequently send numeric values as strings or numbers as strings
                if expected_type == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
                    continue
                if expected_type == "number" and isinstance(value, str):
                    try:
                        float(value.strip())
                        continue
                    except ValueError:
                        pass
                if expected_type == "boolean" and isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
                    continue
                if expected_type == "string" and isinstance(value, (int, float, bool)):
                    continue
                return False, (
                    f"Parameter '{param_name}' must be {expected_type}, "
                    f"got {type(value).__name__}"
                )

        return True, ""

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={list(self._skills)})"


# Module-level singleton — import and use directly
registry = SkillRegistry()
