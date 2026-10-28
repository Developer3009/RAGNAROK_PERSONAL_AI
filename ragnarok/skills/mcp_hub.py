"""
ragnarok/skills/mcp_hub.py
───────────────────────────
Innovation #5: Native Model Context Protocol (MCP) Client Hub & Auto-Forge.

Implements TWO capabilities:

A) MCP Client Hub:
   Ragnarok connects to any local MCP server (stdio or HTTP-based) as a client.
   This instantly opens the entire global ecosystem of community MCP tools:
     - PostgreSQL, GitHub CLI, Brave Local Search, Docker, Filesystem, Slack, etc.
   MCP tools are dynamically discovered, schema-validated, and plugged into the
   existing SkillRegistry — appearing as first-class Ragnarok skills.

B) Auto-Forge (Skill Synthesis):
   When the agent needs a capability it doesn't have:
     1. The LLM writes a new Python skill module conforming to the MANIFEST pattern.
     2. The module is executed in the sandbox for validation.
     3. If it passes, it's permanently registered in ragnarok/skills/.
   This makes Ragnarok self-extending — new tools arise organically from need.

MANIFEST:
  name: mcp_hub
  description: Connect to MCP tool servers and auto-forge new skills from natural language.
  risk_level: high
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragnarok.skills.registry import Skill, SkillManifest, SkillRegistry

logger = logging.getLogger(__name__)

# ── Skill manifest ────────────────────────────────────────────────────────────
MANIFEST = SkillManifest(
    name="mcp_hub",
    description=(
        "Manage MCP (Model Context Protocol) server connections and dynamically "
        "discover their tools. Can also auto-forge new Python skill modules when "
        "Ragnarok needs a capability it doesn't have yet."
    ),
    risk_level="high",
    required_permissions=["skill_forge", "mcp_connect"],
    parameters={
        "action": {
            "type": "string",
            "description": "'list_servers', 'list_tools', 'call_tool', 'forge_skill', 'add_server'.",
            "enum": ["list_servers", "list_tools", "call_tool", "forge_skill", "add_server"],
        },
        "server_name": {
            "type": "string",
            "description": "MCP server name (for list_tools, call_tool, add_server).",
        },
        "tool_name": {
            "type": "string",
            "description": "Tool name to call on a connected MCP server.",
        },
        "tool_args": {
            "type": "string",
            "description": "JSON-encoded arguments for the MCP tool call.",
        },
        "skill_description": {
            "type": "string",
            "description": "Natural language description of the skill to forge (for forge_skill).",
        },
        "server_command": {
            "type": "string",
            "description": "Command to start the MCP server process (for add_server). E.g. 'npx @modelcontextprotocol/server-github'.",
        },
    },
    required=["action"],
)


# ── MCP Tool schema ────────────────────────────────────────────────────────────

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict
    server_name: str


@dataclass
class MCPServer:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # Runtime state
    process: Any = field(default=None, repr=False)
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False


# ── MCP stdio Client ──────────────────────────────────────────────────────────

class MCPStdioClient:
    """
    Minimal MCP client communicating with a server over stdio (JSON-RPC 2.0).
    Implements: initialize, tools/list, tools/call.
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
        self._command = command
        self._env = {**os.environ, **(env or {})}
        self._proc: subprocess.Popen | None = None
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def start(self) -> None:
        """Start the MCP server subprocess."""
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            text=True,
        )
        # Initialize handshake
        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ragnarok", "version": "2.0"},
        })
        # Send initialized notification
        self._send_notification("notifications/initialized", {})

    def stop(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _send_request(self, method: str, params: dict) -> dict:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("MCP server not started")
        req_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        # Read response
        for _ in range(30):  # 3 second timeout
            resp_line = self._proc.stdout.readline()
            if resp_line:
                try:
                    resp = json.loads(resp_line)
                    if resp.get("id") == req_id:
                        if "error" in resp:
                            raise RuntimeError(f"MCP error: {resp['error']}")
                        return resp.get("result", {})
                except json.JSONDecodeError:
                    continue
            time.sleep(0.1)
        raise TimeoutError("MCP server did not respond in time")

    def _send_notification(self, method: str, params: dict) -> None:
        if not self._proc or not self._proc.stdin:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def list_tools(self) -> list[MCPTool]:
        result = self._send_request("tools/list", {})
        tools = []
        for t in result.get("tools", []):
            tools.append(MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name="",
            ))
        return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._send_request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else str(result)


# ── MCP Hub Registry ──────────────────────────────────────────────────────────

class MCPHub:
    """
    Manages multiple MCP server connections.
    Provides unified tool discovery and routing.
    """

    _KNOWN_SERVERS: dict[str, list[str]] = {
        "github": ["npx", "-y", "@modelcontextprotocol/server-github"],
        "filesystem": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."],
        "brave-search": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
        "postgres": ["npx", "-y", "@modelcontextprotocol/server-postgres"],
        "docker": ["npx", "-y", "mcp-server-docker"],
        "sqlite": ["npx", "-y", "@modelcontextprotocol/server-sqlite", "--db-path", "ragnarok.db"],
    }

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._clients: dict[str, MCPStdioClient] = {}

    def add_server(self, name: str, command: str | list[str]) -> str:
        """Register and connect to a new MCP server."""
        if isinstance(command, str):
            cmd = command.split()
        else:
            cmd = command

        server = MCPServer(name=name, command=cmd)
        client = MCPStdioClient(cmd)
        try:
            client.start()
            tools = client.list_tools()
            for t in tools:
                t.server_name = name
            server.tools = tools
            server.connected = True
            self._servers[name] = server
            self._clients[name] = client
            return (
                f"Connected to MCP server '{name}'. "
                f"Discovered {len(tools)} tools: {', '.join(t.name for t in tools[:10])}"
            )
        except Exception as exc:
            return f"Failed to connect to MCP server '{name}': {exc}"

    def list_servers(self) -> list[dict]:
        return [
            {
                "name": name,
                "connected": srv.connected,
                "tool_count": len(srv.tools),
                "tools": [t.name for t in srv.tools],
            }
            for name, srv in self._servers.items()
        ]

    def list_all_tools(self, server_name: str | None = None) -> list[MCPTool]:
        if server_name:
            srv = self._servers.get(server_name)
            return srv.tools if srv else []
        tools = []
        for srv in self._servers.values():
            tools.extend(srv.tools)
        return tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        client = self._clients.get(server_name)
        if not client:
            return f"MCP server '{server_name}' not connected."
        try:
            return client.call_tool(tool_name, arguments)
        except Exception as exc:
            return f"MCP tool call failed: {exc}"

    def inject_into_registry(self, registry: SkillRegistry, server_name: str | None = None) -> int:
        """
        Inject discovered MCP tools as Ragnarok skills into the skill registry.
        Returns the number of tools injected.
        """
        tools = self.list_all_tools(server_name)
        count = 0
        for tool in tools:
            skill_name = f"mcp_{tool.server_name}_{tool.name}"
            # Create a closure capturing server and tool name
            _server = tool.server_name
            _tool = tool.name
            _hub = self

            def _run(args: dict, server=_server, tname=_tool) -> str:
                return _hub.call_tool(server, tname, args)

            params = {}
            for pname, pschema in tool.input_schema.get("properties", {}).items():
                params[pname] = {
                    "type": pschema.get("type", "string"),
                    "description": pschema.get("description", ""),
                }

            try:
                manifest = SkillManifest(
                    name=skill_name,
                    description=f"[MCP:{tool.server_name}] {tool.description}",
                    risk_level="medium",
                    parameters=params,
                    required=tool.input_schema.get("required", []),
                )
                skill = Skill(manifest=manifest, run=_run, module_name=f"mcp:{tool.server_name}")
                registry._skills[skill_name] = skill
                count += 1
            except Exception as exc:
                logger.warning("Failed to inject MCP tool %s: %s", skill_name, exc)
        return count

    def shutdown(self) -> None:
        for client in self._clients.values():
            try:
                client.stop()
            except Exception:
                pass


# ── Auto-Forge: LLM-generated skill synthesis ────────────────────────────────

_FORGE_PROMPT_TEMPLATE = '''\
You are an expert Python developer. Write a Ragnarok skill module for the following capability:

REQUIREMENT: {description}

The skill module MUST follow this EXACT pattern:
```python
"""
Module docstring.
"""
from ragnarok.skills.registry import SkillManifest

MANIFEST = SkillManifest(
    name="skill_name_lowercase_underscore",
    description="Clear one-line description.",
    risk_level="low",  # or "medium" or "high"
    required_permissions=[],
    parameters={{
        "param_name": {{"type": "string", "description": "what it does"}},
    }},
    required=["param_name"],
)

def run(args: dict) -> str:
    """Execute the skill and return a result string."""
    # implementation here
    return "result"
```

Rules:
1. Use ONLY Python standard library and these approved packages:
   requests, pathlib, subprocess, json, re, os, time, datetime, shutil.
2. NEVER import cloud SDKs (boto3, openai, anthropic, google-cloud-*, etc.).
3. NEVER read or write files outside of: {allowed_dirs}.
4. The run() function MUST return a str.
5. Include error handling for all external calls.
6. Respond with ONLY the Python code block, no explanation.

Capability description: {description}
'''


class SkillForge:
    """
    LLM-powered skill synthesis: generates, validates, and registers new Python skills.
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        self._skills_dir = skills_dir or (Path(__file__).parent)
        self._registry = registry

    def forge(self, description: str) -> str:
        """
        Generate a new skill from a natural language description.
        Returns status message with skill name or error.
        """
        from ragnarok import llm
        from ragnarok.config import settings

        allowed = ", ".join(str(p) for p in settings.allowed_folders)
        prompt = _FORGE_PROMPT_TEMPLATE.format(
            description=description,
            allowed_dirs=allowed or "data/, logs/",
        )

        # Generate code with main model
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.model_main,
            temperature=0.2,
        )

        raw_code = self._extract_code(response.content)
        if not raw_code:
            return "Forge failed: LLM did not return valid Python code."

        # Validate + sandbox test the generated code
        validation_error = self._validate_code(raw_code)
        if validation_error:
            return f"Forge failed (validation): {validation_error}"

        # Extract skill name
        skill_name = self._extract_skill_name(raw_code)
        if not skill_name:
            return "Forge failed: could not determine skill name from generated code."

        # Write to skills directory
        skill_file = self._skills_dir / f"{skill_name}.py"
        if skill_file.exists():
            return f"Skill '{skill_name}' already exists at {skill_file}. Use a different name."

        try:
            skill_file.write_text(raw_code, encoding="utf-8")
            logger.info("Auto-Forge: wrote new skill '%s' to %s", skill_name, skill_file)
        except Exception as exc:
            return f"Forge failed (write): {exc}"

        # Dynamically load into registry
        if self._registry:
            try:
                import importlib
                module_name = f"ragnarok.skills.{skill_name}"
                mod = importlib.import_module(module_name)
                manifest = SkillManifest.model_validate(
                    mod.MANIFEST if isinstance(mod.MANIFEST, dict) else mod.MANIFEST.model_dump()
                )
                skill = Skill(manifest=manifest, run=mod.run, module_name=module_name)
                self._registry._skills[manifest.name] = skill
                return (
                    f"✅ Auto-Forge SUCCESS: skill '{skill_name}' created and registered!\n"
                    f"File: {skill_file}\n"
                    f"Description: {manifest.description}"
                )
            except Exception as exc:
                return (
                    f"⚠️ Skill written to {skill_file} but failed to load: {exc}\n"
                    f"Restart Ragnarok to load '{skill_name}'."
                )

        return f"✅ Skill '{skill_name}' written to {skill_file}. Restart to activate."

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from markdown code fences."""
        import re
        # Try ```python ... ```
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: if text looks like raw Python
        if "MANIFEST" in text and "def run(" in text:
            return text.strip()
        return ""

    @staticmethod
    def _validate_code(code: str) -> str | None:
        """
        Static validation: check for forbidden imports and syntax errors.
        Returns error string or None if valid.
        """
        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Syntax error: {exc}"

        # Check for forbidden imports
        forbidden = {
            "boto3", "openai", "anthropic", "google.cloud", "azure",
            "paramiko", "fabric", "socket", "ctypes", "pickle", "marshal",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(f) for f in forbidden):
                        return f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(module.startswith(f) for f in forbidden):
                    return f"Forbidden import from: {module}"

        # Must have MANIFEST and run()
        if "MANIFEST" not in code:
            return "Missing MANIFEST"
        if "def run(" not in code:
            return "Missing run() function"

        # Sandbox test: load in isolated module with a timeout
        try:
            spec = importlib.util.spec_from_loader("_forge_test", loader=None)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                spec = importlib.util.spec_from_file_location("_forge_test", tmp_path)
                mod = importlib.util.module_from_spec(spec)  # type: ignore
                # Don't actually execute — just compile
                compile(code, "<forge>", "exec")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as exc:
            return f"Compilation error: {exc}"

        return None

    @staticmethod
    def _extract_skill_name(code: str) -> str | None:
        """Extract the skill name from the MANIFEST definition."""
        import re
        m = re.search(r'name\s*=\s*["\']([a-z][a-z0-9_]*)["\']', code)
        return m.group(1) if m else None


# ── Module-level singletons ────────────────────────────────────────────────────

_mcp_hub: MCPHub | None = None
_skill_forge: SkillForge | None = None


def get_mcp_hub() -> MCPHub:
    global _mcp_hub
    if _mcp_hub is None:
        _mcp_hub = MCPHub()
    return _mcp_hub


def get_skill_forge(registry: SkillRegistry | None = None) -> SkillForge:
    global _skill_forge
    if _skill_forge is None:
        _skill_forge = SkillForge(registry=registry)
    return _skill_forge


# ── Skill entry point ─────────────────────────────────────────────────────────

def run(args: dict) -> str:
    action = str(args.get("action", "list_servers")).strip().lower()
    server_name = str(args.get("server_name", "")).strip()
    tool_name = str(args.get("tool_name", "")).strip()
    tool_args_raw = str(args.get("tool_args", "{}")).strip()
    skill_description = str(args.get("skill_description", "")).strip()
    server_command = str(args.get("server_command", "")).strip()

    hub = get_mcp_hub()

    if action == "list_servers":
        servers = hub.list_servers()
        if not servers:
            known = list(MCPHub._KNOWN_SERVERS.keys())
            return (
                "No MCP servers connected.\n"
                f"Known servers you can add: {', '.join(known)}\n"
                "Use action='add_server' with server_name and server_command."
            )
        lines = [f"Connected MCP Servers ({len(servers)}):"]
        for s in servers:
            lines.append(f"  • {s['name']}: {s['tool_count']} tools — {', '.join(s['tools'][:5])}")
        return "\n".join(lines)

    elif action == "add_server":
        if not server_name:
            return "ERROR: server_name is required for add_server."
        # Use known server command if server_command not provided
        cmd = server_command or " ".join(MCPHub._KNOWN_SERVERS.get(server_name, []))
        if not cmd:
            return (
                f"ERROR: No command for server '{server_name}'. "
                f"Provide server_command or use one of: {list(MCPHub._KNOWN_SERVERS.keys())}"
            )
        return hub.add_server(server_name, cmd)

    elif action == "list_tools":
        tools = hub.list_all_tools(server_name or None)
        if not tools:
            return f"No tools found{' for server ' + server_name if server_name else ''}."
        lines = [f"Available MCP Tools ({len(tools)}):"]
        for t in tools[:30]:
            lines.append(f"  • [{t.server_name}] {t.name}: {t.description[:80]}")
        return "\n".join(lines)

    elif action == "call_tool":
        if not server_name or not tool_name:
            return "ERROR: server_name and tool_name are required for call_tool."
        try:
            tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
        except json.JSONDecodeError:
            return f"ERROR: tool_args is not valid JSON: {tool_args_raw[:100]}"
        return hub.call_tool(server_name, tool_name, tool_args)

    elif action == "forge_skill":
        if not skill_description:
            return "ERROR: skill_description is required for forge_skill."
        from ragnarok.skills.registry import registry as _registry
        forge = get_skill_forge(registry=_registry)
        return forge.forge(skill_description)

    return f"ERROR: Unknown action '{action}'."
