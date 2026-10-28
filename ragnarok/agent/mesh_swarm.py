"""
ragnarok/agent/mesh_swarm.py
─────────────────────────────
Innovation #3: P2P Local Mesh Swarm — "Valhalla Cluster"

Enables multiple Ragnarok instances across a local network (LAN) to discover
each other via mDNS/Zeroconf and form a peer-to-peer mesh where:
  - The local node acts as gateway/planner.
  - Heavy inference tasks (large models, vision) are routed to the most
    capable peer (e.g., desktop with GPU) over AES-encrypted sockets.
  - No internet connection required — pure LAN communication over TCP.
  - Zero data leaves the local network.

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │   [Laptop Node] ──mDNS─→ discovers [Desktop Node]        │
  │   ValhallaMeshNode.plan() decides: local vs. remote peer  │
  │   Remote task → AES-encrypted JSON over TCP socket        │
  │   [Desktop Node] → uses its local Ollama (32B model)      │
  │   → returns result encrypted → [Laptop] surfaces to user  │
  └──────────────────────────────────────────────────────────┘

Security:
  - All socket communication encrypted with Fernet (AES-128).
  - Peers authenticate with a shared mesh_secret from config.toml.
  - Connections restricted to RFC-1918 private addresses only.

Dependencies:
  - zeroconf (pip install zeroconf)  ← for mDNS peer discovery
  - cryptography (already in requirements)
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_RAGNAROK_SERVICE_TYPE = "_ragnarok._tcp.local."
_MESH_PORT_DEFAULT = 47890
_PRIVATE_IP_RE = re.compile(
    r"^(10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+|127\.\d+\.\d+\.\d+)$"
)


# ── Peer model ────────────────────────────────────────────────────────────────

@dataclass
class MeshPeer:
    """Represents a discovered Ragnarok peer on the LAN."""
    host: str
    port: int
    node_id: str
    capabilities: list[str] = field(default_factory=list)
    # Capabilities: ["gpu", "vision", "large_model", "fast_cpu"]
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 999.0
    available: bool = True

    def is_private_address(self) -> bool:
        return bool(_PRIVATE_IP_RE.match(self.host))

    def ping(self, timeout: float = 2.0) -> float:
        """Measure round-trip time to this peer. Returns latency in ms or -1 on failure."""
        if not self.is_private_address():
            logger.warning("Refusing to ping non-private address: %s", self.host)
            return -1.0
        try:
            t0 = time.perf_counter()
            s = socket.create_connection((self.host, self.port), timeout=timeout)
            # Send a PING frame
            s.sendall(b'{"cmd":"ping"}\n')
            resp = s.recv(256)
            s.close()
            latency = (time.perf_counter() - t0) * 1000
            if b"pong" in resp:
                self.latency_ms = latency
                return latency
        except Exception:
            self.available = False
        return -1.0


@dataclass
class MeshTaskRequest:
    """A task sent from the local node to a remote peer."""
    task_id: str
    user_input: str
    model_hint: str = ""      # e.g. "llama3:70b" or "llava"
    max_tokens: int = 1024
    requester_id: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "cmd": "run_task",
            "task_id": self.task_id,
            "user_input": self.user_input,
            "model_hint": self.model_hint,
            "max_tokens": self.max_tokens,
            "requester_id": self.requester_id,
        })


@dataclass
class MeshTaskResult:
    """Result returned from a remote peer after executing a task."""
    task_id: str
    answer: str
    peer_node_id: str
    latency_ms: float = 0.0
    error: str = ""
    model_used: str = ""

    @property
    def success(self) -> bool:
        return not self.error


# ── Encrypted transport ────────────────────────────────────────────────────────

class MeshTransport:
    """
    AES-128 Fernet encrypted TCP transport layer for mesh communication.
    All messages are length-prefixed: [4-byte big-endian length][encrypted payload]
    """

    def __init__(self, secret_key: bytes | None = None) -> None:
        from cryptography.fernet import Fernet
        if secret_key:
            self._fernet = Fernet(secret_key)
        else:
            # Generate a new key — in production, this comes from config
            self._fernet = Fernet(Fernet.generate_key())

    @classmethod
    def from_passphrase(cls, passphrase: str) -> "MeshTransport":
        import base64, hashlib
        key = base64.urlsafe_b64encode(
            hashlib.scrypt(passphrase.encode(), salt=b"ragnarok_mesh_v1", n=16384, r=8, p=1, dklen=32)
        )
        return cls(secret_key=key)

    def encrypt(self, data: str) -> bytes:
        return self._fernet.encrypt(data.encode("utf-8"))

    def decrypt(self, data: bytes) -> str:
        return self._fernet.decrypt(data).decode("utf-8")

    def send(self, sock: socket.socket, payload: str) -> None:
        encrypted = self.encrypt(payload)
        length = len(encrypted).to_bytes(4, "big")
        sock.sendall(length + encrypted)

    def recv(self, sock: socket.socket) -> str:
        # Read 4-byte length header
        raw_len = self._recvn(sock, 4)
        if not raw_len:
            raise ConnectionError("Connection closed by peer")
        msg_len = int.from_bytes(raw_len, "big")
        if msg_len > 10 * 1024 * 1024:  # 10MB safety cap
            raise ValueError(f"Message too large: {msg_len} bytes")
        raw = self._recvn(sock, msg_len)
        return self.decrypt(raw)

    @staticmethod
    def _recvn(sock: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return data
            data += chunk
        return data


# ── mDNS Peer Discovery ────────────────────────────────────────────────────────

class MeshDiscovery:
    """
    Advertises this Ragnarok node via mDNS and discovers peers on the LAN.
    Uses the `zeroconf` library. Falls back gracefully if not installed.
    """

    def __init__(
        self,
        node_id: str,
        port: int = _MESH_PORT_DEFAULT,
        capabilities: list[str] | None = None,
    ) -> None:
        self._node_id = node_id
        self._port = port
        self._capabilities = capabilities or ["cpu"]
        self._peers: dict[str, MeshPeer] = {}
        self._lock = threading.Lock()
        self._zeroconf = None
        self._service_info = None
        self._browser = None

    def start(self) -> bool:
        """Start mDNS advertisement and discovery. Returns True if successful."""
        try:
            from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser
            import socket as _socket

            local_ip = self._get_local_ip()
            self._zeroconf = Zeroconf()

            self._service_info = ServiceInfo(
                type_=_RAGNAROK_SERVICE_TYPE,
                name=f"{self._node_id}.{_RAGNAROK_SERVICE_TYPE}",
                addresses=[_socket.inet_aton(local_ip)],
                port=self._port,
                properties={
                    "node_id": self._node_id,
                    "capabilities": ",".join(self._capabilities),
                    "version": "2.0",
                },
            )
            self._zeroconf.register_service(self._service_info)

            # Browse for other peers
            self._browser = ServiceBrowser(
                self._zeroconf,
                _RAGNAROK_SERVICE_TYPE,
                handlers=[self._on_service_change],
            )
            logger.info(
                "Valhalla Mesh: advertising as %s on %s:%d",
                self._node_id, local_ip, self._port,
            )
            return True
        except ImportError:
            logger.info(
                "zeroconf not installed — mesh discovery disabled. "
                "Install with: pip install zeroconf"
            )
            return False
        except Exception as exc:
            logger.warning("Mesh discovery start failed: %s", exc)
            return False

    def stop(self) -> None:
        if self._zeroconf and self._service_info:
            try:
                self._zeroconf.unregister_service(self._service_info)
                self._zeroconf.close()
            except Exception:
                pass

    def _on_service_change(self, zeroconf, service_type, name, state_change) -> None:
        from zeroconf import ServiceStateChange
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info and info.properties:
                props = {k.decode(): v.decode() for k, v in info.properties.items()}
                peer_id = props.get("node_id", name)
                if peer_id == self._node_id:
                    return  # Skip self

                import socket as _socket
                host = _socket.inet_ntoa(info.addresses[0]) if info.addresses else ""

                if not host or not _PRIVATE_IP_RE.match(host):
                    logger.warning("Rejecting non-private mesh peer: %s", host)
                    return

                caps = props.get("capabilities", "").split(",")
                peer = MeshPeer(
                    host=host,
                    port=info.port,
                    node_id=peer_id,
                    capabilities=caps,
                )
                with self._lock:
                    self._peers[peer_id] = peer
                logger.info("Valhalla Mesh: discovered peer %s at %s:%d", peer_id, host, info.port)

        elif state_change is ServiceStateChange.Removed:
            with self._lock:
                self._peers = {k: v for k, v in self._peers.items() if k not in name}

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    @property
    def peers(self) -> dict[str, MeshPeer]:
        with self._lock:
            return dict(self._peers)


# ── Mesh Node (server + client) ───────────────────────────────────────────────

class ValhallaMeshNode:
    """
    The main Valhalla Mesh Node:
      - Starts a TCP server that accepts tasks from other peers.
      - Uses MeshDiscovery to find peers.
      - Can route tasks to the best available peer.
    """

    def __init__(
        self,
        node_id: str | None = None,
        port: int = _MESH_PORT_DEFAULT,
        capabilities: list[str] | None = None,
        mesh_secret: str = "ragnarok-valhalla-mesh",
        local_executor: Callable[[str], str] | None = None,
    ) -> None:
        import os
        self._node_id = node_id or f"ragnarok-{os.getpid()}"
        self._port = port
        self._capabilities = capabilities or self._detect_capabilities()
        self._transport = MeshTransport.from_passphrase(mesh_secret)
        self._local_executor = local_executor
        self._discovery = MeshDiscovery(self._node_id, port, self._capabilities)
        self._server_thread: threading.Thread | None = None
        self._running = False

    @staticmethod
    def _detect_capabilities() -> list[str]:
        """Auto-detect hardware capabilities."""
        caps = ["cpu"]
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                caps.append("gpu")
                caps.append("large_model")
        except Exception:
            pass
        return caps

    def start(self) -> None:
        """Start the mesh server and discovery."""
        self._running = True
        self._discovery.start()
        self._server_thread = threading.Thread(
            target=self._serve, daemon=True, name="ragnarok-mesh-server"
        )
        self._server_thread.start()
        logger.info("Valhalla Mesh Node '%s' started on port %d", self._node_id, self._port)

    def stop(self) -> None:
        self._running = False
        self._discovery.stop()

    def _serve(self) -> None:
        """TCP server loop — accepts connections from remote peers."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", self._port))
            srv.listen(5)
            srv.settimeout(1.0)
            while self._running:
                try:
                    conn, addr = srv.accept()
                    # Security: only accept from private addresses
                    client_ip = addr[0]
                    if not _PRIVATE_IP_RE.match(client_ip):
                        logger.warning("Mesh: rejected connection from %s (non-private)", client_ip)
                        conn.close()
                        continue
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn, addr),
                        daemon=True,
                    ).start()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if self._running:
                        logger.error("Mesh server error: %s", exc)

    def _handle_connection(self, conn: socket.socket, addr: tuple) -> None:
        """Handle one incoming peer connection."""
        try:
            with conn:
                raw = self._transport.recv(conn)
                msg = json.loads(raw)
                cmd = msg.get("cmd", "")

                if cmd == "ping":
                    self._transport.send(conn, json.dumps({"cmd": "pong", "node_id": self._node_id}))

                elif cmd == "run_task":
                    user_input = msg.get("user_input", "")
                    task_id = msg.get("task_id", "")
                    model_hint = msg.get("model_hint", "")
                    t0 = time.perf_counter()

                    if self._local_executor:
                        try:
                            answer = self._local_executor(user_input)
                            error = ""
                        except Exception as exc:
                            answer = ""
                            error = str(exc)
                    else:
                        # Fallback: use Ollama directly
                        answer = self._run_local_llm(user_input, model_hint)
                        error = ""

                    latency = (time.perf_counter() - t0) * 1000
                    resp = {
                        "task_id": task_id,
                        "answer": answer,
                        "peer_node_id": self._node_id,
                        "latency_ms": latency,
                        "error": error,
                        "model_used": model_hint,
                    }
                    self._transport.send(conn, json.dumps(resp))
                else:
                    self._transport.send(conn, json.dumps({"error": f"Unknown command: {cmd}"}))
        except Exception as exc:
            logger.error("Mesh: error handling connection from %s: %s", addr, exc)

    @staticmethod
    def _run_local_llm(prompt: str, model: str) -> str:
        """Run a prompt against the local Ollama instance."""
        import requests
        from ragnarok.config import settings
        payload = {
            "model": model or settings.model_main,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def dispatch_to_peer(
        self, request: MeshTaskRequest, peer: MeshPeer, timeout: float = 120.0
    ) -> MeshTaskResult:
        """Send a task to a specific peer and wait for the result."""
        if not peer.is_private_address():
            raise ValueError(f"Refusing to connect to non-private address: {peer.host}")
        try:
            with socket.create_connection((peer.host, peer.port), timeout=10) as sock:
                sock.settimeout(timeout)
                self._transport.send(sock, request.to_json())
                raw = self._transport.recv(sock)
                data = json.loads(raw)
                return MeshTaskResult(
                    task_id=data.get("task_id", request.task_id),
                    answer=data.get("answer", ""),
                    peer_node_id=data.get("peer_node_id", peer.node_id),
                    latency_ms=data.get("latency_ms", 0.0),
                    error=data.get("error", ""),
                    model_used=data.get("model_used", ""),
                )
        except Exception as exc:
            peer.available = False
            return MeshTaskResult(
                task_id=request.task_id,
                answer="",
                peer_node_id=peer.node_id,
                error=str(exc),
            )

    def best_peer_for(self, capability: str = "gpu") -> MeshPeer | None:
        """
        Find the best available peer with the given capability.
        Ranked by: capability match > latency > last_seen.
        """
        peers = self._discovery.peers
        candidates = [
            p for p in peers.values()
            if p.available and capability in p.capabilities
        ]
        if not candidates:
            return None
        # Sort by latency (ping first for freshness)
        for p in candidates:
            if p.latency_ms > 900:
                p.ping()
        candidates.sort(key=lambda p: p.latency_ms)
        return candidates[0] if candidates else None

    def route_or_local(
        self,
        user_input: str,
        model_hint: str = "",
        prefer_capability: str = "gpu",
    ) -> str:
        """
        Intelligently route a task: use a capable peer if available, else run locally.
        """
        import uuid as _uuid
        peer = self.best_peer_for(prefer_capability)
        if peer:
            logger.info("Mesh: routing task to peer %s (%s)", peer.node_id, peer.host)
            req = MeshTaskRequest(
                task_id=str(_uuid.uuid4()),
                user_input=user_input,
                model_hint=model_hint,
                requester_id=self._node_id,
            )
            result = self.dispatch_to_peer(req, peer)
            if result.success:
                return result.answer
            else:
                logger.warning("Mesh: peer %s failed (%s), falling back to local", peer.node_id, result.error)

        # Local fallback
        if self._local_executor:
            return self._local_executor(user_input)
        return self._run_local_llm(user_input, model_hint)

    def status(self) -> dict[str, Any]:
        """Return mesh status information."""
        peers = self._discovery.peers
        return {
            "node_id": self._node_id,
            "port": self._port,
            "capabilities": self._capabilities,
            "active_peers": len(peers),
            "peers": [
                {
                    "node_id": p.node_id,
                    "host": p.host,
                    "capabilities": p.capabilities,
                    "latency_ms": p.latency_ms,
                    "available": p.available,
                }
                for p in peers.values()
            ],
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_mesh_node: ValhallaMeshNode | None = None


def get_mesh_node(
    local_executor: Callable[[str], str] | None = None,
) -> ValhallaMeshNode:
    global _mesh_node
    if _mesh_node is None:
        try:
            from ragnarok.config import settings
            mesh_secret = settings._raw.get("mesh", {}).get("secret", "ragnarok-valhalla-mesh")
            port = int(settings._raw.get("mesh", {}).get("port", _MESH_PORT_DEFAULT))
        except Exception:
            mesh_secret = "ragnarok-valhalla-mesh"
            port = _MESH_PORT_DEFAULT

        _mesh_node = ValhallaMeshNode(
            port=port,
            mesh_secret=mesh_secret,
            local_executor=local_executor,
        )
    return _mesh_node
