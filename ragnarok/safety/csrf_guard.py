"""
ragnarok/safety/csrf_guard.py
──────────────────────────────
CSRF / Origin Guard for the Ragnarok FastAPI server.

Security hardening additions:
  - Only accepts requests from localhost origins (127.0.0.1 / ::1)
  - Blocks requests with mismatched Host headers (DNS rebinding defence)
  - Rejects requests that have both a foreign Origin AND no token
  - Provides a FastAPI dependency: require_local_origin()
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

_LOCALHOST_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "[::1]",
    "testserver",
}


def _host_is_local(host: str) -> bool:
    """Return True if the host header indicates a local connection."""
    # strip port
    base = host.rsplit(":", 1)[0].strip("[]").lower()
    return base in _LOCALHOST_HOSTS


def require_local_origin(request: Request) -> None:
    """
    FastAPI dependency that rejects non-local origins.

    Raises HTTP 403 if:
      - Origin header is present and not localhost
      - Host header is present and not localhost
    """
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")

    if origin:
        # Extract hostname from full origin URL (e.g. http://evil.com:80)
        try:
            from urllib.parse import urlparse
            parsed_host = urlparse(origin).hostname or ""
        except Exception:
            parsed_host = ""
        if parsed_host and parsed_host.lower() not in _LOCALHOST_HOSTS:
            logger.warning("Blocked non-local origin: %s", origin)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cross-origin requests are not permitted. Ragnarok only accepts local connections.",
            )

    if host and not _host_is_local(host.split(":")[0]):
        logger.warning("Blocked non-local Host header: %s", host)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Rejected DNS-rebinding attempt: Host '{host}' is not localhost.",
        )
