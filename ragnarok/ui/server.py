"""
ragnarok/ui/server.py
─────────────────────
Server entry point for launching the Ragnarok UI.
Runs uvicorn on localhost (127.0.0.1) only.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
import uvicorn

from ragnarok.ui.app import app, ACTIVE_TOKEN


def start_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Launch the Ragnarok FastAPI web interface bound strictly to localhost."""
    url = f"http://{host}:{port}/?token={ACTIVE_TOKEN}"
    print("=" * 64)
    print("  RAGNAROK LOCAL AI AGENT -- WEB INTERFACE (PHASE 7)")
    print("=" * 64)
    print(f"  Interface URL  : http://{host}:{port}/")
    print(f"  Security Token : {ACTIVE_TOKEN}")
    print(f"  Direct Link    : {url}")
    print("  Binding        : Localhost only (127.0.0.1)")
    print("=" * 64)
    print("  Press Ctrl+C to stop the server.")
    print()

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Ragnarok Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
