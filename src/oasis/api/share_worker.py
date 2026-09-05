"""Private relay worker: Gradio's launch(share=True) tunnel, without a Gradio UI.

The parent owns stdin; EOF or SIGTERM closes the FRP child. Keeping blocking Gradio
startup here allows the parent to enforce a deadline and kill the entire process
group on Linux, including a stalled FRP executable.
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
import secrets
import signal
import subprocess
import sys
from types import FrameType


def _terminate(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def run(host: str, port: int) -> int:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    networking = importlib.import_module("gradio.networking")
    tunneling = importlib.import_module("gradio.tunneling")
    token = secrets.token_urlsafe(32)  # Per-launch routing key, not an account credential.
    try:
        # This is the exact helper called by Blocks.launch(share=True) in 6.13.0.
        url = networking.setup_tunnel(
            local_host=host,
            local_port=port,
            share_token=token,
            share_server_address=None,
            share_server_tls_certificate=None,
        )
        print(json.dumps({"status": "ready", "url": url}), flush=True)
        sys.stdin.buffer.read()
        return 0
    finally:
        # Also clean up a partially started tunnel if setup_tunnel raised.
        for tunnel in tuple(tunneling.CURRENT_TUNNELS):
            if tunnel.share_token != token:
                continue
            proc = tunnel.proc
            tunnel.kill()
            atexit.unregister(tunnel.kill)
            if proc is not None:
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
            tunneling.CURRENT_TUNNELS.remove(tunnel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    try:
        code = run(args.host, args.port)
    except Exception:
        # Never forward dependency tracebacks, routing keys, or credentials.
        print(json.dumps({"status": "error"}), flush=True)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
