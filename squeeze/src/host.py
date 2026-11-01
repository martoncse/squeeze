"""Run a command with this project's Modal model servers up for its duration, then tear them down.

The multi-model version of the starter repo's `poseidon-host` (poseidon_compute.hosted_model): runs
modal_serve.py as an ephemeral Modal app, passes the endpoints to the command via SQUEEZE_ENDPOINTS,
and stops the app on exit, so no GPU is left running. Models boot lazily on their first request.

    python -m src.host -- python -m src.exp1_matrix --smoke
    python -m src.host --warm qwen,judge -- python -m src.gen_dataset

For a long session (several scripts), `modal deploy modal_serve.py` instead and run scripts directly.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from .common import ROOT, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--warm", default="", help="comma-separated model keys to boot before the command starts")
    ap.add_argument("--ready-timeout", type=int, default=1500)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="the command, after `--`")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("provide a command after `--`, e.g.  -- python -m src.exp1_matrix")

    cfg = load_config()
    from poseidon_compute.hosted_model import _check_hf_access, _wait_ready

    for key, m in cfg["models"].items():  # fail fast on gated models we cannot download
        if m.get("provider") == "modal":
            _check_hf_access(m["id"])

    import modal

    sys.path.insert(0, str(ROOT))
    import modal_serve

    with modal.enable_output(), modal_serve.app.run():
        eps = {k: f.get_web_url().rstrip("/") + "/v1" for k, f in modal_serve.SERVERS.items()}
        print(f"[host] {len(eps)} model endpoints up (each boots on first request)", flush=True)
        warm = [k for k in args.warm.split(",") if k]
        if warm:
            key = os.environ.get("MODAL_SERVE_API_KEY")
            with ThreadPoolExecutor(len(warm)) as ex:
                list(ex.map(lambda k: _wait_ready(eps[k][: -len("/v1")], key, args.ready_timeout), warm))
            print(f"[host] warm: {', '.join(warm)}", flush=True)
        rc = subprocess.run(cmd, env={**os.environ, "SQUEEZE_ENDPOINTS": json.dumps(eps)}).returncode
    print("[host] torn down.", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
