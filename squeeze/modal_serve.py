"""vLLM servers on Modal: one OpenAI-compatible endpoint per model in config.yaml with `provider: modal`.

Modelled on the starter repo's util/poseidon_compute/modal_serve.py, extended to serve several models
from one app (Exp 1 needs four senders, four same-family siblings and a judge at the same time).
Each model is its own Modal function, so it boots only when first called, and scales to zero after
`modal.scaledown_window` seconds idle.

    modal deploy modal_serve.py           # persistent app; URLs are looked up by name at run time
    modal app stop squeeze-models          # when you are done (idle models already cost nothing)

or run ephemerally around one command (torn down on exit), like poseidon's hosted_model:

    python -m src.host -- python -m src.exp1_matrix

SETUP (one-time): `pip install modal && modal setup`. Gated models (Llama) need
`modal secret create huggingface HF_TOKEN=hf_...` and `huggingface` listed in `modal.secrets`.
"""
import os
import subprocess
import threading
from pathlib import Path

import modal
import yaml

HERE = Path(__file__).resolve().parent
CFG_PATH = Path(os.environ.get("SQUEEZE_CONFIG", HERE / "config.yaml"))
VLLM_PORT = 8000

with open(CFG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
MC = CFG["modal"]

app = modal.App(MC["app_name"])

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM does not cap transformers; a newer transformers breaks vLLM's import, so pin a matching one
    .pip_install(f"vllm=={MC['vllm_version']}", f"transformers=={MC['transformers_version']}",
                 "huggingface_hub[hf_transfer]", "pyyaml")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(CFG_PATH, "/root/config.yaml")  # in case the container re-imports this module
)
hf_cache = modal.Volume.from_name(MC["hf_cache_volume"], create_if_missing=True)
secrets = [modal.Secret.from_name(n) for n in MC.get("secrets", [])]


def _make_server(model_id, max_model_len, extra_args):
    def serve():
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", model_id,
               "--served-model-name", model_id, "--host", "0.0.0.0", "--port", str(VLLM_PORT),
               "--max-model-len", str(max_model_len), "--gpu-memory-utilization", "0.90", *extra_args]
        key = os.environ.get("MODAL_SERVE_API_KEY")
        if key:
            cmd += ["--api-key", key]
        proc = subprocess.Popen(cmd)

        def watchdog():
            # if vLLM dies, kill the container so callers fail fast instead of waiting for the port
            code = proc.wait()
            print(f"[modal_serve] vLLM exited with code {code}; stopping container", flush=True)
            os._exit(1)

        threading.Thread(target=watchdog, daemon=True).start()
    return serve


SERVERS = {}
for key, m in CFG["models"].items():
    if m.get("provider") != "modal":
        continue
    fn = _make_server(m["id"], m.get("max_model_len", MC["max_model_len"]), m.get("vllm_args", []))
    fn = modal.web_server(port=VLLM_PORT, startup_timeout=MC["startup_timeout"])(fn)
    fn = modal.concurrent(max_inputs=MC["max_concurrent_requests"])(fn)
    SERVERS[key] = app.function(
        name=f"serve_{key}", image=image, gpu=m["gpu"], serialized=True, secrets=secrets,
        volumes={"/root/.cache/huggingface": hf_cache}, timeout=24 * 3600,
        scaledown_window=MC["scaledown_window"], max_containers=1,
    )(fn)


def endpoints_from_deployment(keys=None):
    """{model_key: base_url} for an already deployed app."""
    out = {}
    for key in keys or SERVERS:
        url = modal.Function.from_name(MC["app_name"], f"serve_{key}").get_web_url()
        out[key] = url.rstrip("/") + "/v1"
    return out
