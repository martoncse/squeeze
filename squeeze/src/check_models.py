"""Boot each Modal model with one tiny request and report whether it answers, and how long it took.

Run this after every `modal deploy`, before a long experiment, so a broken server costs one cold
start instead of hours of retries.

    python -m src.check_models                 # every model in config.yaml
    python -m src.check_models qwen judge      # just these
"""
import asyncio
import sys
import time

from .common import ROOT, load_config
from .llm import LLM


async def main(keys):
    cfg = load_config()
    keys = keys or list(cfg["models"])
    llm = LLM(cfg, ROOT, "check_models")

    async def one(k):
        t0 = time.time()
        msg = [{"role": "user", "content": f"Reply with the single word OK. (health check {t0:.0f})"}]
        try:
            text, _ = await llm.chat(k, msg, temperature=0.0, max_tokens=5, tag="health")
            return k, True, time.time() - t0, text.strip()[:40]
        except Exception as e:
            return k, False, time.time() - t0, f"{type(e).__name__}: {e}"[:300]

    results = await asyncio.gather(*(one(k) for k in keys))
    for k, ok, dt, info in results:
        print(f"{'OK  ' if ok else 'FAIL'} {k:<14} {cfg['models'][k]['id']:<40} {dt:6.0f}s  {info}")
    print(llm.report())
    return all(ok for _, ok, _, _ in results)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main(sys.argv[1:])) else 1)
