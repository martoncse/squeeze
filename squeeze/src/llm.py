"""Unified async LLM client. Default: open-weight models served by vLLM on Modal (modal_serve.py).
Also supports Nebius, OpenRouter, OpenAI, Anthropic, Google and any OpenAI-compatible vLLM URL, plus
an offline mock.

Every call goes through the SQLite cache, so reruns cost nothing. Spend is tracked per run (GPU time
for Modal models, tokens for API providers) and the run aborts (BudgetExceeded) at `budget_usd`.
"""
import asyncio
import json
import math
import os
import random
import re
import time
from pathlib import Path

from .cache import Cache, call_key

OPENAI_COMPAT = {
    "modal": (None, "MODAL_SERVE_API_KEY"),
    "vllm": (None, "VLLM_API_KEY"),
    "nebius": ("https://api.studio.nebius.com/v1", "NEBIUS_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}
KEYLESS = ("modal", "vllm")  # endpoints that may run without a bearer key


class BudgetExceeded(RuntimeError):
    pass


class SetupError(RuntimeError):
    """Misconfiguration (missing key or deployment): fail at once, never retried."""


def load_env():
    """Load the repo-root compute.env the same way every starter-repo project does (poseidon_compute)."""
    try:
        from poseidon_compute.llm_config import _load
    except ImportError:
        return None
    return _load()


def extract_json(text):
    """First {...} object in a reply (models often wrap JSON in prose or code fences)."""
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate a trailing object after the first one
        depth = 0
        s = m.group(0)
        for i, ch in enumerate(s):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                try:
                    return json.loads(s[: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LLM:
    def __init__(self, cfg, root: Path, run_name: str, mock=False):
        self.cfg = cfg
        self.models = cfg["models"]
        self.mock = mock
        self.run_name = run_name
        self.budget = float(cfg["budget_usd"])
        self.cache = Cache(root / ("cache_mock.sqlite" if mock else "cache.sqlite"))
        self.sem = asyncio.Semaphore(int(cfg.get("concurrency", 8)))
        self.spent = 0.0  # token-billed providers
        self.gpu_span = {}  # model key -> [first request start, last response] (Modal models)
        self.n_calls = 0
        self.n_cached = 0
        self.by_tag = {}  # tag -> [calls, cost, in_tokens, out_tokens]
        self._clients = {}
        if not mock:
            load_env()

    # ------------------------------------------------------------------ providers
    async def _modal_url(self, mkey):
        eps = json.loads(os.environ.get("SQUEEZE_ENDPOINTS", "{}"))  # set by `python -m src.host`
        if mkey in eps:
            return eps[mkey]
        import modal  # otherwise look the deployed app's endpoint up by name

        app_name = self.cfg["modal"]["app_name"]
        try:
            url = await modal.Function.from_name(app_name, f"serve_{mkey}").get_web_url.aio()
        except Exception as e:
            raise SetupError(f"no Modal endpoint for {mkey!r}: run `modal deploy modal_serve.py`, or wrap the "
                             f"command in `python -m src.host -- ...` ({type(e).__name__}: {e})") from None
        return url.rstrip("/") + "/v1"

    async def _client(self, mkey):
        mcfg = self.models[mkey]
        provider = mcfg["provider"]
        ck = mkey if provider == "modal" else provider
        if ck in self._clients:
            return self._clients[ck]
        if provider == "anthropic":
            import anthropic  # optional dependency

            # pass base_url explicitly so an ANTHROPIC_BASE_URL set for other tools is not picked up
            client = anthropic.AsyncAnthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ.get("SQUEEZE_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            )
        else:
            from openai import AsyncOpenAI

            base, key_var = OPENAI_COMPAT[provider]
            base = await self._modal_url(mkey) if provider == "modal" else mcfg.get("base_url", base)
            key = os.environ.get(key_var) or ("EMPTY" if provider in KEYLESS else None)
            if not key:
                raise SetupError(f"{key_var} is not set (put it in the repo-root compute.env)")
            timeout = self.cfg.get("modal", {}).get("request_timeout", 1800) if provider == "modal" else 120
            client = AsyncOpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=0)
        self._clients[ck] = client
        return client

    async def _raw_call(self, mkey, messages, temperature, max_tokens, seed, top_logprobs=None):
        """Return (text, usage). With top_logprobs, text is a JSON {token: prob} for the first
        generated token instead of the generated text."""
        mcfg = self.models[mkey]
        provider = mcfg["provider"]
        if self.mock:
            from .mock import mock_reply

            text = mock_reply(mkey, self.models, messages, seed)
            n_in = sum(len(m["content"]) for m in messages) // 4
            return text, {"in": n_in, "out": len(text) // 4, "cost": 0.0}
        client = await self._client(mkey)
        if provider == "anthropic":
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            rest = [m for m in messages if m["role"] != "system"]
            r = await client.messages.create(
                model=mcfg["id"], system=system or None, messages=rest,
                temperature=temperature, max_tokens=max_tokens,
            )
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens, "cost": None}
        else:
            extra = dict(mcfg.get("extra_body", {}))
            if provider == "openrouter":
                extra.setdefault("usage", {"include": True})
            kwargs = dict(model=mcfg["id"], messages=messages, temperature=temperature,
                          max_tokens=max_tokens, extra_body=extra or None)
            if provider != "google":
                kwargs["seed"] = seed
            if top_logprobs:
                kwargs.update(logprobs=True, top_logprobs=top_logprobs)
            r = await client.chat.completions.create(**kwargs)
            text = r.choices[0].message.content or ""
            if top_logprobs:
                lp = r.choices[0].logprobs
                first = lp.content[0].top_logprobs if lp and lp.content else []
                dist = {}
                for t in first:  # merge tokens that differ only by whitespace
                    dist[t.token] = dist.get(t.token, 0.0) + math.exp(t.logprob)
                text = json.dumps(dist)
            u = r.usage
            cost = getattr(u, "cost", None) if u is not None else None
            if cost is None and u is not None and isinstance(getattr(u, "model_extra", None), dict):
                cost = u.model_extra.get("cost")
            usage = {"in": getattr(u, "prompt_tokens", 0) or 0,
                     "out": getattr(u, "completion_tokens", 0) or 0, "cost": cost}
        if usage["cost"] is None:
            # API providers bill tokens; Modal bills GPU time, which gpu_cost() tracks instead
            pin, pout = mcfg.get("price", [0, 0])
            usage["cost"] = (usage["in"] * pin + usage["out"] * pout) / 1e6
        return text, usage

    # ------------------------------------------------------------------ public API
    def key(self, mkey, messages, temperature, max_tokens, seed, top_logprobs=None):
        mcfg = self.models[mkey]
        return call_key(provider=mcfg["provider"], model=mcfg["id"], messages=messages,
                        temperature=temperature, max_tokens=max_tokens, seed=seed,
                        extra=mcfg.get("extra_body"), mock=self.mock,
                        **({"top_logprobs": top_logprobs} if top_logprobs else {}))

    def cached(self, mkey, messages, *, temperature, max_tokens, seed=0, top_logprobs=None):
        """The cached reply for exactly this call, or None. Never calls a model."""
        hit = self.cache.get(self.key(mkey, messages, temperature, max_tokens, seed, top_logprobs))
        return None if hit is None else hit[0]

    async def chat(self, mkey, messages, *, temperature, max_tokens, seed=0, tag="misc", top_logprobs=None):
        """Return (text, token_cost_usd_of_this_call). Cached calls cost 0."""
        mcfg = self.models[mkey]
        key = self.key(mkey, messages, temperature, max_tokens, seed, top_logprobs)
        hit = self.cache.get(key)
        stats = self.by_tag.setdefault(tag, [0, 0.0, 0, 0])
        if hit is not None:
            self.n_cached += 1
            return hit[0], 0.0
        if self.total_spent() >= self.budget:
            raise BudgetExceeded(f"spent ${self.total_spent():.2f} >= budget ${self.budget:.2f}")
        async with self.sem:
            span = self.gpu_span.setdefault(mkey, [time.time(), time.time()])
            last_err = None
            # total wait per call is bounded, so a model that never comes up fails instead of hanging
            deadline = time.time() + (self.cfg.get("modal", {}).get("request_timeout", 1500)
                                      if mcfg["provider"] == "modal" else 600)
            text = None
            for attempt in range(6):
                if attempt and time.time() > deadline:
                    break
                try:
                    text, usage = await self._raw_call(mkey, messages, temperature, max_tokens, seed,
                                                       top_logprobs)
                    span[1] = time.time()
                    break
                except SetupError:
                    raise
                except Exception as e:  # rate limits, 5xx, timeouts, cold starts
                    last_err = e
                    status = getattr(e, "status_code", None)
                    if status in (400, 401, 403, 404):
                        raise
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
            if text is None:
                raise RuntimeError(f"{mkey}: call failed after retries: {last_err!r}")
        self.cache.put(key, text, {**usage, "model": mcfg["id"], "tag": tag, "t": time.time()})
        self.spent += usage["cost"]
        self.n_calls += 1
        stats[0] += 1
        stats[1] += usage["cost"]
        stats[2] += usage["in"]
        stats[3] += usage["out"]
        if self.total_spent() > self.budget:
            raise BudgetExceeded(f"spent ${self.total_spent():.2f} > budget ${self.budget:.2f}")
        return text, usage["cost"]

    async def token_dist(self, mkey, messages, *, seed=0, tag="misc", top_logprobs=20):
        """Probability of each candidate first token (greedy, one token). Returns ({token: prob}, cost).
        Scoring from token probabilities means a model cannot hedge with a stated 0.5."""
        text, cost = await self.chat(mkey, messages, temperature=0.0, max_tokens=1, seed=seed, tag=tag,
                                     top_logprobs=top_logprobs)
        try:
            return json.loads(text), cost
        except json.JSONDecodeError:
            return {}, cost

    async def chat_json(self, mkey, messages, *, validate, temperature, max_tokens, seed=0, tag="misc"):
        """Structured call: parse JSON, retry once on failure, else return (None, cost). Never imputes."""
        total = 0.0
        for attempt in range(2):
            msgs = messages if attempt == 0 else messages + [
                {"role": "user", "content": "Reply with only the JSON object in the required format."}]
            text, cost = await self.chat(mkey, msgs, temperature=temperature, max_tokens=max_tokens,
                                         seed=seed, tag=tag)
            total += cost
            obj = extract_json(text)
            try:
                if obj is not None:
                    return validate(obj), total
            except (KeyError, TypeError, ValueError):
                pass
        return None, total

    def gpu_cost(self):
        """Estimated Modal GPU spend: time each model was in use (plus its idle scaledown) x hourly rate.
        An upper bound for one script: it ignores that models running in parallel share wall time only
        when they are actually busy."""
        if self.mock:
            return {}
        idle = self.cfg.get("modal", {}).get("scaledown_window", 300)
        return {k: (b - a + idle) / 3600 * self.models[k].get("gpu_usd_per_hour", 0)
                for k, (a, b) in self.gpu_span.items() if self.models[k]["provider"] == "modal"}

    def total_spent(self):
        return self.spent + sum(self.gpu_cost().values())

    def report(self):
        gpu = self.gpu_cost()
        lines = [f"[cost] run={self.run_name} new_calls={self.n_calls} cached={self.n_cached} "
                 f"spent=${self.total_spent():.4f} (tokens ${self.spent:.4f}, est. Modal GPU "
                 f"${sum(gpu.values()):.4f}) budget=${self.budget:.2f}"]
        for k, c in sorted(gpu.items()):
            a, b = self.gpu_span[k]
            lines.append(f"  gpu {k:<16} {self.models[k].get('gpu', ''):<6} in use {(b - a) / 60:6.1f} min  ~${c:.3f}")
        for tag, (n, c, i, o) in sorted(self.by_tag.items()):
            if n:
                lines.append(f"  {tag:<22} calls={n:<6} ${c:.4f}  ({i / n:.0f} in / {o / n:.0f} out tok per call)")
        return "\n".join(lines)
