"""Config, paths, JSONL I/O, the resumable stage runner and the leak filter."""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import yaml

from .llm import LLM, BudgetExceeded

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(line_buffering=True)  # progress shows up live when output goes to a log file


def load_config(path=None):
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def base_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="offline mock models; writes to results_mock/")
    ap.add_argument("--limit-templates", type=int, default=None,
                    help="use only the first N templates of each split (smoke tests)")
    ap.add_argument("--budget", type=float, default=None, help="override budget_usd for this run")
    ap.add_argument("--analyze-only", action="store_true", help="skip API calls, rebuild tables/figures")
    return ap


class Ctx:
    """Everything a script needs: config, client, directories, dataset."""

    def __init__(self, args, run_name):
        self.args = args
        self.cfg = load_config(args.config)
        if args.budget is not None:
            self.cfg["budget_usd"] = args.budget
        self.mock = args.mock
        self.res = ROOT / ("results_mock" if args.mock else "results")
        self.raw = self.res / "raw"
        self.tables = self.res / "tables"
        self.figs = self.res / "figures"
        for d in (self.raw, self.tables, self.figs):
            d.mkdir(parents=True, exist_ok=True)
        self.data = ROOT / ("data_mock" if args.mock else "data")
        self.llm = LLM(self.cfg, ROOT, run_name, mock=args.mock)
        self.run_name = run_name
        self.t0 = time.time()

    def cvs(self, split=None, limit=None):
        rows = read_jsonl(self.data / "cv_pairs.jsonl")
        if not rows:
            sys.exit(f"no dataset at {self.data}; run `python -m src.gen_dataset{' --mock' if self.mock else ''}` first")
        limit = limit if limit is not None else self.args.limit_templates
        out = []
        for s in ([split] if isinstance(split, str) else split or ["test", "heldout"]):
            part = [r for r in rows if r["split"] == s]
            tids = sorted({r["template_id"] for r in part})
            if limit:
                tids = set(tids[:limit])
            out += [r for r in part if r["template_id"] in tids]
        return out

    def finish(self):
        print(self.llm.report())
        led = self.res / "cost_ledger.jsonl"
        append_jsonl(led, {"run": self.run_name, "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "spent_usd": round(self.llm.total_spent(), 5),
                           "gpu_usd_est": {k: round(v, 4) for k, v in self.llm.gpu_cost().items()},
                           "new_calls": self.llm.n_calls, "cached_calls": self.llm.n_cached,
                           "by_tag": {k: [v[0], round(v[1], 5)] for k, v in self.llm.by_tag.items()},
                           "wall_s": round(time.time() - self.t0, 1)})
        total = sum(r["spent_usd"] for r in read_jsonl(led))
        print(f"[cost] all runs so far in {led.parent.name}/: ${total:.4f}")


# ---------------------------------------------------------------- JSONL
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def index(rows, keys):
    return {tuple(r[k] for k in keys): r for r in rows}


async def run_stage(ctx, name, items, keys, work):
    """Resumable stage: skip items whose key is already in raw/<name>.jsonl, run the rest
    concurrently, append each finished row immediately. `work(item)` returns a dict of new fields."""
    path = ctx.raw / f"{name}.jsonl"
    done = {tuple(r[k] for k in keys) for r in read_jsonl(path)}
    seen, todo = set(), []
    for it in items:
        k = tuple(it[k] for k in keys)
        if k not in done and k not in seen:
            seen.add(k)
            todo.append(it)
    print(f"[{name}] {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return read_jsonl(path)

    async def one(it):
        out = await work(it)
        return None if out is None else {**{k: it[k] for k in it if not k.startswith("_")}, **out}

    tasks = [asyncio.ensure_future(one(it)) for it in todo]
    n, step, err = 0, max(1, len(todo) // 10), None
    try:
        for fut in asyncio.as_completed(tasks):
            row = await fut
            if row is not None:
                append_jsonl(path, row)
            n += 1
            if n % step == 0 or n == len(todo):
                print(f"[{name}] {n}/{len(todo)}  spent=${ctx.llm.total_spent():.3f}", flush=True)
    except BudgetExceeded as e:
        err = e
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"[{name}] ABORTED: {e}. Finished rows are saved; rerun with a higher budget to resume.")
        ctx.finish()
        sys.exit(2)
    return read_jsonl(path)


# ---------------------------------------------------------------- leak filter
def load_universities():
    with open(ROOT / "data" / "universities.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LeakFilter:
    def __init__(self):
        u = load_universities()
        self.tier_a, self.tier_b = u["tier_a"], u["tier_b"]
        pats = [re.escape(x) for x in self.tier_a + self.tier_b] + u["leak_patterns"]
        # leak_patterns are regexes; ETH/MIT need case-sensitive matching to avoid "method", "Seth"
        self.cs = re.compile(r"\b(" + "|".join(p for p in pats if p in ("MIT", "ETH")) + r")\b")
        self.ci = re.compile(r"\b(" + "|".join(p for p in pats if p not in ("MIT", "ETH")) + r")\b", re.I)

    def match(self, text):
        m = self.cs.search(text or "") or self.ci.search(text or "")
        return m.group(0) if m else None
