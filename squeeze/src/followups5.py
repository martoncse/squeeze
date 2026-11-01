"""Rescore decoder results with each decoder's P(tier A) averaged over the four labelings (original, letters
swapped, order reversed, both), so a decoder's letter and position bias cancels.

    python -m src.followups5 --scope exp1,qwen14,presence --estimate   # count uncached calls and price each scope
    python -m src.followups5 --scope exp1,qwen14 [--budget USD]        # decode the missing labelings, then rescore
"""
import asyncio
import json
from collections import Counter, defaultdict

import numpy as np

from . import prompts as PR
from .common import Ctx, base_args, read_jsonl, run_stage
from .exp1_matrix import write_csv
from .followups import schelling_texts
from .followups2 import blocked_drafts, ext16, loop_texts, price, scrub
from .followups4 import QWEN14_READERS, VARIANTS
from .metrics import TemplateBoot

EXP1_CONDS = ["honest", "schelling", "codebook"]


def text_index(ctx):
    """(sender, condition, cv_id, paraphraser) -> row with text, for every test-split text that was decoded."""
    idx = {}
    add = lambda r, v: idx.setdefault((r["sender"], r["condition"], r["cv_id"], v), {**r, "paraphraser": v})
    summ = read_jsonl(ctx.raw / "summaries.jsonl")
    s16 = {(t["sender"], t["cv_id"]): t for t in schelling_texts(ctx, 16, ext16(ctx))}
    for r in summ:
        if r["condition"] == "schelling" and (r["sender"], r["cv_id"]) in s16:
            r = s16[(r["sender"], r["cv_id"])]
        if not r["leak_dropped"]:
            add(r, "none")
    for r in read_jsonl(ctx.raw / "paraphrases.jsonl"):
        if not r["leak_dropped"]:
            add(r, r["paraphraser"])
    blind = {(r["sender"], r["cv_id"]): r for r in read_jsonl(ctx.raw / "summaries_blind8.jsonl")}
    fb = {(r["sender"], r["cv_id"]): r for r in read_jsonl(ctx.raw / "summaries_feedback8.jsonl")}
    loops = loop_texts([r for r in summ if r["sender"] in ("falcon", "granite", "qwen") and r["condition"] == "schelling"],
                       blind, fb)
    for v, ts in loops.items():
        for t in ts:
            if not t["leak_dropped"]:
                add(t, v)
    for n in (1, 2):
        for r in read_jsonl(ctx.raw / f"summaries_bl{n}.jsonl"):
            if not r["leak_dropped"]:
                add(r, f"bl{n}")
    bd, _ = blocked_drafts(ctx)
    for cv, d in bd.items():
        for v, src in (("redact_first", d["first"]), ("redact_last", d["last"])):
            add({**d["row"], "text": scrub(src)}, v)
    return {k: v for k, v in idx.items() if v["split"] == "test"}


def complete(rows):
    tids = Counter(r["template_id"] for r in rows)
    return [r for r in rows if tids[r["template_id"]] == 2]


def scope_texts(ctx, scope, idx):
    """{scope: (texts, decoders_for)} - the texts each analysis needs, complete pairs only."""
    S = ctx.cfg["senders"]
    if scope == "exp1":  # the original Exp 1 texts (rewrite budget 4)
        rows = [r for r in read_jsonl(ctx.raw / "summaries.jsonl") if r["split"] == "test" and r["sender"] in S
                and r["condition"] in EXP1_CONDS and not r["leak_dropped"]]
        groups = defaultdict(list)
        for r in rows:
            groups[(r["sender"], r["condition"])].append({**r, "paraphraser": "none_b4"})
        return [t for g in groups.values() for t in complete(g)], lambda t: S
    if scope == "qwen14":
        rows = [r for r in read_jsonl(ctx.raw / "summaries.jsonl") if r["split"] == "test" and r["sender"] == "qwen_big"
                and not r["leak_dropped"]]
        groups = defaultdict(list)
        for r in rows:
            groups[r["condition"]].append({**r, "paraphraser": "none"})
        return [t for g in groups.values() for t in complete(g)], lambda t: QWEN14_READERS
    if scope in ("presence", "presence_core"):  # core: as-run texts only, no paraphrase/rewrite variants
        groups = defaultdict(list)
        for (s, c, cv, v), r in idx.items():
            if s in S and (scope == "presence" or v == "none"):
                groups[(s, c, v)].append(r)
        return [t for g in groups.values() for t in complete(g)], lambda t: S
    raise ValueError(scope)


def msgs(t, d, v):
    cb = t["condition"] if (d == t["sender"] and t["condition"] in PR.CODEBOOKS) else None
    return PR.decoder_variant_messages(v, t["text"], cb)


def is_cached(ctx, t, d, v):
    return ctx.llm.cached(d, msgs(t, d, v), temperature=0.0, max_tokens=1, seed=0, top_logprobs=20) is not None


async def decode_all(ctx, texts, decoders_for, stage):
    """Every (text, decoder, labeling). Identical prompts are cache hits, so overlap between scopes is free."""
    items = [{"sender": t["sender"], "condition": t["condition"], "cv_id": t["cv_id"], "paraphraser": t["paraphraser"],
              "template_id": t["template_id"], "Y": t["Y"], "decoder": d, "variant": v, "_t": t}
             for t in texts for d in decoders_for(t) for v in VARIANTS]

    async def work(it):
        dist, _ = await ctx.llm.token_dist(it["decoder"], msgs(it["_t"], it["decoder"], it["variant"]),
                                           tag=f"decode:{it['decoder']}")
        return {"p_tier_a": PR.score_variant(it["variant"], dist)}

    keys = ("sender", "condition", "cv_id", "paraphraser", "decoder", "variant")
    rows = await run_stage(ctx, stage, items, keys, work)
    want = {tuple(i[k] for k in keys) for i in items}
    return [r for r in rows if tuple(r[k] for k in keys) in want]


def pair_outcomes(rows, how):
    """how = 'original' (the main runs' labeling) or 'avg4' (mean P(tier A) over the four labelings)."""
    P = defaultdict(dict)
    for r in rows:
        P[(r["sender"], r["condition"], r["paraphraser"], r["decoder"], r["cv_id"])][r["variant"]] = r["p_tier_a"]

    def score(k):
        v = P.get(k, {})
        if how == "original":
            return v.get("decoder_lp")
        vals = [v.get(x) for x in VARIANTS]
        return None if any(x is None for x in vals) else float(np.mean(vals))

    out = []
    for (s, c, para, d, cv) in {k for k in P if k[-1].endswith("_A")}:
        tid = cv[:-2]
        a, b = score((s, c, para, d, f"{tid}_A")), score((s, c, para, d, f"{tid}_B"))
        if a is not None and b is not None:
            out.append({"sender": s, "condition": c, "paraphraser": para, "decoder": d, "template_id": tid,
                        "correct": 1.0 if a > b else 0.5 if a == b else 0.0})
    return out


def cell_table(ctx, rows, keyf, decoders, self_of=lambda k: k[0]):
    """Leakage per (group, decoder) and self-minus-others gap, before (original labeling) and after (avg of 4)."""
    out = []
    pairs = {h: pair_outcomes(rows, h) for h in ("original", "avg4")}
    groups = sorted({keyf(p) for p in pairs["original"]})
    for g in groups:
        row = {"group": " / ".join(map(str, g))}
        for h in ("original", "avg4"):
            ps = [p for p in pairs[h] if keyf(p) == g]
            boot = TemplateBoot(sorted({p["template_id"] for p in ps}), ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
            row["pairs"] = len({p["template_id"] for p in ps})
            for d in decoders:
                sel = [p for p in ps if p["decoder"] == d]
                row[f"{h}:{d}"] = f"{boot.leakage(sel).point:+.2f}" if sel else ""
            me = self_of(g)
            selfs = [p for p in ps if p["decoder"] == me]
            others = [p for p in ps if p["decoder"] != me and ctx.cfg["models"][p["decoder"]]["family"]
                      != ctx.cfg["models"][me]["family"]]
            if selfs and others:
                gap = boot.leakage(selfs) - boot.leakage(others)
                row[f"{h}:gap"] = gap.fmt()
                row[f"{h}:gap_sig"] = gap.excludes_zero()
            pooled = [p for p in ps if p["decoder"] in ctx.cfg["senders"]]
            row[f"{h}:presence"] = f"{abs(boot.leakage(pooled).point):.2f}"
        out.append(row)
    return out


async def main():
    ap = base_args(__doc__)
    ap.add_argument("--scope", default="exp1,qwen14,presence")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--bh", action="store_true", help="BH on the averaged-labeling privacy gaps (no model calls)")
    args = ap.parse_args()
    ctx = Ctx(args, f"followup5{'_estimate' if args.estimate else ''}{'_bh' if args.bh else ''}")
    if args.bh:
        run_bh(ctx)
        ctx.finish()
        return
    scopes = args.scope.split(",")
    idx = text_index(ctx) if any(sc.startswith("presence") for sc in scopes) else {}
    seen, total = set(), Counter()
    for sc in scopes:
        texts, dec_for = scope_texts(ctx, sc, idx)
        new = Counter()
        for t in texts:
            for d in dec_for(t):
                for v in VARIANTS:
                    k = (d, t["text"], t["condition"] if (d == t["sender"] and t["condition"] in PR.CODEBOOKS) else "", v)
                    if k in seen:
                        continue
                    seen.add(k)
                    if not is_cached(ctx, t, d, v):
                        new[d] += 1
        print(f"[{sc}] {len(texts)} texts ({len(texts) // 2} pairs); new decodes beyond earlier scopes: {dict(new)}")
        if args.estimate:
            price(ctx, f"rescoring scope {sc}", {}, dict(new))
        total.update(new)
    if args.estimate:
        price(ctx, "all requested scopes together", {}, dict(total))
        ctx.finish()
        return
    S = ctx.cfg["senders"]
    for sc in scopes:
        texts, dec_for = scope_texts(ctx, sc, idx)
        rows = await decode_all(ctx, texts, dec_for, f"decodes_avg4")
        if sc == "exp1":
            tab = cell_table(ctx, rows, lambda p: (p["sender"], p["condition"]), S)
        elif sc == "qwen14":
            tab = cell_table(ctx, rows, lambda p: ("qwen_big", p["condition"]), QWEN14_READERS)
        else:  # presence and presence_core
            tab = cell_table(ctx, rows, lambda p: (p["sender"], p["condition"], p["paraphraser"]), S)
        write_csv(ctx.tables / f"followup5_avg4_{sc}.csv", tab, sorted({k for r in tab for k in r},
                  key=lambda k: (k not in ("group", "pairs"), k)))
    ctx.finish()



# ------------------------------------------------------------------------------------------- BH on averaged gaps
def gap_tests(ctx, rows, keyf, self_of):
    """Privacy gap (self minus cross-family readers) per group, before and after averaging, with bootstrap p."""
    from .followups4 import boot_p

    models = ctx.cfg["models"]
    pairs = {h: pair_outcomes(rows, h) for h in ("original", "avg4")}
    out = []
    for g in sorted({keyf(p) for p in pairs["avg4"]}):
        me = self_of(g)
        rec = {"group": " / ".join(map(str, g))}
        for h in ("original", "avg4"):
            ps = [p for p in pairs[h] if keyf(p) == g]
            selfs = [p for p in ps if p["decoder"] == me]
            others = [p for p in ps if p["decoder"] != me and models[p["decoder"]]["family"] != models[me]["family"]]
            if not selfs or not others:
                continue
            boot = TemplateBoot(sorted({p["template_id"] for p in ps}), ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
            gap = boot.leakage(selfs) - boot.leakage(others)
            pooled = boot.leakage([p for p in ps if p["decoder"] in ctx.cfg["senders"]])
            rec.update({"pairs": len({p["template_id"] for p in selfs}), f"{h}_gap": round(gap.point, 3),
                        f"{h}_lo": round(gap.ci()[0], 3), f"{h}_hi": round(gap.ci()[1], 3), f"{h}_p": boot_p(gap.boot),
                        f"{h}_presence": round(abs(pooled.point), 3)})
        if "avg4_gap" in rec and "original_gap" in rec:
            out.append(rec)
    return out


def run_bh(ctx):
    from .followups4 import bh

    rows = read_jsonl(ctx.raw / "decodes_avg4.jsonl")
    S = ctx.cfg["senders"]
    fam = {
        "exp1": gap_tests(ctx, [r for r in rows if r["paraphraser"] == "none_b4"], lambda p: (p["sender"], p["condition"]),
                          lambda g: g[0]),
        "qwen14": gap_tests(ctx, [r for r in rows if r["sender"] == "qwen_big"], lambda p: ("qwen_big", p["condition"]),
                            lambda g: g[0]),
        "presence": gap_tests(ctx, [r for r in rows if r["sender"] in S and r["paraphraser"] != "none_b4"],
                              lambda p: (p["sender"], p["condition"], p["paraphraser"]), lambda g: g[0]),
    }
    out, summ = [], {}
    for f, tests in fam.items():
        for h in ("original", "avg4"):
            rej, adj = bh([t[f"{h}_p"] for t in tests])
            for t, r, a in zip(tests, rej, adj):
                t[f"{h}_bh_adj_p"], t[f"{h}_bh_survives"] = round(float(a), 4), bool(r)
        summ[f] = {"tests": len(tests),
                   **{f"{h}_{k}": int(sum(bool(v) for v in (t[f"{h}_{kk}"] < 0.05 if kk == "p" else t[f"{h}_{kk}"]
                                                             for t in tests)))
                      for h in ("original", "avg4") for k, kk in (("nominal", "p"), ("bh", "bh_survives"))},
                   "avg4_bh_survivors": [f"{t['group']}: {t['avg4_gap']:+.3f} [{t['avg4_lo']:+.3f}, {t['avg4_hi']:+.3f}] "
                                         f"(n {t['pairs']}, adj p {t['avg4_bh_adj_p']})" for t in tests if t["avg4_bh_survives"]]}
        out += [{"family": f, **t} for t in tests]
    write_csv(ctx.tables / "followup5_bh.csv", out, sorted({k for r in out for k in r},
              key=lambda k: (k not in ("family", "group", "pairs"), k)))
    (ctx.res / "followup5_bh_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
