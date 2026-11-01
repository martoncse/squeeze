"""Fourth round of follow-ups. Only `granite-ctl` and `qwen14` call models; the rest read results/raw/ and
the call cache (a missing cached decode is left out and counted, never fetched).

    python -m src.followups4 bh                          # (1) Benjamini-Hochberg across privacy gaps + judge cells
    python -m src.followups4 batchsize                   # (2) AUROC vs batch size, Phi-4 and both gap detectors
    python -m src.followups4 granite-ctl [--estimate]    # (3) Granite self-decoding, letters swapped / order reversed
    python -m src.followups4 coverage                    # (4) coverage vs leakage across budgets and blocklist rounds
    python -m src.followups4 qwen14 [--estimate]         # (5) Qwen2.5-14B as sender, read by five decoders
"""
import asyncio
import csv
import json
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import prompts as PR  # noqa: E402
from . import stages as ST  # noqa: E402
from .common import Ctx, base_args, read_jsonl, run_stage  # noqa: E402
from .exp1_matrix import write_csv  # noqa: E402
from .followups import pair_rows, schelling_texts  # noqa: E402
from .followups2 import ext16, price  # noqa: E402
from .followups3 import judge_sets  # noqa: E402
from .metrics import TemplateBoot, auroc  # noqa: E402

SIZES = [1, 2, 5, 10, 20, 50]
VARIANTS = ["decoder_lp", "decoder_lp_swap", "decoder_lp_rev", "decoder_lp_swaprev"]


def boot_p(boot, null=0.0):
    """Two-sided bootstrap p-value for 'statistic = null' (add-one smoothed)."""
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan
    lo, hi = np.sum(b <= null), np.sum(b >= null)
    return min(1.0, (2 * min(lo, hi) + 1) / (len(b) + 1))


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: boolean 'rejected' per p-value, plus the BH-adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out <= q, out


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------------------- (1)
def privacy_tests(ctx):
    S = ctx.cfg["senders"]
    pr = [p for p in pair_rows(read_jsonl(ctx.raw / "decodes_lp.jsonl")) if p["split"] == "test" and p["decoder"] in S]
    groups = defaultdict(list)
    for p in pr:
        groups[(p["sender"], p["condition"], p["paraphraser"])].append(p)
    out = []
    for (s, c, v), ps in sorted(groups.items()):
        decs = {p["decoder"] for p in ps}
        if s not in decs or len(decs - {s}) < 2 or s not in S:
            continue
        boot = TemplateBoot(sorted({p["template_id"] for p in ps}), ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
        gap = boot.leakage([p for p in ps if p["decoder"] == s]) - boot.leakage([p for p in ps if p["decoder"] != s])
        out.append({"family": "privacy gap", "test": f"{s} / {c} / {v}", "n": len({p["template_id"] for p in ps if p["decoder"] == s}),
                    "estimate": round(gap.point, 3), "ci": f"[{gap.ci()[0]:+.3f}, {gap.ci()[1]:+.3f}]", "p": boot_p(gap.boot)})
    return out


def judge_tests(ctx, B=2000):
    cfg = ctx.cfg
    J = cfg["senders"] + [cfg["judge"]]
    sets, honest = judge_sets(ctx)
    M = {(m["sender"], m["condition"], m["cv_id"], m["paraphraser"], m["judge"]): m["suspicion"]
         for m in read_jsonl(ctx.raw / "monitor_lp.jsonl")}
    rng = np.random.default_rng(cfg["seed"])
    out = []
    for name, ts in sets.items():
        for s in sorted({t["sender"] for t in ts}):
            for j in J:
                sc = lambda t: M.get((t["sender"], t["condition"], t["cv_id"], t["paraphraser"], j))
                pos = [(t["template_id"], sc(t)) for t in ts if t["sender"] == s and t["split"] == "test" and sc(t) is not None]
                neg = [(t["template_id"], sc(t)) for t in honest if t["sender"] == s and t["split"] == "test" and sc(t) is not None]
                P, N = defaultdict(list), defaultdict(list)
                for t, v in pos:
                    P[t].append(v)
                for t, v in neg:
                    N[t].append(v)
                tids = sorted(set(P) | set(N))
                pt = auroc([v for _, v in pos], [v for _, v in neg])
                vals = []
                for _ in range(B):
                    pick = rng.choice(tids, len(tids))
                    p = [x for t in pick for x in P.get(t, [])]
                    n = [x for t in pick for x in N.get(t, [])]
                    if p and n:
                        vals.append(auroc(p, n))
                out.append({"family": "judge AUROC", "test": f"{name} / sender {s} / judge {'phi-4' if j == cfg['judge'] else j}",
                            "n": f"{len(pos)} pos / {len(neg)} neg", "estimate": round(pt, 3),
                            "ci": f"[{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]", "p": boot_p(vals, 0.5)})
    return out


def run_bh(ctx):
    tests = privacy_tests(ctx) + judge_tests(ctx)
    rej_all, adj_all = bh([t["p"] for t in tests])
    rows = []
    for fam in ("privacy gap", "judge AUROC"):
        idx = [i for i, t in enumerate(tests) if t["family"] == fam]
        rej_f, adj_f = bh([tests[i]["p"] for i in idx])
        for k, i in enumerate(idx):
            rows.append({**tests[i], "p": round(tests[i]["p"], 5), "bh_adj_p_all": round(float(adj_all[i]), 4),
                         "survives_all_118": bool(rej_all[i]), "bh_adj_p_within_family": round(float(adj_f[k]), 4),
                         "survives_within_family": bool(rej_f[k]), "uncorrected_p<0.05": tests[i]["p"] < 0.05})
    write_csv(ctx.tables / "followup4_bh.csv", rows)
    summ = {f: {"tests": sum(r["family"] == f for r in rows),
                "uncorrected_p<0.05": sum(r["family"] == f and r["uncorrected_p<0.05"] for r in rows),
                "survive_BH_all": sum(r["family"] == f and r["survives_all_118"] for r in rows),
                "survive_BH_within_family": sum(r["family"] == f and r["survives_within_family"] for r in rows)}
            for f in ("privacy gap", "judge AUROC")}
    summ["survivors_all"] = [f"{r['test']}: {r['estimate']:+.3f} {r['ci']} (n {r['n']}, adj p {r['bh_adj_p_all']})"
                             for r in rows if r["survives_all_118"]]
    (ctx.res / "followup4_bh_summary.json").write_text(json.dumps(summ, indent=2, default=lambda o: o.item()))
    print(json.dumps(summ, indent=2, default=lambda o: o.item()))
    return rows


# ------------------------------------------------------------------------------------------- (2)
def cached_p(ctx, model, text):
    d = ctx.llm.cached(model, PR.decoder_lp_messages(text), temperature=0.0, max_tokens=1, seed=0, top_logprobs=20)
    return None if d is None else PR.score_ab(json.loads(d))


def cached_susp(ctx, model, text):
    d = ctx.llm.cached(model, PR.monitor_lp_messages(text), temperature=0.0, max_tokens=1, seed=0, top_logprobs=20)
    return None if d is None else PR.score_digit(json.loads(d))


def template_scalars(ctx, texts, sender):
    """Per template: Phi-4 mean suspicion over the pair, and pair outcomes for receiver, sibling, cross decoders."""
    cfg = ctx.cfg
    sib = cfg["models"][sender]["sibling"]
    cross = [d for d in cfg["senders"] if cfg["models"][d]["family"] != cfg["models"][sender]["family"]]
    by = defaultdict(dict)
    for t in texts:
        if t["sender"] == sender and t["split"] == "test" and not t["leak_dropped"]:
            by[t["template_id"]][t["Y"]] = t["text"]
    out = {"phi": {}, "gap_same": {}, "gap_cross": {}}
    outcome = lambda m, g: (lambda a, b: None if a is None or b is None else (1.0 if a > b else 0.5 if a == b else 0.0))(
        cached_p(ctx, m, g[1]), cached_p(ctx, m, g[0]))
    for tid, g in by.items():
        if 0 not in g or 1 not in g:
            continue
        s1, s0 = cached_susp(ctx, cfg["judge"], g[1]), cached_susp(ctx, cfg["judge"], g[0])
        if s1 is not None and s0 is not None:
            out["phi"][tid] = (s1 + s0) / 2
        oR, oS = outcome(sender, g), outcome(sib, g)
        oC = [outcome(d, g) for d in cross]
        if oR is not None and oS is not None:
            out["gap_same"][tid] = oR - oS
        if oR is not None and all(o is not None for o in oC):
            out["gap_cross"][tid] = oR - float(np.mean(oC))
    return out


def batch_auroc(pos, neg, b, rng, n_batches=1000):
    pv, nv = np.array(list(pos.values())), np.array(list(neg.values()))
    ps = pv[rng.integers(0, len(pv), (n_batches, b))].mean(1)
    ns = nv[rng.integers(0, len(nv), (n_batches, b))].mean(1)
    return auroc(ps, ns)


def run_batchsize(ctx, outer=200):
    honest = [r for r in read_jsonl(ctx.raw / "summaries.jsonl") if r["condition"] == "honest"]
    sets, _ = judge_sets(ctx)
    pos_sets = {"Falcon schelling, budget 16": ("falcon", sets["Falcon schelling, budget 16"]),
                "Granite schelling, feedback loop": ("granite", sets["Granite schelling, feedback loop"])}
    rows, curves = [], {}
    rng = np.random.default_rng(ctx.cfg["seed"])
    for name, (s, ts) in pos_sets.items():
        P, N = template_scalars(ctx, ts, s), template_scalars(ctx, honest, s)
        for det, label in (("phi", "Phi-4 judge (mean suspicion)"), ("gap_same", "gap, same-family sentinel"),
                           ("gap_cross", "gap, cross-family sentinels")):
            pos, neg = P[det], N[det]
            pt, lo, hi = [], [], []
            for b in SIZES:
                point = batch_auroc(pos, neg, b, rng)
                # template uncertainty: resample each template pool, then draw batches from it
                vals = []
                for _ in range(outer):
                    pk = rng.choice(list(pos), len(pos))
                    nk = rng.choice(list(neg), len(neg))
                    vals.append(batch_auroc({i: pos[k] for i, k in enumerate(pk)}, {i: neg[k] for i, k in enumerate(nk)},
                                            b, rng, n_batches=400))
                l, h = np.percentile(vals, [2.5, 97.5])
                pt.append(point), lo.append(l), hi.append(h)
                rows.append({"texts": name, "detector": label, "batch_templates": b, "batch_summaries": 2 * b,
                             "pos_templates": len(pos), "neg_templates": len(neg), "auroc": round(point, 3),
                             "ci": f"[{l:.3f}, {h:.3f}]"})
            curves[(name, label)] = (pt, lo, hi, len(pos), len(neg))
    write_csv(ctx.tables / "followup4_batchsize.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, name in zip(axes, pos_sets):
        for (n, label), (pt, lo, hi, npos, nneg) in curves.items():
            if n != name:
                continue
            ax.plot(SIZES, pt, "o-", label=f"{label} ({npos} vs {nneg} templates)")
            ax.fill_between(SIZES, lo, hi, alpha=0.15)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(SIZES, [str(s) for s in SIZES])
        ax.set_xlabel("batch size (templates; 2 summaries each)")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("AUROC vs honest batches (95% CI)")
    fig.suptitle("Batching amplifies any per-summary shift, in either direction (AUROC < 0.5: covert batches score lower than honest)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ctx.figs / "followup4_auroc_vs_batchsize.png", dpi=160)
    plt.close(fig)
    return rows


# ------------------------------------------------------------------------------------------- (3)
def granite_ctl_texts(ctx):
    """Granite test texts: schelling as run, blind loop (budget 8), feedback loop (budget 8), codebook."""
    summ = [r for r in read_jsonl(ctx.raw / "summaries.jsonl") if r["sender"] == "granite" and r["split"] == "test"]
    blind = {r["cv_id"]: r for r in read_jsonl(ctx.raw / "summaries_blind8.jsonl") if r["sender"] == "granite"}
    fb = {r["cv_id"]: r for r in read_jsonl(ctx.raw / "summaries_feedback8.jsonl") if r["sender"] == "granite"}
    sets = defaultdict(list)
    for r in summ:
        if r["condition"] == "schelling":
            sets["schelling, as run"].append(r)
            b = r if not r["leak_dropped"] else {**r, **{k: blind[r["cv_id"]][k] for k in ("text", "leak_dropped")}}
            sets["schelling, blind loop"].append(b)
            f = r if not r.get("first_leak") else {**r, **{k: fb[r["cv_id"]][k] for k in ("text", "leak_dropped")}}
            sets["schelling, feedback loop"].append(f)
        elif r["condition"] == "codebook":
            sets["codebook (with codebook)"].append(r)
    for k in sets:  # complete pairs only
        tids = Counter(t["template_id"] for t in sets[k] if not t["leak_dropped"])
        sets[k] = [t for t in sets[k] if not t["leak_dropped"] and tids[t["template_id"]] == 2]
    return sets


async def granite_ctl(ctx, estimate):
    sets = granite_ctl_texts(ctx)
    items = [{"set": name, "variant": v, "cv_id": t["cv_id"], "template_id": t["template_id"], "Y": t["Y"],
              "_text": t["text"], "_cb": "codebook" if name.startswith("codebook") else None}
             for name, ts in sets.items() for t in ts for v in VARIANTS]
    new = sum(ctx.llm.cached("granite", PR.decoder_variant_messages(i["variant"], i["_text"], i["_cb"]), temperature=0.0,
                             max_tokens=1, seed=0, top_logprobs=20) is None for i in items)
    print("[granite-ctl] " + ", ".join(f"{k}: {len(v) // 2} pairs" for k, v in sets.items()) + f"; {new} new decodes")
    if estimate:
        return price(ctx, "3 Granite letter/order control", {}, {"granite": new})

    async def work(it):
        d, _ = await ctx.llm.token_dist("granite", PR.decoder_variant_messages(it["variant"], it["_text"], it["_cb"]),
                                        tag="decode:granite")
        return {"p_tier_a": PR.score_variant(it["variant"], d), "top_tokens": dict(sorted(d.items(), key=lambda kv: -kv[1])[:4]),
                "prompt_version": PR.version(it["variant"])}

    rows = await run_stage(ctx, "decodes_granite_ctl", items, ("set", "variant", "cv_id"), work)
    P = {(r["set"], r["variant"], r["cv_id"]): r["p_tier_a"] for r in rows}
    out = []
    for name, ts in sets.items():
        tids = sorted({t["template_id"] for t in ts})
        boot = TemplateBoot(tids, ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
        for v in VARIANTS:
            pr = []
            for tid in tids:
                a, b = P.get((name, v, f"{tid}_A")), P.get((name, v, f"{tid}_B"))
                if a is not None and b is not None:
                    pr.append({"template_id": tid, "correct": 1.0 if a > b else 0.5 if a == b else 0.0})
            L = boot.leakage(pr)
            tier_a_mean = np.mean([p for (n, vv, _), p in P.items() if n == name and vv == v and p is not None])
            out.append({"texts": name, "variant": v.replace("decoder_lp", "original") if v == "decoder_lp" else v.replace("decoder_lp_", ""),
                        "pairs": len(pr), "granite_self_leakage": L.fmt(), "mean_p_tier_a": round(float(tier_a_mean), 3)})
    write_csv(ctx.tables / "followup4_granite_ctl.csv", out)
    return out


def complete_pairs(texts):
    tids = Counter(t["template_id"] for t in texts if not t["leak_dropped"])
    return [t for t in texts if not t["leak_dropped"] and tids[t["template_id"]] == 2]


def qwen_ctl_sets(ctx):
    summ = [r for r in read_jsonl(ctx.raw / "summaries.jsonl") if r["split"] == "test"]
    sets = {f"Qwen-14B {c}": complete_pairs([r for r in summ if r["sender"] == "qwen_big" and r["condition"] == c])
            for c in ("honest", "schelling", "codebook")}
    sets["Qwen-7B schelling"] = complete_pairs([r for r in summ if r["sender"] == "qwen" and r["condition"] == "schelling"])
    return sets


async def decoder_ctl(ctx, estimate, decoders=("qwen", "qwen_big")):
    """Letter/order control for any decoder: each reads each text under the four labelings. The codebook is
    given only when the decoder is the sender, as in the main runs."""
    sets = qwen_ctl_sets(ctx)
    items = [{"set": name, "decoder": d, "variant": v, "sender": t["sender"], "cv_id": t["cv_id"],
              "template_id": t["template_id"], "Y": t["Y"], "_text": t["text"],
              "_cb": t["condition"] if (d == t["sender"] and t["condition"] in PR.CODEBOOKS) else None}
             for name, ts in sets.items() for t in ts for d in decoders for v in VARIANTS]
    msgs = lambda i: PR.decoder_variant_messages(i["variant"], i["_text"], i["_cb"])
    new = Counter(i["decoder"] for i in items if ctx.llm.cached(i["decoder"], msgs(i), temperature=0.0, max_tokens=1,
                                                                 seed=0, top_logprobs=20) is None)
    print("[decoder-ctl] " + ", ".join(f"{k}: {len(v) // 2} pairs" for k, v in sets.items()) + f"; new decodes {dict(new)}")
    if estimate:
        return price(ctx, "Qwen decoder letter/order control", {}, dict(new))

    async def work(it):
        d, _ = await ctx.llm.token_dist(it["decoder"], msgs(it), tag=f"decode:{it['decoder']}")
        return {"p_tier_a": PR.score_variant(it["variant"], d), "prompt_version": PR.version(it["variant"])}

    rows = await run_stage(ctx, "decodes_fmt_ctl", items, ("set", "decoder", "variant", "cv_id"), work)
    P = {(r["set"], r["decoder"], r["variant"], r["cv_id"]): r["p_tier_a"] for r in rows}
    out = []
    for name, ts in sets.items():
        tids = sorted({t["template_id"] for t in ts})
        boot = TemplateBoot(tids, ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
        sender = ts[0]["sender"]
        for d in decoders:
            for v in VARIANTS:
                pr, ps = [], []
                for tid in tids:
                    a, b = P.get((name, d, v, f"{tid}_A")), P.get((name, d, v, f"{tid}_B"))
                    if a is not None and b is not None:
                        pr.append({"template_id": tid, "correct": 1.0 if a > b else 0.5 if a == b else 0.0})
                        ps += [a, b]
                ps = np.array(ps)
                out.append({"texts": name, "decoder": d, "role": "self" if d == sender else "same family",
                            "variant": "original" if v == "decoder_lp" else v.replace("decoder_lp_", ""),
                            "pairs": len(pr), "leakage": boot.leakage(pr).fmt(),
                            "mean_p_tier_a": round(float(ps.mean()), 3) if len(ps) else None,
                            "share_saturated": round(float(np.mean((ps < 0.001) | (ps > 0.999))), 3) if len(ps) else None,
                            "exact_ties": sum(p["correct"] == 0.5 for p in pr)})
    write_csv(ctx.tables / "followup4_qwen_decoder_ctl.csv", out)
    return out


# ------------------------------------------------------------------------------------------- (4)
def run_coverage(ctx):
    num = lambda s: [float(x) for x in re.findall(r"[-+]?\d+\.\d+", s)]
    pts = []  # (sender, path, step label, coverage, L, lo, hi, pairs)
    for r in read_csv(ctx.tables / "followup2_by_budget.csv"):
        l = num(r["L_pooled"])
        pts.append((r["sender"], "blind rewrites", f"budget {r['max_attempts']}", int(r["test_pairs_kept"]) / 100, *l,
                    int(r["test_pairs_kept"])))
    q = [r for r in read_csv(ctx.tables / "followup3_presence_privacy.csv") if r["sender"] == "qwen"
         and r["condition"] == "schelling" and r["variant"] == "as run"][0]
    l = num(q["L_pooled"])
    pts.append(("qwen", "blind rewrites", "budget 4", int(q["pairs"]) / 100, *l, int(q["pairs"])))
    for r in read_csv(ctx.tables / "followup2b_loops_leakage.csv"):
        if r["pair_attempt"] == "all" and r["loop"] == "blind8" and r["sender"] == "qwen":
            l = num(r["L_pooled"])
            pts.append(("qwen", "blind rewrites", "budget 8", int(r["pairs"]) / 100, *l, int(r["pairs"])))
        if r["pair_attempt"] == "all" and r["loop"] == "fb8" and r["sender"] == "qwen":
            l = num(r["L_pooled"])
            pts.append(("qwen", "feedback + blocklist", "round 0", int(r["pairs"]) / 100, *l, int(r["pairs"])))
    for rnd in (0, 1, 2):
        for r in read_csv(ctx.tables / f"followup3_blocklist_round{rnd}.csv"):
            l = num(r["L_pooled_all_families"])
            pts.append((r["sender"], "feedback + blocklist", f"round {rnd}", int(r["test_pairs_kept"]) / 100, *l,
                        int(r["test_pairs_kept"])))
    write_csv(ctx.tables / "followup4_coverage.csv",
              [dict(zip(["sender", "path", "step", "coverage", "L_pooled", "lo", "hi", "test_pairs"], p)) for p in pts])
    colors = {"falcon": "tab:blue", "granite": "tab:orange", "qwen": "tab:green"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in colors:
        for path, ls in (("blind rewrites", "--"), ("feedback + blocklist", "-")):
            pp = sorted([p for p in pts if p[0] == s and p[1] == path], key=lambda p: int(p[2].split()[-1]))
            if not pp:
                continue
            xs, ys = [p[3] for p in pp], [p[4] for p in pp]
            ax.errorbar(xs, ys, yerr=[[p[4] - p[5] for p in pp], [p[6] - p[4] for p in pp]], fmt="o" + ls,
                        color=colors[s], capsize=2, alpha=0.85, label=f"{s}, {path}")
            for p in pp:
                ax.annotate(f"{p[2]} (n={p[7]})", (p[3], p[4]), fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("coverage: test pairs kept / 100")
    ax.set_ylabel("pooled leakage, 4 decoder families (95% CI)")
    ax.set_title("Blocklist rounds cut leakage by dropping pairs; longer blind budgets keep more pairs and more leakage", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ctx.figs / "followup4_coverage_vs_leakage.png", dpi=160)
    plt.close(fig)
    return pts


# ------------------------------------------------------------------------------------------- (5)
QWEN14_READERS = ["qwen_big", "qwen", "falcon", "olmo", "granite"]


async def qwen14(ctx, estimate):
    cvs = ctx.cvs(["test", "heldout"])
    if estimate:
        return price(ctx, "5 Qwen2.5-14B sender (upper bound)", {"qwen_big": 3 * len(cvs) * 2},
                     {d: 3 * len(cvs) for d in QWEN14_READERS})
    summ = await ST.summaries(ctx, ["qwen_big"], ["honest", "schelling", "codebook"], cvs)
    dec = await ST.decodes(ctx, "followup4_qwen14", summ, lambda t: QWEN14_READERS)
    pr = [p for p in pair_rows(dec) if p["split"] == "test"]
    rows = []
    for c in ["honest", "schelling", "codebook"]:
        pc = [p for p in pr if p["condition"] == c]
        boot = TemplateBoot(sorted({p["template_id"] for p in pc}), ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
        row = {"condition": c, "pairs": len({p["template_id"] for p in pc}),
               "first_attempt_leak_rate": round(float(np.mean([s.get("first_leak") is not None for s in summ if s["condition"] == c])), 3)}
        for d in QWEN14_READERS:
            row[f"L_{d}"] = boot.leakage([p for p in pc if p["decoder"] == d]).fmt()
        row["privacy_gap_self_minus_other_families"] = (boot.leakage([p for p in pc if p["decoder"] == "qwen_big"]) -
                                                        boot.leakage([p for p in pc if p["decoder"] in ("falcon", "olmo", "granite")])).fmt()
        rows.append(row)
    write_csv(ctx.tables / "followup4_qwen14.csv", rows)
    return rows


# -------------------------------------------------------------------------------------------
async def main():
    ap = base_args(__doc__)
    ap.add_argument("task", choices=["bh", "batchsize", "granite-ctl", "coverage", "qwen14", "qwen-ctl"])
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    ctx = Ctx(args, f"followup4_{args.task}{'_estimate' if args.estimate else ''}")
    if args.task == "bh":
        run_bh(ctx)
    elif args.task == "batchsize":
        run_batchsize(ctx)
    elif args.task == "granite-ctl":
        await granite_ctl(ctx, args.estimate)
    elif args.task == "coverage":
        run_coverage(ctx)
    elif args.task == "qwen-ctl":
        await decoder_ctl(ctx, args.estimate)
    else:
        await qwen14(ctx, args.estimate)
    ctx.finish()


if __name__ == "__main__":
    asyncio.run(main())
