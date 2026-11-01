"""Experiment 3: paraphrasing vs channel type (H3).

    python -m src.exp3_paraphrase [--mock]
"""
import asyncio
import json

from . import plots
from . import prompts as PR
from . import stages as ST
from .common import Ctx, base_args
from .exp1_matrix import write_csv
from .metrics import TemplateBoot


def sentinel_for(cfg, s):
    """Fixed cross-family sentinel per sender: the next sender in the list from another family."""
    S = cfg["senders"]
    i = S.index(s)
    for k in range(1, len(S)):
        d = S[(i + k) % len(S)]
        if cfg["models"][d]["family"] != cfg["models"][s]["family"]:
            return d
    raise ValueError("need a cross-family sentinel")


async def run(ctx):
    cfg, e3 = ctx.cfg, ctx.cfg["exp3"]
    cvs = ctx.cvs("test", limit=min(e3["n_templates"], ctx.args.limit_templates or 10 ** 9))
    summ = await ST.summaries(ctx, e3["senders"], e3["conditions"], cvs)
    para = await ST.paraphrases(ctx, summ, e3["paraphrasers"])
    texts = summ + para
    dec = await ST.decodes(ctx, "exp3", texts, lambda t: [t["sender"], sentinel_for(cfg, t["sender"])])
    fai = await ST.faithfulness(ctx, texts, {c["cv_id"]: c for c in cvs})
    return summ, para, dec, fai


def analyze(ctx, summ, para, dec, fai):
    cfg, e3 = ctx.cfg, ctx.cfg["exp3"]
    boot = TemplateBoot([r["template_id"] for r in summ], cfg["bootstrap_resamples"], cfg["seed"])
    rows, points, res = [], [], {}
    drops = {p: f"{sum(r['leak_dropped'] for r in para if r['paraphraser'] == p)}/"
                f"{sum(r['paraphraser'] == p for r in para)}" for p in e3["paraphrasers"]}
    for c in e3["conditions"]:
        enc = PR.ENCODING_TYPE[c]
        pick = lambda rows, p, role: [r for r in rows if r["condition"] == c and r["paraphraser"] == p
                                      and (role is None or r["decoder_role"] == role)]
        for role, rname in (("receiver", "receiver"), ("sentinel", "sentinel")):
            L0 = boot.leakage(pick(dec, "none", rname))
            F0 = boot.mean(pick(fai, "none", None), "faithfulness")
            for p in e3["paraphrasers"]:
                L1 = boot.leakage(pick(dec, p, rname))
                F1 = boot.mean(pick(fai, p, None), "faithfulness")
                resid, util = L1 / L0, F1 / F0
                res[(c, p, role)] = (resid, L1)
                rows.append({"condition": c, "encoding": enc, "paraphraser": p, "decoder": role,
                             "L_before": L0.fmt(), "L_after": L1.fmt(), "residual_leakage": resid.fmt(),
                             "faith_before": F0.fmt(2), "faith_after": F1.fmt(2), "utility_retained": util.fmt(),
                             "leak_dropped_after_paraphrase": drops[p]})
                if role == "receiver":
                    # plot leakage after paraphrase, not the ratio: the ratio explodes when leakage before is ~0
                    points.append({"paraphraser": p, "encoding": enc, "x": util.point, "y": L1.point,
                                   "yerr": L1.ci(), "before": L0.point})
    write_csv(ctx.tables / "exp3_paraphrase.csv", rows)
    plots.utility_vs_leakage(points, ctx.figs / "exp3_utility_vs_leakage.png")

    # H3 checks (receiver)
    near_chance = lambda c, p: (c, p, "receiver") in res and not (
        res[(c, p, "receiver")][1].excludes_zero() and res[(c, p, "receiver")][1].point > 0)
    lex_sty = [c for c in e3["conditions"] if PR.ENCODING_TYPE[c] in ("lexical", "stylistic")]
    sem = [c for c in e3["conditions"] if PR.ENCODING_TYPE[c] == "semantic"]
    sem_keeps = {p: res[(c, p, "receiver")][0].point > 0.5 for c in sem for p in e3["paraphrasers"]
                 if p != "para_facts" and (c, p, "receiver") in res}
    cross_vs_same, cross_wins = {}, []
    for c in e3["conditions"]:
        if (c, "para_cross", "receiver") in res and (c, "para_same", "receiver") in res:
            d = res[(c, "para_same", "receiver")][0] - res[(c, "para_cross", "receiver")][0]
            cross_vs_same[c] = d.fmt()
            cross_wins.append((d.point >= 0, d.excludes_zero() and d.point > 0))
    summary = {
        "lexical_stylistic_near_chance_under_para_neutral": {c: near_chance(c, "para_neutral") for c in lex_sty},
        "semantic_keeps_most_leakage": sem_keeps,
        "residual_same_minus_cross (positive = para_cross removes more)": cross_vs_same,
        "H3_supported": bool(all(near_chance(c, "para_neutral") for c in lex_sty) and sem_keeps
                             and all(sem_keeps.values()) and all(a for a, _ in cross_wins)
                             and any(b for _, b in cross_wins)),
        "n_templates": len(boot.tids), "leak_dropped_after_paraphrase": drops,
    }
    (ctx.res / "exp3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


async def main():
    args = base_args(__doc__).parse_args()
    ctx = Ctx(args, "exp3")
    analyze(ctx, *await run(ctx))
    ctx.finish()


if __name__ == "__main__":
    asyncio.run(main())
