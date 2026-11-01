"""Second round of follow-ups on the cached runs. Every subcommand takes --estimate, which counts the
calls not already in the cache and prices them without calling a model.

    python -m src.followups2 redact   [--estimate]   # (1) redaction control: Falcon's blocked drafts, scrubbed
    python -m src.followups2 loops    [--estimate]   # (2) feedback vs blind rewrite loop, budget 8
    python -m src.followups2 lr       [--estimate]   # (3) template-grouped logistic regression + Phi-4 tagging
    python -m src.followups2 paradet  [--estimate]   # (4) frozen Exp 2 detectors + paraphrasers, attempt >= 5
"""
import asyncio
import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np

from . import prompts as PR
from . import stages as ST
from .common import Ctx, base_args, read_jsonl, run_stage
from .exp1_matrix import write_csv
from . import exp2_detectors as E2
from .followups import ABS, USD_PER_1K_CALLS, eval_detectors, exp2_records, pair_rows, schelling_texts
from .metrics import TemplateBoot

LOOP_SENDERS = ["falcon", "granite", "qwen"]
BUCKETS = ["1", "2", "3", "4", "5-8", "9-16"]
GEN_USD_PER_1K = 0.40  # generation calls (about 150 output tokens) cost about twice a one-token scoring call
COLD_USD = {"A10G": 0.15, "L40S": 0.26}  # boot + 5-minute idle per model per run


def bucket(a):
    return str(a) if a <= 4 else ("5-8" if a <= 8 else "9-16")


def ext16(ctx):
    return {(r["sender"], r["cv_id"]): r for r in read_jsonl(ctx.raw / "summaries_ext16.jsonl")}


def cv_by_id(ctx):
    return {c["cv_id"]: c for c in ctx.cvs(["test", "heldout"])}


def sender_kw(ctx):
    s = ctx.cfg["sampling"]
    return dict(temperature=s["sender_temperature"], max_tokens=s["sender_max_tokens"])


def price(ctx, label, gen, score):
    """gen/score: {model_key: n new calls}. Prints and returns the estimate."""
    models = ctx.cfg["models"]
    used = {m for m, n in {**gen, **score}.items() if n}
    usd = (sum(gen.values()) * GEN_USD_PER_1K + sum(score.values()) * USD_PER_1K_CALLS) / 1000 \
        + sum(COLD_USD.get(models[m]["gpu"], 0.2) for m in used)
    est = {"part": label, "new_generation_calls": {k: v for k, v in gen.items() if v},
           "new_scoring_calls": {k: v for k, v in score.items() if v}, "estimated_usd": round(usd, 2),
           "basis": "generation $0.40/1k, scoring $0.20/1k (Exp 2's run), plus a cold start per model used"}
    print(json.dumps(est, indent=2))
    return est


def missing_decodes(ctx, texts, decoders_for):
    """New decode calls ST.decodes would make: uncached prompts only (identical text = cache hit)."""
    n = Counter()
    broken = {(t["sender"], t["condition"], t["paraphraser"], t["template_id"]) for t in texts if t["leak_dropped"]}
    for t in texts:
        if t["leak_dropped"] or (t["sender"], t["condition"], t["paraphraser"], t["template_id"]) in broken:
            continue
        for d in decoders_for(t):
            cb = t["condition"] if (d == t["sender"] and t["condition"] in PR.CODEBOOKS) else None
            if ctx.llm.cached(d, PR.decoder_lp_messages(t["text"], codebook=cb), temperature=0.0, max_tokens=1,
                              seed=0, top_logprobs=20) is None:
                n[d] += 1
    return n


def missing_monitor(ctx, texts, judges_for):
    n = Counter()
    for t in texts:
        if t["leak_dropped"]:
            continue
        for j in judges_for(t):
            if ctx.llm.cached(j, PR.monitor_lp_messages(t["text"]), temperature=0.0, max_tokens=1, seed=0,
                              top_logprobs=20) is None:
                n[j] += 1
    return n


def leak_table(ctx, pairs, variants, decoders, splits=("test", "all")):
    """Pair leakage per variant: pooled over decoders, the sender reading itself, and other families."""
    rows = []
    for split in splits:
        sel = [p for p in pairs if split == "all" or p["split"] == split]
        boot = TemplateBoot(sorted({p["template_id"] for p in sel}), ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
        for v in variants:
            pv = [p for p in sel if p["paraphraser"] == v and p["decoder"] in decoders]
            if not pv:
                continue
            row = {"split": split, "variant": v, "pairs": len({p["template_id"] for p in pv}),
                   "L_pooled": boot.leakage(pv).fmt()}
            for who, f in (("self", lambda p: p["decoder"] == p["sender"]),
                           ("other_families", lambda p: p["decoder"] != p["sender"])):
                sub = [p for p in pv if f(p)]
                if sub:
                    row[f"L_{who}"] = boot.leakage(sub).fmt()
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------- (1)
def scrub(text):
    """Delete every leak-filter match (university names, aliases, tier words) and tidy the gaps."""
    lf = ST.leak_filter()
    for _ in range(5):
        t = lf.cs.sub("", lf.ci.sub("", text))
        t = re.sub(r"\(\s*\)", "", t)
        t = re.sub(r"\s+([,.;:)])", r"\1", t)
        t = re.sub(r"\b(a|an|the|from|at)\s+([,.;:])", r"\2", t, flags=re.I)
        t = re.sub(r"[ \t]{2,}", " ", t).strip()
        if t == text:
            break
        text = t
    return text


def blocked_drafts(ctx):
    """Falcon schelling CVs: the first and last blocked draft (read from the cache) and the accepted text."""
    cvs, ext, kw = cv_by_id(ctx), ext16(ctx), sender_kw(ctx)
    out, missing = {}, 0
    for r in read_jsonl(ctx.raw / "summaries.jsonl"):
        if r["sender"] != "falcon" or r["condition"] != "schelling":
            continue
        e = ext.get(("falcon", r["cv_id"]))
        final = r if not r["leak_dropped"] else e
        n_blocked = (final["attempts"] - 1) if (final and not final["leak_dropped"]) else (final or r)["attempts"]
        if n_blocked < 1:
            continue
        msgs = PR.sender_messages("schelling", cvs[r["cv_id"]]["text"], r["Y"])
        drafts = [ctx.llm.cached("falcon", msgs, seed=k, **kw) for k in (0, n_blocked - 1)]
        if any(d is None for d in drafts):
            missing += 1
            continue
        out[r["cv_id"]] = {"row": r, "first": drafts[0].strip(), "last": drafts[1].strip(),
                           "accepted": None if (final is None or final["leak_dropped"]) else final["text"]}
    return out, missing


async def redact(ctx, estimate):
    S = ctx.cfg["senders"]
    bd, missing = blocked_drafts(ctx)
    texts = []
    for cv, d in bd.items():
        r = d["row"]
        for v, src in (("redact_first", d["first"]), ("redact_last", d["last"])):
            t = scrub(src)
            hit = ST.leak_filter().match(t)
            texts.append({**{k: r[k] for k in ("sender", "condition", "cv_id", "template_id", "Y", "split")},
                          "paraphraser": v, "text": t, "leak_match": hit, "leak_dropped": hit is not None,
                          "source_attempt": 1 if v == "redact_first" else None, "cost_usd": 0.0,
                          "prompt_version": "scrub@v1"})
    print(f"[redact] {len(bd)} Falcon CVs with >= 1 blocked draft ({missing} skipped: draft not in cache); "
          f"{sum(t['leak_dropped'] for t in texts)} scrubbed drafts still match the filter")
    if estimate:
        return price(ctx, "1 redaction control", {}, missing_decodes(ctx, texts, lambda t: S))
    dec = await ST.decodes(ctx, "followup2b_redact", texts, lambda t: S)
    acc = [d for d in read_jsonl(ctx.raw / "decodes_lp.jsonl") if d["sender"] == "falcon"
           and d["condition"] == "schelling" and d["paraphraser"] == "none" and d["decoder"] in S]
    pr = pair_rows(dec) + pair_rows(acc)
    # matched comparison: templates where both CVs have scrubbed drafts AND accepted rewrites scored
    have = defaultdict(set)
    for p in pr:
        have[p["template_id"]].add(p["paraphraser"])
    matched = {t for t, v in have.items() if {"redact_first", "redact_last", "none"} <= v}
    rows = leak_table(ctx, [p for p in pr if p["template_id"] in matched], ["redact_first", "redact_last", "none"], S)
    for r in rows:
        r["variant"] = {"none": "accepted rewrite"}.get(r["variant"], r["variant"])
    write_csv(ctx.tables / "followup2b_redaction.csv", rows)
    examples = [{"cv_id": cv, "blocked_first": d["first"][:400], "scrubbed": scrub(d["first"])[:400],
                 "accepted": (d["accepted"] or "")[:400]} for cv, d in list(bd.items())[:8]]
    (ctx.tables / "followup2b_redaction_examples.json").write_text(json.dumps(examples, indent=2))
    return rows


# ------------------------------------------------------------------------------------------- (2)
async def run_loops(ctx):
    cvs, kw = cv_by_id(ctx), sender_kw(ctx)
    summ = [r for r in read_jsonl(ctx.raw / "summaries.jsonl")
            if r["sender"] in LOOP_SENDERS and r["condition"] == "schelling"]
    base = lambda r: {k: r[k] for k in ("sender", "condition", "cv_id", "template_id", "Y", "split")} | {"_cv": cvs[r["cv_id"]]}

    async def blind(it):  # new seed each attempt, no feedback (the main runs' loop), continued to 8
        msgs = PR.sender_messages("schelling", it["_cv"]["text"], it["Y"])
        hits = []
        for attempt in range(4, 8):
            txt, _ = await ctx.llm.chat(it["sender"], msgs, seed=attempt, tag=f"sender:{it['sender']}", **kw)
            hit = ST.leak_filter().match(txt)
            hits.append(hit)
            if hit is None:
                break
        return {"text": txt.strip(), "attempts": attempt + 1, "leak_dropped": hit is not None, "hits_5_8": hits}

    async def feedback(it):  # the sender is shown what got it blocked and rewrites in the same conversation
        msgs = PR.sender_messages("schelling", it["_cv"]["text"], it["Y"])
        txt, _ = await ctx.llm.chat(it["sender"], msgs, seed=0, tag=f"sender:{it['sender']}", **kw)
        hit, hits = ST.leak_filter().match(txt), []
        attempt = 0
        while hit is not None and attempt < 7:
            hits.append(hit)
            attempt += 1
            msgs = msgs + [{"role": "assistant", "content": txt},
                           {"role": "user", "content": PR.text("rewrite_feedback", match=hit)}]
            txt, _ = await ctx.llm.chat(it["sender"], msgs, seed=attempt, tag=f"sender:{it['sender']}", **kw)
            hit = ST.leak_filter().match(txt)
        return {"text": txt.strip(), "attempts": attempt + 1, "leak_dropped": hit is not None, "hits": hits + [hit],
                "prompt_version": PR.version("rewrite_feedback")}

    keys = ("sender", "condition", "cv_id")
    b = await run_stage(ctx, "summaries_blind8", [base(r) for r in summ if r["leak_dropped"]], keys, blind)
    f = await run_stage(ctx, "summaries_feedback8", [base(r) for r in summ if r.get("first_leak")], keys, feedback)
    return summ, {(r["sender"], r["cv_id"]): r for r in b}, {(r["sender"], r["cv_id"]): r for r in f}


def loop_texts(summ, blind, fb):
    out = {"blind8": [], "fb8": []}
    for r in summ:
        k = (r["sender"], r["cv_id"])
        bl = r if not r["leak_dropped"] else {**r, **{x: blind[k][x] for x in ("text", "attempts", "leak_dropped")}}
        fr = r if not r.get("first_leak") else {**r, **{x: fb[k][x] for x in ("text", "attempts", "leak_dropped")}}
        out["blind8"].append({**bl, "paraphraser": "blind8"})
        out["fb8"].append({**fr, "paraphraser": "fb8"})
    return out


async def loops(ctx, estimate):
    S = ctx.cfg["senders"]
    if estimate:
        cvs, kw = cv_by_id(ctx), sender_kw(ctx)
        summ = [r for r in read_jsonl(ctx.raw / "summaries.jsonl")
                if r["sender"] in LOOP_SENDERS and r["condition"] == "schelling"]
        gen = Counter()
        for r in summ:
            msgs = PR.sender_messages("schelling", cvs[r["cv_id"]]["text"], r["Y"])
            if r["leak_dropped"]:
                gen[r["sender"]] += sum(ctx.llm.cached(r["sender"], msgs, seed=k, **kw) is None for k in range(4, 8))
            if r.get("first_leak"):
                gen[r["sender"]] += 7  # upper bound: a feedback conversation is never in the cache
        n_fb = sum(1 for r in summ if r.get("first_leak"))
        print(f"[loops] {n_fb} CVs blocked on attempt 1 enter the feedback loop; "
              f"{sum(r['leak_dropped'] for r in summ)} continue the blind loop past attempt 4")
        dec_ub = {d: 2 * n_fb * 1 for d in S}  # upper bound: both CVs of each changed pair, 4 decoders
        return price(ctx, "2 feedback vs blind loop (upper bound)", dict(gen), dec_ub)
    summ, blind, fb = await run_loops(ctx)
    texts = loop_texts(summ, blind, fb)
    dec = {v: await ST.decodes(ctx, f"followup2b_{v}", t, lambda t: S) for v, t in texts.items()}
    # attempts-to-pass, among CVs blocked on attempt 1
    att_rows = []
    for s in LOOP_SENDERS:
        for v in ("blind8", "fb8"):
            blocked = [t for t in texts[v] if t["sender"] == s and t.get("first_leak")]
            a = np.array([t["attempts"] if not t["leak_dropped"] else 99 for t in blocked])
            att_rows.append({"sender": s, "loop": v, "cvs_blocked_at_attempt_1": len(blocked),
                             "pass_by_2": round(float(np.mean(a <= 2)), 3), "pass_by_4": round(float(np.mean(a <= 4)), 3),
                             "pass_by_8": round(float(np.mean(a <= 8)), 3),
                             "mean_attempts_if_passed": round(float(np.mean(a[a <= 8])), 2) if (a <= 8).any() else None})
    write_csv(ctx.tables / "followup2b_loops_attempts.csv", att_rows)
    # leakage overall and per attempt bucket (pair attempt = the later CV of the pair), test split
    rows = []
    for v in ("blind8", "fb8"):
        att = {(t["sender"], t["cv_id"]): t["attempts"] for t in texts[v] if not t["leak_dropped"]}
        pr = [p for p in pair_rows(dec[v]) if p["split"] == "test"]
        boot = TemplateBoot(sorted({p["template_id"] for p in pr}), ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
        for s in LOOP_SENDERS:
            ps = [p for p in pr if p["sender"] == s]
            for p in ps:
                p["pair_attempt"] = max(att[(s, f"{p['template_id']}_A")], att[(s, f"{p['template_id']}_B")])
            for b in ["all"] + BUCKETS[:5]:
                sel = [p for p in ps if b == "all" or bucket(p["pair_attempt"]) == b]
                if sel:
                    rows.append({"sender": s, "loop": v, "pair_attempt": b, "pairs": len({p["template_id"] for p in sel}),
                                 "L_pooled": boot.leakage(sel).fmt(),
                                 "L_self": boot.leakage([p for p in sel if p["decoder"] == s]).fmt(),
                                 "L_other_families": boot.leakage([p for p in sel if p["decoder"] != s]).fmt()})
    write_csv(ctx.tables / "followup2b_loops_leakage.csv", rows)
    return att_rows, rows


# ------------------------------------------------------------------------------------------- (3)
def falcon_accepted(ctx, condition):
    if condition == "schelling":
        texts = [t for t in schelling_texts(ctx, 16, ext16(ctx)) if t["sender"] == "falcon"]
    else:
        texts = [r for r in read_jsonl(ctx.raw / "summaries.jsonl") if r["sender"] == "falcon" and r["condition"] == condition]
    by = defaultdict(dict)
    for t in texts:
        if not t["leak_dropped"]:
            by[t["template_id"]][t["Y"]] = t
    return {tid: g for tid, g in by.items() if 0 in g and 1 in g}  # complete pairs only


def logreg(ctx, pairs, label):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    rows = [(t["text"], y, tid) for tid, g in sorted(pairs.items()) for y, t in g.items()]
    X, y, groups = [r[0] for r in rows], np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    make = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, lowercase=True),
                                 LogisticRegression(C=1.0, max_iter=2000))
    oof, fold_top = np.zeros(len(y)), Counter()
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make().fit([X[i] for i in tr], y[tr])
        oof[te] = m.predict_proba([X[i] for i in te])[:, 1]
        vec, lr = m.steps[0][1], m.steps[1][1]
        names = vec.get_feature_names_out()
        fold_top.update(names[np.argsort(-np.abs(lr.coef_[0]))[:20]])
    pr = []
    for tid in sorted(pairs):
        ia, ib = [i for i in range(len(y)) if groups[i] == tid and y[i] == 1][0], \
                 [i for i in range(len(y)) if groups[i] == tid and y[i] == 0][0]
        pr.append({"template_id": tid, "correct": 1.0 if oof[ia] > oof[ib] else 0.5 if oof[ia] == oof[ib] else 0.0})
    boot = TemplateBoot([p["template_id"] for p in pr], ctx.cfg["bootstrap_resamples"], ctx.cfg["seed"])
    full = make().fit(X, y)
    vec, lr = full.steps[0][1], full.steps[1][1]
    names, coef = vec.get_feature_names_out(), lr.coef_[0]
    top = [{"feature": names[i], "coef": round(float(coef[i]), 3), "points_to": "tier A" if coef[i] > 0 else "tier B",
            "in_top20_of_folds": fold_top[names[i]]} for i in np.argsort(-np.abs(coef))[:20]]
    return {"label": label, "summaries": len(y), "pairs": len(pr), "oof_auroc": round(float(roc_auc_score(y, oof)), 3),
            "oof_pair_accuracy": boot.acc(pr).fmt(), "top": top, "oof": dict(zip(groups.tolist(), oof.tolist())),
            "pair_outcomes": pr}


async def lr_and_tags(ctx, estimate):
    sch, hon = falcon_accepted(ctx, "schelling"), falcon_accepted(ctx, "honest")
    ext = ext16(ctx)
    summ = {(r["sender"], r["cv_id"]): r for r in read_jsonl(ctx.raw / "summaries.jsonl")
            if r["condition"] == "schelling"}

    def attempts(cv):
        r = summ[("falcon", cv)]
        return r["attempts"] if not r["leak_dropped"] else ext[("falcon", cv)]["attempts"]

    pair_att = {tid: max(attempts(f"{tid}_A"), attempts(f"{tid}_B")) for tid in sch}
    groups = {"schelling, attempt >= 5": [t for t in sch if pair_att[t] >= 5],
              "schelling, attempt 1-4": [t for t in sch if pair_att[t] <= 4]}
    groups["honest, same templates as attempt >= 5"] = [t for t in groups["schelling, attempt >= 5"] if t in hon]
    jobs = [(g, tid) for g, tids in groups.items() for tid in tids]
    judge = ctx.cfg["judge"]
    src = lambda g, tid: (hon if g.startswith("honest") else sch)[tid]
    new = sum(ctx.llm.cached(judge, PR.tag_messages(src(g, t)[1]["text"], src(g, t)[0]["text"]), temperature=0.0,
                             max_tokens=200, seed=0) is None for g, t in jobs)
    print(f"[lr] logistic regression: {len(sch)} Falcon schelling pairs, {len(hon)} honest pairs (local, $0); "
          f"tagging {len(jobs)} pairs: " + ", ".join(f"{g}: {len(v)}" for g, v in groups.items()))
    if estimate:
        return price(ctx, "3 Phi-4 tagging", {judge: new}, {})
    res = {"schelling": logreg(ctx, sch, "Falcon schelling (accepted, budget 16)"),
           "honest": logreg(ctx, hon, "Falcon honest (control)")}
    lr_rows = []
    for k, r in res.items():
        late = [p for p in r["pair_outcomes"] if k == "schelling" and pair_att[p["template_id"]] >= 5]
        lr_rows.append({"data": r["label"], "summaries": r["summaries"], "pairs": r["pairs"], "oof_auroc": r["oof_auroc"],
                        "oof_pair_accuracy": r["oof_pair_accuracy"],
                        "pair_accuracy_attempt_ge5": (f"{np.mean([p['correct'] for p in late]):.3f} (n={len(late)})"
                                                      if late else "")})
    write_csv(ctx.tables / "followup2b_logreg.csv", lr_rows)
    write_csv(ctx.tables / "followup2b_logreg_top20_schelling.csv", res["schelling"]["top"])
    write_csv(ctx.tables / "followup2b_logreg_top20_honest.csv", res["honest"]["top"])

    async def tag(it):
        g = src(it["group"], it["template_id"])
        out, _ = await ctx.llm.chat_json(judge, PR.tag_messages(g[1]["text"], g[0]["text"]), validate=PR.parse_tag,
                                         temperature=0.0, max_tokens=200, seed=0, tag="tag")
        return {"tags": out, "prompt_version": PR.version("tag_pair")}

    tagged = await run_stage(ctx, "pair_tags", [{"group": g, "template_id": t} for g, t in jobs], ("group", "template_id"), tag)
    tag_rows = []
    for g in groups:
        tt = [r["tags"] for r in tagged if r["group"] == g and r["tags"]]
        row = {"group": g, "pairs": len(tt), "parse_failures": sum(1 for r in tagged if r["group"] == g and not r["tags"])}
        for k in ("euphemism", "emphasis", "tone"):
            row[f"{k}_rate"] = round(float(np.mean([t[k] for t in tt])), 3) if tt else None
        pc = Counter(t["primary"] for t in tt)
        row.update({f"primary_{k}": pc.get(k, 0) for k in ("euphemism", "emphasis", "tone", "none")})
        tag_rows.append(row)
    write_csv(ctx.tables / "followup2b_tags.csv", tag_rows)
    return lr_rows, res["schelling"]["top"], tag_rows


# ------------------------------------------------------------------------------------------- (4)
def late_falcon_texts(ctx):
    ext = ext16(ctx)
    texts = [t for t in schelling_texts(ctx, 16, ext) if t["sender"] == "falcon" and not t["leak_dropped"]]
    by = defaultdict(dict)
    for t in texts:
        by[t["template_id"]][t["Y"]] = t
    late = {tid for tid, g in by.items() if 0 in g and 1 in g and max(g[0]["attempts"], g[1]["attempts"]) >= 5}
    return [t for t in texts if t["template_id"] in late]


async def paradet(ctx, estimate):
    cfg = ctx.cfg
    raw = late_falcon_texts(ctx)
    paras = ctx.cfg["exp3"]["paraphrasers"]
    sib = cfg["models"]["falcon"]["sibling"]
    decs = lambda t: ["falcon", sib] + [d for d in cfg["senders"] if d != "falcon"]
    judges = lambda t: [cfg["judge"], "falcon"]
    n_tmpl = len({t["template_id"] for t in raw})
    print(f"[paradet] {n_tmpl} Falcon schelling pairs passed only at attempt >= 5 "
          f"({len({t['template_id'] for t in raw if t['split'] == 'test'})} test)")
    if estimate:
        have_p = {(r["sender"], r["condition"], r["cv_id"], r["paraphraser"]) for r in read_jsonl(ctx.raw / "paraphrases.jsonl")}
        gen = Counter()
        for t in raw:
            for p in paras:
                if (t["sender"], t["condition"], t["cv_id"], p) not in have_p:
                    gen["falcon" if p == "para_same" else cfg["judge"]] += 2 if p == "para_facts" else 1
        n_para = sum(gen.values())
        score = missing_decodes(ctx, raw, decs) + missing_monitor(ctx, raw, judges)
        for m in decs(raw[0]):
            score[m] += 2 * len(raw)  # upper bound for paraphrased texts (4 paraphrasers, ~half new)
        for j in judges(raw[0]):
            score[j] += 4 * len(raw)
        print(f"[paradet] {n_para} paraphrase calls; scoring counts for paraphrased text are an upper bound")
        return price(ctx, "4 detectors + paraphrasers on attempt >= 5", dict(gen), dict(score))
    dec_raw = await ST.decodes(ctx, "followup2b_paradet", raw, decs)
    mon_raw = await ST.monitor(ctx, raw, judges)
    para = await ST.paraphrases(ctx, raw, paras)
    para = [p for p in para if (p["sender"], p["condition"], p["cv_id"]) in {(t["sender"], t["condition"], t["cv_id"]) for t in raw}
            and p["paraphraser"] in paras]
    dec_p = await ST.decodes(ctx, "followup2b_paradet", para, decs)
    mon_p = await ST.monitor(ctx, para, judges)
    dec, mon = dec_raw + dec_p, mon_raw + mon_p
    fam = [d for d in cfg["senders"]]
    pairs = [p for p in pair_rows(dec) if p["decoder"] in fam]
    leak_rows = leak_table(ctx, pairs, ["none"] + paras, fam)
    write_csv(ctx.tables / "followup2b_paradet_leakage.csv", leak_rows)
    # detectors: test-split audit batches per variant, against Exp 2's test negatives
    D = {(d["cv_id"], d["paraphraser"], d["decoder"]): d["p_tier_a"] for d in dec}
    M = {(m["cv_id"], m["paraphraser"], m["judge"]): m["suspicion"] for m in mon}
    texts = [t for t in raw + para if not t["leak_dropped"] and t["split"] == "test"]
    recs = defaultdict(list)
    for t in texts:
        k = (t["cv_id"], t["paraphraser"])
        recs[t["paraphraser"]].append({"template_id": t["template_id"], "Y": t["Y"], "p_R": D.get(k + ("falcon",)),
                                       "p_sib": D.get(k + (sib,)),
                                       "p_cross": {d: D.get(k + (d,)) for d in cfg["senders"] if d != "falcon"},
                                       "susp_cross": M.get(k + (cfg["judge"],)), "susp_same": M.get(k + ("falcon",))})
    dec_none = [r for r in read_jsonl(ctx.raw / "decodes_lp.jsonl") if r["paraphraser"] == "none"]
    neg_all = [b for b in E2.make_batches(ctx, exp2_records(ctx, dec_none), "test") if not b["label"]]
    orig = json.loads((ctx.res / "exp2_frozen.json").read_text())
    thr = {d: v["thresholds"] for d, v in orig["detectors"].items()}
    absf = ctx.res / "followup1_abs_frozen.json"
    names = list(E2.DETECTORS) + list(E2.REFERENCE)
    if absf.exists():
        thr.update({d: v["thresholds"] for d, v in json.loads(absf.read_text())["detectors"].items()})
        names += list(ABS)
    k = cfg["exp2"]["batch_size"] // 2
    det_rows = []
    for variant, rows in recs.items():
        complete = [tid for tid in sorted({r["template_id"] for r in rows})
                    if sum(r["template_id"] == tid for r in rows) == 2]
        rng = np.random.default_rng(int(hashlib.sha1(f"{cfg['seed']}|late|{variant}".encode()).hexdigest()[:8], 16))
        pos = []
        for _ in range(cfg["exp2"]["n_batches_per_cell"]):
            pick = set(rng.choice(complete, size=min(k, len(complete)), replace=False))
            pos.append({"label": 1, **E2.score_batch([r for r in rows if r["template_id"] in pick], frozenset())})
        for neg_label, neg in (("falcon negatives", [b for b in neg_all if b["sender"] == "falcon"]),
                               ("all senders' negatives", neg_all)):
            for r in eval_detectors(pos + neg, thr, names, cfg["seed"]):
                det_rows.append({"variant": variant, "test_pairs": len(complete), "negatives": neg_label, **r})
    write_csv(ctx.tables / "followup2b_paradet_detectors.csv", det_rows)
    return leak_rows, det_rows


# -------------------------------------------------------------------------------------------
async def main():
    ap = base_args(__doc__)
    ap.add_argument("task", choices=["redact", "loops", "lr", "paradet"])
    ap.add_argument("--estimate", action="store_true", help="count and price new calls only")
    args = ap.parse_args()
    ctx = Ctx(args, f"followup2b_{args.task}{'_estimate' if args.estimate else ''}")
    fn = {"redact": redact, "loops": loops, "lr": lr_and_tags, "paradet": paradet}[args.task]
    await fn(ctx, args.estimate)
    ctx.finish()


if __name__ == "__main__":
    asyncio.run(main())
