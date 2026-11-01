"""Generate 150 fictional CV templates and the 300 paired CVs (tier A / tier B university line only).

    python -m src.gen_dataset [--mock]
"""
import asyncio
import random
import re

from . import prompts as PR
from .common import Ctx, LeakFilter, base_args, load_universities, read_jsonl, write_jsonl
from .llm import extract_json

FIELDS = ["software engineering", "data science", "mechanical engineering", "corporate finance",
          "digital marketing", "product management", "civil engineering", "biomedical research",
          "UX design", "supply chain management", "cybersecurity", "economics consulting",
          "electrical engineering", "human resources", "environmental science"]
SENIORITY = ["junior", "mid-level", "senior", "lead", "early-career"]
FIRST = ["Aisha", "Ben", "Carmen", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal", "Kofi",
         "Lena", "Mateo", "Nadia", "Oscar", "Priya", "Quinn", "Rosa", "Samuel", "Tara", "Uma", "Victor",
         "Wen", "Ximena", "Yusuf", "Zoe", "Arjun", "Beatriz", "Chloe", "Daniel"]
LAST = ["Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak", "Silva", "Kowalski", "Mensah", "Reyes",
        "Fischer", "Patel", "O'Brien", "Nguyen", "Romano", "Kim", "Abebe", "Larsen", "Costa", "Ivanova"]
INSTITUTION = re.compile(r"\b(universit\w*|college|school|institute|academy|polytechnic)\b", re.I)


def validate(obj):
    for k in ("name", "headline", "skills", "jobs", "projects", "statement", "degree", "grad_year"):
        if k not in obj:
            raise KeyError(k)
    assert 2 <= len(obj["jobs"]) <= 3 and 1 <= len(obj["projects"]) <= 3
    obj["grad_year"] = int(obj["grad_year"])
    return obj


def render(t, university):
    jobs = "\n".join(
        f"- {j['title']}, {j['company']} ({j.get('start_year', '')}-{j.get('end_year', '')})\n"
        + "\n".join(f"    * {b}" for b in j.get("bullets", []))
        for j in t["jobs"])
    projects = "\n".join(f"- {p['name']}: {p['description']}" for p in t["projects"])
    return (f"{t['name']}\n{t.get('location', '')}\n{t['headline']}\n\n"
            f"Personal statement: {t['statement']}\n\n"
            f"Experience:\n{jobs}\n\nProjects:\n{projects}\n\n"
            f"Skills: {', '.join(t['skills'])}\n\n"
            f"Education: {t['degree']}, {university} ({t['grad_year']})")


async def main():
    ap = base_args(__doc__)
    args = ap.parse_args()
    ctx = Ctx(args, "gen_dataset")
    cfg, ds = ctx.cfg, ctx.cfg["dataset"]
    ctx.data.mkdir(exist_ok=True)
    leak = LeakFilter()
    tpath = ctx.data / "cv_templates.jsonl"
    have = {r["template_id"]: r for r in read_jsonl(tpath)}
    n = args.limit_templates or ds["n_templates"]

    async def make(i):
        field, sen = FIELDS[i % len(FIELDS)], SENIORITY[(i // len(FIELDS)) % len(SENIORITY)]
        msgs = [{"role": "user", "content": PR.text("cv_template", seniority=sen, field=field)}]
        for attempt in range(4):
            txt, _ = await ctx.llm.chat(cfg["dataset_generator"], msgs, temperature=ds["temperature"],
                                        max_tokens=1200, seed=cfg["seed"] + i + 1000 * attempt,
                                        tag="dataset")
            obj = extract_json(txt)
            try:
                obj = validate(obj)
            except Exception:
                continue
            flat = str(obj)
            if INSTITUTION.search(flat) or leak.match(flat):
                continue
            return {"template_id": f"t{i:03d}", "field": field, "seniority": sen, **obj,
                    "prompt_version": PR.version("cv_template")}
        print(f"  template {i} failed validation 4 times; skipped")
        return None

    todo = [i for i in range(n) if f"t{i:03d}" not in have]
    print(f"[dataset] {len(have)} templates exist, generating {len(todo)}")
    for r in await asyncio.gather(*(make(i) for i in todo)):
        if r:
            have[r["template_id"]] = r
    templates = [have[k] for k in sorted(have)]
    # small models reuse a handful of names; give every template a unique, seeded one instead
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    random.Random(cfg["seed"]).shuffle(names)
    for t, name in zip(templates, names):
        t.setdefault("generated_name", t["name"])
        t["name"] = name
    write_jsonl(tpath, templates)

    # pairs + split (seeded, by template)
    u = load_universities()
    rng = random.Random(cfg["seed"])
    tids = [t["template_id"] for t in templates]
    order = tids[:]
    rng.shuffle(order)
    heldout = set(order[: min(ds["n_heldout"], len(order) // 3)])
    pairs = []
    for t in templates:
        r = random.Random(f"{cfg['seed']}-{t['template_id']}")
        for y, uni in ((1, r.choice(u["tier_a"])), (0, r.choice(u["tier_b"]))):
            pairs.append({"template_id": t["template_id"], "cv_id": f"{t['template_id']}_{'A' if y else 'B'}",
                          "Y": y, "university": uni, "text": render(t, uni),
                          "split": "heldout" if t["template_id"] in heldout else "test"})
    write_jsonl(ctx.data / "cv_pairs.jsonl", pairs)
    print(f"[dataset] {len(templates)} templates -> {len(pairs)} CVs "
          f"({sum(p['split'] == 'test' for p in pairs) // 2} test / {len(heldout)} heldout templates), "
          f"split seed {cfg['seed']}")
    ctx.finish()


if __name__ == "__main__":
    asyncio.run(main())
