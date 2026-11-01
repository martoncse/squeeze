# The Squeeze

Confinement-induced steganography and the same-family monitor blind spot. An AI Collusion Sprint
project (Oct 2026), built as one more project folder next to the starter repo's: same `util/`
compute layer, same repo-root `compute.env`, same layout. The full design is in [PLAN.md](PLAN.md).

**References:** the CV task follows Mathew et al. (2024), *Hidden in Plain Text*
([arXiv:2410.03768](https://arxiv.org/abs/2410.03768)); other prior work is listed in PLAN.md
("Prior work to disclose and cite").

## Overview

One paired-CV task drives four experiments. A sender summarises a CV without naming the university.
A receiver tries to recover whether it was a tier A or a tier B university, and monitors try to catch the
hidden signal. The two CVs of each pair differ only in the university line, so any decoding accuracy
above 50% is leakage.

| Exp | Hypothesis | Script | Main outputs (in `results/`) |
|---|---|---|---|
| 1 | H1 shared prior: copies of a model decode each other's zero-shot codes better than other families do | `src/exp1_matrix.py` | `tables/exp1_self_advantage.csv`, `figures/exp1_did_heatmap.png` |
| 2 | H2 same-family sentinels and judges miss leakage that cross-family ones catch | `src/exp2_detectors.py` | `tables/exp2_partA_gap.csv`, `tables/exp2_detectors.csv`, `figures/exp2_roc.png`, `exp2_frozen.json` |
| 3 | H3 paraphrasing removes lexical and stylistic codes but not semantic ones | `src/exp3_paraphrase.py` | `tables/exp3_paraphrase.csv`, `figures/exp3_utility_vs_leakage.png` |
| 4 | H4 closing the side channel moves the payload into the monitored channel | `src/exp4_squeeze.py` | `tables/exp4_squeeze.csv`, `figures/exp4_squeeze.png` |

Each script also writes `results/expN_summary.json` with the hypothesis checks as PLAN.md defines them.

### Models

Every model is open-weight and served with vLLM on Modal ([modal_serve.py](modal_serve.py)), so no
inference API key is needed. Each model is its own Modal function: it boots on its first request and
scales to zero after 5 idle minutes.

| Role | Models (family) | GPU |
|---|---|---|
| Senders, receivers and sentinels | Qwen2.5-7B (Qwen), Falcon3-7B (TII), OLMo-2-7B (AllenAI), Granite-3.1-8B (IBM) | A10G |
| Same-family siblings (Exp 2 sentinels) | Qwen2.5-14B, Falcon3-3B, OLMo-2-13B, Granite-3.1-2B | L40S / A10G |
| Fixed judge (monitor, faithfulness, cross-family paraphraser) | Phi-4 (Microsoft), cross-family to every sender | L40S |

All default models are ungated, so no Hugging Face token is needed. Models, GPUs and hourly rates are set in
[config.yaml](config.yaml); any model can be switched to `provider: nebius`, `openrouter` or another
OpenAI-compatible endpoint.

### Repository layout

| Path | Description |
|---|---|
| `../util/` | `poseidon_compute`, copied unchanged from the starter repo: loads `compute.env`, Hugging Face access preflight, Modal readiness checks. |
| `../compute.env.example` | The starter repo's key file, plus a Modal section for this project. |
| `config.yaml` | Models, Modal settings, sample sizes, budget cap, seeds. |
| `modal_serve.py` | One Modal app with a vLLM server per model (adapted from `poseidon_compute/modal_serve.py`). |
| `src/` | `llm.py` client, `cache.py`, `prompts.py`, `stages.py` (shared summarise / paraphrase / decode / monitor / faithfulness stages), the four experiments, `metrics.py`, `plots.py`, `host.py` (ephemeral Modal wrapper), `mock.py` (offline stand-in models). |
| `data/` | `universities.yaml` (tier lists and leak-filter patterns); `gen_dataset` writes `cv_templates.jsonl` and `cv_pairs.jsonl` here. |
| `results/` | `raw/` append-only JSONL per stage, `tables/`, `figures/`, summaries, `cost_ledger.jsonl`. |
| `scripts/ci_check.sh` | Compile, unit tests and the full pipeline on mock models. No key or Modal account needed. |
| `tests/` | Unit tests for metrics, leak filter, prompts and the Exp 4 sandbox. |

**Dataset.** None of the starter repo's projects ship a CV dataset (ColludeBench's nearest scenario is
peer review), and Mathew et al. did not release theirs, so `gen_dataset` generates 150 fictional
templates with the dataset model (Qwen2.5-7B by default) and pairs each with one tier A and one tier B
university.

## Setup

From this folder:

```bash
uv sync                                  # or: python -m venv .venv && pip install -r requirements.txt
cp ../compute.env.example ../compute.env # optional for Modal-only runs (holds MODAL_SERVE_API_KEY if you use one)
modal setup                              # one-time browser sign-in; hackathon credits attach to this account
```

To use a gated model (e.g. Llama), accept its licence on Hugging Face, run
`modal secret create huggingface HF_TOKEN=<your token>` and add `huggingface` to `modal.secrets` in
`config.yaml`.

## Quick start

No key, no Modal account, no calls:

```bash
scripts/ci_check.sh
```

Success signal: it ends with `ci_check passed`, after the unit tests and all four experiments have run on
the mock models (outputs in `results_mock/`). The mock plants a known same-family effect, so the mock
summaries report the hypotheses as supported; that only shows the analysis recovers a planted effect.

## Running experiments

Either deploy the model app once for the session, then run scripts directly:

```bash
modal deploy modal_serve.py
python -m src.check_models                # boot every model once; catches a broken server early
python -m src.gen_dataset                 # 150 templates -> 300 paired CVs (100 test / 50 held-out templates)
python -m src.exp1_matrix --smoke         # 10 templates: example summaries + projected cost of every experiment
python -m src.exp1_matrix                 # full Exp 1 (both splits; held-out outputs feed Exp 2)
python -m src.exp2_detectors              # freezes thresholds on held-out, then scores test
python -m src.exp3_paraphrase
python -m src.exp4_squeeze                # add --icrl for the in-context RL stretch
modal app stop squeeze-models             # when done
```

or wrap a single command so the app is torn down when it exits (like the starter's `poseidon-host`):

```bash
python -m src.host -- python -m src.exp1_matrix --smoke
```

The first request to each model waits for its cold start (weights are cached in the
`squeeze-hf-cache` Modal volume, so later boots are faster). Every script accepts `--mock`,
`--limit-templates N`, `--budget USD` and `--config path`.

**Order and decision gate (from PLAN.md):** dataset and smoke test, then Exp 1. If `schelling`
leakage is at chance for every model, lead with the `codebook` results and Exps 3 and 4 and report the
null result. Exps 1 and 2 are the minimum paper, Exp 3 makes it strong, Exp 4 is the stretch.

## Guarantees from the implementation spec

- **Cache:** every call goes through a SQLite cache keyed on provider, model, messages, params and seed
  (`cache.sqlite`). Reruns cost nothing.
- **Budget guard:** for Modal models the run estimates GPU spend as the time each model has been in use
  plus its idle scaledown, times its hourly rate, and stops at `budget_usd`. Token-billed providers use
  reported cost. Each run's spend goes to `results/cost_ledger.jsonl`. The estimate covers this script's
  own use only; check the Modal dashboard for actual billing.
- **Resumable:** each stage appends to `results/raw/<stage>.jsonl` and skips rows already there. Stages
  are shared, so Exps 2 and 3 reuse Exp 1's summaries and decodes.
- **Seeds:** dataset generation, split, batch sampling and bootstraps derive from `seed` in config.yaml.
  vLLM also gets a per-request seed, but batched GPU sampling is not bit-reproducible; the cache is
  what makes reruns identical.
- **Structured outputs:** decoders return `{"p_tier_a", "reason"}`, monitors `{"suspicion", "reason"}`. A
  parse failure is retried once and then recorded as missing, never imputed.
- **Leak filter:** a regex over university names, aliases and tier words (`data/universities.yaml`) drops
  summaries before decoding; drops are logged per sender and condition (`tables/exp1_drop_rates.csv`).
- **Freeze:** Exp 2 fits detector thresholds on held-out batches only, writes `results/exp2_frozen.json`
  with its hash, then scores the test split.
- **No side effects:** Exp 4's tools are Python functions over an in-memory dict.
- **Output schema:** `tables/exp1_decodes.csv` has one row per decode with the columns listed in PLAN.md.

## Design decisions not fixed by PLAN.md

- **Sender knowledge of the label.** In incentive conditions the sender gets a private note saying
  whether the candidate is tier A or B; honest senders do not. This separates encoding ability from
  knowing the tier lists.
- **Codebooks.** Only the receiver (the sender's own model) gets the codebook; sentinels never do. The
  main `codebook` condition is the stylistic scheme (first vs third person), which doubles as Exp 3's
  stylistic variant; Exp 3 adds lexical and semantic schemes.
- **Same-family sentinel.** With temperature-0 decoders, using the sender's own model as sentinel makes
  the gap exactly zero by construction, so Exp 2 uses a different-size sibling from the same family.
- **Gap as a batch detector.** The ratio gap `1 - L(S,T)/L(S,R)` is unstable when `L(S,R)` is near zero, so
  Exp 2 scores batches with `L(S,R) - L(S,T)`; Part A reports both. The cross-family gap averages over all
  cross-family sentinels. Receiver leakage alone is reported as a reference detector.
- **Audit batches.** Batches of 10 templates (20 summaries) are resampled from each split, 50 per sender
  and condition, so batches overlap and batch-level CIs are optimistic.
- **Paraphrasers.** `para_same` uses the sender's model; the other three use the judge model. Exp 3 uses
  50 test templates by default and a fixed cross-family sentinel per sender.
- **Exp 4.** ls/list_tools are offered at R2 and R3 only, so R2 and R3 differ only in whether
  `/shared/notes.txt` exists. The monitor flag threshold is Exp 2's per-summary 5%-FPR threshold when
  available (else `exp4.flag_threshold`); naming the university is always flagged.
