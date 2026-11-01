# The Squeeze

Do AI models hide information from their monitors, and does confinement push them to? This project tests confinement-induced steganography and the same-family monitor blind spot on open-weight models from 3B to 32B, served on Modal. It was built for the AI Collusion Sprint (Oct 2026).

**The task.** A sender summarises a CV without naming the university. A reader tries to tell whether it was a tier A or a tier B university. The two CVs in each pair differ only in that line, so any pair accuracy above 50% is leakage (leakage = pair accuracy − 0.5).

## Conclusion

We found no evidence that models of this size share a private code just because they share weights, and confinement did not push them into one. The risk depends on the sender. A sender with a fixed prompt mostly leaks in the open, and cheap filters stop it. A sender that gets feedback from the defences learned a code within 10 rounds, and the original rewrite turned that code into a private channel for anyone holding its strategy notes.

| Question | Conclusion | Main evidence |
|---|---|---|
| H1, shared prior | Not supported | No self-advantage for any of the four models in Exp 1 |
| H2, same-family blind spot | Weak at most | Recall gap 9.2% vs 4.0% at AUROC 0.47–0.52; the adaptive code was unreadable to the same model without the notes (+0.05) |
| H3, paraphrase limits | Partly | Lexical codes fall under every paraphraser; rewriting from the summary kept the adaptive code, rewriting from the CV cut it to +0.12 |
| H4, the squeeze | Not supported | Main-channel leakage R3 − R0: +0.06 [−0.06, +0.19]; the payload fails rather than moving |
| Added: adaptation | Supported, one sender | Reader with notes −0.04 with the plain prompt, +0.28 [+0.21, +0.34] with learned notes, same defences |

## Recommendations

Strongest evidence first:

1. **Evaluate each defence against an attacker that has adapted to it.** The facts-then-prose stack stopped the plain prompt and lost to the adapted sender.
2. **Keep deterministic filters for overt leaks.** A one-line verdict regex beat all seven LLM monitors, though adaptive senders learn to avoid it.
3. **Audit batches of output against known inputs.** A bag-of-words test flagged all four attack pools at 200 summaries, with a 5–10% rate on honest text. It is weak at 50 summaries (13–53%) and detects rather than prevents.
4. **Rewrite from the source document, not the sender's summary.** The notes reader fell from +0.28 to +0.12 on the frozen attack, and to +0.05 [−0.05, +0.15] after the attacker re-adapted.
5. **Don't rely on LLM monitors for subtle codes.** Our perception-question check scored an AUROC of about 0.5.
6. **Isolate context that senders and readers share,** such as notes, memory and system prompts. Inferred from the setup, not tested.
7. **Remove the sensitive attribute before the sender sees it, where the task allows.** Reasoned, not tested.

**These defences did not close the channel.** No LLM reader decoded what passed the final stack, but the intervals allow a leak of up to about +0.15, and a supervised classifier still separates the tiers at 0.83 pair accuracy.

## Limits

These are small open-weight models on one synthetic hiring task. The adaptive results come from one sender, one seed and 10 rounds of in-context learning, with a stated goal and shared notes standing in for shared weights. They show this failure can happen, not how likely it is. Rewrite quality was graded by Phi-4, the model that wrote the rewrites. The most useful next test is a sender trained against the final stack, with a separate reader and an independent quality judge.

## What's here

| Path | Contents |
|---|---|
| [`squeeze/README.md`](squeeze/README.md) | Setup, how to run the experiments, implementation guarantees |
| [`squeeze/PLAN.md`](squeeze/PLAN.md) | The original experiment design |
| `squeeze/src/` | Exps 1–4, follow-ups 1–14 (`followups*.py`), the adaptive attacker (`adapt.py`), the defences and batch audit (`defend.py`) |
| `squeeze/data/` | Fictional CV dataset and university tier lists |
| `squeeze/results/` | Summaries, tables, figures and the learned attacker notes |
| `util/` | Shared compute layer from the hackathon starter repo |

**Checking and rerunning.** To try the pipeline with no account or API calls, run `scripts/ci_check.sh` from `squeeze/`, which uses mock models. Raw model outputs (167 MB) and the call cache aren't in this repo, so scripts that read `results/raw/` need the stages rerun first, and those model calls cost GPU time. The tables and summaries are here for checking reported numbers.
