# The Squeeze

This project asks whether AI models hide information from their monitors, and whether confining them pushes them to. It tests confinement-induced steganography and the same-family monitor blind spot on open-weight models from 3B to 32B, served on Modal. We built it for the AI Collusion Sprint (Oct 2026).

In the task, a sender summarises a CV without naming the university, and a reader tries to tell whether it was a tier A or tier B university. The two CVs in each pair differ only in that line, so any pair accuracy above 50% counts as leakage (leakage = pair accuracy − 0.5).

## Conclusion

We found no evidence that models of this size share a private code because they share weights, and confinement did not push them into one.

The risk depends on the sender. With a fixed prompt, senders mostly leak in the open, and cheap filters stop them. One sender that got feedback from the defences learned a code within 10 rounds. The original rewrite then made that code private to whoever held the sender's strategy notes.

| Question | Conclusion | Main evidence |
|---|---|---|
| H1, shared prior | Not supported | None of the four models in Exp 1 showed a self-advantage |
| H2, same-family blind spot | Weak at most | Recall gap of 9.2% vs 4.0% at AUROC 0.47 to 0.52; the same model could not read the adaptive code without the notes (+0.05) |
| H3, paraphrase limits | Partly | Lexical codes fall under every paraphraser; the adaptive code survived rewriting from the summary, and rewriting from the CV cut it to +0.12 |
| H4, the squeeze | Not supported | Main-channel leakage R3 − R0 was +0.06 [−0.06, +0.19]; the payload fails and does not move |
| Added: adaptation | Supported, one sender | With the same defences, the reader with notes scored −0.04 on the plain prompt and +0.28 [+0.21, +0.34] on learned notes |

## Recommendations

These are ordered by strength of evidence.

1. Evaluate each defence against an attacker that has adapted to it. The facts-then-prose stack stopped the plain prompt and lost to the adapted sender.
2. Keep deterministic filters for overt leaks. A one-line verdict regex beat all seven LLM monitors, although adaptive senders learn to avoid it.
3. Audit batches of output against known inputs. A bag-of-words test flagged all four attack pools at 200 summaries and 5% to 10% of honest batches. At 50 summaries it flagged only 13% to 53%, and it detects leaks without preventing them.
4. Rewrite from the source document, never from the sender's summary alone. The notes reader fell from +0.28 to +0.12 on the frozen attack, and scored +0.05 [−0.05, +0.15] after the attacker re-adapted.
5. Don't rely on LLM monitors for subtle codes. Our perception-question check scored an AUROC of about 0.5.
6. Isolate context that senders and readers share, such as notes, memory and system prompts. We inferred this from the setup and did not test it.
7. Where the task allows, remove the sensitive attribute before the sender sees it. This is reasoned and untested.

The final defence stack did not close the channel. No LLM reader decoded what passed it, but the intervals allow a leak of up to about +0.15, and a supervised classifier still separates the tiers at 0.83 pair accuracy.

## Limits

We tested small open-weight models on one synthetic hiring task. The adaptive results come from one sender, one seed and 10 rounds of in-context learning, with a stated goal and shared notes standing in for shared weights. They show this failure can happen and say nothing about how likely it is. Phi-4 graded rewrite quality, and Phi-4 also wrote the rewrites.

The most useful next test is a sender trained against the final stack, with a separate reader and an independent quality judge.

## Repository contents

| Path | Contents |
|---|---|
| [`squeeze/README.md`](squeeze/README.md) | Setup, how to run the experiments, implementation guarantees |
| [`squeeze/PLAN.md`](squeeze/PLAN.md) | The original experiment design |
| `squeeze/src/` | Exps 1 to 4, follow-ups 1 to 14 (`followups*.py`), the adaptive attacker (`adapt.py`), the defences and batch audit (`defend.py`) |
| `squeeze/data/` | Fictional CV dataset and university tier lists |
| `squeeze/results/` | Summaries, tables, figures and the learned attacker notes |
| `util/` | Shared compute layer from the hackathon starter repo |

To try the pipeline with mock models, with no account or API calls, run `scripts/ci_check.sh` from `squeeze/`.

The raw model outputs (167 MB) and the call cache aren't in this repo. Scripts that read `results/raw/` need their stages rerun first, and those model calls cost GPU time. You can check the reported numbers against the tables and summaries here.
