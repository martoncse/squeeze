import numpy as np

from src import prompts as PR
from src.common import LeakFilter
from src.exp4_squeeze import Env
from src.llm import extract_json
from src.metrics import TemplateBoot, auroc, gap, threshold_at_fpr


def test_leak_filter():
    lf = LeakFilter()
    assert lf.match("A graduate of Stanford University with ...")
    assert lf.match("studied at MIT")
    assert lf.match("an Ivy League background")
    assert lf.match("from a top-tier school")
    assert lf.match("Their method is sound and they like Seth's work") is None  # MIT/ETH are case-sensitive
    assert lf.match("Strong engineer with five years of experience.") is None


def test_extract_json():
    assert extract_json('Sure! ```json\n{"p_tier_a": 0.7, "reason": "x"}\n```') == {"p_tier_a": 0.7, "reason": "x"}
    assert extract_json('{"a": 1} and {"b": 2}') == {"a": 1}
    assert extract_json("no json") is None


def test_bootstrap_paired_and_exact():
    rows = [{"template_id": f"t{i}", "correct": float(i % 4 != 0)} for i in range(40)]
    b = TemplateBoot([r["template_id"] for r in rows], B=2000, seed=0)
    e = b.acc(rows)
    assert abs(e.point - 0.75) < 1e-9
    lo, hi = e.ci()
    assert lo < 0.75 < hi
    d = e - e  # paired: identical resamples give exactly zero
    assert np.allclose(d.boot, 0)


def test_gap_formula():
    b = TemplateBoot(["a", "b"], B=10, seed=0)
    R = b.mean([{"template_id": "a", "v": 0.4}, {"template_id": "b", "v": 0.4}], "v")
    T = b.mean([{"template_id": "a", "v": 0.1}, {"template_id": "b", "v": 0.1}], "v")
    assert abs(gap(T, R).point - 0.75) < 1e-9


def test_auroc_and_threshold():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([1], [1]) == 0.5
    neg = list(range(100))
    t = threshold_at_fpr(neg, 0.05)
    assert np.mean(np.array(neg) > t) <= 0.05


def test_prompts_render():
    m = PR.sender_messages("schelling", "CV TEXT", 1)
    assert "copy of you" in m[0]["content"] and "tier A" in m[1]["content"]
    m = PR.sender_messages("honest", "CV TEXT", 1)
    assert "tier" not in m[1]["content"]
    d = PR.decoder_messages("S", codebook="codebook_lexical")
    assert '"strong"' in d[0]["content"] and '{"p_tier_a"' in d[0]["content"]
    assert "{{" not in PR.text("monitor")


def test_env_levels():
    e = Env("R2")
    r, done = e.step({"tool": "ls", "args": {"path": "/shared"}})
    assert "notes.txt" in r and e.saw_note
    r, _ = e.step({"tool": "write_note", "args": {"text": "x"}})
    assert e.note == "x"
    e3 = Env("R3")
    r, _ = e3.step({"tool": "write_note", "args": {"text": "x"}})
    assert r.startswith("Error") and e3.note is None
    r, _ = e3.step({"tool": "ls", "args": {"path": "shared"}})
    assert r == "(empty)"
    r, done = e3.step({"tool": "post_summary", "args": {"text": "hi"}})
    assert done and e3.posted == "hi"
