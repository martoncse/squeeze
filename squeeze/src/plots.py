"""The four figures named in PLAN.md."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def heatmap(mat, senders, decoders, title, path, annot=None, cbar_label="", center=True):
    fig, ax = plt.subplots(figsize=(1.4 * len(decoders) + 2, 1.1 * len(senders) + 1.6))
    v = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1
    im = ax.imshow(mat, cmap="RdBu_r" if center else "viridis", vmin=-v if center else None,
                   vmax=v if center else None)
    ax.set_xticks(range(len(decoders)), decoders)
    ax.set_yticks(range(len(senders)), senders)
    ax.set_xlabel("decoder")
    ax.set_ylabel("sender")
    for i in range(len(senders)):
        for j in range(len(decoders)):
            txt = annot[i][j] if annot else f"{mat[i, j]:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i, j]) > 0.6 * v else "black")
            if senders[i] == decoders[j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=2.5, ec="black"))
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc(curves, path, fprs=(0.01, 0.05), title="Exp 2: detectors on test-split audit batches"):
    """curves: {label: (fpr_array, tpr_array, auroc)}"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (f, t, a) in curves.items():
        ax.plot(f, t, lw=1.8, label=f"{label} (AUROC {a:.2f})", drawstyle="steps-post")
    for x in fprs:
        ax.axvline(x, ls="--", lw=0.8, color="grey")
        ax.text(x, 0.02, f" {x:.0%} FPR", rotation=90, fontsize=7, color="grey")
    ax.plot([0, 1], [0, 1], ":", color="lightgrey")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("recall")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def utility_vs_leakage(points, path):
    """points: list of dict(paraphraser, encoding, x, y, xerr=(lo,hi), yerr=(lo,hi))"""
    markers = {"zero-shot": "o", "stylistic": "s", "lexical": "^", "semantic": "D"}
    paras = sorted({p["paraphraser"] for p in points})
    colors = dict(zip(paras, plt.cm.tab10.colors))
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for p in points:
        if not (np.isfinite(p["x"]) and np.isfinite(p["y"])):
            continue
        yerr = np.array([[p["y"] - p["yerr"][0]], [p["yerr"][1] - p["y"]]]).clip(0)
        ax.errorbar(p["x"], p["y"], yerr=yerr, fmt=markers.get(p["encoding"], "o"),
                    color=colors[p["paraphraser"]], ms=7, capsize=2, alpha=0.9)
    for name, c in colors.items():
        ax.plot([], [], "o", color=c, label=name)
    for enc, m in markers.items():
        ax.plot([], [], m, color="black", mfc="none", label=enc)
    ax.axhline(0, color="grey", lw=0.8)
    for enc in markers:  # dotted line: each encoding's leakage before paraphrasing
        b = [p["before"] for p in points if p["encoding"] == enc and "before" in p]
        if b:
            ax.axhline(b[0], color="grey", lw=0.7, ls=":")
            ax.text(ax.get_xlim()[0] if False else 1.0, b[0], f" {enc} before", fontsize=6, color="grey",
                    va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("utility retained (faithfulness after / before)")
    ax.set_ylabel("receiver leakage after paraphrase (pair accuracy - 0.5)")
    ax.set_title("Exp 3: which encodings survive paraphrasing", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def squeeze(series, levels, path, baseline=None):
    """series: {receiver_type: {"side_use": [..], "leak": [(pt, lo, hi)..], "success": [..]}}"""
    fig, axes = plt.subplots(1, len(series), figsize=(5.2 * len(series), 4.2), sharey=True, squeeze=False)
    x = np.arange(len(levels))
    for ax, (rtype, s) in zip(axes[0], series.items()):
        ax.bar(x, s["side_use"], color="lightsteelblue", label="side-channel use rate")
        pts = np.array(s["leak"], float)
        ax.errorbar(x, pts[:, 0], yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]], fmt="o-",
                    color="crimson", capsize=3, label="main-channel leakage (acc - 0.5)")
        ax.plot(x, s["success"], "s--", color="black", label="total payload success")
        if baseline and rtype in baseline:
            ax.axhline(baseline[rtype], color="crimson", ls=":", lw=1, label="honest baseline leakage (Exp 1)")
        ax.set_xticks(x, levels)
        ax.set_title(f"receiver: {rtype}", fontsize=10)
        ax.set_ylim(-0.1, 1.35)
        ax.legend(fontsize=7, loc="upper center", ncol=2)
    axes[0][0].set_ylabel("rate")
    fig.suptitle("Exp 4: route choice under confinement", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def leakage_by_attempt(series, buckets, path):
    """series: {sender: [None | (point, lo, hi, n_pairs) per bucket]}"""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(buckets))
    k = len(series)
    for j, (s, pts) in enumerate(series.items()):
        off = (j - (k - 1) / 2) * 0.12
        xs = [xi + off for xi, p in zip(x, pts) if p]
        ys = np.array([p[0] for p in pts if p])
        lo = np.array([p[1] for p in pts if p])
        hi = np.array([p[2] for p in pts if p])
        if not len(xs):
            continue
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o-", capsize=3, label=s, alpha=0.9)
        for xi, p in zip(x, pts):
            if p:
                ax.annotate(f"{p[3]}", (xi + off, p[2]), textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=6, color="grey")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x, buckets)
    ax.set_xlabel("rewrite attempt at which the pair got past the leak filter (later CV of the pair)")
    ax.set_ylabel("pair leakage, pooled over 4 decoders")
    ax.set_title("Exp 1 schelling: leakage by rewrite attempt (small numbers = pairs)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def presence_privacy(points, path):
    """points: dict(x=|pooled leakage|, xlo, xhi, y=self minus others, ylo, yhi, label, condition, sig, n)"""
    conds = sorted({p["condition"] for p in points})
    colors = dict(zip(conds, plt.cm.tab10.colors))
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for p in points:
        c = colors[p["condition"]]
        ax.errorbar(p["x"], p["y"], xerr=[[max(p["x"] - p["xlo"], 0)], [max(p["xhi"] - p["x"], 0)]],
                    yerr=[[max(p["y"] - p["ylo"], 0)], [max(p["yhi"] - p["y"], 0)]], fmt="o", color=c,
                    ms=6 if p["sig"] else 4, alpha=0.85 if p["sig"] else 0.45, elinewidth=0.7, capsize=0)
        if p["sig"] or p["x"] > 0.15:
            ax.annotate(p["label"], (p["x"], p["y"]), fontsize=6, xytext=(4, 3), textcoords="offset points")
    for c, col in colors.items():
        ax.plot([], [], "o", color=col, label=c)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("presence: |pooled leakage| over 4 families (pair accuracy - 0.5)")
    ax.set_ylabel("privacy gap: sender reading itself - other families")
    ax.set_title("Presence vs privacy, every decoded condition (test split); large dots: gap CI excludes 0", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
