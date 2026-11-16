"""Phase 5: publication figures (PDF + PNG) from results/metrics.json and score files.

Usage:
    python -m src.plots

Figures (figures/):
    fig1_auroc_comparison   AUROC with 95% CIs, every method, both datasets
    fig2_abstention         accuracy vs coverage when abstaining on the most uncertain
    fig3_cost_vs_auroc      forward passes per question (log) vs AUROC
    fig4_layer_sweep        probe test AUROC per layer, both token positions
    fig5_transfer           train-domain x test-domain AUROC heatmaps

Color assignment is fixed per method across all figures (a color always means the same
method). The four method hues were checked with a colorblind-safety validator for
all-pairs use; every series also has a distinct marker and a text label, so identity
never relies on color alone. Self-consistency and the answer-identity control are drawn
in neutral grays: the first is near-identical to semantic entropy in cost and AUROC,
the second is a reference, not a method.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .evaluate import accuracy_coverage, subset_scores  # noqa: E402
from .utils import ROOT, load_config, results_dir  # noqa: E402

# ----------------------------------------------------------------------------- style
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"  # primary / secondary / muted text
GRID = "#e6e5e1"
METHODS = {  # key: (label, color, marker)
    "probe_answer": ("Answer-token probe", "#2a78d6", "o"),
    "probe_prompt": ("Prompt-only probe", "#1baf7a", "s"),
    "neg_mean_logprob": ("Token log-prob", "#eda100", "D"),
    "semantic_entropy": ("Semantic entropy", "#e87ba4", "^"),
    "self_consistency": ("Self-consistency", "#52514e", "v"),
    "answer_identity": ("Answer identity (control)", "#a8a7a2", "P"),
}
DATASET_LABEL = {"trivia_qa": "TriviaQA", "pubmed_qa": "PubMedQA"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
})

FULL_W = 6.75  # inches: full text width of a two-column workshop paper


def _metrics(cfg) -> dict:
    return json.loads((results_dir(cfg) / "metrics.json").read_text(encoding="utf-8"))


def _save(fig, name: str, cfg) -> None:
    out = ROOT / cfg["paths"]["figures"]
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


def _methods_for(ds: str) -> list[str]:
    ms = ["probe_answer", "probe_prompt", "neg_mean_logprob", "self_consistency", "semantic_entropy"]
    return ms + (["answer_identity"] if ds == "pubmed_qa" else [])


# ----------------------------------------------------------------------------- fig 1
def fig_auroc_comparison(cfg, M) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.3), sharex=True)
    for ax, ds in zip(axes, ("trivia_qa", "pubmed_qa")):
        res = M["comparison"][ds]["subset"]
        ms = _methods_for(ds)[::-1]  # best-known first at the top
        for yi, m in enumerate(ms):
            label, color, marker = METHODS[m]
            r = res[m]
            ax.plot(r["ci95"], [yi, yi], color=color, lw=2, solid_capstyle="round")
            ax.plot(r["auroc"], yi, marker=marker, ms=6, color=color, mec="white", mew=1.2)
            ax.text(r["ci95"][1] + 0.008, yi, f"{r['auroc']:.3f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(ms)), [METHODS[m][0] for m in ms])
        ax.set_title(f"{DATASET_LABEL[ds]} (n={res['probe_answer']['n']})", loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.6, 0.95)
        ax.set_xlabel("AUROC for detecting incorrect answers (95% CI)")
    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig1_auroc_comparison", cfg)


# ----------------------------------------------------------------------------- fig 2
def fig_abstention(cfg, M) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    for ax, ds in zip(axes, ("trivia_qa", "pubmed_qa")):
        y, scores = subset_scores(ds, cfg)
        n = len(y)
        cov = np.arange(1, n + 1) / n
        lo = int(0.1 * n) - 1  # below 10% coverage only ~50 questions remain: too noisy
        oracle = accuracy_coverage(y, y.astype(float))
        ax.plot(cov[lo:], oracle[lo:], color=INK3, lw=1.0, label="Oracle")
        ax.axhline(1 - y.mean(), color=INK3, lw=0.8)
        # x-axis is inverted, so ha="right" grows the label leftward, back inside the axes
        ax.text(0.105, 1 - y.mean() + 0.008, "no abstention", fontsize=6.5, color=INK3,
                ha="right", va="bottom")
        for m in ["answer_identity", "semantic_entropy", "neg_mean_logprob", "probe_prompt", "probe_answer"]:
            if m not in scores:
                continue
            label, color, marker = METHODS[m]
            acc = accuracy_coverage(y, scores[m])
            marks = np.linspace(lo, n - 1, 6).astype(int)
            ax.plot(cov[lo:], acc[lo:], color=color, lw=2, label=label, marker=marker,
                    markevery=list(marks - lo), ms=4.5, mec="white", mew=0.8)
        ax.set_xlim(1.0, 0.1)  # read left -> right as "abstain more"
        ax.set_xlabel("Coverage (fraction of questions answered)")
        ax.set_ylabel("Accuracy on answered questions")
        ax.set_title(DATASET_LABEL[ds], loc="left", color=INK)
    h, l = axes[1].get_legend_handles_labels()
    order = [l.index(k) for k in ["Answer-token probe", "Prompt-only probe", "Token log-prob",
                                   "Semantic entropy", "Answer identity (control)", "Oracle"]]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.06), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig2_abstention", cfg)


# ----------------------------------------------------------------------------- fig 3
# Label anchors in data coordinates (x, y, ha), placed by hand to avoid collisions:
# beside the point where there is room, otherwise just beyond the end of its CI whisker.
LABEL_POS = {
    "trivia_qa": {"probe_answer": (3.4, 0.884, "right"), "neg_mean_logprob": (4.6, 0.835, "left"),
                  "probe_prompt": (1.0, 0.768, "center"), "self_consistency": (23.55, 0.870, "center"),
                  "semantic_entropy": (32.6, 0.768, "center")},
    "pubmed_qa": {"probe_answer": (2.35, 0.845, "left"), "answer_identity": (2.35, 0.760, "left"),
                  "neg_mean_logprob": (2.35, 0.722, "left"), "probe_prompt": (1.0, 0.879, "center"),
                  "self_consistency": (10.4, 0.669, "right"), "semantic_entropy": (15.0, 0.676, "left")},
}


def fig_cost_vs_auroc(cfg, M) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True)
    for ax, ds in zip(axes, ("trivia_qa", "pubmed_qa")):
        res, costs = M["comparison"][ds]["subset"], M["costs"][ds]["methods"]
        for m in _methods_for(ds):
            label, color, marker = METHODS[m]
            x = costs[m]["llm_forward_passes"] + costs[m]["nli_forward_passes"]
            r = res[m]
            ax.plot([x, x], r["ci95"], color=color, lw=1.5, solid_capstyle="round")
            ax.plot(x, r["auroc"], marker=marker, ms=7, color=color, mec="white", mew=1.2, ls="none")
            lx, ly, ha = LABEL_POS[ds][m]
            ax.text(lx, ly, label, ha=ha, va="center", fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 70)
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_xlabel("Forward passes per question (LLM + NLI, log scale)")
        ax.set_title(DATASET_LABEL[ds], loc="left", color=INK)
    axes[0].set_ylabel("AUROC (95% CI)")
    fig.tight_layout(w_pad=2.0)
    _save(fig, "fig3_cost_vs_auroc", cfg)


# ----------------------------------------------------------------------------- fig 4
def fig_layer_sweep(cfg, M) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), sharey=True)
    for ax, ds in zip(axes, ("trivia_qa", "pubmed_qa")):
        P = M["probes"][ds]
        for pos, m in (("answer", "probe_answer"), ("prompt", "probe_prompt")):
            label, color, marker = METHODS[m]
            rows = P["layer_sweep"][pos]
            L = np.array([r["layer"] for r in rows])
            a = np.array([r["test"]["auroc"] for r in rows])
            lo = np.array([r["test"]["ci95"][0] for r in rows])
            hi = np.array([r["test"]["ci95"][1] for r in rows])
            ax.fill_between(L, lo, hi, color=color, alpha=0.13, lw=0)
            ax.plot(L, a, color=color, lw=2, marker=marker, ms=4.5, mec="white", mew=0.8, label=label)
            best = P["best"][pos]["layer"]  # chosen on VALIDATION, shown on test
            bi = list(L).index(best)
            ax.plot(best, a[bi], marker="o", ms=11, mfc="none", mec=color, mew=1.4)
            if ds == "trivia_qa":  # line ends are well separated here
                ax.annotate(label.split(" probe")[0], (L[-1], a[-1]), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=7, color=INK)
            else:  # ends overlap on PubMedQA; label at layer 12 where the lines are far apart
                k = list(L).index(12)
                ax.text(12, a[k] + (0.035 if pos == "answer" else -0.045), label.split(" probe")[0],
                        ha="center", va="center", fontsize=7, color=INK)
        ref = P["baselines_test"]["neg_mean_logprob"]["auroc"]
        ax.axhline(ref, color=METHODS["neg_mean_logprob"][1], lw=1.2,
                   label="Token log-prob (same test split)")
        above = ds == "trivia_qa"  # put the label where neither probe line runs
        ax.text(4, ref + (0.008 if above else -0.012), f"token log-prob ({ref:.3f})",
                fontsize=6.5, color=INK2, va="bottom" if above else "top")
        ax.set_xticks(L)
        ax.set_xlim(2, 44)
        ax.set_xlabel("Layer")
        ax.set_title(f"{DATASET_LABEL[ds]} (test n={P['split_sizes']['test']})", loc="left", color=INK)
    axes[0].set_ylabel("Probe test AUROC")
    axes[0].set_ylim(0.4, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=2.2)
    fig.text(0.99, -0.02, "Ring = layer selected on the validation split. Band = 95% bootstrap CI.",
             ha="right", fontsize=6.5, color=INK3)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "fig4_layer_sweep", cfg)


# ----------------------------------------------------------------------------- fig 5
def fig_transfer(cfg, M) -> None:
    T = M["transfer"]
    ds = ("trivia_qa", "pubmed_qa")
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.72, 2.4), sharey=True)
    for ax, (pos, title) in zip(axes, (("prompt", "Prompt-only probe"), ("answer", "Answer-token probe"))):
        A = np.array([[T[f"{s}->{t}"][pos]["test"]["auroc"] for t in ds] for s in ds])
        im = ax.imshow(A, cmap=cmap, vmin=0.5, vmax=1.0)
        for i, s in enumerate(ds):
            for j, t in enumerate(ds):
                r = T[f"{s}->{t}"][pos]["test"]
                dark = (A[i, j] - 0.5) / 0.5 > 0.55
                ax.text(j, i - 0.08, f"{r['auroc']:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if dark else INK, fontweight="bold")
                ax.text(j, i + 0.2, f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]", ha="center",
                        va="center", fontsize=6.5, color="white" if dark else INK2)
        ax.set_xticks([0, 1], [DATASET_LABEL[d] for d in ds])
        ax.set_yticks([0, 1], [DATASET_LABEL[d] for d in ds])
        ax.set_xlabel("Test domain")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Train domain")
    axes[1].tick_params(labelleft=False)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cb.set_label("Test AUROC", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    _save(fig, "fig5_transfer", cfg)


def main():
    cfg = load_config()
    M = _metrics(cfg)
    fig_auroc_comparison(cfg, M)
    fig_abstention(cfg, M)
    fig_cost_vs_auroc(cfg, M)
    fig_layer_sweep(cfg, M)
    fig_transfer(cfg, M)


if __name__ == "__main__":
    main()
