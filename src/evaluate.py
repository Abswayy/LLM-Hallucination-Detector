"""Evaluation utilities: AUROC with paired bootstrap CIs (Phase 3+), and more in Phase 5.

Convention throughout the project: the target is y = 1 if the answer is INCORRECT,
and every score is "higher = more likely incorrect".
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def bootstrap_indices(n: int, n_boot: int = 1000, seed: int = 42) -> np.ndarray:
    """[n_boot, n] resample indices. Reusing the same indices for every method makes the
    CIs *paired*, i.e. the methods are compared on identical resamples."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def auroc_with_ci(y: np.ndarray, s: np.ndarray, idx: np.ndarray) -> dict:
    """Point AUROC plus 95% percentile bootstrap CI over the given resample indices."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    boots = []
    for b in idx:
        yb = y[b]
        if yb.min() == yb.max():  # a resample with one class has no AUROC; skip it
            continue
        boots.append(roc_auc_score(yb, s[b]))
    boots = np.array(boots)
    return {"auroc": auroc(y, s),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(len(y)), "n_incorrect": int(y.sum())}


# --------------------------------------------------------------------------- cost accounting
# What each detector needs at inference time, in terms of the measured stages
# (results/metrics.json ["costs"][<dataset>]["llm"] from src/timing.py, and
# ["nli_clustering"] from src/baselines.py).
#   "generations"  number of LLM answers decoded (greedy + samples)
#   "llm_forward"  LLM forward passes = decoding steps (+1 prefill for prompt-only)
#   "nli_forward"  DeBERTa (premise, hypothesis) pairs
METHOD_STAGES = {
    "neg_mean_logprob": ["greedy"],
    "neg_min_logprob": ["greedy"],
    "answer_identity": ["greedy"],
    "self_consistency": ["greedy", "sample5"],
    "semantic_entropy": ["greedy", "sample5", "nli"],
    "probe_prompt": ["prefill"],           # can run before any answer is generated
    "probe_answer": ["greedy_hooks"],      # hidden state read during normal decoding
}


def method_costs(costs: dict) -> dict:
    """Per-method cost for one dataset from measured stage costs.

    `seconds_total` is the full cost of obtaining the score. `seconds_overhead` is the
    extra cost over producing the greedy answer, which a deployed system pays anyway.
    For the prompt-only probe the overhead is reported as the full prefill, because it is
    meant to run *before* (or instead of) generation.
    """
    llm, nli = costs["llm"], costs["nli_clustering"]
    fp = llm["forward_passes_per_question"]
    sec = {"prefill": llm["prefill"]["seconds_per_question"],
           "greedy": llm["greedy"]["seconds_per_question"],
           "greedy_hooks": llm["greedy_hooks"]["seconds_per_question"],
           "sample5": llm["sample5"]["seconds_per_question"],
           "nli": nli["seconds_per_question"]}
    out = {}
    for m, stages in METHOD_STAGES.items():
        total = sum(sec[s] for s in stages)
        llm_fp = sum(fp[{"greedy_hooks": "greedy"}.get(s, s)] for s in stages if s != "nli")
        out[m] = {
            "generations": sum(1 for s in stages if s in ("greedy", "greedy_hooks"))
            + (5 if "sample5" in stages else 0),
            "llm_forward_passes": llm_fp,
            "nli_forward_passes": nli["nli_calls_per_question"] if "nli" in stages else 0.0,
            "seconds_total": total,
            "seconds_overhead": total if m == "probe_prompt" else total - sec["greedy"],
        }
    return out


def auroc_diff_ci(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, idx: np.ndarray) -> dict:
    """AUROC(s1) - AUROC(s2) with a paired 95% bootstrap CI (same resamples for both).
    If the CI excludes 0, the difference is unlikely to be resampling noise."""
    y, s1, s2 = np.asarray(y), np.asarray(s1, float), np.asarray(s2, float)
    diffs = []
    for b in idx:
        yb = y[b]
        if yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(yb, s1[b]) - roc_auc_score(yb, s2[b]))
    return {"delta": auroc(y, s1) - auroc(y, s2),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


# --------------------------------------------------------------------------- abstention
def accuracy_coverage(incorrect: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Accuracy when the model answers only its k most-confident questions, k = 1..n.

    Returns acc with acc[k-1] = accuracy at coverage k/n. Questions with tied scores
    (common for semantic entropy, which takes few distinct values) are handled exactly:
    we report the *expected* accuracy if ties were broken uniformly at random, so the
    curve does not depend on arbitrary sort order.
    """
    correct = 1.0 - np.asarray(incorrect, float)
    order = np.argsort(np.asarray(score, float), kind="stable")
    s, c = np.asarray(score, float)[order], correct[order]
    n = len(s)
    _, starts = np.unique(s, return_index=True)
    ends = np.append(starts[1:], n)
    acc = np.empty(n)
    kept_correct = 0.0
    for st, en in zip(starts, ends):
        j = np.arange(1, en - st + 1)
        acc[st:en] = (kept_correct + j * c[st:en].mean()) / (st + j)
        kept_correct += c[st:en].sum()
    return acc


ABSTENTION_METHODS = ["probe_answer", "probe_prompt", "neg_mean_logprob", "neg_min_logprob",
                      "self_consistency", "semantic_entropy", "answer_identity"]


def subset_scores(name: str, cfg: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(incorrect, {method: score}) on the 500-question subset where all methods exist.
    Probe scores are the out-of-fold (cross-fitted) scores from src/probe.py."""
    from .utils import read_jsonl, results_dir

    rd = results_dir(cfg) / name
    base = {r["id"]: r for r in read_jsonl(rd / "baseline_scores.jsonl")}
    probe = {r["id"]: r for r in read_jsonl(rd / "probe_scores.jsonl")}
    ids = [i for i, r in base.items() if r["in_sample_subset"]]
    y = np.array([base[i]["incorrect"] for i in ids])
    scores = {}
    for m in ABSTENTION_METHODS:
        src = probe if m.startswith("probe_") else base
        if m in src[ids[0]]:
            scores[m] = np.array([src[i][m] for i in ids], float)
    return y, scores


def abstention_metrics(name: str, cfg: dict) -> dict:
    y, scores = subset_scores(name, cfg)
    n = len(y)
    out = {"n": n, "accuracy_full_coverage": float(1 - y.mean()), "methods": {}}
    for m, s in {**scores, "oracle": y.astype(float)}.items():
        acc = accuracy_coverage(y, s)
        out["methods"][m] = {
            "aurac": float(acc.mean()),  # area under the accuracy-coverage curve
            "accuracy_at_coverage": {f"{c:.1f}": float(acc[int(round(c * n)) - 1])
                                     for c in (0.2, 0.5, 0.8)},
        }
    return out


def main():
    """Phase 5 numbers: abstention (accuracy vs coverage) for every method."""
    from .utils import load_config, update_metrics

    cfg = load_config()
    for name in ("trivia_qa", "pubmed_qa"):
        res = abstention_metrics(name, cfg)
        update_metrics(cfg, ["abstention", name], res)
        print(f"[{name}] n={res['n']}  accuracy at full coverage {res['accuracy_full_coverage']:.3f}")
        for m, v in res["methods"].items():
            a = v["accuracy_at_coverage"]
            print(f"  {m:18s} AURAC {v['aurac']:.3f} | acc@80% {a['0.8']:.3f}  "
                  f"acc@50% {a['0.5']:.3f}  acc@20% {a['0.2']:.3f}")


if __name__ == "__main__":
    main()
