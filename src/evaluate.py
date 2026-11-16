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
