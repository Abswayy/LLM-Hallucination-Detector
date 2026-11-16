"""Phase 3: uncertainty baselines. Every score is "higher = more likely incorrect".

Usage:
    python -m src.baselines

Methods
-------
neg_mean_logprob   -(mean log-prob of the greedy answer tokens)
neg_min_logprob    -(min log-prob of the greedy answer tokens)
self_consistency   1 - (fraction of the 5 samples that agree with the greedy answer)
semantic_entropy   entropy of meaning-clusters over {greedy + 5 samples}, clusters
                   found by bidirectional entailment with microsoft/deberta-large-mnli
answer_identity    PubMedQA-only control: the error rate of the predicted class
                   (yes/no/maybe), cross-fitted over 5 folds so no example's own label
                   is used to score it

Outputs:
    results/<dataset>/semantic_entropy.jsonl   clusters + NLI call counts (cached)
    results/<dataset>/baseline_scores.jsonl    every score for every question
    results/metrics.json  ["baselines"][<dataset>]  AUROC + 95% bootstrap CI
"""
from __future__ import annotations

import gc
import time
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .evaluate import auroc_with_ci, bootstrap_indices
from .label import _contains_span, normalize_answer, parse_yes_no_maybe
from .utils import load_config, read_jsonl, results_dir, update_metrics, write_jsonl

NLI_MODEL = "microsoft/deberta-large-mnli"


# --------------------------------------------------------------------------- token probs
def logprob_scores(row: dict) -> dict:
    lp = row["token_logprobs"]
    if not lp:  # empty answer: no tokens to score (does not occur in our runs)
        return {"neg_mean_logprob": float("nan"), "neg_min_logprob": float("nan")}
    return {"neg_mean_logprob": -float(np.mean(lp)), "neg_min_logprob": -float(np.min(lp))}


# --------------------------------------------------------------------------- self-consistency
def same_answer(a: str, b: str, dataset: str) -> bool:
    """Agreement test using the labeling normalization.

    TriviaQA: equal after normalization, or one contains the other as whole words
    (the labeling rule, applied in both directions since neither side is "gold").
    PubMedQA: same parsed yes/no/maybe label (unparseable never agrees).
    """
    if dataset == "pubmed_qa":
        pa, pb = parse_yes_no_maybe(a)[0], parse_yes_no_maybe(b)[0]
        return pa is not None and pa == pb
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    return na == nb or _contains_span(na, nb) or _contains_span(nb, na)


def self_consistency(greedy: str, samples: list[str], dataset: str) -> float:
    agree = sum(same_answer(greedy, s, dataset) for s in samples)
    return 1.0 - agree / len(samples)


# --------------------------------------------------------------------------- semantic entropy
class EntailmentJudge:
    """Bidirectional-entailment test with DeBERTa-large-MNLI, as in Kuhn et al. (2023)."""

    def __init__(self, device: str = "cuda"):
        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()
        self.device = device
        ent = [i for i, lab in self.model.config.id2label.items() if lab.lower() == "entailment"]
        assert len(ent) == 1, self.model.config.id2label
        self.ent_id = ent[0]
        self.n_calls = 0  # number of (premise, hypothesis) pairs scored

    @torch.no_grad()
    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        if not pairs:
            return []
        enc = self.tok([p for p, _ in pairs], [h for _, h in pairs], return_tensors="pt",
                       padding=True, truncation=True, max_length=256).to(self.device)
        pred = self.model(**enc).logits.argmax(-1)
        self.n_calls += len(pairs)
        return (pred == self.ent_id).tolist()


def semantic_clusters(question: str, answers: list[str], judge: EntailmentJudge,
                      dataset: str) -> tuple[list[int], int]:
    """Greedy clustering: each answer joins the first cluster whose representative it
    mutually entails (given the question), else starts a new cluster.

    Answers identical after normalization are merged without an NLI call (an answer
    trivially entails itself). Returns (cluster id per answer, NLI calls used).
    """
    calls0 = judge.n_calls
    norm = [normalize_answer(a) for a in answers]
    if dataset == "pubmed_qa":  # "Yes" vs "yes." are literally the same answer
        norm = [parse_yes_no_maybe(a)[0] or n for a, n in zip(answers, norm)]
    reps: list[int] = []  # index of each cluster's first member
    assign: list[int] = []
    for i, a in enumerate(answers):
        hit = next((c for c, r in enumerate(reps) if norm[r] == norm[i]), None)
        if hit is None and reps:
            # One batched NLI call: (rep -> a) and (a -> rep) for every existing cluster.
            pairs = []
            for r in reps:
                pairs += [(f"{question} {answers[r]}", f"{question} {a}"),
                          (f"{question} {a}", f"{question} {answers[r]}")]
            ent = judge.entails(pairs)
            hit = next((c for c in range(len(reps)) if ent[2 * c] and ent[2 * c + 1]), None)
        if hit is None:
            reps.append(i)
            hit = len(reps) - 1
        assign.append(hit)
    return assign, judge.n_calls - calls0


def entropy_from_assignments(assign: list[int]) -> float:
    """Discrete semantic entropy: Shannon entropy (nats) of cluster frequencies."""
    counts = np.array(list(Counter(assign).values()), dtype=float)
    p = counts / counts.sum()
    return float(max(0.0, -(p * np.log(p)).sum()))  # max() avoids printing -0.0


def run_semantic_entropy(name: str, cfg: dict) -> dict[str, dict]:
    """Compute (or load cached) semantic entropy for the sampled subset of one dataset."""
    path = results_dir(cfg) / name / "semantic_entropy.jsonl"
    if path.exists():
        return {r["id"]: r for r in read_jsonl(path)}
    judge = EntailmentJudge()
    rows = read_jsonl(results_dir(cfg) / name / "samples.jsonl")
    # Warm-up so CUDA initialization is not billed to the first question.
    judge.entails([("a b", "a b")])
    judge.n_calls = 0
    out = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for r in tqdm(rows, desc=f"{name}:semantic_entropy", unit="q"):
        answers = [r["greedy"], *r["samples"]]
        assign, calls = semantic_clusters(r["question"], answers, judge, name)
        out.append({"id": r["id"], "answers": answers, "clusters": assign,
                    "n_clusters": len(set(assign)), "nli_calls": calls,
                    "semantic_entropy": entropy_from_assignments(assign)})
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    write_jsonl(path, out)
    update_metrics(cfg, ["costs", name, "nli_clustering"],
                   {"seconds_total": secs, "seconds_per_question": secs / len(out),
                    "nli_calls_per_question": float(np.mean([o["nli_calls"] for o in out])),
                    "n_questions": len(out), "model": NLI_MODEL,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3})
    del judge
    gc.collect()
    torch.cuda.empty_cache()
    return {o["id"]: o for o in out}


# --------------------------------------------------------------------------- control
def answer_identity_scores(parsed: list[str | None], incorrect: np.ndarray, seed: int,
                           k: int = 5) -> np.ndarray:
    """Cross-fitted P(incorrect | predicted class): each fold is scored with error rates
    estimated on the other folds, so no example's own label leaks into its score."""
    keys = np.array([str(p) for p in parsed])
    folds = np.random.default_rng(seed).permutation(len(keys)) % k
    out = np.empty(len(keys))
    for f in range(k):
        tr, te = folds != f, folds == f
        prior = incorrect[tr].mean()
        for c in np.unique(keys[te]):
            m = tr & (keys == c)
            out[te & (keys == c)] = incorrect[m].mean() if m.any() else prior
    return out


# --------------------------------------------------------------------------- driver
def score_dataset(name: str, cfg: dict) -> None:
    rd = results_dir(cfg) / name
    greedy = {r["id"]: r for r in read_jsonl(rd / "greedy.jsonl")}
    labels = read_jsonl(rd / "labels.jsonl")
    samples = {r["id"]: r for r in read_jsonl(rd / "samples.jsonl")}
    se = run_semantic_entropy(name, cfg)

    y = np.array([lab["incorrect"] for lab in labels])
    rows = []
    for lab in labels:
        g = greedy[lab["id"]]
        r = {"id": lab["id"], "incorrect": lab["incorrect"],
             "in_sample_subset": lab["in_sample_subset"], **logprob_scores(g)}
        if lab["id"] in samples:
            assert samples[lab["id"]]["greedy"] == g["answer"]
            r["self_consistency"] = self_consistency(g["answer"], samples[lab["id"]]["samples"], name)
            r["semantic_entropy"] = se[lab["id"]]["semantic_entropy"]
        rows.append(r)
    if name == "pubmed_qa":
        ai = answer_identity_scores([lab["parsed"] for lab in labels], y.astype(float), cfg["seed"])
        for r, s in zip(rows, ai):
            r["answer_identity"] = float(s)
    write_jsonl(rd / "baseline_scores.jsonl", rows)

    full_methods = ["neg_mean_logprob", "neg_min_logprob"] + (
        ["answer_identity"] if name == "pubmed_qa" else [])
    sub_methods = full_methods + ["self_consistency", "semantic_entropy"]
    res = {"full": {}, "sample_subset": {}}
    idx_full = bootstrap_indices(len(rows), seed=cfg["seed"])
    for m in full_methods:
        res["full"][m] = auroc_with_ci(y, np.array([r[m] for r in rows]), idx_full)
    sub = [r for r in rows if r["in_sample_subset"]]
    ys = np.array([r["incorrect"] for r in sub])
    idx_sub = bootstrap_indices(len(sub), seed=cfg["seed"])
    for m in sub_methods:
        res["sample_subset"][m] = auroc_with_ci(ys, np.array([r[m] for r in sub]), idx_sub)
    se_rows = list(se.values())
    res["semantic_entropy_stats"] = {
        "mean_n_clusters": float(np.mean([o["n_clusters"] for o in se_rows])),
        "frac_single_cluster": float(np.mean([o["n_clusters"] == 1 for o in se_rows])),
        "mean_nli_calls": float(np.mean([o["nli_calls"] for o in se_rows])),
    }
    update_metrics(cfg, ["baselines", name], res)

    print(f"\n[{name}]  (y = incorrect; n_full={len(rows)}, n_subset={len(sub)})")
    for split, d in res.items():
        if split == "semantic_entropy_stats":
            continue
        for m, v in d.items():
            print(f"  {split:13s} {m:18s} AUROC {v['auroc']:.3f}  "
                  f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print(f"  SE stats: {res['semantic_entropy_stats']}")


def main():
    cfg = load_config()
    for name in ("trivia_qa", "pubmed_qa"):
        score_dataset(name, cfg)


if __name__ == "__main__":
    main()
