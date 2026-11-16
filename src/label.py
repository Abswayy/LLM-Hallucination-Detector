"""Phase 2: correctness labels for greedy answers.

Usage:
    python -m src.label            # label both datasets, print 30 examples for review

Outputs:
    results/<dataset>/labels.jsonl   id, correct, incorrect (0/1 target used later),
                                     how the decision was made, and ambiguity flags
    results/metrics.json             ["labeling"][<dataset>]: accuracy + counts

Rules
-----
TriviaQA: correct if the normalized answer equals a normalized gold alias, OR a
normalized gold alias appears inside the normalized answer as a whole-word span.
Normalization is SQuAD-style: lowercase, drop punctuation, drop the articles
a/an/the, collapse whitespace. (Whole-word containment keeps "11" from matching
"110".)

PubMedQA: the answer is parsed to yes/no/maybe (see `parse_yes_no_maybe`) and compared
with the gold label. Unparseable answers count as incorrect.
"""
from __future__ import annotations

import random
import re
import string
from collections import Counter

import numpy as np

from .utils import load_config, read_jsonl, results_dir, update_metrics, write_jsonl

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—", "…"}


# --------------------------------------------------------------------------- TriviaQA
def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation and articles, fix whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _contains_span(haystack: str, needle: str) -> bool:
    """Whole-word containment of `needle` inside `haystack` (both normalized)."""
    return bool(needle) and f" {needle} " in f" {haystack} "


def token_f1(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def label_trivia(answer: str, aliases: list[str]) -> dict:
    pred = normalize_answer(answer)
    golds = [g for g in dict.fromkeys(normalize_answer(a) for a in aliases) if g]
    flags = []
    match, matched_alias = "none", None
    if pred and pred in golds:
        match, matched_alias = "exact", pred
    else:
        hits = [g for g in golds if _contains_span(pred, g)]
        if hits:
            match, matched_alias = "contains", max(hits, key=len)
    correct = match != "none"

    # Ambiguity flags: cases where string matching is most likely to be wrong.
    if match == "contains":
        if len(pred.split()) - len(matched_alias.split()) >= 3:
            flags.append("long_answer_contains_alias")  # e.g. "not Paris but Lyon"
        if len(matched_alias) <= 3 or matched_alias.isdigit():
            flags.append("short_or_numeric_alias_match")
    if not correct and pred:
        best_f1 = max((token_f1(pred, g) for g in golds), default=0.0)
        if best_f1 >= 0.5:
            flags.append("high_overlap_but_no_match")  # e.g. "8 minutes 20 seconds"
        if any(_contains_span(g, pred) for g in golds):
            flags.append("answer_is_part_of_alias")  # e.g. "Bowie" vs "Jim Bowie"
    return {"correct": correct, "match": match, "matched_alias": matched_alias,
            "normalized_answer": pred, "flags": flags}


# --------------------------------------------------------------------------- PubMedQA
_LABELS = ("yes", "no", "maybe")


def parse_yes_no_maybe(answer: str) -> tuple[str | None, str]:
    """Map a free-text answer to yes/no/maybe.

    1. If the first word is a label, use it ("Yes.", "no, because ...").
    2. Otherwise, if exactly one distinct label word appears anywhere, use it.
    3. Otherwise unparseable (None).
    Returns (label or None, how it was parsed).
    """
    words = normalize_answer(answer).split()
    if not words:
        return None, "empty"
    if words[0] in _LABELS:
        return words[0], "first_word"
    found = {w for w in words if w in _LABELS}
    if len(found) == 1:
        return found.pop(), "unique_word"
    return None, "multiple_labels" if found else "no_label"


def label_pubmed(answer: str, gold: str) -> dict:
    parsed, how = parse_yes_no_maybe(answer)
    flags = [] if how == "first_word" else [f"parse_{how}"]
    return {"correct": parsed == gold, "parsed": parsed, "parse": how, "flags": flags}


# --------------------------------------------------------------------------- driver
def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def label_dataset(name: str, cfg: dict) -> list[dict]:
    rows = read_jsonl(results_dir(cfg) / name / "greedy.jsonl")
    out = []
    for r in rows:
        if name == "trivia_qa":
            lab = label_trivia(r["answer"], r["gold"]["aliases"])
        else:
            lab = label_pubmed(r["answer"], r["gold"])
        out.append({"id": r["id"], "question": r["question"], "gold": r["gold"],
                    "answer": r["answer"], "in_sample_subset": r["in_sample_subset"],
                    "incorrect": int(not lab["correct"]), **lab})
    write_jsonl(results_dir(cfg) / name / "labels.jsonl", out)

    acc = np.array([r["correct"] for r in out], dtype=float)
    sub = np.array([r["correct"] for r in out if r["in_sample_subset"]], dtype=float)
    lo, hi = bootstrap_ci(acc, seed=cfg["seed"])
    flag_counts = Counter(f for r in out for f in r["flags"])
    m = {"n": len(out), "n_correct": int(acc.sum()), "accuracy": float(acc.mean()),
         "accuracy_ci95": [lo, hi], "accuracy_sample_subset": float(sub.mean()),
         "n_sample_subset": len(sub), "n_flagged": sum(bool(r["flags"]) for r in out),
         "flag_counts": dict(flag_counts)}
    if name == "trivia_qa":
        m["match_counts"] = dict(Counter(r["match"] for r in out))
    else:
        m["n_unparseable"] = sum(r["parsed"] is None for r in out)
        m["parsed_counts"] = dict(Counter(str(r["parsed"]) for r in out))
        m["gold_counts"] = dict(Counter(r["gold"] for r in out))
        m["accuracy_by_parsed"] = {
            k: float(np.mean([r["correct"] for r in out if r["parsed"] == k]))
            for k in _LABELS if any(r["parsed"] == k for r in out)}
    update_metrics(cfg, ["labeling", name], m)
    return out


def _gold_str(g) -> str:
    return g["value"] if isinstance(g, dict) else g


def print_review(labels: dict[str, list[dict]], n_total: int = 30, seed: int = 42) -> None:
    """Print a seeded, mixed sample of correct/incorrect labels for manual checking."""
    rng = random.Random(seed)
    plan = {"trivia_qa": 20, "pubmed_qa": n_total - 20}
    for name, k in plan.items():
        rows = labels[name]
        cor = [r for r in rows if r["correct"]]
        inc = [r for r in rows if not r["correct"]]
        pick = rng.sample(cor, k // 2) + rng.sample(inc, k - k // 2)
        print(f"\n=== {name}: {k} random examples ({k // 2} labelled correct, "
              f"{k - k // 2} incorrect) ===")
        for r in pick:
            tag = "CORRECT  " if r["correct"] else "INCORRECT"
            how = r.get("match") or f"parsed={r.get('parsed')}"
            extra = f" | aliases: {r['gold']['aliases'][:4]}" if name == "trivia_qa" else ""
            flag = f"  <-- FLAG: {', '.join(r['flags'])}" if r["flags"] else ""
            print(f"[{tag}] ({how}) Q: {r['question'][:90]}\n"
                  f"     model: {r['answer']!r} | gold: {_gold_str(r['gold'])!r}{extra}{flag}")


def main():
    cfg = load_config()
    labels = {name: label_dataset(name, cfg) for name in ("trivia_qa", "pubmed_qa")}
    for name, rows in labels.items():
        acc = np.mean([r["correct"] for r in rows])
        print(f"{name}: accuracy {acc:.3f} ({sum(r['correct'] for r in rows)}/{len(rows)}), "
              f"flagged {sum(bool(r['flags']) for r in rows)}")
    print_review(labels)


if __name__ == "__main__":
    main()
