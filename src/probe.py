"""Phase 4: linear probes on hidden states to predict "the answer is incorrect".

Usage:
    python -m src.probe

Protocols
---------
A. Fixed 70/15/15 question split (per dataset, seeded):
   * layer sweep: for every saved layer and both token positions, fit a standardized
     logistic regression on train, pick C on val, report val and test AUROC;
   * best layer per position chosen on **val**; reported on **test**;
   * shuffled-label control at the chosen layer (must give test AUROC ~ 0.5);
   * cross-domain transfer: probe trained on one dataset's train split (layer and C chosen
     on that dataset's val split), evaluated on the other dataset's test split;
   * baselines evaluated on the same test split for reference.
B. 5-fold cross-fitting (per dataset): every question gets an out-of-fold probe score
   from a probe that never saw it. Layer and C are selected on an inner validation split
   inside each fold. This lets the probe be compared with the sampling baselines on the
   full 500-question subset where all methods exist.

Nothing is ever selected on test data. Each split is disjoint by question; this is
asserted.

Outputs:
    results/<dataset>/probe_scores.jsonl   out-of-fold probe scores (protocol B)
    results/metrics.json ["probes"], ["transfer"], ["comparison"], ["costs"][..]["methods"]
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .evaluate import auroc_diff_ci, auroc_with_ci, bootstrap_indices, method_costs
from .label import normalize_answer
from .utils import load_config, read_jsonl, results_dir, update_metrics, write_jsonl

DATASETS = ("trivia_qa", "pubmed_qa")
POSITIONS = ("prompt", "answer")  # last prompt token, last generated answer token


# --------------------------------------------------------------------------- data
def load_arrays(name: str, cfg: dict) -> dict:
    rd = results_dir(cfg) / name
    h = torch.load(rd / "hidden.pt")
    labels = read_jsonl(rd / "labels.jsonl")
    base = {r["id"]: r for r in read_jsonl(rd / "baseline_scores.jsonl")}
    ids = [r["id"] for r in labels]
    assert ids == h["ids"], "hidden states and labels are not aligned"
    return {
        "name": name,
        "ids": ids,
        "questions": [r["question"] for r in labels],
        "y": np.array([r["incorrect"] for r in labels]),
        "X": {"prompt": h["prompt_last"].float().numpy(), "answer": h["answer_last"].float().numpy()},
        "layers": list(h["layers"]),
        "in_subset": np.array([r["in_sample_subset"] for r in labels]),
        "parsed": [r.get("parsed") for r in labels],
        "baselines": base,
    }


def split_by_question(n: int, seed: int, fracs=(0.70, 0.15, 0.15)) -> np.ndarray:
    """Return an array of 'train'/'val'/'test' per question (seeded permutation)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(round(fracs[0] * n)), int(round(fracs[1] * n))
    split = np.empty(n, dtype=object)
    split[perm[:n_tr]] = "train"
    split[perm[n_tr : n_tr + n_va]] = "val"
    split[perm[n_tr + n_va :]] = "test"
    return split


def assert_disjoint(ids: list[str], questions: list[str], split: np.ndarray) -> None:
    """No question (by id or by normalized text) may appear in more than one split."""
    parts = sorted(set(split))
    for i, a in enumerate(parts):
        for b in parts[i + 1 :]:
            ia = {ids[k] for k in np.where(split == a)[0]}
            ib = {ids[k] for k in np.where(split == b)[0]}
            qa = {normalize_answer(questions[k]) for k in np.where(split == a)[0]}
            qb = {normalize_answer(questions[k]) for k in np.where(split == b)[0]}
            assert not (ia & ib), f"question ids shared between {a} and {b}"
            assert not (qa & qb), f"question texts shared between {a} and {b}"


# --------------------------------------------------------------------------- probe
class Probe:
    """Standardize with train statistics, then L2-regularized logistic regression.
    score() returns the logit for "incorrect" (higher = more likely wrong)."""

    def __init__(self, C: float, seed: int):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
        self.C = C

    def fit(self, X, y):
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, X, scaler: StandardScaler | None = None):
        return self.clf.decision_function((scaler or self.scaler).transform(X))


def fit_select_C(Xtr, ytr, Xva, yva, Cs, seed) -> tuple[Probe, float]:
    """Fit one probe per C on train; keep the C with the best val AUROC."""
    best, best_auc = None, -1.0
    for C in Cs:
        p = Probe(C, seed).fit(Xtr, ytr)
        auc = roc_auc_score(yva, p.score(Xva))
        if auc > best_auc:
            best, best_auc = p, auc
    return best, best_auc


def sweep_layers(X3, y, tr, va, Cs, seed) -> list[dict]:
    """X3: [N, n_layers, d]. Returns per-layer fitted probe + val AUROC."""
    out = []
    for li in range(X3.shape[1]):
        p, auc = fit_select_C(X3[tr, li], y[tr], X3[va, li], y[va], Cs, seed)
        out.append({"layer_index": li, "probe": p, "val_auroc": auc})
    return out


# --------------------------------------------------------------------------- protocol A
def fixed_split_experiments(D: dict, cfg: dict) -> dict:
    pc = cfg["probe"]
    seed, Cs = cfg["seed"], pc["C_grid"]
    y, n = D["y"], len(D["y"])
    split = split_by_question(n, seed)
    assert_disjoint(D["ids"], D["questions"], split)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    idx_te = bootstrap_indices(int(te.sum()), seed=seed)
    res = {"split_sizes": {s: int((split == s).sum()) for s in ("train", "val", "test")},
           "test_n_incorrect": int(y[te].sum()), "layers": D["layers"],
           "layer_sweep": {}, "best": {}, "shuffled_control": {}}
    models = {}
    for pos in POSITIONS:
        X3 = D["X"][pos]
        sweep = sweep_layers(X3, y, tr, va, Cs, seed)
        rows = []
        for s in sweep:
            te_scores = s["probe"].score(X3[te, s["layer_index"]])
            rows.append({"layer": D["layers"][s["layer_index"]], "C": s["probe"].C,
                         "val_auroc": s["val_auroc"],
                         "test": auroc_with_ci(y[te], te_scores, idx_te)})
        res["layer_sweep"][pos] = rows
        bi = int(np.argmax([s["val_auroc"] for s in sweep]))  # selected on VAL only
        best = sweep[bi]
        res["best"][pos] = {"layer": D["layers"][bi], **{k: rows[bi][k] for k in ("C", "val_auroc", "test")}}
        models[pos] = {"probe": best["probe"], "layer_index": bi}

        # Shuffled-label control: same pipeline (C chosen on val) with permuted train/val
        # labels; evaluated on the TRUE test labels. Should be ~0.5.
        ctrl = []
        for k in range(pc["n_shuffles"]):
            rng = np.random.default_rng(seed + 100 + k)
            ytr_s, yva_s = rng.permutation(y[tr]), rng.permutation(y[va])
            p, _ = fit_select_C(X3[tr, bi], ytr_s, X3[va, bi], yva_s, Cs, seed)
            ctrl.append(roc_auc_score(y[te], p.score(X3[te, bi])))
        res["shuffled_control"][pos] = {"layer": D["layers"][bi], "test_aurocs": ctrl,
                                        "mean": float(np.mean(ctrl)), "std": float(np.std(ctrl))}

    # Baselines on the same test split (token log-prob exists for every question; the
    # answer-identity control is re-estimated from the TRAIN split only).
    b = D["baselines"]
    ids_te = [D["ids"][i] for i in np.where(te)[0]]
    res["baselines_test"] = {m: auroc_with_ci(y[te], np.array([b[i][m] for i in ids_te]), idx_te)
                             for m in ("neg_mean_logprob", "neg_min_logprob")}
    if D["name"] == "pubmed_qa":
        res["baselines_test"]["answer_identity"] = auroc_with_ci(
            y[te], answer_identity_from_train(D["parsed"], y, tr, te), idx_te)
    return {"results": res, "models": models, "split": split}


def answer_identity_from_train(parsed, y, tr, te) -> np.ndarray:
    keys = np.array([str(p) for p in parsed])
    rate = {c: y[tr & (keys == c)].mean() for c in np.unique(keys[tr])}
    return np.array([rate.get(k, y[tr].mean()) for k in keys[te]])


def transfer_experiments(Ds: dict, fixed: dict, cfg: dict) -> dict:
    """Train on source (its train split; layer/C chosen on its val split), test on the
    target's test split. Also reports target-standardized transfer, which re-fits only the
    feature standardization on the *unlabeled* target test features (no target labels)."""
    seed = cfg["seed"]
    out = {}
    for src in DATASETS:
        for tgt in DATASETS:
            Dt = Ds[tgt]
            te = fixed[tgt]["split"] == "test"
            yt = Dt["y"][te]
            idx = bootstrap_indices(int(te.sum()), seed=seed)
            key = f"{src}->{tgt}"
            out[key] = {}
            for pos in POSITIONS:
                m = fixed[src]["models"][pos]
                Xt = Dt["X"][pos][te, m["layer_index"]]
                r = {"layer": Ds[src]["layers"][m["layer_index"]], "C": m["probe"].C,
                     "test": auroc_with_ci(yt, m["probe"].score(Xt), idx)}
                if src != tgt:
                    r["test_target_standardized"] = auroc_with_ci(
                        yt, m["probe"].score(Xt, scaler=StandardScaler().fit(Xt)), idx)
                out[key][pos] = r
    return out


def transfer_layer_sweep(Ds: dict, fixed: dict, cfg: dict) -> dict:
    """Cross-domain AUROC at EVERY layer (C chosen on the source val split), on the
    target's test split and on all target questions. Shows how strongly transfer depends
    on the layer, and whether source-only layer selection finds a layer that transfers."""
    Cs, seed = cfg["probe"]["C_grid"], cfg["seed"]
    out = {}
    for src in DATASETS:
        for tgt in DATASETS:
            if src == tgt:
                continue
            Ss, Dt = fixed[src]["split"], Ds[tgt]
            tr, va = Ss == "train", Ss == "val"
            te = fixed[tgt]["split"] == "test"
            rows = {}
            for pos in POSITIONS:
                Xs, Xt = Ds[src]["X"][pos], Dt["X"][pos]
                rows[pos] = []
                for li, L in enumerate(Ds[src]["layers"]):
                    p, v = fit_select_C(Xs[tr, li], Ds[src]["y"][tr], Xs[va, li], Ds[src]["y"][va], Cs, seed)
                    s = p.score(Xt[:, li])
                    rows[pos].append({"layer": L, "source_val_auroc": v,
                                      "target_test_auroc": float(roc_auc_score(Dt["y"][te], s[te])),
                                      "target_all_auroc": float(roc_auc_score(Dt["y"], s))})
            out[f"{src}->{tgt}"] = rows
    return out


# --------------------------------------------------------------------------- protocol B
def crossfit_scores(D: dict, cfg: dict) -> dict:
    """Out-of-fold probe scores for every question (5 folds by question)."""
    pc = cfg["probe"]
    seed, Cs, K = cfg["seed"], pc["C_grid"], pc["crossfit_folds"]
    n, y = len(D["y"]), D["y"]
    fold = np.random.default_rng(seed + 7).permutation(n) % K
    scores = {pos: np.full(n, np.nan) for pos in POSITIONS}
    chosen = {pos: [] for pos in POSITIONS}
    for k in range(K):
        te = fold == k
        rest = np.where(~te)[0]
        # Inner train/val split of the remaining folds, in the same 70:15 proportion.
        perm = np.random.default_rng(seed + 1000 + k).permutation(rest)
        n_va = int(round(len(rest) * 0.15 / 0.85))
        va = np.zeros(n, bool)
        va[perm[:n_va]] = True
        tr = ~te & ~va
        split = np.where(te, "test", np.where(va, "val", "train"))
        assert_disjoint(D["ids"], D["questions"], split)
        for pos in POSITIONS:
            X3 = D["X"][pos]
            sweep = sweep_layers(X3, y, tr, va, Cs, seed)
            bi = int(np.argmax([s["val_auroc"] for s in sweep]))
            scores[pos][te] = sweep[bi]["probe"].score(X3[te, bi])
            chosen[pos].append({"fold": k, "layer": D["layers"][bi], "C": sweep[bi]["probe"].C})
    assert all(np.isfinite(s).all() for s in scores.values())
    return {"scores": scores, "chosen": chosen, "fold": fold}


BASELINE_ORDER = ["neg_mean_logprob", "neg_min_logprob", "answer_identity", "self_consistency",
                  "self_consistency_10", "semantic_entropy", "semantic_entropy_lenient",
                  "semantic_entropy_10", "semantic_entropy_10_lenient"]

# Target for the semantic-entropy probes, fixed BEFORE looking at any SEP result:
# the standard (strict) semantic entropy with the larger sample budget.
SEP_TARGET = "semantic_entropy_10"


def binarize_se(se_train: np.ndarray) -> float:
    """Threshold that splits SE values into two groups with minimal total within-group
    squared error (a 1-D two-means split), the "best split" binarization described for
    Semantic Entropy Probes (Kossen et al., 2024). Computed on training questions only."""
    vals = np.unique(se_train)
    if len(vals) == 1:
        return float(vals[0])
    best_t, best_sse = vals[0], np.inf
    for t in vals[:-1]:
        lo, hi = se_train[se_train <= t], se_train[se_train > t]
        sse = ((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum()
        if sse < best_sse:
            best_t, best_sse = t, sse
    return float(best_t)


def _fit_select(X3, target, tr, va, Cs, seed):
    """Layer + C chosen on val AUROC against `target`. Returns (layer_index, probe)."""
    sweep = sweep_layers(X3, target, tr, va, Cs, seed)
    bi = int(np.argmax([s["val_auroc"] for s in sweep]))
    return bi, sweep[bi]["probe"]


def sep_vs_accuracy_probe(Ds: dict, cfg: dict) -> dict:
    """Semantic Entropy Probes (Kossen et al., 2024) vs our accuracy probes.

    Both kinds of probe are trained on the SAME questions (the 500 sampled questions,
    the only ones with semantic entropy) so data size is matched:
      * accuracy probe : target = "answer is incorrect" (needs gold labels)
      * SEP            : target = binarized semantic entropy (needs no gold labels, but
                         needs 11 generations + NLI per training question)
    Layer and C are selected on an inner validation split against each probe's OWN
    target (an SEP never sees correctness labels). Evaluated as detectors of incorrect
    answers: (a) in-domain by 5-fold cross-fitting, (b) cross-domain (train on one
    dataset's 500 questions, test on the other's 500).
    """
    pc, seed = cfg["probe"], cfg["seed"]
    Cs, K = pc["C_grid"], pc["crossfit_folds"]
    out = {"target": SEP_TARGET, "in_domain": {}, "transfer": {}}
    data = {}
    for name, D in Ds.items():
        sub = np.where(D["in_subset"])[0]
        se = np.array([D["baselines"][D["ids"][i]][SEP_TARGET] for i in sub])
        data[name] = {"sub": sub, "se": se, "y": D["y"][sub]}

    def train(name, pos, kind, tr_mask, va_mask):
        D, d = Ds[name], data[name]
        X3 = D["X"][pos][d["sub"]]
        if kind == "accuracy":
            target = d["y"]
        else:
            t = binarize_se(d["se"][tr_mask])
            target = (d["se"] > t).astype(int)
            if target[tr_mask].min() == target[tr_mask].max() or target[va_mask].min() == target[va_mask].max():
                raise RuntimeError("degenerate SE binarization")
        bi, p = _fit_select(X3, target, tr_mask, va_mask, Cs, seed)
        return bi, p

    for name in Ds:
        d = data[name]
        n = len(d["sub"])
        fold = np.random.default_rng(seed + 11).permutation(n) % K
        idx = bootstrap_indices(n, seed=seed)
        out["in_domain"][name] = {}
        for pos in POSITIONS:
            X3 = Ds[name]["X"][pos][d["sub"]]
            scores = {"accuracy": np.full(n, np.nan), "sep": np.full(n, np.nan)}
            for k in range(K):
                te = fold == k
                rest = np.where(~te)[0]
                perm = np.random.default_rng(seed + 2000 + k).permutation(rest)
                va = np.zeros(n, bool)
                va[perm[: int(round(len(rest) * 0.15 / 0.85))]] = True
                tr = ~te & ~va
                for kind in scores:
                    bi, p = train(name, pos, kind, tr, va)
                    scores[kind][te] = p.score(X3[te, bi])
            res = {f"{kind}_probe": auroc_with_ci(d["y"], s, idx) for kind, s in scores.items()}
            res["delta_accuracy_minus_sep"] = auroc_diff_ci(d["y"], scores["accuracy"], scores["sep"], idx)
            out["in_domain"][name][pos] = res

    for src in Ds:
        for tgt in Ds:
            if src == tgt:
                continue
            ds, dt = data[src], data[tgt]
            n = len(ds["sub"])
            perm = np.random.default_rng(seed + 3000).permutation(n)
            va = np.zeros(n, bool)
            va[perm[: int(round(0.15 * n))]] = True
            tr = ~va
            idx = bootstrap_indices(len(dt["sub"]), seed=seed)
            out["transfer"][f"{src}->{tgt}"] = {}
            for pos in POSITIONS:
                Xt = Ds[tgt]["X"][pos][dt["sub"]]
                sc = {}
                for kind in ("accuracy", "sep"):
                    bi, p = train(src, pos, kind, tr, va)
                    sc[kind] = p.score(Xt[:, bi])
                r = {f"{kind}_probe": auroc_with_ci(dt["y"], s, idx) for kind, s in sc.items()}
                r["delta_accuracy_minus_sep"] = auroc_diff_ci(dt["y"], sc["accuracy"], sc["sep"], idx)
                out["transfer"][f"{src}->{tgt}"][pos] = r
    return out


def split_robustness(D: dict, cfg: dict, seeds=(42, 43, 44, 45, 46)) -> dict:
    """Re-run the fixed-split protocol (layer and C chosen on val) for several random
    70/15/15 splits; report the spread of test AUROC. Shows the headline numbers do not
    depend on one lucky split."""
    Cs = cfg["probe"]["C_grid"]
    y, n = D["y"], len(D["y"])
    b = D["baselines"]
    lp = np.array([b[i]["neg_mean_logprob"] for i in D["ids"]])
    per = {"probe_answer": [], "probe_prompt": [], "neg_mean_logprob": [], "best_layers": []}
    for s in seeds:
        split = split_by_question(n, s)
        assert_disjoint(D["ids"], D["questions"], split)
        tr, va, te = split == "train", split == "val", split == "test"
        layers = {}
        for pos in POSITIONS:
            bi, p = _fit_select(D["X"][pos], y, tr, va, Cs, cfg["seed"])
            per[f"probe_{pos}"].append(float(roc_auc_score(y[te], p.score(D["X"][pos][te, bi]))))
            layers[pos] = D["layers"][bi]
        per["neg_mean_logprob"].append(float(roc_auc_score(y[te], lp[te])))
        per["best_layers"].append(layers)
    summ = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "min": float(np.min(v)),
                "max": float(np.max(v)), "values": v}
            for m, v in per.items() if m != "best_layers"}
    summ["best_layers"] = per["best_layers"]
    summ["seeds"] = list(seeds)
    return summ


def compare_on_subset(D: dict, cf: dict, cfg: dict) -> dict:
    """Probe (out-of-fold) vs every baseline on the 500-question subset, paired bootstrap."""
    seed = cfg["seed"]
    b = D["baselines"]
    sub = np.where(D["in_subset"])[0]
    y = D["y"][sub]
    methods = {"probe_answer": cf["scores"]["answer"][sub],
               "probe_prompt": cf["scores"]["prompt"][sub]}
    first = b[D["ids"][sub[0]]]
    names = [m for m in BASELINE_ORDER if m in first]  # every baseline that was computed
    for m in names:
        methods[m] = np.array([b[D["ids"][i]][m] for i in sub])
    idx = bootstrap_indices(len(sub), seed=seed)
    res = {"subset": {m: auroc_with_ci(y, s, idx) for m, s in methods.items()}, "deltas": {}}
    for p in ("probe_answer", "probe_prompt"):
        for m in names:
            res["deltas"][f"{p} - {m}"] = auroc_diff_ci(y, methods[p], methods[m], idx)

    # Out-of-fold AUROC on ALL questions (more data, only token-prob baselines exist).
    yall = D["y"]
    idx_all = bootstrap_indices(len(yall), seed=seed)
    full = {"probe_answer": cf["scores"]["answer"], "probe_prompt": cf["scores"]["prompt"],
            "neg_mean_logprob": np.array([b[i]["neg_mean_logprob"] for i in D["ids"]])}
    if D["name"] == "pubmed_qa":
        full["answer_identity"] = np.array([b[i]["answer_identity"] for i in D["ids"]])
    res["full_oof"] = {m: auroc_with_ci(yall, s, idx_all) for m, s in full.items()}

    # PubMedQA: does the probe know more than which word was said? AUROC *within* each
    # predicted class, where the answer-identity control is constant by construction.
    if D["name"] == "pubmed_qa":
        res["within_answer_class"] = {}
        keys = np.array([str(p) for p in D["parsed"]])
        for c in np.unique(keys):
            m = keys == c
            if yall[m].min() == yall[m].max():
                continue
            idx_c = bootstrap_indices(int(m.sum()), seed=seed)
            res["within_answer_class"][c] = {
                meth: auroc_with_ci(yall[m], s[m], idx_c)
                for meth, s in full.items() if meth != "answer_identity"}
    return res


def probe_head_seconds(D: dict, model: dict, pos: str, reps: int = 20) -> float:
    """Wall-clock of applying a trained probe (standardize + dot product) per question."""
    X = D["X"][pos][:, model["layer_index"]]
    t0 = time.perf_counter()
    for _ in range(reps):
        model["probe"].score(X)
    return (time.perf_counter() - t0) / (reps * len(X))


# --------------------------------------------------------------------------- driver
def _fmt(r):
    return f"{r['auroc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


def main():
    cfg = load_config()
    Ds = {name: load_arrays(name, cfg) for name in DATASETS}
    fixed = {}
    for name, D in Ds.items():
        fixed[name] = fixed_split_experiments(D, cfg)
        r = fixed[name]["results"]
        update_metrics(cfg, ["probes", name], r)
        print(f"\n[{name}] split {r['split_sizes']}  (test incorrect: {r['test_n_incorrect']})")
        for pos in POSITIONS:
            print(f"  {pos:6s} layer sweep (val / test AUROC): " + "  ".join(
                f"L{s['layer']}:{s['val_auroc']:.3f}/{s['test']['auroc']:.3f}" for s in r["layer_sweep"][pos]))
            b = r["best"][pos]
            print(f"  {pos:6s} best-on-val layer {b['layer']} (C={b['C']}): test AUROC {_fmt(b['test'])}")
            c = r["shuffled_control"][pos]
            lo, hi = np.percentile(c["test_aurocs"], [5, 95])
            print(f"  {pos:6s} shuffled-label control ({len(c['test_aurocs'])} shuffles): "
                  f"mean {c['mean']:.3f} (sd {c['std']:.3f}, 5-95% [{lo:.3f}, {hi:.3f}])")
        for m, v in r["baselines_test"].items():
            print(f"  baseline on same test split  {m:18s} {_fmt(v)}")

    tr = transfer_experiments(Ds, fixed, cfg)
    update_metrics(cfg, ["transfer"], tr)
    tsweep = transfer_layer_sweep(Ds, fixed, cfg)
    update_metrics(cfg, ["transfer_layer_sweep"], tsweep)
    for key, d in tsweep.items():
        for pos, rows in d.items():
            print(f"  transfer sweep {key:22s} {pos:6s} (target, all questions): " + " ".join(
                f"L{r['layer']}:{r['target_all_auroc']:.3f}" for r in rows))
    print("\nTransfer (test split of target):")
    for key, d in tr.items():
        for pos, r in d.items():
            extra = (f" | target-standardized {_fmt(r['test_target_standardized'])}"
                     if "test_target_standardized" in r else "")
            print(f"  {key:22s} {pos:6s} L{r['layer']}: {_fmt(r['test'])}{extra}")

    for name, D in Ds.items():
        cf = crossfit_scores(D, cfg)
        comp = compare_on_subset(D, cf, cfg)
        comp["crossfit_chosen"] = cf["chosen"]
        update_metrics(cfg, ["comparison", name], comp)
        write_jsonl(results_dir(cfg) / name / "probe_scores.jsonl", [
            {"id": i, "fold": int(cf["fold"][k]), "probe_prompt": float(cf["scores"]["prompt"][k]),
             "probe_answer": float(cf["scores"]["answer"][k])} for k, i in enumerate(D["ids"])])
        print(f"\n[{name}] probe vs baselines on the 500-question subset (out-of-fold probe scores):")
        for m, v in comp["subset"].items():
            print(f"  {m:18s} {_fmt(v)}")
        for k, v in comp["deltas"].items():
            print(f"  delta {k:38s} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
        print("  full set (out-of-fold): " + ", ".join(f"{m} {_fmt(v)}" for m, v in comp["full_oof"].items()))
        for c, d in comp.get("within_answer_class", {}).items():
            print(f"  within class '{c}': " + ", ".join(f"{m} {_fmt(v)} (n={v['n']})" for m, v in d.items()))
        print("  layers chosen per fold: " + str({p: [c["layer"] for c in v] for p, v in cf["chosen"].items()}))

        # Costs of all methods incl. probes (probe head time added to the LLM stage time).
        metrics = json.loads((results_dir(cfg) / "metrics.json").read_text(encoding="utf-8"))
        if "llm" not in metrics.get("costs", {}).get(name, {}):
            continue  # no timing run for this setup (the 7B replication is not re-timed)
        costs = method_costs(metrics["costs"][name])
        for pos in POSITIONS:
            head = probe_head_seconds(D, fixed[name]["models"][pos], pos)
            m = f"probe_{pos}"
            costs[m]["probe_head_seconds"] = head
            costs[m]["seconds_total"] += head
            costs[m]["seconds_overhead"] += head
        if name != "pubmed_qa":
            costs.pop("answer_identity", None)
        update_metrics(cfg, ["costs", name, "methods"], costs)

    # Robustness: spread of test AUROC over 5 different random 70/15/15 splits.
    print("\nRobustness over 5 random splits (test AUROC, mean +- sd [min, max]):")
    for name, D in Ds.items():
        rob = split_robustness(D, cfg)
        update_metrics(cfg, ["robustness", name], rob)
        for m in ("probe_answer", "probe_prompt", "neg_mean_logprob"):
            v = rob[m]
            print(f"  {name:9s} {m:16s} {v['mean']:.3f} +- {v['std']:.3f} [{v['min']:.3f}, {v['max']:.3f}]")

    # Semantic Entropy Probes (Kossen et al., 2024) vs accuracy probes, matched data.
    if all(SEP_TARGET in Ds[n]["baselines"][Ds[n]["ids"][int(np.where(Ds[n]["in_subset"])[0][0])]]
           for n in Ds):
        sep = sep_vs_accuracy_probe(Ds, cfg)
        update_metrics(cfg, ["sep_comparison"], sep)
        print(f"\nAccuracy probe vs SEP (target: binarized {SEP_TARGET}), trained on the 500 sampled questions:")
        for scope in ("in_domain", "transfer"):
            for key, d in sep[scope].items():
                for pos, r in d.items():
                    dd = r["delta_accuracy_minus_sep"]
                    print(f"  {scope:9s} {key:22s} {pos:6s} accuracy {_fmt(r['accuracy_probe'])} | "
                          f"SEP {_fmt(r['sep_probe'])} | delta {dd['delta']:+.3f} "
                          f"[{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
