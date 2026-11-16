"""Generate every reported number and table from results/metrics.json (and the 7B run).

Usage:
    python -m src.report

Outputs:
    paper/numbers.tex          \\newcommand macros for every number quoted in the paper text
    paper/tables/*.tex         LaTeX tables
    results/tables.md          the same tables in Markdown (pasted into README.md)

Nothing in the README or paper is typed by hand: numbers come from here.
"""
from __future__ import annotations

import json
from pathlib import Path

from .evaluate import method_costs
from .utils import ROOT

DS = {"trivia_qa": ("Tq", "TriviaQA"), "pubmed_qa": ("Pm", "PubMedQA")}
METHODS = [  # (key, macro stem, LaTeX label, Markdown label)
    ("probe_answer", "ProbeAns", "Answer-token probe (ours)", "**Answer-token probe (ours)**"),
    ("probe_prompt", "ProbePrompt", "Prompt-only probe (ours)", "**Prompt-only probe (ours)**"),
    ("neg_mean_logprob", "LogProb", "Token log-prob (mean)", "Token log-prob (mean)"),
    ("answer_identity", "AnsId", "Answer identity (control)", "Answer identity (control)"),
    ("self_consistency", "SCfive", "Self-consistency, 5 samples", "Self-consistency, 5 samples"),
    ("self_consistency_10", "SCten", "Self-consistency, 10 samples", "Self-consistency, 10 samples"),
    ("semantic_entropy", "SEfive", "Semantic entropy, 5 samples", "Semantic entropy, 5 samples"),
    ("semantic_entropy_10", "SEten", "Semantic entropy, 10 samples", "Semantic entropy, 10 samples"),
    ("semantic_entropy_10_lenient", "SEtenLen", "Semantic entropy, 10, lenient",
     "Semantic entropy, 10 samples, lenient"),
]
STEM = {k: s for k, s, _, _ in METHODS}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def f3(x):
    return f"{x:.3f}"


def ci(r):
    return f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


def sgn(x):
    return f"{x:+.3f}"


def pct(x):
    """Percentage with exact round-half-up (0.4255 -> 42.6, not 42.5 from float error)."""
    from decimal import ROUND_HALF_UP, Decimal

    return str((Decimal(repr(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def ms(x):
    return f"{1000 * x:.1f}"


def costs_for(M: dict, ds: str) -> dict:
    """Per-method costs, recomputed from the measured stages (so late timing runs are
    included), with the probe-head time recorded by src/probe.py added to the probes."""
    c = method_costs(M["costs"][ds])
    for p in ("probe_answer", "probe_prompt"):
        head = M["costs"][ds].get("methods", {}).get(p, {}).get("probe_head_seconds", 0.0)
        c[p]["seconds_total"] += head
        c[p]["seconds_overhead"] += head
    return c


def build_macros(M: dict, M7: dict | None) -> dict[str, str]:
    mac = {}
    for ds, (p, _) in DS.items():
        lab = M["labeling"][ds]
        mac[f"{p}N"] = f"{lab['n']:,}"
        mac[f"{p}Acc"] = pct(lab["accuracy"])
        mac[f"{p}AccCI"] = f"{pct(lab['accuracy_ci95'][0])}--{pct(lab['accuracy_ci95'][1])}"
        sub = M["comparison"][ds]["subset"]
        mac[f"{p}SubN"] = str(sub["probe_answer"]["n"])
        for k, r in sub.items():
            if k in STEM:
                mac[f"{p}Auc{STEM[k]}"] = f3(r["auroc"])
                mac[f"{p}Ci{STEM[k]}"] = ci(r)
        for name, d in M["comparison"][ds]["deltas"].items():
            a, b = name.split(" - ")
            if a in STEM and b in STEM:
                mac[f"{p}D{STEM[a]}vs{STEM[b]}"] = sgn(d["delta"])
                mac[f"{p}DCi{STEM[a]}vs{STEM[b]}"] = f"[{sgn(d['ci95'][0])}, {sgn(d['ci95'][1])}]"
        for k, c in costs_for(M, ds).items():
            if k in STEM:
                mac[f"{p}Ms{STEM[k]}"] = ms(c["seconds_total"])
                mac[f"{p}Over{STEM[k]}"] = ms(c["seconds_overhead"])
                mac[f"{p}Fp{STEM[k]}"] = f"{c['llm_forward_passes'] + c['nli_forward_passes']:.1f}"
        best = M["probes"][ds]["best"]
        for pos, stem in (("answer", "ProbeAns"), ("prompt", "ProbePrompt")):
            mac[f"{p}Test{stem}"] = f3(best[pos]["test"]["auroc"])
            mac[f"{p}TestCi{stem}"] = ci(best[pos]["test"])
            mac[f"{p}Layer{stem}"] = str(best[pos]["layer"])
            sc = M["probes"][ds]["shuffled_control"][pos]
            mac[f"{p}Shuf{stem}"] = f3(sc["mean"])
            mac[f"{p}ShufSd{stem}"] = f3(sc["std"])
            rob = M["robustness"][ds][f"probe_{pos}"]
            mac[f"{p}Rob{stem}"] = f"{rob['mean']:.3f} $\\pm$ {rob['std']:.3f}"
        rob = M["robustness"][ds]["neg_mean_logprob"]
        mac[f"{p}RobLogProb"] = f"{rob['mean']:.3f} $\\pm$ {rob['std']:.3f}"
        ab = M["abstention"][ds]
        mac[f"{p}AccAll"] = pct(ab["accuracy_full_coverage"])
        for k, v in ab["methods"].items():
            if k in STEM:
                mac[f"{p}Aurac{STEM[k]}"] = f3(v["aurac"])
                mac[f"{p}HalfAcc{STEM[k]}"] = pct(v['accuracy_at_coverage']['0.5'])
        st = M["baselines"][ds]["semantic_entropy_stats"]
        mac[f"{p}SEsingle"] = f"{100 * st['semantic_entropy_10']['frac_single_cluster']:.0f}"
    for tag, X in (("Three", M), ("Seven", M7)):
        if X and "runtime" in X:
            mac[f"Gpu{tag}Min"] = f"{X['runtime']['total_gpu_minutes']:.0f}"
            mac[f"Vram{tag}"] = f"{max(d['peak_vram_gb'] for k, d in X['runtime'].items() if k in DS):.1f}"
    # PubMedQA within-class analysis (answer-identity confound)
    wc = M["comparison"]["pubmed_qa"]["within_answer_class"]
    for c, d in wc.items():
        C = c.capitalize()
        mac[f"PmWithin{C}N"] = str(d["probe_answer"]["n"])
        for k in ("probe_answer", "probe_prompt", "neg_mean_logprob"):
            mac[f"PmWithin{C}{STEM[k]}"] = f3(d[k]["auroc"])
    # Transfer
    for key, d in M["transfer"].items():
        s, t = key.split("->")
        if s == t:
            continue
        tag = DS[s][0] + "to" + DS[t][0]
        for pos, stem in (("answer", "ProbeAns"), ("prompt", "ProbePrompt")):
            mac[f"Tr{tag}{stem}"] = f3(d[pos]["test"]["auroc"])
            mac[f"TrCi{tag}{stem}"] = ci(d[pos]["test"])
    # Transfer at every layer (C chosen on source val), AUROC on ALL target questions
    for key, d in M.get("transfer_layer_sweep", {}).items():
        s, t = key.split("->")
        tag = DS[s][0] + "to" + DS[t][0]
        for pos, stem in (("answer", "ProbeAns"), ("prompt", "ProbePrompt")):
            rows = d[pos]
            vals = [r["target_all_auroc"] for r in rows]
            sel = M["probes"][s]["best"][pos]["layer"]  # chosen on source val
            r_sel = next(r for r in rows if r["layer"] == sel)
            r_best = max(rows, key=lambda r: r["target_all_auroc"])
            mac[f"TrSweep{tag}{stem}Min"] = f3(min(vals))
            mac[f"TrSweep{tag}{stem}Max"] = f3(max(vals))
            mac[f"TrSweep{tag}{stem}Sel"] = f"layer {sel}: {f3(r_sel['target_all_auroc'])}"
            mac[f"TrSweep{tag}{stem}Best"] = f"layer {r_best['layer']}: {f3(r_best['target_all_auroc'])}"
    # SEP comparison
    sep = M.get("sep_comparison")
    if sep:
        for scope, block in (("In", sep["in_domain"]), ("Tr", sep["transfer"])):
            for key, d in block.items():
                tag = DS[key][0] if scope == "In" else "".join(DS[x][0] for x in key.split("->"))
                for pos, stem in (("answer", "Ans"), ("prompt", "Prompt")):
                    r = d[pos]
                    mac[f"Sep{scope}{tag}{stem}Acc"] = f3(r["accuracy_probe"]["auroc"])
                    mac[f"Sep{scope}{tag}{stem}Sep"] = f3(r["sep_probe"]["auroc"])
                    dd = r["delta_accuracy_minus_sep"]
                    mac[f"Sep{scope}{tag}{stem}D"] = sgn(dd["delta"])
                    mac[f"Sep{scope}{tag}{stem}DCi"] = f"[{sgn(dd['ci95'][0])}, {sgn(dd['ci95'][1])}]"
    # 7B replication
    if M7:
        for ds, (p, _) in DS.items():
            if ds not in M7.get("comparison", {}):
                continue
            mac[f"Sv{p}Acc"] = pct(M7["labeling"][ds]["accuracy"])
            for k, r in M7["comparison"][ds]["subset"].items():
                if k in STEM:
                    mac[f"Sv{p}Auc{STEM[k]}"] = f3(r["auroc"])
                    mac[f"Sv{p}Ci{STEM[k]}"] = ci(r)
            for name, d in M7["comparison"][ds]["deltas"].items():
                a, b = name.split(" - ")
                if a in STEM and b in STEM:
                    mac[f"Sv{p}D{STEM[a]}vs{STEM[b]}"] = sgn(d["delta"])
                    mac[f"Sv{p}DCi{STEM[a]}vs{STEM[b]}"] = f"[{sgn(d['ci95'][0])}, {sgn(d['ci95'][1])}]"
            best = M7["probes"][ds]["best"]
            for pos, stem in (("answer", "ProbeAns"), ("prompt", "ProbePrompt")):
                mac[f"Sv{p}Test{stem}"] = f3(best[pos]["test"]["auroc"])
                mac[f"Sv{p}Layer{stem}"] = str(best[pos]["layer"])
                mac[f"Sv{p}Shuf{stem}"] = f3(M7["probes"][ds]["shuffled_control"][pos]["mean"])
                rob = M7["robustness"][ds][f"probe_{pos}"]
                mac[f"Sv{p}Rob{stem}"] = f"{rob['mean']:.3f} $\\pm$ {rob['std']:.3f}"
        for key, d in M7.get("transfer", {}).items():
            s, t = key.split("->")
            if s != t:
                for pos, stem in (("answer", "ProbeAns"), ("prompt", "ProbePrompt")):
                    mac[f"SvTr{DS[s][0]}to{DS[t][0]}{stem}"] = f3(d[pos]["test"]["auroc"])
    # Split sizes and the PubMedQA prompt-selection facts (from the archived v1 run + pilot)
    for ds, (p, _) in DS.items():
        mac[f"{p}TestN"] = str(M["probes"][ds]["split_sizes"]["test"])
    ps = M.get("prompt_selection")
    if ps:
        mac["PilotN"] = str(ps["n_pilot"])
        mac["PmVOneMaybePct"] = f"{100 * ps['v1_maybe_rate']:.0f}"
        mac["PmGoldMaybePct"] = f"{100 * ps['gold_maybe_rate']:.0f}"
        mac["PmVThreeMaybePct"] = f"{100 * ps['final_maybe_rate']:.0f}"
    return mac


def record_prompt_selection(results: Path) -> None:
    """Store the facts behind the PubMedQA prompt change in metrics.json, computed from
    the archived v1 outputs, the pilot file and the final outputs."""
    from .label import parse_yes_no_maybe
    from .utils import read_jsonl, update_metrics

    v1 = results / "archive" / "pubmed_qa_prompt_v1" / "greedy.jsonl"
    pilot = load(results / "pubmed_prompt_pilot.json")
    if not v1.exists() or not pilot:
        return
    rows_v1, rows_v3 = read_jsonl(v1), read_jsonl(results / "pubmed_qa" / "greedy.jsonl")
    rate = lambda rows: sum(parse_yes_no_maybe(r["answer"])[0] == "maybe" for r in rows) / len(rows)  # noqa: E731
    update_metrics({"paths": {"results": str(results.relative_to(ROOT))}}, ["prompt_selection"], {
        "n_pilot": pilot["n_pilot"], "pilot": {k: v for k, v in pilot.items() if k.startswith("v")},
        "v1_maybe_rate": rate(rows_v1), "final_maybe_rate": rate(rows_v3),
        "gold_maybe_rate": sum(r["gold"] == "maybe" for r in rows_v3) / len(rows_v3)})


def write_macros(mac: dict, path: Path) -> None:
    lines = ["% AUTO-GENERATED by src/report.py from results/metrics.json. Do not edit.\n"]
    for k in sorted(mac):
        assert k.isalpha(), f"LaTeX macro names must be letters only: {k}"
        lines.append(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main_tables(M: dict) -> tuple[str, str]:
    """Main comparison: AUROC [CI] and extra latency per method, both datasets."""
    costs = {ds: costs_for(M, ds) for ds in DS}
    best = {ds: max(v["auroc"] for k, v in M["comparison"][ds]["subset"].items() if k in STEM)
            for ds in DS}
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{TriviaQA} & \multicolumn{2}{c}{PubMedQA} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Method & AUROC [95\% CI] & +ms & AUROC [95\% CI] & +ms \\", r"\midrule"]
    md = ["| Method | TriviaQA AUROC [95% CI] | +ms | PubMedQA AUROC [95% CI] | +ms |",
          "|---|---|---|---|---|"]
    for k, _, tl, ml in METHODS:
        cells_t, cells_m = [], []
        for ds in DS:
            r = M["comparison"][ds]["subset"].get(k)
            c = costs[ds].get(k)
            if r is None:
                cells_t += ["--", "--"]
                cells_m += ["—", "—"]
                continue
            a = f3(r["auroc"])
            bold = abs(r["auroc"] - best[ds]) < 1e-12
            over = ms(c["seconds_overhead"]) if c else "--"
            if k == "probe_prompt" and c:
                over = f"({ms(c['seconds_total'])})"  # runs before answering: full prefill cost
            cells_t += [(rf"\textbf{{{a}}}" if bold else a) + f" {ci(r)}", over]
            cells_m += [(f"**{a}**" if bold else a) + f" {ci(r)}", over if c else "—"]
        tex.append(f"{tl} & " + " & ".join(cells_t) + r" \\")
        if k == "answer_identity":
            tex.append(r"\midrule")
        md.append(f"| {ml} | " + " | ".join(cells_m) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex) + "\n", "\n".join(md) + "\n"


def transfer_tables(M: dict) -> tuple[str, str]:
    rows = [("trivia_qa->trivia_qa", "TriviaQA $\\to$ TriviaQA (in-domain)"),
            ("pubmed_qa->pubmed_qa", "PubMedQA $\\to$ PubMedQA (in-domain)"),
            ("trivia_qa->pubmed_qa", "TriviaQA $\\to$ PubMedQA"),
            ("pubmed_qa->trivia_qa", "PubMedQA $\\to$ TriviaQA")]
    tex = [r"\begin{tabular}{lcc}", r"\toprule", r"Train $\to$ test & Prompt-only probe & Answer-token probe \\",
           r"\midrule"]
    md = ["| Train → test | Prompt-only probe | Answer-token probe |", "|---|---|---|"]
    for key, label in rows:
        d = M["transfer"][key]
        tex.append(f"{label} & {f3(d['prompt']['test']['auroc'])} {ci(d['prompt']['test'])} & "
                   f"{f3(d['answer']['test']['auroc'])} {ci(d['answer']['test'])} \\\\")
        md.append(f"| {label.replace(chr(36) + chr(92) + 'to' + chr(36), '→')} | {f3(d['prompt']['test']['auroc'])} "
                  f"{ci(d['prompt']['test'])} | {f3(d['answer']['test']['auroc'])} {ci(d['answer']['test'])} |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex) + "\n", "\n".join(md) + "\n"


def sep_tables(M: dict) -> tuple[str, str]:
    sep = M["sep_comparison"]
    tex = [r"\begin{tabular}{llccc}", r"\toprule",
           r"Setting & Token & Accuracy probe & SEP & $\Delta$ [95\% CI] \\", r"\midrule"]
    md = ["| Setting | Token | Accuracy probe | SEP | Δ (accuracy − SEP) [95% CI] |", "|---|---|---|---|---|"]
    entries = [(f"{DS[k][1]} (in-domain)", v) for k, v in sep["in_domain"].items()] + \
              [(" $\\to$ ".join(DS[x][1] for x in k.split("->")), v) for k, v in sep["transfer"].items()]
    for label, d in entries:
        for pos, pl in (("prompt", "prompt"), ("answer", "answer")):
            r = d[pos]
            dd = r["delta_accuracy_minus_sep"]
            dci = f"[{sgn(dd['ci95'][0])}, {sgn(dd['ci95'][1])}]"
            tex.append(f"{label} & {pl} & {f3(r['accuracy_probe']['auroc'])} & {f3(r['sep_probe']['auroc'])} & "
                       f"{sgn(dd['delta'])} {dci} \\\\")
            md.append(f"| {label.replace(' $' + chr(92) + 'to$ ', ' → ')} | {pl} | {f3(r['accuracy_probe']['auroc'])} | "
                      f"{f3(r['sep_probe']['auroc'])} | {sgn(dd['delta'])} {dci} |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex) + "\n", "\n".join(md) + "\n"


def scale_tables(M: dict, M7: dict) -> tuple[str, str]:
    keys = ["probe_answer", "probe_prompt", "neg_mean_logprob", "self_consistency_10", "semantic_entropy_10"]
    lbl = {k: (tl, ml) for k, _, tl, ml in METHODS}
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{TriviaQA} & \multicolumn{2}{c}{PubMedQA} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & 3B & 7B (4-bit) & 3B & 7B (4-bit) \\", r"\midrule"]
    md = ["| | TriviaQA 3B | TriviaQA 7B (4-bit) | PubMedQA 3B | PubMedQA 7B (4-bit) |", "|---|---|---|---|---|"]
    acc = [f"{pct(X['labeling'][ds]['accuracy'])}\\%" for ds in DS for X in (M, M7)]
    tex.append("Model accuracy & " + " & ".join(acc) + r" \\")
    md.append("| Model accuracy | " + " | ".join(a.replace("\\%", "%") for a in acc) + " |")
    tex.append(r"\midrule")
    for k in keys:
        vals = []
        for ds in DS:
            for X in (M, M7):
                r = X["comparison"][ds]["subset"].get(k)
                vals.append(f3(r["auroc"]) if r else "--")
        tex.append(f"{lbl[k][0]} & " + " & ".join(vals) + r" \\")
        md.append(f"| {lbl[k][1]} | " + " | ".join(vals) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex) + "\n", "\n".join(md) + "\n"


def render_readme(mac: dict, md_tables: dict) -> None:
    """README.template.md -> README.md: {{Name}} becomes the macro value, {{TABLE:x}} the
    Markdown table x. Fails loudly on any placeholder that has no value."""
    import re

    tpl = ROOT / "README.template.md"
    if not tpl.exists():
        return
    text = tpl.read_text(encoding="utf-8")

    def sub(m):
        key = m.group(1)
        if key.startswith("TABLE:"):
            return md_tables[key[6:]].strip()
        if key not in mac:
            raise KeyError(f"README placeholder without a value: {key}")
        return mac[key].replace("$\\pm$", "±").replace("--", "–")

    out = re.sub(r"\{\{([A-Za-z:_]+)\}\}", sub, text)
    (ROOT / "README.md").write_text(
        "<!-- AUTO-GENERATED from README.template.md by src/report.py. Edit the template. -->\n" + out,
        encoding="utf-8")


def record_runtime(results: Path) -> dict | None:
    """Sum the measured generation + NLI wall-clock of one run into metrics.json."""
    from .utils import update_metrics

    if not (results / "metrics.json").exists():
        return None
    M = load(results / "metrics.json")
    rt = {}
    for ds in DS:
        meta, extra = load(results / ds / "generation_meta.json"), load(results / ds / "generation_meta_extra.json")
        if not meta:
            continue
        nli = M.get("costs", {}).get(ds, {})
        nli_s = nli.get("nli_clustering", {}).get("seconds_total", 0.0) + sum(
            v["seconds_total"] for v in nli.get("nli_clustering_variants", {}).values())
        rt[ds] = {"greedy_and_hidden_s": meta["greedy"]["seconds"], "sampling_5_s": meta["sampling"]["seconds"],
                  "sampling_extra_5_s": extra["seconds"] if extra else 0.0, "nli_all_variants_s": nli_s,
                  "peak_vram_gb": max(meta["greedy"]["peak_alloc_gb"], meta["sampling"]["peak_alloc_gb"])}
    rt["total_gpu_minutes"] = sum(sum(v for k, v in d.items() if k.endswith("_s")) for d in rt.values()) / 60
    cfg = {"paths": {"results": str(results.relative_to(ROOT))}}
    update_metrics(cfg, ["runtime"], rt)
    return rt


def main():
    rt3, rt7 = record_runtime(ROOT / "results"), record_runtime(ROOT / "results_7b")
    record_prompt_selection(ROOT / "results")
    M = load(ROOT / "results" / "metrics.json")
    M7 = load(ROOT / "results_7b" / "metrics.json")
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    mac = build_macros(M, M7)
    write_macros(mac, ROOT / "paper" / "numbers.tex")
    md_parts = []
    for name, fn, args in [("main", main_tables, (M,)), ("transfer", transfer_tables, (M,)),
                           ("sep", sep_tables, (M,)) if "sep_comparison" in M else (None, None, None),
                           ("scale", scale_tables, (M, M7)) if M7 and "comparison" in M7 else (None, None, None)]:
        if name is None:
            continue
        tex, md = fn(*args)
        (ROOT / "paper" / "tables" / f"{name}.tex").write_text(tex, encoding="utf-8")
        md_parts.append(f"### {name}\n\n{md}")
    (ROOT / "results" / "tables.md").write_text("\n".join(md_parts), encoding="utf-8")
    render_readme(mac, {p.split("\n", 1)[0][4:]: p.split("\n", 1)[1] for p in md_parts})
    print(f"wrote {len(mac)} macros to paper/numbers.tex; tables: "
          f"{[p.name for p in (ROOT / 'paper' / 'tables').glob('*.tex')]}")


if __name__ == "__main__":
    main()
