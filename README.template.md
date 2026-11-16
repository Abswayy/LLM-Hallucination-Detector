# Cheap Hallucination Detection with Hidden-State Probes

Can a single linear probe on an LLM's hidden state catch hallucinated answers as well as
expensive sampling-based methods like semantic entropy? We test this on
**Qwen2.5-3B-Instruct** (plus a 4-bit 7B replication) on TriviaQA and PubMedQA, running
everything on one 12 GB laptop GPU. We measure accuracy and real cost, include the
controls these comparisons often skip, and report what doesn't work as well as what does.

## Headline result

On the same 500 questions per dataset, a logistic-regression probe on the **last answer
token** detects incorrect answers with **AUROC {{TqAucProbeAns}}** (TriviaQA) and
**{{PmAucProbeAns}}** (PubMedQA). That **beats semantic entropy with 10 samples**
({{TqAucSEten}} / {{PmAucSEten}}; paired-bootstrap Δ {{TqDProbeAnsvsSEten}} {{TqDCiProbeAnsvsSEten}} and
{{PmDProbeAnsvsSEten}} {{PmDCiProbeAnsvsSEten}}) while adding **{{TqOverProbeAns}} ms** per question instead of
**{{TqOverSEten}} ms** (TriviaQA). A probe that reads only the **prompt**, before any token
is generated, reaches **{{TqAucProbePrompt}}**, statistically tied with 10-sample semantic entropy
(Δ {{TqDProbePromptvsSEten}} {{TqDCiProbePromptvsSEten}}) at {{TqMsProbePrompt}} ms instead of {{TqMsSEten}} ms.

![Cost vs AUROC](figures/fig3_cost_vs_auroc.png)

## Results (Qwen2.5-3B-Instruct, 500 questions per dataset)

AUROC for detecting incorrect answers, with 95% bootstrap CIs. **+ms** is the measured
latency added on top of generating the answer (RTX 4080 Laptop GPU, same batching for
every method). The prompt-only probe shows its full cost in parentheses, because it runs
*before* generation. Probe scores are out-of-fold (5-fold cross-fitting by question).

{{TABLE:main}}

Model accuracy: TriviaQA {{TqAcc}}% (n={{TqN}}), PubMedQA {{PmAcc}}% (n={{PmN}}).

![AUROC comparison](figures/fig1_auroc_comparison.png)

## What else we found

**It is not a lucky split.** Over 5 random 70/15/15 splits (layer and regularization
chosen on validation only), the answer-token probe scores {{TqRobProbeAns}} on TriviaQA
(token log-prob: {{TqRobLogProb}}). A 200-permutation shuffled-label control averages
{{TqShufProbeAns}} (TriviaQA) and {{PmShufProbeAns}} (PubMedQA): no leakage.

**PubMedQA needs an answer-identity control.** Errors depend heavily on which word the
model says (it is usually wrong when it says "maybe"). Predicting errors from the
yes/no/maybe word alone already gives {{PmAucAnsId}}. The probes beat this control
(Δ {{PmDProbeAnsvsAnsId}} {{PmDCiProbeAnsvsAnsId}}). The extra signal comes mostly from
spotting wrong "no" answers: within "no" answers the prompt probe scores {{PmWithinNoProbePrompt}}.

**Accuracy probes vs. Semantic Entropy Probes (Kossen et al., 2024)**, trained on the same
500 questions. In-domain, accuracy probes match or beat SE probes. Out of domain the
result depends on direction, consistent in part with Kossen et al.'s claim that SE probes
generalize better.

{{TABLE:sep}}

**Cross-domain transfer is fragile.** A probe trained on TriviaQA and tested on PubMedQA
ranges from {{TrSweepTqtoPmProbeAnsMin}} to {{TrSweepTqtoPmProbeAnsMax}} AUROC depending on
the layer. The layer that is best in-domain ({{TrSweepTqtoPmProbeAnsSel}}) is not the one
that transfers best ({{TrSweepTqtoPmProbeAnsBest}}).

{{TABLE:transfer}}

**7B replication (4-bit NF4).** Same questions, prompts and protocol:

{{TABLE:scale}}

## What does *not* work, or needs care

- **Transfer.** Probes need in-domain labels. Choosing the layer on source-domain data
  does not reliably find a layer that transfers.
- **Prompt-only on PubMedQA.** The last prompt token's final layer *produces* the
  one-word answer, so "before answering" there means "before the answer is emitted". The
  long abstract also makes the time saving small.
- **Token log-prob on 7B PubMedQA** is close to chance: the 7B model hedges with "maybe"
  and is about as confident when hedging as when not.
- **Short answers only.** Semantic entropy was designed for, and may do better on,
  longer generations.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
bash scripts/run_all.sh                               # 3B pipeline, every step cached and resumable
HP_CONFIG=config_7b.yaml bash scripts/run_all.sh      # 7B (4-bit) replication into results_7b/
python -m src.report                                  # regenerates paper/numbers.tex, tables, this README
```

Every number in this README and in `paper/main.tex` is generated from
`results/metrics.json` by `src/report.py` (this file is rendered from
`README.template.md`).

**Hardware and runtime.** RTX 4080 Laptop GPU (12 GB), Windows 11, Python 3.12, PyTorch
2.14 (CUDA 12.6). Generation, sampling and NLI took about **{{GpuThreeMin}} GPU-minutes**
for 3B (peak {{VramThree}} GB) and **{{GpuSevenMin}} GPU-minutes** for 7B in 4-bit (peak
{{VramSeven}} GB). Probe training runs on CPU.

## Repository layout

```
config.yaml / config_7b.yaml   every setting (seed 42)
src/data.py         dataset sampling + prompt formatting
src/generate.py     greedy answers, per-token log-probs, hidden states, sampled answers
src/label.py        correctness labels (SQuAD-normalized alias match; yes/no/maybe parsing)
src/baselines.py    token log-prob, self-consistency, semantic entropy (5/10 samples, strict/lenient)
src/probe.py        probes, layer sweep, prompt-only, transfer, shuffled control, SEP comparison
src/timing.py       clean per-stage latency on an idle GPU
src/evaluate.py     AUROC + bootstrap CIs, abstention curves, cost accounting
src/plots.py        figures (PDF + PNG)
src/report.py       paper macros, tables and this README from metrics.json
NOTES.md            plain-English explanation of every step and decision
paper/main.tex      4-page workshop-style write-up
```

## Limitations

One model family; two short-answer datasets. Labels come from string matching: noisy
TriviaQA aliases cause some false positives and formatting differences some false
negatives (not quantified; see `NOTES.md`). PubMedQA test splits are small
({{PmTestN}} questions), so its CIs are wide. The PubMedQA prompt was adjusted once, using
only the answer distribution on {{PilotN}} questions, before any labels existed. The 7B model is
4-bit quantized, which may perturb hidden states. Full discussion: `NOTES.md` and the
paper's Limitations section.
