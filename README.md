<!-- AUTO-GENERATED from README.template.md by src/report.py. Edit the template. -->
# LLM Hallucination Detector

**Catching an LLM's wrong answers from a single hidden state: more accurate than semantic entropy, with ~434× less added latency.**

A linear probe reads one internal activation of the model and predicts whether the answer
it just gave is wrong. On Qwen2.5 (3B, and a 4-bit 7B replication), across 3,000 trivia
and biomedical questions, it beats semantic entropy, a leading sampling-based method, on accuracy at a
fraction of the cost. Everything runs on a single 12 GB laptop GPU.

📄 **Paper:** [Research Report.pdf](Research%20Report.pdf) (4 pages) · 🧪 **Every number here is generated from** [`results/metrics.json`](results/metrics.json)

---

## Highlights

| | TriviaQA | PubMedQA |
|---|---|---|
| **Hidden-state probe** (ours), AUROC | **0.881** | **0.839** |
| Semantic entropy, 10 samples, AUROC | 0.833 | 0.701 |
| Difference (paired bootstrap, 95% CI) | +0.048 [+0.016, +0.085] | +0.138 [+0.090, +0.186] |
| Added latency per question: probe vs. semantic entropy | **0.8 ms** vs. 337.0 ms | **1.8 ms** vs. 610.9 ms |

- 🎯 **More accurate and far cheaper.** The probe beats semantic entropy and
  self-consistency (both with 10 samples), and every difference is statistically
  significant.
- ⏱️ **Predicts errors before the model answers.** From the prompt alone, the probe reaches
  **0.821** on TriviaQA, statistically tied with 10-sample semantic entropy, at
  6.8 ms instead of 361.1 ms.
- 📈 **Holds at 7B.** With Qwen2.5-7B (4-bit), the probe scores 0.889 vs. 0.830
  for semantic entropy on TriviaQA (Δ +0.059 [+0.027, +0.093]).
- 🧯 **Practical payoff.** If the model answers only the half of TriviaQA questions the probe
  is most confident about, accuracy rises from 43.8% to 72.8%.

![Detection quality vs. cost](figures/fig3_cost_vs_auroc.png)

---

## How it works

```mermaid
flowchart LR
    Q[Question] --> LLM[LLM generates<br/>one answer]
    LLM -->|hidden state of the<br/>last answer token| P[Linear probe<br/>2,048 weights]
    P --> R[Risk that the<br/>answer is wrong]
    Q -.->|prompt only, before answering| LLM2[LLM reads<br/>the question] -.->|hidden state of the<br/>last prompt token| P2[Prompt-only probe] -.-> R2[Risk that the<br/>answer will be wrong]
```

**Semantic entropy**, the baseline, asks the model the same question 10 more times, groups
the answers by meaning with an NLI model (DeBERTa-large-MNLI), and scores disagreement.
That is accurate but costs about 10 extra generations plus NLI calls per query.

**The probe** is a logistic regression on the model's residual stream at one layer. Because
it is linear, it can only work if the model already represents "I'm probably wrong"
internally. It costs one dot product (microseconds) on an activation the model computes
anyway.

---

## Results

AUROC for detecting incorrect answers on the **same 500 questions per dataset**, with 95%
bootstrap CIs (Qwen2.5-3B-Instruct). **+ms** is measured latency added on top of
generating the answer (RTX 4080 Laptop GPU, identical batching for every method). The
prompt-only probe shows its full cost in parentheses, since it runs *before* generation.

| Method | TriviaQA AUROC [95% CI] | +ms | PubMedQA AUROC [95% CI] | +ms |
|---|---|---|---|---|
| **Answer-token probe (ours)** | **0.881** [0.852, 0.908] | 0.8 | **0.839** [0.800, 0.873] | 1.8 |
| **Prompt-only probe (ours)** | 0.821 [0.781, 0.858] | (6.8) | 0.830 [0.791, 0.864] | (57.6) |
| Token log-prob (mean) | 0.837 [0.802, 0.870] | 0.0 | 0.731 [0.685, 0.775] | 0.0 |
| Answer identity (control) | — | — | 0.753 [0.708, 0.794] | 0.0 |
| Self-consistency, 5 samples | 0.823 [0.787, 0.858] | 87.6 | 0.669 [0.624, 0.712] | 311.5 |
| Self-consistency, 10 samples | 0.845 [0.809, 0.878] | 149.7 | 0.698 [0.649, 0.741] | 588.9 |
| Semantic entropy, 5 samples | 0.817 [0.782, 0.854] | 169.2 | 0.671 [0.624, 0.715] | 327.3 |
| Semantic entropy, 10 samples | 0.833 [0.799, 0.867] | 337.0 | 0.701 [0.652, 0.745] | 610.9 |
| Semantic entropy, 10 samples, lenient | 0.835 [0.802, 0.869] | 342.6 | 0.702 [0.655, 0.747] | 611.7 |

Model accuracy: TriviaQA 42.6% (n=2,000), PubMedQA 61.6% (n=1,000).

<p align="center"><img src="figures/fig1_auroc_comparison.png" width="90%"></p>

**3B vs. 7B (4-bit NF4).** Same questions, prompts and protocol:

| | TriviaQA 3B | TriviaQA 7B (4-bit) | PubMedQA 3B | PubMedQA 7B (4-bit) |
|---|---|---|---|---|
| Model accuracy | 42.6% | 53.2% | 61.6% | 62.1% |
| **Answer-token probe (ours)** | 0.881 | 0.889 | 0.839 | 0.870 |
| **Prompt-only probe (ours)** | 0.821 | 0.824 | 0.830 | 0.852 |
| Token log-prob (mean) | 0.837 | 0.846 | 0.731 | 0.519 |
| Self-consistency, 10 samples | 0.845 | 0.840 | 0.698 | 0.616 |
| Semantic entropy, 10 samples | 0.833 | 0.830 | 0.701 | 0.615 |

---

## Rigor: why these numbers can be trusted

- **No leakage.** Train, validation and test splits are disjoint by question (asserted in
  code). Layer and regularization are chosen on validation only. A 200-permutation
  shuffled-label control averages **0.505** (TriviaQA) and **0.497**
  (PubMedQA), i.e. chance.
- **Not a lucky split.** Over 5 random splits the probe scores 0.896 ± 0.011 on TriviaQA
  (token log-prob: 0.840 ± 0.015).
- **Fair comparison.** All methods are scored on the same questions. Probe scores are
  out-of-fold via 5-fold cross-fitting. Differences use a paired bootstrap.
- **Strong baselines.** Semantic entropy was strengthened to 10 samples, and a lenient
  clustering variant was tested too. Costs are measured wall-clock, not estimated.
- **Confounds controlled.** On PubMedQA (yes/no/maybe), *which word* the model says
  predicts errors well on its own (0.753 AUROC). The probe beats this
  answer-identity control (Δ +0.085 [+0.053, +0.117]).

<p align="center"><img src="figures/fig4_layer_sweep.png" width="90%"></p>

---

## What doesn't work (reported honestly)

- **Cross-domain transfer is fragile.** A probe trained on TriviaQA and tested on PubMedQA
  ranges from 0.573 to 0.766 AUROC depending on the
  layer. The best in-domain layer (layer 24: 0.611) is not the one that transfers
  best (layer 28: 0.766). Probes need in-domain labels.
- **Token probability can fail completely.** On PubMedQA, the 7B model hedges with "maybe"
  and is about as confident when hedging as when not, so token log-prob drops to
  0.519 (near chance).
- **"Before answering" has limits.** For PubMedQA's one-word answers, the last prompt token
  already *produces* the answer, so the prompt-only probe there predicts before the answer
  is emitted, not before it is decided.
- **Semantic Entropy Probes** ([Kossen et al., 2024](https://arxiv.org/abs/2406.15927)) were
  compared at matched training data. Our accuracy probes match or beat them in-domain;
  out of domain the winner depends on direction ([`results/tables.md`](results/tables.md)).

---

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

HP_CONFIG=config_7b.yaml bash scripts/run_all.sh      # optional: 7B (4-bit) replication -> results_7b/
bash scripts/run_all.sh                               # 3B pipeline -> results/, figures/, README.md
```

Each step caches its outputs and resumes after a crash. Hardware: RTX 4080 Laptop GPU
(12 GB), Python 3.12, PyTorch 2.14 (CUDA 12.6). Generation, sampling and NLI took about
**15 GPU-minutes** for 3B (peak 9.5 GB VRAM) and **26 GPU-minutes**
for 7B in 4-bit (peak 9.3 GB). Probe training runs on CPU.

<details>
<summary><b>Repository layout</b></summary>

```
config.yaml / config_7b.yaml   every setting (seed 42)
src/data.py         dataset sampling + prompt formatting
src/generate.py     greedy answers, token log-probs, hidden states, sampled answers
src/label.py        correctness labels (SQuAD-normalized alias match; yes/no/maybe parsing)
src/baselines.py    token log-prob, self-consistency, semantic entropy (5/10 samples, strict/lenient)
src/probe.py        probes, layer sweep, prompt-only, transfer, shuffled control, SE-probe comparison
src/timing.py       clean per-stage latency on an idle GPU
src/evaluate.py     AUROC + bootstrap CIs, abstention curves, cost accounting
src/plots.py        figures (PDF + PNG)
src/report.py       paper numbers/tables and this README, generated from metrics.json
Research Report.pdf 4-page workshop-style write-up of the study
```
</details>

## Limitations

One model family (Qwen2.5) and two short-answer datasets; semantic entropy was designed
for, and may do better on, longer generations. Correctness labels come from string
matching: noisy TriviaQA aliases cause some false positives, and formatting differences
some false negatives. PubMedQA test splits are small (150 questions), so its CIs are
wide. The PubMedQA prompt was adjusted once, using only the answer distribution on
100 questions, before any correctness labels existed. The 7B model is 4-bit
quantized. Details: the Limitations section of [Research Report.pdf](Research%20Report.pdf).

## Built with

PyTorch · Hugging Face Transformers & Datasets · bitsandbytes · scikit-learn · matplotlib ·
Qwen2.5-3B/7B-Instruct · DeBERTa-large-MNLI · TriviaQA · PubMedQA
