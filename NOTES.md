# Project notes

Plain-English explanations written at each checkpoint. Numbers quoted here are copied
from files in `results/`; the file each number comes from is named next to it.

---

## Checkpoint 1 — Data and generation

### What the code does

1. **Pick the questions** (`src/data.py`).
   - TriviaQA (`rc.nocontext`, validation): we drop duplicate questions, then draw 2,000 at
     random with seed 42. Each has a canonical answer plus a list of accepted aliases
     (e.g. "The Hague" / "The Hague, Netherlands").
   - PubMedQA (`pqa_labeled`): all 1,000 expert-labelled questions. The model sees the
     abstract's sections (BACKGROUND / METHODS / RESULTS ...) but **not** the abstract's
     conclusion (`long_answer`), which would give the answer away. Abstracts longer than
     768 tokens would keep their first and last halves; in practice none were that long.
   - A seeded 500-question subset of each dataset is marked for sampling (step 4).
   - The chosen questions are frozen to `results/data_<dataset>.jsonl`, so every later
     phase uses exactly the same ones.

2. **Ask the model** (`src/generate.py`). Each question is wrapped in Qwen's chat template
   (system: "You are a helpful assistant."), asking for one short phrase (TriviaQA) or
   one word yes/no/maybe (PubMedQA). The model answers with **greedy decoding** (always
   take the most likely next token), up to 32 new tokens. For every answer token we save
   its log-probability, i.e. how confident the model was in that token.

3. **Save hidden states.** A transformer processes text as a stack of 36 layers. At each
   layer, every token is represented by a vector of 2,048 numbers (the "residual stream"),
   which is the model's internal working memory at that point. We save that vector at
   layers 4, 8, 12, ..., 36 (every 4th layer, which includes the final one) for two tokens:
   - **the last prompt token**: the position right before the model writes its first
     answer token;
   - **the last generated token**: the final token of the answer (not counting the
     end-of-message token).

   Vectors are stored as float16, shape `[questions, 9 layers, 2048]` per position, in
   `results/<dataset>/hidden.pt`.

4. **Sample alternative answers.** For the 500-question subset we also draw 5 answers with
   randomness turned on (temperature 1.0, nucleus top-p 0.95). Phase 3 uses them for the
   self-consistency and semantic-entropy baselines, which ask: does the model give the
   same answer if you ask again?

### Why the *last* token?

Qwen, like GPT-style models, is *causal*: each token can only look at tokens before it.
So the last token of a sequence is the only position whose vector has "seen" the whole
input. That is why it is the natural place to read out a summary of what the model knows.

- The **last prompt token** has read the whole question but not any answer yet. If a probe
  can predict hallucinations from it, the model "knows" before it speaks that it is likely
  to get this one wrong, and we could skip generation or route the question elsewhere.
  Fun detail: with the chat template, this token is literally the same newline character
  for every question, so any difference between questions comes purely from what the
  model has mixed in from earlier context. That is visible in the data: at layer 4 these
  vectors barely differ between questions, while at layer 36 they differ a lot.
- The **last generated token** has read the question *and* the model's own answer, so it
  can reflect "how sure was I about what I just said". We expect it to be at least as
  informative as the prompt token.

### Implementation choices worth knowing

- **Truly greedy decoding.** Qwen's default generation config has sampling on
  (temperature 0.7, top-k 20, repetition penalty 1.05). We replace it with neutral
  defaults, so the only decoding settings in effect are the ones in `config.yaml`. The code
  asserts that each first greedy token is the argmax of the raw logits.
- **How hidden states are captured.** After generation, we run one extra forward pass over
  prompt + answer, with hooks that keep only the two positions we need. This avoids
  holding every layer's activations for every token in memory (several GB for PubMedQA's
  long prompts). In deployment these activations would come for free during generation;
  the extra pass is an implementation convenience, not part of the method's cost.
- **"Layer L"** is the residual stream after decoder block L. For layer 36 we take it
  *before* the final normalization, so all layers are treated the same way.
- **Crash safety.** Work is cached in chunks of 128 questions under
  `results/<dataset>/_chunks/`; a re-run skips finished chunks.
- **Batching.** Prompts are sorted longest-first and processed in batches of 16. On a CUDA
  out-of-memory error the batch size halves automatically. PyTorch is capped at 10.5 GB,
  because on Windows exceeding VRAM silently spills to system RAM (very slow) instead of
  raising an error.
- **Known caveat.** Batched bf16 inference with padding can very occasionally choose a
  different greedy token than unbatched inference would, because of floating-point
  rounding. This affects reproducibility at the level of single examples, not the method.

### Run facts

From `results/<dataset>/generation_meta.json` (PubMedQA column: final prompt v3; RTX 4080 Laptop GPU, 12 GB; bf16; batch 16
greedy / 8 prompts x 5 samples; no OOM retries were needed):

| | TriviaQA | PubMedQA |
|---|---|---|
| Questions (greedy + hidden states) | 2,000 | 1,000 |
| Questions with 5 samples | 500 | 500 |
| Mean prompt length (tokens) | 60.3 | 412.2 |
| Greedy time (incl. hidden-state pass) | 75.6 s | 113.4 s |
| Sampling time | 44.1 s | 153.0 s |
| Mean greedy decode steps / question | 4.06 | 2.00 |
| Mean sampling decode steps / question (5 samples) | 19.64 | 10.00 |
| Peak VRAM allocated (reserved) | 6.43 GB (6.93) | 9.48 GB (10.36) |
| Answers hitting the 32-token limit / empty | 0 / 0 | 0 / 0 |
| Abstracts truncated | n/a | 0 |
| `hidden.pt` size | 141 MB | 71 MB |

These timings were measured with the GPU otherwise idle. A first smoke test that ran
while another process was using the GPU was 10–18x slower, so timings are only
comparable when measured on an idle GPU.

### PubMedQA prompt change ("maybe" bias)

**Problem.** With the first prompt (v1), the model answered **"maybe" on 698/1,000**
PubMedQA questions, but only **110/1,000** gold labels are "maybe" (gold: 552 yes /
338 no / 110 maybe). That causes two problems: accuracy would mostly measure hedging,
and a probe on the last answer token could score well just by spotting the word "maybe".

**What we did.** `scripts/pubmed_prompt_pilot.py` compared four wordings on the first
100 questions, judging **only the yes/no/maybe distribution** (no correctness labels had
been computed yet, so the prompt was not picked for accuracy or AUROC). Results are in
`results/pubmed_prompt_pilot.json`:

| Prompt (first 100 questions) | yes | no | maybe |
|---|---|---|---|
| v1 original | 24 | 5 | 71 |
| v2 "maybe only if inconclusive" | 22 | 6 | 72 |
| **v3 same instruction, placed after the abstract (chosen)** | 41 | 40 | 19 |
| v4 yes/no framing, maybe as escape hatch | 31 | 68 | 1 |
| gold | 58 | 26 | 16 |

Telling the model *when* to say "maybe" did nothing while the instruction came before
the long abstract; moving it after the abstract fixed most of the bias. Small models pay
more attention to recent text. v4 overcorrected into "no".

**Full run with v3:** 478 no / 334 yes / 188 maybe (case-folded; from
`results/pubmed_qa/greedy.jsonl`). "Maybe" is much closer to the gold rate, but the model
now over-predicts "no". So "incorrect" is still partly predictable from *which word*
was said. Phases 3–4 therefore add an **answer-identity control**: how well can we
predict errors from the yes/no/maybe word alone? The hidden-state probe has to beat that
to show it detects more than the answer's identity.

The v1 outputs are kept in `results/archive/pubmed_qa_prompt_v1/` for the record.

**Methodological caveat.** The pilot questions are also part of the evaluation set.
Because selection used only the answer distribution, not correctness, we consider the
leakage negligible, but it is mentioned in the paper's limitations.

---

## Checkpoint 2 — Labeling

### What the code does (`src/label.py`)

Each greedy answer gets a label, *correct* or *incorrect*. Later phases try to predict
"incorrect", so `incorrect = 1` is the target.

- **TriviaQA.** Both the model's answer and each gold alias are *normalized*: lowercase,
  punctuation removed, the words "a/an/the" removed, extra spaces collapsed. So
  "The U.S. Virgin Islands." becomes "us virgin islands". The answer is correct if
  (a) it equals some normalized alias, or (b) some alias appears *inside* the answer as
  whole words ("Sun Records" contains "sun"). Whole-word matching stops "11" from matching
  "110".
- **PubMedQA.** The answer is parsed into yes/no/maybe: use the first word if it is one
  of the three; otherwise use the only label word that appears; otherwise
  "unparseable", which counts as incorrect. **0/1,000 answers were unparseable**; every
  answer was a bare label word (sometimes capitalized).

### Results (from `results/metrics.json` → `labeling`)

| | TriviaQA | PubMedQA |
|---|---|---|
| Accuracy | 42.6% (851/2,000) | 61.6% (616/1,000) |
| 95% bootstrap CI | 40.4–44.7% | 58.7–64.4% |
| Accuracy on the 500 sampled questions | 43.8% | 62.2% |
| How matched | 784 exact, 67 containment | 1,000 parsed from first word |

TriviaQA accuracy is inside the expected 35–55% range for a 3B model.

PubMedQA accuracy depends heavily on which word the model said: **yes 88.0% correct,
no 60.7%, maybe 17.0%**. So the yes/no/maybe word alone carries a lot of information
about correctness, which is why Phases 3–4 include the answer-identity control.

### Failure modes of string-match labeling (TriviaQA)

The code flags 101/2,000 TriviaQA labels as worth a second look
(`flags` field in `results/trivia_qa/labels.jsonl`). Manual inspection found:

1. **False positives from noisy aliases.** TriviaQA's alias lists are built from
   Wikipedia redirects, and some redirects point to pages covering several things. The
   alias list for "Dr Pangloss" contains "Candide" and "Cacambo", and the list for
   "Vitalstatistix" contains "Getafix", so those wrong answers count as correct. Among the
   35 "correct" answers with zero word overlap with the canonical answer, most are
   legitimate variants (Two/2, Clothing/Clothes, Jinn/Djinn), and 2 are clearly wrong.
2. **False positives from containment.** A long answer can contain the right words
   without being right. Flagged as `long_answer_contains_alias` (17) and
   `short_or_numeric_alias_match` (5). Most are fine on inspection ("Ursine refers to
   bears."), but e.g. "George Herbert Walker Bush" counts as correct for gold "Walker".
3. **False negatives from formatting.** A right answer can fail to match because the
   gold has extra text, e.g. "Peter Townsend" vs "(Group Captain) Peter Townsend", or
   "Frankenstein" vs "Frankenstein (Victor)". Also stripping parentheticals from aliases
   would flip **10/2,000** labels to correct; most flips are genuine fixes, a few are
   debatable ("Asia" for "South East Asia"). We keep the specified rule unchanged and
   report this as label noise.
4. **False negatives from paraphrase or precision.** "8 minutes and 20 seconds" for
   "About 8 minutes"; "Scott" for "Montgomery Scott". Flagged as
   `high_overlap_but_no_match` (77) and `answer_is_part_of_alias` (31). Most of these
   are truly wrong ("Edward II" vs "Edward the First", "Long Island" vs "Staten Island").
5. **Accents are not folded** (SQuAD normalization keeps "é"). This is harmless here
   because alias lists usually include both spellings.

**Net effect.** Label noise is small (on the order of 1% of TriviaQA labels from the
cases above) but not zero. Noise in the target caps the AUROC any detector can reach and
adds variance; it does not favour any one method.

PubMedQA labels are exact (a closed set of three words), so the only "noise" is
disagreement with the expert annotators.

---

## Checkpoint 3 — Baselines

### What the code does (`src/baselines.py`, `src/timing.py`)

Every method produces one number per question; **higher means "more likely wrong"**.
We then measure **AUROC**: the probability that a randomly chosen wrong answer gets a
higher score than a randomly chosen right answer. 0.5 is a coin flip; 1.0 is perfect.
95% CIs come from 1,000 bootstrap resamples. The same resamples are used for every
method (a "paired" bootstrap), so methods are compared on identical data.

1. **Token probability (mean / min).** While writing its answer, the model assigns a
   probability to each token it picks. A hesitant model gives low probabilities. The
   score is the negated mean (or minimum) log-probability of the answer tokens. It is
   free: the numbers come out of the normal answer generation.
2. **Self-consistency.** Ask the same question 5 more times with randomness on. If the
   model keeps giving the greedy answer it is probably confident. The score is
   1 − (fraction of the 5 samples that agree with the greedy answer). "Agree" uses the
   labeling normalization: equal, or one contains the other as whole words (so "Niger"
   agrees with "Niger River"). For PubMedQA, it means the same yes/no/maybe word.
3. **Semantic entropy** (explained below).
4. **Answer-identity control (PubMedQA only).** A deliberately "dumb" score: the error
   rate of the word the model said (yes/no/maybe), estimated on other folds (5-fold
   cross-fitting) so each question's own label is never used to score it. It is not a
   real detector. It shows how much of PubMedQA's errors can be predicted just from
   *which* answer was given.

### Semantic entropy in plain words

We collect 6 answers to the same question (the greedy one plus 5 samples) and group
those that *mean the same thing*. Two answers are put in the same group if an NLI model
(DeBERTa-large-MNLI), given the question, judges that each one implies the other. So
"Niger" and "Niger River" share a group, but "Niger" and "Senegal River" do not. The
score is the entropy of the group sizes: all six answers in one group gives 0 (the
model consistently means the same thing), while six different meanings gives the
maximum, log 6 ≈ 1.79. Unlike plain self-consistency, it does not penalize the model for
phrasing the same answer differently.

### Results (from `results/metrics.json`: `baselines`, `costs`)

AUROC for predicting "incorrect" on the 500-question subset where every method is
available. Costs are per question, measured on an idle GPU with the same batch sizes as
generation (16 for greedy; 8 prompts x 5 samples), median of 3 repeats. "Overhead" is
the cost on top of producing the greedy answer, which a deployed system pays anyway.

**TriviaQA** (500 questions, 281 incorrect)

| Method | AUROC [95% CI] | LLM generations | LLM forward passes | NLI passes | Total ms | Overhead ms |
|---|---|---|---|---|---|---|
| −mean token log-prob | 0.837 [0.802, 0.870] | 1 | 3.91 | 0 | 24.1 | 0 |
| −min token log-prob | 0.833 [0.797, 0.868] | 1 | 3.91 | 0 | 24.1 | 0 |
| Self-consistency | 0.823 [0.787, 0.858] | 6 | 23.55 | 0 | 111.8 | 87.6 |
| Semantic entropy | 0.817 [0.782, 0.854] | 6 | 23.55 | 9.03 | 193.3 | 169.2 |

On the full 2,000 questions: −mean log-prob 0.843 [0.825, 0.860], −min 0.837 [0.820, 0.854].

**PubMedQA** (500 questions, 189 incorrect)

| Method | AUROC [95% CI] | LLM generations | LLM forward passes | NLI passes | Total ms | Overhead ms |
|---|---|---|---|---|---|---|
| −mean token log-prob | 0.731 [0.685, 0.775] | 1 | 2.00 | 0 | 63.4 | 0 |
| −min token log-prob | 0.731 [0.685, 0.775] | 1 | 2.00 | 0 | 63.4 | 0 |
| Answer identity (control) | 0.753 [0.708, 0.794] | 1 | 2.00 | 0 | 63.4 | 0 |
| Self-consistency | 0.669 [0.624, 0.712] | 6 | 12.00 | 0 | 374.9 | 311.5 |
| Semantic entropy | 0.671 [0.624, 0.715] | 6 | 12.00 | 0.85 | 390.7 | 327.3 |

On the full 1,000 questions: −mean log-prob 0.707 [0.675, 0.739], answer identity
0.744 [0.714, 0.775].

### What to take from this

- **All methods beat chance (> 0.5)**, and none falls outside the 0.55–0.95 range that
  would trigger an investigation.
- **The expensive methods did not beat token probability here.** On TriviaQA the four
  methods are within a few points of each other with overlapping CIs; on PubMedQA the
  sampling methods are clearly *worse* than token probability. Plausible reasons:
  answers are very short (often 1–4 tokens), where token probability already captures
  most of the uncertainty, and we use only 5 samples (Farquhar et al. used 10). For
  PubMedQA there are only three possible answers, so 6 answers carry little information
  beyond what the single token probability already says.
- **PubMedQA's answer-identity control beats every uncertainty method.** "The model said
  maybe" predicts errors better than any of them, so PubMedQA AUROCs largely reflect
  which word was said. Phase 4's probes must be compared against this control.
- **Mean = min on PubMedQA** because every answer is one token.
- **Semantic-entropy clustering is strict.** DeBERTa puts "Shakespeare" and
  "William Shakespeare" in different groups (the short answer does not strictly entail
  the long one). This is the standard method (Kuhn et al.); we note it as a limitation.
  On TriviaQA it used 9.0 NLI calls and 2.8 clusters per question on average; 36.8% of
  questions had a single cluster.
- **Costs.** The sampling methods cost 4–6x the greedy answer in wall-clock time and
  6x in generations. On PubMedQA, sampling is especially expensive relative to greedy
  decoding: the long abstract prompt is processed for every sample, and we do not share
  its cache across samples.

### Implementation notes

- Answers identical after normalization are merged without calling the NLI model; only
  genuinely different strings cost NLI passes, and those calls are counted exactly.
- `src/timing.py` frees cached GPU memory between stages; without that, the largest
  PubMedQA sampling batch ran out of memory because of fragmentation (it fits on its
  own, as in Phase 1).

---

## Checkpoint 4 — Probes (the main contribution)

### What a linear probe is

A **probe** is a deliberately simple classifier that reads the model's internal state
and tries to predict something. Here, "will this answer be wrong?". We use **logistic
regression**: it learns one weight per hidden-state dimension (2,048 weights), adds up
weight x value, and squashes the sum into a probability. Because it is *linear*, it can
only succeed if the information is already laid out plainly in the hidden state; it
cannot do complex computation of its own. That makes a working probe evidence that the
model itself represents its likelihood of being wrong. It is also extremely cheap to
run: one dot product, about 4–7 microseconds per question here.

Details (`src/probe.py`):
- **Split by question.** 70% train / 15% val / 15% test, seeded. The code asserts no
  question, by id or by normalized text, is in two splits.
- **Standardization.** Each of the 2,048 features is rescaled using the *train* mean and
  standard deviation only.
- **Regularization strength C.** Chosen on val from 8 values (1e-4 … 1). The chosen
  values are small (3e-4 to 3e-3): with 2,048 features and only 700–1,400 training
  questions, strong regularization prevents memorization.
- **Layer selection.** One probe per saved layer (4, 8, …, 36) and per token position.
  The **best layer is picked on val**, then reported once on test.
- **Cross-fitting for the comparison.** To compare with semantic entropy on all 500
  sampled questions, 5-fold cross-fitting gives every question a score from a probe that
  never saw it. Layer and C are again chosen on an inner validation split inside each
  fold.

### Why the shuffled-label control matters

If you shuffle the labels, there is no real relationship left between hidden states and
"wrong", so an honest pipeline must score ~0.5 on the (unshuffled) test labels. If it
scores higher, information is leaking: for example the same question appears in train
and test, or test data influenced a choice. It is a check of the *pipeline*, not of the
model.

**What happened.** With the first 5 shuffles, the PubMedQA answer-token control averaged
0.415 (one run 0.293). That is below chance, which is equally suspicious. We tested two
explanations:
- 200 shuffles: the mean was ~0.50 in every case.
- 200 random untrained directions: same wide spread.

A shuffled probe learns an essentially random direction, and when the real signal is
strong, a random direction can align with it or against it by chance. With only 150
test questions that produces a wide spread around 0.5. Five draws were simply too few, so
the control now uses 200 shuffles:

| Shuffled-label control (200 shuffles) | mean | sd | 5–95% |
|---|---|---|---|
| TriviaQA, prompt token (L28) | 0.499 | 0.053 | 0.413–0.591 |
| TriviaQA, answer token (L24) | 0.505 | 0.070 | 0.387–0.646 |
| PubMedQA, prompt token (L36) | 0.500 | 0.080 | 0.365–0.640 |
| PubMedQA, answer token (L32) | 0.497 | 0.091 | 0.344–0.651 |

All four are centred on 0.5: **no evidence of leakage.**

### Results (from `results/metrics.json`: `probes`, `transfer`, `comparison`, `costs`)

**A. Fixed split, best layer chosen on val, reported on test**
(TriviaQA test n=300, 172 incorrect; PubMedQA test n=150, 55 incorrect)

| | TriviaQA | PubMedQA |
|---|---|---|
| Answer-token probe | **0.909** [0.875, 0.938] (L24) | **0.828** [0.747, 0.898] (L32) |
| Prompt-only probe | **0.837** [0.794, 0.882] (L28) | **0.827** [0.751, 0.896] (L36) |
| −mean token log-prob (same test split) | 0.845 [0.801, 0.887] | 0.760 [0.677, 0.839] |
| Answer identity (same test split) | — | 0.776 [0.697, 0.848] |

Layer sweep (test AUROC; the full sweep with val AUROC is in `metrics.json`):

| Layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 |
|---|---|---|---|---|---|---|---|---|---|
| TriviaQA prompt | .686 | .685 | .729 | .722 | .783 | .795 | **.837** | .810 | .799 |
| TriviaQA answer | .626 | .634 | .732 | .829 | .890 | **.909** | .902 | .875 | .867 |
| PubMedQA prompt | .507 | .564 | .484 | .529 | .555 | .583 | .803 | .819 | **.827** |
| PubMedQA answer | .786 | .776 | .778 | .790 | .802 | .826 | .832 | **.828** | .835 |

(Bold = layer chosen on val. Information about correctness appears in the middle-to-late
layers; on PubMedQA the prompt-token signal appears abruptly at layer 28.)

**B. Probe vs every baseline on the same 500-question subset** (out-of-fold probe
scores; ΔAUROC = probe − baseline with paired-bootstrap 95% CI. If the CI excludes 0,
the difference is not resampling noise.)

TriviaQA:

| Method | AUROC [95% CI] | Δ answer probe − method | Δ prompt probe − method |
|---|---|---|---|
| **Answer-token probe** | **0.881** [0.852, 0.908] | | |
| Prompt-only probe | 0.821 [0.781, 0.858] | | |
| −mean token log-prob | 0.837 [0.802, 0.870] | +0.044 [+0.011, +0.081] | −0.016 [−0.053, +0.025] |
| −min token log-prob | 0.833 [0.797, 0.868] | +0.048 [+0.015, +0.082] | −0.012 [−0.048, +0.026] |
| Self-consistency | 0.823 [0.787, 0.858] | +0.058 [+0.023, +0.098] | −0.002 [−0.046, +0.041] |
| Semantic entropy | 0.817 [0.782, 0.854] | +0.064 [+0.029, +0.102] | +0.004 [−0.036, +0.045] |

PubMedQA:

| Method | AUROC [95% CI] | Δ answer probe − method | Δ prompt probe − method |
|---|---|---|---|
| **Answer-token probe** | **0.839** [0.800, 0.873] | | |
| Prompt-only probe | 0.830 [0.791, 0.864] | | |
| −mean token log-prob | 0.731 [0.685, 0.775] | +0.108 [+0.059, +0.156] | +0.099 [+0.056, +0.145] |
| Answer identity (control) | 0.753 [0.708, 0.794] | +0.085 [+0.053, +0.117] | +0.077 [+0.037, +0.116] |
| Self-consistency | 0.669 [0.624, 0.712] | +0.170 [+0.119, +0.218] | +0.162 [+0.113, +0.206] |
| Semantic entropy | 0.671 [0.624, 0.715] | +0.168 [+0.118, +0.216] | +0.159 [+0.110, +0.205] |

On all questions (out-of-fold): TriviaQA answer probe 0.894 [0.881, 0.907], prompt probe
0.826 [0.807, 0.843], −mean log-prob 0.843 [0.825, 0.860]. PubMedQA answer probe 0.824
[0.796, 0.851], prompt probe 0.822 [0.795, 0.849], −mean log-prob 0.707 [0.675, 0.739],
answer identity 0.744 [0.714, 0.775].

**C. Cost per question** (`costs.<dataset>.methods`; idle RTX 4080 Laptop GPU)

| Method | LLM generations | LLM forward passes (Trivia / PubMed) | Total ms (Trivia / PubMed) | Extra ms over the greedy answer (Trivia / PubMed) |
|---|---|---|---|---|
| Prompt-only probe | 0 (prefill only) | 1 / 1 | 6.8 / 57.6 | (runs *before* answering) |
| Answer-token probe | 1 | 3.91 / 2.00 | 24.9 / 65.3 | 0.8 / 1.8 |
| Token log-prob | 1 | 3.91 / 2.00 | 24.1 / 63.4 | 0 |
| Self-consistency | 6 | 23.55 / 12.00 | 111.8 / 374.9 | 87.6 / 311.5 |
| Semantic entropy | 6 (+ NLI) | 23.55 / 12.00 (+9.03 / 0.85 NLI) | 193.3 / 390.7 | 169.2 / 327.3 |

**D. Cross-domain transfer** (target's test split; layer and C chosen on the *source's* val)

| Train → test | Prompt-only probe | Answer-token probe |
|---|---|---|
| TriviaQA → TriviaQA (in-domain) | 0.837 [0.794, 0.882] | 0.909 [0.875, 0.938] |
| PubMedQA → PubMedQA (in-domain) | 0.827 [0.751, 0.896] | 0.828 [0.747, 0.898] |
| **TriviaQA → PubMedQA** | **0.741** [0.653, 0.823] | 0.540 [0.448, 0.637] |
| **PubMedQA → TriviaQA** | 0.496 [0.429, 0.561] | 0.577 [0.515, 0.641] |

Re-standardizing features with the (unlabeled) target data did not help (0.738, 0.508,
0.506, 0.578 respectively).

### What to take from this (honestly)

1. **The answer-token probe is the best detector on both datasets, at essentially the
   cost of token probability.** On TriviaQA it beats every baseline, including semantic
   entropy, by +0.04 to +0.06 AUROC (CIs exclude 0). On PubMedQA it beats all of them by
   +0.09 to +0.17. It adds ~1–2 ms to a 25–65 ms answer, versus +88–327 ms for the
   sampling methods.
2. **Hallucinations are largely predictable before the model answers.** From the prompt
   alone, with one forward pass and no decoding, the probe matches token probability,
   self-consistency and semantic entropy on TriviaQA (all Δ CIs include 0). It does that
   at ~1/28 of semantic entropy's wall-clock (6.8 vs 193 ms), and it beats them all on
   PubMedQA.
   - *Important nuance:* the last prompt token's final-layer state is exactly what
     produces the first answer token. On PubMedQA the whole answer is that one token, so
     "before answering" there means "before the answer is *emitted*", not "before it is
     *decided*".
   - The prompt probe saves little time on PubMedQA (57.6 vs 63.4 ms), because the long
     abstract dominates.
3. **The PubMedQA probes are not just reading the answer word.** They beat the
   answer-identity control by +0.08 (CI excludes 0). Within the model's "no" answers
   (n=478) they still separate right from wrong (prompt probe 0.789 [0.749, 0.829],
   answer probe 0.750 [0.706, 0.794], token log-prob 0.688 [0.637, 0.735]).
   - Within "yes" (n=334) the answer probe is at chance: 0.463 [0.366, 0.563], below
     token log-prob's 0.653.
   - Within "maybe" (n=188) every method is near chance.
   - So the PubMedQA advantage comes mostly from recognizing wrong "no" answers.
4. **Transfer is weak and asymmetric: a mostly negative result.**
   - A *prompt* probe trained on TriviaQA keeps some signal on PubMedQA (0.741, below
     in-domain 0.827 and close to PubMedQA's own token log-prob 0.760).
   - Answer-token probes do not transfer (0.540 and 0.577), and the PubMedQA prompt
     probe does not transfer to TriviaQA at all (0.496).
   - The "correctness direction" learned by these probes is largely domain-specific.
5. **Not too good to be true.** The highest AUROC is 0.909, below the 0.95 investigate
   threshold; the shuffled-label controls are at 0.5; layer/C selection never touched
   test data.

---

## Phase 5 — Analysis and figures

### What the code does (`src/evaluate.py`, `src/plots.py`)

**Abstention ("selective answering").** Suppose the model may decline to answer. We
rank the 500 sampled questions from most to least confident by each method's score,
let the model answer only the top fraction ("coverage"), and measure accuracy on the
questions it does answer. A good uncertainty score makes accuracy rise quickly as
coverage drops. We summarize each curve by its area (**AURAC**, area under the
accuracy-coverage curve; higher is better). Ties are handled exactly: for discrete
scores like semantic entropy we report the *expected* accuracy if tied questions were
dropped in random order. That was checked against a 20,000-run simulation (max
difference 0.001).

From `results/metrics.json` → `abstention` (500-question subset; probe scores are
out-of-fold):

| | TriviaQA AURAC | acc @ 50% coverage | PubMedQA AURAC | acc @ 50% coverage |
|---|---|---|---|---|
| Answer everything | 0.438 | 0.438 | 0.622 | 0.622 |
| **Answer-token probe** | **0.722** | **0.728** | **0.839** | **0.888** |
| Prompt-only probe | 0.691 | 0.676 | 0.838 | 0.868 |
| Token log-prob (mean) | 0.700 | 0.680 | 0.782 | 0.772 |
| Self-consistency | 0.670 | 0.682 | 0.719 | 0.743 |
| Semantic entropy | 0.658 | 0.699 | 0.720 | 0.743 |
| Answer identity (control) | — | — | 0.787 | 0.773 |
| Oracle (perfect ranking) | 0.799 | 0.876 | 0.917 | 1.000 |

Reading it: if the system answers only the half of TriviaQA questions the answer-token
probe is most confident about, accuracy rises from 43.8% to 72.8%. On PubMedQA it rises
from 62.2% to 88.8%.

Semantic entropy's curve goes flat at low coverage. With 6 answers its score takes only
a few distinct values, and many questions tie at 0 (all six answers agree): 64% of
PubMedQA questions and 37% of TriviaQA questions. Once only those questions remain, it
has no way to rank them further. This is a real practical limitation of sampling-based
scores with few samples, not a plotting artifact.

### Figures (`figures/`, PDF + PNG)

| File | What it shows |
|---|---|
| `fig1_auroc_comparison` | AUROC with 95% CI for every method on the same 500 questions, both datasets |
| `fig2_abstention` | Accuracy vs coverage (abstain on the most uncertain first) |
| `fig3_cost_vs_auroc` | Forward passes per question (LLM + NLI, log scale) vs AUROC |
| `fig4_layer_sweep` | Probe test AUROC per layer, both token positions; ring = layer chosen on val |
| `fig5_transfer` | Train-domain x test-domain AUROC heatmaps for both probes |

Design choices:
- Each method keeps the same color and marker in every figure.
- The four method colors were checked with a colorblind-safety validator for the
  all-pairs case (scatter plots), and every series also has a distinct marker and a
  text label, so nothing depends on color alone.
- Self-consistency and the answer-identity control are neutral grays.
- Self-consistency is left out of the abstention plot because it is nearly identical
  to semantic entropy; its numbers are in the table above.
- Figure 3 counts NLI forward passes together with LLM passes for semantic entropy.
  The NLI model (DeBERTa-large, 0.4B parameters) is much smaller than the LLM, so this
  slightly *overstates* semantic entropy's cost in passes. Wall-clock costs (Checkpoint
  4, table C) are the more faithful comparison.

---

## Extension — stronger baselines, SE probes, 7B replication, write-up

All numbers are in `results/metrics.json` (3B) and `results_7b/metrics.json` (7B); the
paper and README quote them through `src/report.py`, so they are never typed by hand.

### 1. Stronger semantic-entropy baseline
**Why.** With 5 samples and strict clustering, a reader could argue the probe only beat a
handicapped baseline.

**What.** We drew 5 more samples per question (`samples_extra.jsonl`; i.i.d. with the
first 5, so together they are exactly 10 samples) and added a *lenient* clustering rule:
two answers are the same meaning if neither direction is a contradiction and they are not
neutral both ways. That lets "Shakespeare" and "William Shakespeare" cluster together.
10-sample generation was timed separately on an idle GPU.

**Result.** 10 samples help the sampling methods (TriviaQA SE 0.817 → 0.833,
self-consistency 0.823 → 0.845). The lenient rule barely matters. The answer-token probe
still beats every variant with CIs excluding 0 (TriviaQA: +0.048 over 10-sample SE,
+0.035 over 10-sample self-consistency). The prompt-only probe stays statistically tied
with all of them on TriviaQA.

### 2. Semantic Entropy Probes (Kossen et al., 2024) vs. accuracy probes
**What.** An SE probe (SEP) is trained to predict *high vs. low semantic entropy* (10-sample
SE, binarized with a two-means split on training questions) instead of correctness. It
needs no correctness labels. To compare fairly, both kinds of probe are trained on the
same 500 sampled questions. The SEP target was fixed *before* looking at any SEP result.

**Result (3B).** In-domain, accuracy probes match or beat SEPs (answer token: +0.037 on
TriviaQA, +0.088 on PubMedQA). Out of domain it depends on direction:
PubMedQA → TriviaQA favours the SEP (−0.108), while TriviaQA → PubMedQA favours the
accuracy probe (+0.101). So we reproduce Kossen et al.'s "SEPs generalize better" in one
direction only.

### 3. Robustness over 5 random splits
Answer-token probe test AUROC: TriviaQA 0.896 ± 0.011, PubMedQA 0.819 ± 0.022. Token
log-prob on the same splits: 0.840 ± 0.015 and 0.700 ± 0.069. The headline numbers are not
a lucky split.

### 4. Transfer depends strongly on the layer (a correction to Checkpoint 4)
The SEP comparison gave TriviaQA → PubMedQA answer-probe transfer of 0.743, versus 0.540 in
Checkpoint 4. Their CIs did not overlap, so we investigated. Sweeping all layers (C chosen
on the source val split; AUROC on all 1,000 PubMedQA questions), transfer ranges from
0.573 to 0.766 depending on the layer. The layer that is best *in-domain* (24) transfers
poorly (0.611), while layer 28 transfers best (0.766). The two experiments had simply
selected different layers.

**Revised conclusion.** It is not that answer-token probes "don't transfer". Transfer is
fragile and layer-dependent, and choosing the layer on source data alone does not
reliably pick one that transfers. Stored in `metrics.json` → `transfer_layer_sweep`.

### 5. 7B replication (Qwen2.5-7B-Instruct, 4-bit NF4)
Same questions, prompts and protocol; full datasets rather than 500 questions, because a
500-question split leaves only 75 test questions, too few for meaningful CIs. Hidden size
3,584, 28 layers (saved: 4, 8, …, 28). 7B results were not re-timed; the cost comparison
is a 3B result.

- Accuracy: TriviaQA 53.2% (3B: 42.6%), PubMedQA 62.1% (3B: 61.6%).
- TriviaQA: answer-token probe 0.889 vs 10-sample SE 0.830 (Δ +0.059 [+0.027, +0.093]);
  prompt-only probe ties SE (Δ −0.006 [−0.044, +0.033]). Same pattern as 3B.
- PubMedQA: **token log-prob drops to 0.519 (below the 0.55 investigate threshold, so we
  investigated).** The 7B model answers "maybe" 292 times (gold: 110). Its "maybe" answers
  are right only 13.7% of the time, but it is about as confident when saying "maybe" as
  otherwise (mean token uncertainty 0.363 vs 0.292 for "yes"). So token probability
  carries almost no error signal, while the answer word carries a lot (answer identity
  0.802). Not a pipeline bug: the same code gives 0.853 on 7B TriviaQA. The answer-token
  probe still beats the answer-identity control (+0.068 [+0.035, +0.100]).
- Shuffled-label controls ≈ 0.50; 5-split robustness 0.893 ± 0.016 (TriviaQA).
- Transfer at 7B: TriviaQA → PubMedQA is much better (answer probe 0.821), but
  PubMedQA → TriviaQA stays weak (0.617).
- One run-time note: the first background 7B probe run stopped before the SEP step. That
  step was re-run on its own (same deterministic function, same inputs). `run_all.sh`
  regenerates everything in one pass.

### 6. Write-up machinery
- `src/report.py` turns `metrics.json` into `paper/numbers.tex` (398 LaTeX macros),
  `paper/tables/*.tex`, `results/tables.md`, and `README.md` (rendered from
  `README.template.md`; it fails loudly on any missing value).
- Percentages use exact round-half-up. Float rounding had printed 0.4255 as 42.5%; it is
  42.6%, as in Checkpoint 2.
- `paper/main.tex` compiles with Tectonic (installed in a separate conda env `tex`) to a
  4-page PDF. It uses a two-column `article` layout because the official ACL/NeurIPS style
  files are not on CTAN; moving it into the official template is copy-paste.
- References marked `TODO-VERIFY` need a manual check.
- Several sentences were corrected during the number check: a figure caption said
  "6–15×" more forward passes (actual 11–18×, now "an order of magnitude"), an estimated
  "~1% label noise" was removed (it was an estimate, not a measurement), and an unverified
  claim about Farquhar et al.'s sample count was removed.
