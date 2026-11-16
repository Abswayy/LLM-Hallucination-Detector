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
