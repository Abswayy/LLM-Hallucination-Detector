"""Dataset loading and prompt formatting.

Produces a fixed, seeded list of examples per dataset and caches it to
results/data_<name>.jsonl, so every later phase sees exactly the same questions.

Each example is a dict:
    id        unique question id (string)
    dataset   "trivia_qa" | "pubmed_qa"
    question  question text
    gold      TriviaQA: {"value": str, "aliases": [str, ...]}
              PubMedQA: "yes" | "no" | "maybe"
    context   PubMedQA only: the abstract text shown to the model
"""
from __future__ import annotations

import numpy as np
from datasets import load_dataset

from .utils import read_jsonl, results_dir, write_jsonl

SYSTEM_PROMPT = "You are a helpful assistant."

TRIVIA_TEMPLATE = (
    "Answer the following trivia question with a single short phrase. "
    "Give only the answer, with no explanation.\n\n"
    "Question: {question}"
)

PUBMED_TEMPLATE = (
    "Read the abstract below and answer the research question with exactly one word: "
    "yes, no, or maybe. Reply with that single word only.\n\n"
    "Abstract:\n{context}\n\n"
    "Question: {question}"
)


# --------------------------------------------------------------------------- loading
def _load_trivia_qa(dcfg: dict, seed: int) -> list[dict]:
    ds = load_dataset(dcfg["hf_path"], dcfg["hf_config"], split=dcfg["split"])
    # De-duplicate by question id (and question text) before sampling so that no
    # question can ever appear twice, which keeps the Phase-4 split clean.
    seen_ids, seen_q, keep = set(), set(), []
    for i, (qid, q) in enumerate(zip(ds["question_id"], ds["question"])):
        key = q.strip().lower()
        if qid in seen_ids or key in seen_q:
            continue
        seen_ids.add(qid)
        seen_q.add(key)
        keep.append(i)
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.array(keep), size=dcfg["n_questions"], replace=False)
    rows = []
    for i in idx.tolist():
        ex = ds[i]
        ans = ex["answer"]
        aliases = list(dict.fromkeys([ans["value"], *ans["aliases"]]))  # dedupe, keep order
        rows.append(
            {
                "id": ex["question_id"],
                "dataset": "trivia_qa",
                "question": ex["question"].strip(),
                "gold": {"value": ans["value"], "aliases": aliases},
            }
        )
    return rows


def _format_pubmed_context(ctx: dict) -> str:
    """Join abstract sections as 'LABEL: text' paragraphs."""
    parts = []
    labels = ctx.get("labels") or [None] * len(ctx["contexts"])
    for label, text in zip(labels, ctx["contexts"]):
        text = " ".join(text.split())
        parts.append(f"{label.upper()}: {text}" if label else text)
    return "\n".join(parts)


def _load_pubmed_qa(dcfg: dict, seed: int) -> list[dict]:
    ds = load_dataset(dcfg["hf_path"], dcfg["hf_config"], split=dcfg["split"])
    order = list(range(len(ds)))
    if dcfg.get("n_questions"):
        rng = np.random.default_rng(seed)
        order = sorted(rng.choice(len(ds), size=dcfg["n_questions"], replace=False).tolist())
    rows = []
    for i in order:
        ex = ds[i]
        rows.append(
            {
                "id": str(ex["pubid"]),
                "dataset": "pubmed_qa",
                "question": ex["question"].strip(),
                # NOTE: `long_answer` (the abstract's conclusion) is deliberately NOT
                # shown to the model; it would leak the answer.
                "context": _format_pubmed_context(ex["context"]),
                "gold": ex["final_decision"].strip().lower(),
            }
        )
    assert len({r["id"] for r in rows}) == len(rows), "duplicate PubMedQA ids"
    return rows


LOADERS = {"trivia_qa": _load_trivia_qa, "pubmed_qa": _load_pubmed_qa}


def load_examples(name: str, cfg: dict) -> list[dict]:
    """Load (or build and cache) the fixed example list for one dataset."""
    path = results_dir(cfg) / f"data_{name}.jsonl"
    if path.exists():
        return read_jsonl(path)
    rows = LOADERS[name](cfg["datasets"][name], cfg["seed"])
    write_jsonl(path, rows)
    return rows


def sample_subset_ids(examples: list[dict], cfg: dict) -> list[str]:
    """Seeded subset of question ids that also get sampled answers (Phase 1, step 4)."""
    n = min(cfg["sampling"]["subset_size"], len(examples))
    rng = np.random.default_rng(cfg["seed"] + 1)  # offset: independent of dataset sampling
    idx = sorted(rng.choice(len(examples), size=n, replace=False).tolist())
    return [examples[i]["id"] for i in idx]


# --------------------------------------------------------------------------- prompts
def _truncate_middle(text: str, tokenizer, max_tokens: int) -> tuple[str, bool]:
    """Keep the first and last halves of an over-long abstract.

    The head carries the background/objective; the tail carries the results, which
    are the most informative part for a yes/no/maybe decision.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    half = max_tokens // 2
    head = tokenizer.decode(ids[:half])
    tail = tokenizer.decode(ids[-half:])
    return f"{head} [...] {tail}", True


def build_prompt(example: dict, tokenizer, cfg: dict) -> tuple[str, bool]:
    """Return (chat-formatted prompt string, was_context_truncated)."""
    truncated = False
    if example["dataset"] == "trivia_qa":
        user = TRIVIA_TEMPLATE.format(question=example["question"])
    else:
        ctx, truncated = _truncate_middle(
            example["context"], tokenizer, cfg["datasets"]["pubmed_qa"]["max_context_tokens"]
        )
        user = PUBMED_TEMPLATE.format(context=ctx, question=example["question"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, truncated
