"""PubMedQA prompt pilot (documents how the final prompt was chosen).

The original prompt made Qwen2.5-3B answer "maybe" on ~70% of questions vs 11% gold
"maybe". This script compares candidate prompt wordings on the first 100 PubMedQA
questions, using ONLY the distribution of yes/no/maybe answers. It never looks at
correctness, so the prompt is not selected for accuracy or for downstream AUROC.

Usage: python -m scripts.pubmed_prompt_pilot
"""
from __future__ import annotations

import collections
import json

import torch

from src.data import SYSTEM_PROMPT, _truncate_middle, load_examples
from src.generate import _split_answer, load_model
from src.utils import load_config, results_dir, seed_everything, write_json

N_PILOT = 100

CANDIDATES = {
    "v1_original": (
        "Read the abstract below and answer the research question with exactly one word: "
        "yes, no, or maybe. Reply with that single word only.\n\n"
        "Abstract:\n{context}\n\nQuestion: {question}"
    ),
    "v2_when_maybe": (
        "Read the abstract below and answer the research question with exactly one word: "
        "yes, no, or maybe. Answer \"yes\" or \"no\" whenever the findings in the abstract "
        "support a conclusion; answer \"maybe\" only if the evidence is genuinely mixed or "
        "inconclusive. Reply with that single word only.\n\n"
        "Abstract:\n{context}\n\nQuestion: {question}"
    ),
    # Instruction placed after the abstract (small models weight recent text heavily).
    "v3_instruction_last": (
        "Abstract:\n{context}\n\nQuestion: {question}\n\n"
        "Based on the findings in the abstract, answer the question with exactly one word. "
        "Answer \"yes\" or \"no\" if the findings support a conclusion. Answer \"maybe\" "
        "only if the evidence is genuinely mixed or inconclusive."
    ),
    # Yes/no framing with "maybe" as an explicit escape hatch; instruction last.
    "v4_yes_no_first": (
        "Abstract:\n{context}\n\nQuestion: {question}\n\n"
        "Do the findings of this study answer the question with yes or no? Reply with one "
        "word: \"yes\" or \"no\". Only if the study's results are truly inconclusive, "
        "reply \"maybe\"."
    ),
}


@torch.no_grad()
def greedy_answers(prompts, model, tok, eos_ids, bs=16):
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i : i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, do_sample=False, max_new_tokens=8)
        for row in gen[:, enc["input_ids"].shape[1] :].tolist():
            ans, _ = _split_answer(row, eos_ids)
            out.append(tok.decode(ans, skip_special_tokens=True).strip().lower().rstrip("."))
    return out


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    tok, model, eos_ids = load_model(cfg)
    exs = load_examples("pubmed_qa", cfg)[:N_PILOT]
    max_ctx = cfg["datasets"]["pubmed_qa"]["max_context_tokens"]
    report = {"n_pilot": N_PILOT,
              "gold_distribution": dict(collections.Counter(e["gold"] for e in exs))}
    for name, tmpl in CANDIDATES.items():
        prompts = []
        for e in exs:
            ctx, _ = _truncate_middle(e["context"], tok, max_ctx)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tmpl.format(context=ctx, question=e["question"])}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        dist = collections.Counter(greedy_answers(prompts, model, tok, eos_ids))
        report[name] = dict(dist)
        print(f"{name:22s} {dict(dist)}")
    print(f"{'gold':22s} {report['gold_distribution']}")
    write_json(results_dir(cfg) / "pubmed_prompt_pilot.json", report)


if __name__ == "__main__":
    main()
