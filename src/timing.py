"""Clean wall-clock cost of each LLM stage, measured on the 500-question sampled subset.

Usage:
    python -m src.timing          # run with the GPU otherwise idle

Phase 1 timings include our extra hidden-state pass (an implementation convenience), so
here each stage that a *deployed* detector would need is timed on its own, with the same
batch sizes as Phase 1:

    prefill        one forward pass over the prompt, no decoding. This is the entire LLM
                   cost of the prompt-only probe, which can run before answering.
    greedy         greedy decoding with logits returned. This gives the answer and the
                   token log-probs.
    greedy_hooks   greedy decoding while hooks keep the last position's residual stream
                   at the saved layers. This is the answer-token probe in deployment.
    sample5        5 sampled answers per question (self-consistency / semantic entropy).

Each stage is repeated `repeats` times; the median seconds/question is reported.
Results: results/metrics.json ["costs"][<dataset>]["llm"].
"""
from __future__ import annotations

import argparse
import json
import gc
import statistics
import time

import numpy as np
import torch

from .data import load_examples
from .generate import build_items, layers_to_save, load_model
from .utils import load_config, read_jsonl, results_dir, seed_everything, update_metrics


class LastPositionHooks:
    """Keep h[:, -1] at each saved layer on every forward call (stays on GPU)."""

    def __init__(self, model, layer_ids):
        self.kept = []
        self.handles = [model.model.layers[L - 1].register_forward_hook(self._fn)
                        for L in layer_ids]

    def _fn(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.kept.append(h[:, -1].detach())

    def remove(self):
        for h in self.handles:
            h.remove()


def _batches(items, bs):
    for i in range(0, len(items), bs):
        yield [it["prompt"] for it in items[i : i + bs]]


@torch.no_grad()
def run_stage(stage, items, model, tok, cfg, layer_ids):
    g, s = cfg["generation"], cfg["sampling"]
    # Sampling batches hold the same number of sequences for 5 and 10 samples (8x5 = 4x10).
    bs = {"sample5": s["batch_size"], "sample10": s["batch_size"] // 2}.get(stage, g["batch_size"])
    for prompts in _batches(items, bs):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        if stage == "prefill":
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            hooks = LastPositionHooks(model, layer_ids)
            model.model(**enc, position_ids=pos, use_cache=False)
            hooks.remove()
        elif stage in ("greedy", "greedy_hooks"):
            hooks = LastPositionHooks(model, layer_ids) if stage == "greedy_hooks" else None
            model.generate(**enc, do_sample=False, max_new_tokens=g["max_new_tokens"],
                           output_logits=True, return_dict_in_generate=True)
            if hooks:
                hooks.remove()
        elif stage in ("sample5", "sample10"):
            model.generate(**enc, do_sample=True, temperature=s["temperature"],
                           top_p=s["top_p"], top_k=0, num_return_sequences=int(stage[6:]),
                           max_new_tokens=g["max_new_tokens"])


ALL_STAGES = ["prefill", "greedy", "greedy_hooks", "sample5", "sample10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["trivia_qa", "pubmed_qa"])
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES[:4], choices=ALL_STAGES)
    ap.add_argument("--repeats", type=int, default=None, help="override timing.repeats")
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    qcfg = None
    if cfg["model"].get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
    tok, model, _ = load_model(cfg, quantization_config=qcfg)
    layer_ids = layers_to_save(model.config.num_hidden_layers, cfg["generation"]["layer_stride"])
    repeats = args.repeats or cfg.get("timing", {}).get("repeats", 3)
    stages = args.stages

    for name in args.dataset:
        rd = results_dir(cfg) / name
        sub_ids = {r["id"] for r in read_jsonl(rd / "greedy.jsonl") if r["in_sample_subset"]}
        items = build_items([e for e in load_examples(name, cfg) if e["id"] in sub_ids], tok, cfg)
        n = len(items)
        run_stage("greedy", items[:16], model, tok, cfg, layer_ids)  # warm-up (CUDA kernels)
        # Merge into earlier measurements so timing one new stage keeps the others.
        metrics_path = results_dir(cfg) / "metrics.json"
        prev = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        res = prev.get("costs", {}).get(name, {}).get("llm", {})
        for stage in stages:
            secs = []
            for _ in range(repeats):
                # Release cached blocks from the previous stage: fragmentation otherwise
                # makes the largest sampling batch OOM even though it fits on its own.
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_stage(stage, items, model, tok, cfg, layer_ids)
                torch.cuda.synchronize()
                secs.append((time.perf_counter() - t0) / n)
            res[stage] = {"seconds_per_question": statistics.median(secs),
                          "repeats_seconds_per_question": secs, "repeats": repeats}
            print(f"[{name}] {stage:13s} {1000 * res[stage]['seconds_per_question']:8.2f} ms/question "
                  f"(repeats: {[round(1000 * x, 2) for x in secs]})")
        # Forward passes (decoding steps) per question, from the actual Phase-1 outputs.
        greedy_rows = [r for r in read_jsonl(rd / "greedy.jsonl") if r["in_sample_subset"]]
        sample_rows = read_jsonl(rd / "samples.jsonl")
        res["forward_passes_per_question"] = {
            "prefill": 1.0,
            "greedy": float(np.mean([r["n_forward_passes"] for r in greedy_rows])),
            "sample5": float(np.mean([r["n_forward_passes"] for r in sample_rows])),
        }
        if (rd / "samples_extra.jsonl").exists():
            extra = {r["id"]: r["n_forward_passes"] for r in read_jsonl(rd / "samples_extra.jsonl")}
            res["forward_passes_per_question"]["sample10"] = float(np.mean(
                [r["n_forward_passes"] + extra[r["id"]] for r in sample_rows]))
        res["n_questions"] = n
        res["batch_sizes"] = {"greedy_and_prefill": cfg["generation"]["batch_size"],
                              "sample5_prompts": cfg["sampling"]["batch_size"],
                              "sample10_prompts": cfg["sampling"]["batch_size"] // 2}
        update_metrics(cfg, ["costs", name, "llm"], res)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
