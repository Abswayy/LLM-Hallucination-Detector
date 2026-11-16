"""Phase 1: greedy answers + hidden-state extraction + sampled answers.

Usage:
    python -m src.generate --dataset trivia_qa pubmed_qa            # full run
    python -m src.generate --dataset trivia_qa pubmed_qa --limit 20 # smoke test

Outputs (per dataset, under results/<dataset>/ or results/smoke/<dataset>/):
    greedy.jsonl          one row per question: answer text, token ids, per-token logprobs
    hidden.pt             float16 hidden states, shape [N, n_layers, d_model], for
                          (a) the last prompt token and (b) the last generated answer token
    samples.jsonl         5 sampled answers for the seeded 500-question subset
    generation_meta.json  runtime, peak VRAM, forward-pass counts, settings
    _chunks/              per-chunk caches so a crash never loses finished work

Design notes
------------
* Greedy decoding uses `model.generate` with raw logits returned, so per-token
  log-probabilities are exact (no logits processors are active).
* Hidden states are captured with forward hooks in one extra forward pass over
  prompt + answer. The hooks keep only the two positions we need, so we never
  hold full [layers x seq x d_model] tensors in memory. In deployment, these same
  activations are produced during generation itself; the extra pass is just an
  implementation convenience here.
* "Layer L" means the residual stream *after* decoder block L (1-indexed), i.e.
  HF's `hidden_states[L]`. For the final layer we take the residual stream before
  the final RMSNorm so all layers are on the same footing.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .data import build_prompt, load_examples, sample_subset_ids
from .utils import load_config, read_jsonl, results_dir, seed_everything, write_json, write_jsonl

FP16_MAX = 65504.0


# --------------------------------------------------------------------------- model
def load_model(cfg: dict, quantization_config=None):
    """Load tokenizer + model on GPU with a hard VRAM cap and neutral generation config."""
    assert torch.cuda.is_available(), "CUDA is not available"
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    frac = min(1.0, cfg["model"]["max_vram_gb"] / total_gb)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"GPU: {props.name} | {total_gb:.1f} GB total | allocator capped at {frac * total_gb:.1f} GB")

    name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # left padding => the last position is the real last token
    kwargs = dict(dtype=getattr(torch, cfg["model"]["dtype"]), attn_implementation="sdpa")
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if quantization_config is None:
        model = model.to("cuda")
    model.eval()

    eos_ids = sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")})
    # Qwen ships a generation_config with sampling on, top_k=20, repetition_penalty=1.05.
    # Replace it with neutral defaults so every decoding setting is the one we pass.
    model.generation_config = GenerationConfig(
        eos_token_id=eos_ids, pad_token_id=tok.pad_token_id, bos_token_id=None
    )
    return tok, model, eos_ids


def layers_to_save(n_layers: int, stride: int) -> list[int]:
    layers = list(range(stride, n_layers + 1, stride))
    if layers[-1] != n_layers:
        layers.append(n_layers)
    return layers


class HiddenCapture:
    """Forward hooks that keep the residual stream at chosen (layer, position) pairs only."""

    def __init__(self, model, layer_ids: list[int]):
        self.layer_ids = layer_ids
        self.positions: torch.Tensor | None = None  # [B, 2] token indices; None = inactive
        self.store: dict[int, torch.Tensor] = {}
        self.max_abs = 0.0
        blocks = model.model.layers
        self.handles = [blocks[L - 1].register_forward_hook(self._hook(L)) for L in layer_ids]

    def _hook(self, L: int):
        def fn(module, inputs, output):
            if self.positions is None:
                return
            h = output[0] if isinstance(output, tuple) else output  # [B, T, d]
            rows = torch.arange(h.shape[0], device=h.device)[:, None]
            picked = h[rows, self.positions]  # [B, 2, d]
            self.max_abs = max(self.max_abs, picked.abs().max().item())
            self.store[L] = picked.to(torch.float16).cpu()

        return fn

    def collect(self) -> torch.Tensor:
        """Return [B, n_layers, 2, d] float16 and reset."""
        out = torch.stack([self.store[L] for L in self.layer_ids], dim=1)
        self.store = {}
        return out


# --------------------------------------------------------------------------- batches
def _split_answer(gen_row: list[int], eos_ids: list[int]) -> tuple[list[int], bool]:
    """Cut a generated id list at the first EOS. Returns (answer_ids, ended_with_eos)."""
    for i, t in enumerate(gen_row):
        if t in eos_ids:
            return gen_row[:i], True
    return gen_row, False


@torch.no_grad()
def greedy_batch(batch, model, tok, eos_ids, capture, cfg) -> list[dict]:
    prompts = [b["prompt"] for b in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    P = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        output_logits=True,
        return_dict_in_generate=True,
    )
    gen = out.sequences[:, P:]  # [B, T]
    T = gen.shape[1]
    # Exact per-step log-probs of the chosen tokens from raw (unprocessed) logits.
    step_logp = torch.empty(gen.shape, dtype=torch.float32, device=gen.device)
    for t in range(T):
        lp = torch.log_softmax(out.logits[t].float(), dim=-1)
        step_logp[:, t] = lp.gather(-1, gen[:, t : t + 1]).squeeze(-1)
    # Sanity check: at step 0 (no row finished yet) every chosen token must be the argmax.
    assert bool((out.logits[0].argmax(-1) == gen[:, 0]).all()), "decoding is not greedy"
    del out

    gen_l, logp_l = gen.tolist(), step_logp.tolist()
    attn = enc["attention_mask"].bool()
    rows, seqs = [], []
    for i, b in enumerate(batch):
        ans_ids, ended = _split_answer(gen_l[i], eos_ids)
        k = len(ans_ids)
        prompt_ids = enc["input_ids"][i][attn[i]].tolist()
        seqs.append((prompt_ids, ans_ids))
        rows.append(
            {
                "id": b["id"],
                "answer": tok.decode(ans_ids, skip_special_tokens=True).strip(),
                "answer_token_ids": ans_ids,
                "token_logprobs": [round(x, 6) for x in logp_l[i][:k]],
                "eos_logprob": round(logp_l[i][k], 6) if ended else None,
                "ended_with_eos": ended,
                "n_forward_passes": k + 1 if ended else T,  # decode steps actually run
                "prompt_len": len(prompt_ids),
                "context_truncated": b["truncated"],
            }
        )

    # Extra prompt+answer pass with hooks: grab last-prompt and last-answer-token states.
    L = max(len(p) + len(a) for p, a in seqs)
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    pos = torch.empty((len(seqs), 2), dtype=torch.long)
    for i, (p, a) in enumerate(seqs):
        s = p + a
        ids[i, L - len(s) :] = torch.tensor(s)
        mask[i, L - len(s) :] = 1
        pos[i, 0] = L - 1 - len(a)  # last prompt token ('\n' after '<|im_start|>assistant')
        pos[i, 1] = L - 1  # last answer token (== last prompt token if answer is empty)
    ids, mask, pos = ids.cuda(), mask.cuda(), pos.cuda()
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)  # same positions generate() used
    capture.positions = pos
    try:
        model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False)
    finally:
        capture.positions = None
    hid = capture.collect()  # [B, n_layers, 2, d]
    for i, r in enumerate(rows):
        r["_hidden"] = hid[i]
    return rows


@torch.no_grad()
def sample_batch(batch, model, tok, eos_ids, cfg) -> list[dict]:
    s = cfg["sampling"]
    n = s["n_samples"]
    enc = tok([b["prompt"] for b in batch], return_tensors="pt", padding=True,
              add_special_tokens=False).to("cuda")
    P = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=s["temperature"],
        top_p=s["top_p"],
        top_k=0,  # disable top-k so only nucleus sampling is active
        num_return_sequences=n,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        return_dict_in_generate=True,
    )
    gen = out.sequences[:, P:].tolist()  # [B*n, T], grouped by prompt
    rows = []
    for i, b in enumerate(batch):
        texts, steps = [], 0
        for j in range(n):
            ans, ended = _split_answer(gen[i * n + j], eos_ids)
            texts.append(tok.decode(ans, skip_special_tokens=True).strip())
            steps += len(ans) + 1 if ended else len(gen[i * n + j])
        rows.append({"id": b["id"], "samples": texts, "n_forward_passes": steps})
    return rows


def run_adaptive(fn, items: list, state: dict, min_bs: int, pbar) -> list:
    """Run fn over items in batches; on CUDA OOM halve the batch size and retry."""
    out, i = [], 0
    while i < len(items):
        bs = state["bs"]
        batch = items[i : i + bs]
        try:
            out.extend(fn(batch))
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if bs <= min_bs:
                raise
            state["bs"] = max(min_bs, bs // 2)
            tqdm.write(f"  OOM at batch size {bs} -> retrying with {state['bs']}")
            continue
        i += len(batch)
        pbar.update(len(batch))
    return out


# --------------------------------------------------------------------------- stages
def run_stage(stage, items, model, tok, eos_ids, capture, cfg, out_dir: Path) -> dict:
    """Process items in cached chunks. Returns aggregated timing/VRAM stats."""
    chunk_dir = out_dir / "_chunks" / stage
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cs = cfg["generation"]["chunk_size"]
    gcfg = cfg["generation"]

    # Guard against silently reusing stale caches: fingerprint the exact prompts + settings.
    fp = hashlib.sha256(json.dumps(
        [[it["id"], it["prompt"]] for it in items]
        + [cfg["model"]["name"], gcfg["max_new_tokens"], gcfg["layer_stride"], cfg["sampling"],
           cfg["seed"], cs],
        sort_keys=True).encode()).hexdigest()
    fp_path = chunk_dir / "fingerprint.txt"
    if fp_path.exists() and fp_path.read_text().strip() != fp:
        raise RuntimeError(f"{chunk_dir} was produced with different prompts/settings; "
                           "delete it to regenerate.")
    fp_path.write_text(fp)
    state = {"bs": gcfg["batch_size"] if stage == "greedy" else cfg["sampling"]["batch_size"]}
    fn = {
        "greedy": lambda b: greedy_batch(b, model, tok, eos_ids, capture, cfg),
        "sample": lambda b: sample_batch(b, model, tok, eos_ids, cfg),
    }[stage]

    n_chunks = (len(items) + cs - 1) // cs
    pbar = tqdm(total=len(items), desc=f"{out_dir.name}:{stage}", unit="ex")
    for c in range(n_chunks):
        path = chunk_dir / f"chunk_{c:04d}.pt"
        chunk = items[c * cs : (c + 1) * cs]
        if path.exists():
            pbar.update(len(chunk))
            continue
        # Per-chunk seed so a resumed run draws the same samples as an uninterrupted one
        # (exact only if the batch size did not change because of an OOM).
        seed_everything(cfg["seed"] + 1000 * (stage == "sample") + c)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows = run_adaptive(fn, chunk, state, gcfg["min_batch_size"], pbar)
        torch.cuda.synchronize()
        stats = {
            "seconds": time.perf_counter() - t0,
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "final_batch_size": state["bs"],
        }
        torch.save({"rows": rows, "stats": stats}, path)
    pbar.close()

    # Merge chunks (including ones from earlier, interrupted runs).
    rows, stats = [], []
    for c in range(n_chunks):
        d = torch.load(chunk_dir / f"chunk_{c:04d}.pt", weights_only=False)
        rows.extend(d["rows"])
        stats.append(d["stats"])
    return {
        "rows": rows,
        "seconds": sum(s["seconds"] for s in stats),
        "peak_alloc_gb": max(s["peak_alloc_gb"] for s in stats),
        "peak_reserved_gb": max(s["peak_reserved_gb"] for s in stats),
        "final_batch_size": min(s["final_batch_size"] for s in stats),
    }


def process_dataset(name, cfg, model, tok, eos_ids, layer_ids, limit=None) -> dict:
    examples = load_examples(name, cfg)
    sub_ids = set(sample_subset_ids(examples, cfg))
    if limit:
        examples = examples[:limit]
        sub_ids = {e["id"] for e in examples}  # smoke test: sample everything
        out_dir = results_dir(cfg) / "smoke" / name
    else:
        out_dir = results_dir(cfg) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for e in examples:
        prompt, trunc = build_prompt(e, tok, cfg)
        n_tok = len(tok(prompt, add_special_tokens=False)["input_ids"])
        items.append({"id": e["id"], "prompt": prompt, "truncated": trunc, "n_tok": n_tok})
    # Longest prompts first: less padding per batch, and any OOM shows up immediately.
    order = sorted(range(len(items)), key=lambda i: -items[i]["n_tok"])
    items_sorted = [items[i] for i in order]

    capture = HiddenCapture(model, layer_ids)
    g = run_stage("greedy", items_sorted, model, tok, eos_ids, capture, cfg, out_dir)
    for h in capture.handles:
        h.remove()

    by_id = {r["id"]: r for r in g["rows"]}
    ex_by_id = {e["id"]: e for e in examples}
    greedy_rows, hidden = [], []
    for e in examples:  # restore original order
        r = by_id[e["id"]]
        hidden.append(r.pop("_hidden"))
        r.update(question=e["question"], gold=e["gold"], in_sample_subset=e["id"] in sub_ids)
        greedy_rows.append(r)
    hidden = torch.stack(hidden)  # [N, n_layers, 2, d]
    assert torch.isfinite(hidden).all(), "non-finite hidden states after fp16 cast"
    assert capture.max_abs < FP16_MAX, f"activation {capture.max_abs} overflows float16"
    write_jsonl(out_dir / "greedy.jsonl", greedy_rows)
    torch.save(
        {
            "ids": [e["id"] for e in examples],
            "layers": layer_ids,
            "prompt_last": hidden[:, :, 0].contiguous(),
            "answer_last": hidden[:, :, 1].contiguous(),
            "model": cfg["model"]["name"],
        },
        out_dir / "hidden.pt",
    )

    sub_items = [it for it in items_sorted if it["id"] in sub_ids]
    s = run_stage("sample", sub_items, model, tok, eos_ids, None, cfg, out_dir)
    s_by_id = {r["id"]: r for r in s["rows"]}
    sample_rows = []
    for e in examples:
        if e["id"] in s_by_id:
            r = s_by_id[e["id"]]
            r.update(question=e["question"], gold=e["gold"], greedy=by_id[e["id"]]["answer"])
            sample_rows.append(r)
    write_jsonl(out_dir / "samples.jsonl", sample_rows)

    n = len(greedy_rows)
    meta = {
        "dataset": name,
        "model": cfg["model"]["name"],
        "n_questions": n,
        "n_sampled_questions": len(sample_rows),
        "layers_saved": layer_ids,
        "hidden_dim": int(hidden.shape[-1]),
        "max_abs_activation_saved": capture.max_abs,
        "n_context_truncated": sum(r["context_truncated"] for r in greedy_rows),
        "n_hit_max_new_tokens": sum(not r["ended_with_eos"] for r in greedy_rows),
        "n_empty_answers": sum(r["answer"] == "" for r in greedy_rows),
        "mean_prompt_tokens": sum(r["prompt_len"] for r in greedy_rows) / n,
        "greedy": {k: v for k, v in g.items() if k != "rows"}
        | {"mean_forward_passes": sum(r["n_forward_passes"] for r in greedy_rows) / n},
        "sampling": {k: v for k, v in s.items() if k != "rows"}
        | {"mean_forward_passes": sum(r["n_forward_passes"] for r in sample_rows) / max(1, len(sample_rows))},
        "settings": {"generation": cfg["generation"], "sampling": cfg["sampling"], "seed": cfg["seed"]},
        "torch": torch.__version__,
    }
    write_json(out_dir / "generation_meta.json", meta)
    return {"greedy": greedy_rows, "samples": sample_rows, "meta": meta, "out_dir": out_dir}


def _print_examples(res: dict, k: int) -> None:
    by_id = {r["id"]: r for r in res["samples"]}
    for r in res["greedy"][:k]:
        gold = r["gold"]["value"] if isinstance(r["gold"], dict) else r["gold"]
        print(f"- Q: {r['question'][:110]}")
        print(f"  gold: {gold!r} | greedy: {r['answer']!r} | "
              f"mean logp: {sum(r['token_logprobs']) / max(1, len(r['token_logprobs'])):.3f}")
        if r["id"] in by_id:
            print(f"  samples: {by_id[r['id']]['samples']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["trivia_qa", "pubmed_qa"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N examples")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    tok, model, eos_ids = load_model(cfg)
    layer_ids = layers_to_save(model.config.num_hidden_layers, cfg["generation"]["layer_stride"])
    print(f"Model {cfg['model']['name']}: {model.config.num_hidden_layers} layers, "
          f"d_model={model.config.hidden_size}; saving layers {layer_ids}")

    for name in args.dataset:
        res = process_dataset(name, cfg, model, tok, eos_ids, layer_ids, limit=args.limit)
        m = res["meta"]
        print(f"\n[{name}] {m['n_questions']} questions | greedy {m['greedy']['seconds']:.1f}s | "
              f"sampling {m['sampling']['seconds']:.1f}s | peak VRAM (allocated) "
              f"{max(m['greedy']['peak_alloc_gb'], m['sampling']['peak_alloc_gb']):.2f} GB")
        if args.limit:
            _print_examples(res, args.limit)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
