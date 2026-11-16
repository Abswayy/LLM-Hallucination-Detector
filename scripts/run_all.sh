#!/usr/bin/env bash
# End-to-end reproduction. Every step caches its outputs, so re-running after a crash
# resumes where it stopped instead of recomputing.
#
#   bash scripts/run_all.sh                            # Qwen2.5-3B  -> results/
#   HP_CONFIG=config_7b.yaml bash scripts/run_all.sh   # Qwen2.5-7B (4-bit) -> results_7b/
#
# Run the 7B pipeline before the 3B one (or re-run `python -m src.report` afterwards) so
# the paper numbers include both models. Timing steps need an otherwise idle GPU.
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."

# Use the project venv (Windows layout first, then POSIX layout).
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
RES=$($PY -c "from src.utils import load_config; print(load_config()['paths']['results'])")
mkdir -p "$RES"
# The 7B run uses exactly the questions frozen by the 3B run.
if [ "$RES" != "results" ] && [ -f results/data_trivia_qa.jsonl ]; then
  cp -n results/data_trivia_qa.jsonl results/data_pubmed_qa.jsonl "$RES"/ || true
fi

# Phase 1: greedy answers, hidden states, 5 samples; then 5 more samples (10 in total)
$PY -m src.generate --dataset trivia_qa pubmed_qa
$PY -m src.generate --dataset trivia_qa pubmed_qa --stage extra

# Phase 2: correctness labels (+ prints 30 examples for manual review)
$PY -m src.label | tee "$RES/label_review.txt"

# Phase 3: baselines (token prob, self-consistency, semantic entropy 5/10 x strict/lenient)
$PY -m src.baselines
if [ "$RES" = "results" ]; then      # clean per-stage latency: 3B cost comparison only
  $PY -m src.timing
  $PY -m src.timing --stages sample10 --repeats 1
fi

# Phase 4: probes, prompt-only, transfer (+ layer sweep), shuffled control, robustness, SEPs
$PY -m src.probe | tee "$RES/phase4_log.txt"

# Phase 5: abstention metrics; figures, tables and paper numbers come from the 3B run
$PY -m src.evaluate
if [ "$RES" = "results" ]; then
  $PY -m src.plots
  $PY -m src.report                 # paper/numbers.tex, paper/tables/*.tex, README.md
  # Optional: compile the paper (needs Tectonic or another LaTeX distribution)
  if command -v tectonic >/dev/null 2>&1; then (cd paper && tectonic -X compile main.tex); fi
fi
