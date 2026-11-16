#!/usr/bin/env bash
# End-to-end reproduction. Every step caches its outputs under results/, so re-running
# after a crash resumes where it stopped instead of recomputing.
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."

# Use the project venv (Windows layout first, then POSIX layout).
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

# Phase 1: greedy answers, hidden states, sampled answers
$PY -m src.generate --dataset trivia_qa pubmed_qa

# Phase 2: correctness labels (+ prints 30 examples for manual review)
$PY -m src.label | tee results/label_review.txt

# Phase 3: baselines (token prob, self-consistency, semantic entropy), then clean timing
$PY -m src.baselines
$PY -m src.timing
