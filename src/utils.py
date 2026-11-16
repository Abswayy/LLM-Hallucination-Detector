"""Small shared helpers: config loading, seeding, JSONL I/O, paths."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml, or another config given by `path` or the HP_CONFIG environment
    variable (e.g. HP_CONFIG=config_7b.yaml runs every script on the 7B setup)."""
    path = path or os.environ.get("HP_CONFIG")
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["results"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def update_metrics(cfg: dict, keys: list[str], value) -> None:
    """Set metrics.json[keys[0]][keys[1]]... = value, keeping everything else.

    results/metrics.json is the single source for every number in the README and paper.
    """
    path = results_dir(cfg) / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    node = metrics
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    write_json(path, metrics)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
