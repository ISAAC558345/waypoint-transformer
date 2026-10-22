"""CSV and stat helpers for Phase-9.4.4."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size < 3 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranked = np.empty_like(order, dtype=np.float64)
        ranked[order] = np.arange(values.size, dtype=np.float64)
        return ranked

    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size < 3:
        return float("nan")
    return pearson(ranks(a), ranks(b))


def parse_training_log(path: Path) -> list[dict[str, Any]]:
    pattern = re.compile(r"(deepsets_set|hgnn_set) seed=(\d+) epoch (\d+)/(\d+) loss=([0-9.]+)")
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name, seed, epoch, total, loss = match.groups()
        encoder = "deepsets" if name.startswith("deepsets") else "hgnn"
        rows.append(
            {
                "encoder": encoder,
                "model_seed": int(seed),
                "epoch": int(epoch),
                "epochs": int(total),
                "train_loss": float(loss),
            }
        )
    return rows
