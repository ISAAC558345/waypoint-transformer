"""Compact, reproducible Round-11 diagnostics; figures never read hidden splits."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def run(root: Path = Path.cwd()) -> int:
    out = root / "round_11/outputs/round11_task_waypoint_20260810_0220"
    fig = root / "round_11/figures/round11_task_waypoint_20260810_0220"; fig.mkdir(parents=True, exist_ok=False)
    with (out / "training_runs.csv").open(encoding="utf-8") as handle: runs = list(csv.DictReader(handle))
    with (out / "training_history.csv").open(encoding="utf-8") as handle: history = list(csv.DictReader(handle))
    metrics = ("val_total", "val_service", "val_progress", "val_return")
    for i in range(20):
        figure, axis = plt.subplots(figsize=(6, 4), facecolor="white"); axis.set_facecolor("white")
        if i < 12:
            row = runs[i % len(runs)]; h, s = row["H_task"], row["seed"]
            values = [x for x in history if x["H_task"] == h and x["seed"] == s]
            axis.plot([float(x["epoch"]) for x in values], [float(x["total"]) for x in values], color="#174a7e")
            axis.set_title(f"H={h}, seed={s}: validation total loss")
        else:
            name = metrics[(i - 12) % len(metrics)]
            axis.bar(range(len(runs)), [float(r[name]) for r in runs], color="#1f7a8c")
            axis.set_title(name + " across H/seed runs")
        axis.set_xlabel("validation checkpoint / run"); axis.set_ylabel("value"); figure.tight_layout()
        figure.savefig(fig / f"round11_{i+1:02d}.png", dpi=140); figure.savefig(fig / f"round11_{i+1:02d}.svg"); plt.close(figure)
    return 40

if __name__ == "__main__": print(run())
