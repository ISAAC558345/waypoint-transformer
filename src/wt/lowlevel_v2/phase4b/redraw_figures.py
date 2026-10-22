"""Redraw Phase-4B figures from frozen CSVs. No training."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
P4 = Path(__file__).resolve().parent
OUTS = [P4 / "figures", P4 / "outputs" / "figures"]


def read_csv(name: str) -> list[dict]:
    with (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fv(row: dict, key: str) -> float:
    return float(row[key])


def main() -> None:
    head = {r["tag"]: r for r in read_csv("LLV2_PHASE4B_HIGH_LOW_HEADROOM.csv")}
    regret = read_csv("LLV2_PHASE4B_RL_COMM_REGRET.csv")
    align = read_csv("LLV2_PHASE4B_DIRECTION_ALIGNMENT.csv")
    jac = read_csv("LLV2_PHASE4B_COMM_JACOBIAN.csv")
    hi, lo = head["HIGH"], head["LOW"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [">1%", ">5%", ">10%", "median rel"]
    hi_v = [fv(hi, "frac_gt_1pct"), fv(hi, "frac_gt_5pct"), fv(hi, "frac_gt_10pct"), fv(hi, "median_rel")]
    lo_v = [fv(lo, "frac_gt_1pct"), fv(lo, "frac_gt_5pct"), fv(lo, "frac_gt_10pct"), fv(lo, "median_rel")]
    x = np.arange(4)
    ax.bar(x - 0.2, hi_v, 0.4, label="HIGH")
    ax.bar(x + 0.2, lo_v, 0.4, label="LOW")
    ax.set_xticks(list(x), labels)
    ax.set_title("HIGH vs LOW communication headroom")
    ax.legend()
    fig.tight_layout()
    figs = {"llv2_phase4b_high_low_headroom.png": fig}

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([r["model_seed"] for r in regret], [float(r["mean_regret_high"]) for r in regret])
    ax.set_ylabel("mean team-rate regret HIGH")
    ax.set_title("RL communication regret")
    fig.tight_layout()
    figs["llv2_phase4b_rl_regret.png"] = fig

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = ["sinr", "rate", "interference", "inr"]
    ax.bar(names, [float(np.mean([float(r[f"jac_{n}"]) for r in jac])) for n in names])
    ax.set_ylabel("mean ||d mu / d feat||")
    ax.set_title("feature Jacobian")
    fig.tight_layout()
    figs["llv2_phase4b_jacobian.png"] = fig

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([r["model_seed"] for r in align], [float(r["mean_cosine"]) for r in align])
    ax.axhline(0, color="k")
    ax.set_ylabel("RL vs best-comm cosine")
    ax.set_title("direction alignment")
    fig.tight_layout()
    figs["llv2_phase4b_direction_alignment.png"] = fig

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(["HIGH span", "LOW span"], [fv(hi, "mean_span"), fv(lo, "mean_span")])
    ax.set_title("candidate action utility")
    fig.tight_layout()
    figs["llv2_phase4b_candidate_utility_span.png"] = fig

    for dest in OUTS:
        dest.mkdir(parents=True, exist_ok=True)
        for name, figure in figs.items():
            figure.savefig(dest / name, dpi=130)
    print("wrote", [str(d) for d in OUTS])


if __name__ == "__main__":
    main()
