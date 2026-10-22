"""Plot LL-V2.1 curves from saved CSVs. No training. No environment."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CURVE = ROOT / "outputs" / "LLV2_PHASE1_EVAL_CURVE.csv"
DESIGN = ROOT / "outputs" / "LLV2_PHASE1_DESIGN_METRICS.csv"
OUT = ROOT / "outputs" / "figures"


def main() -> None:
    rows = list(csv.DictReader(CURVE.open(encoding="utf-8")))
    design = list(csv.DictReader(DESIGN.open(encoding="utf-8")))
    xs = [int(r["update"]) for r in rows]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)

    def line(key, ylabel, name):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, [float(r[key]) for r in rows], marker="o")
        ax.set_xlabel("update")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=150)
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)

    line("payload", "payload bits", "payload_learning_curve")
    line("completed_blocks", "completed blocks", "completed_blocks_curve")
    line("projector_intervention_rate", "intervention rate", "projector_intervention_curve")
    line("min_sep", "meters", "min_separation_curve")
    line("mean_residual_norm", "residual norm", "residual_magnitude_curve")
    names = [r["method"] for r in design]
    vals = [float(r["payload"]) for r in design]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color=["#6c757d", "#2a6f97"])
    ax.set_ylabel("payload")
    ax.set_title("L0 vs ICAR-MAPPO payload")
    fig.tight_layout()
    fig.savefig(OUT / "l0_vs_icar_payload.png", dpi=150)
    fig.savefig(OUT / "l0_vs_icar_payload.pdf")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
