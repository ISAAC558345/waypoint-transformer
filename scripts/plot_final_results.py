"""Post-process Phase-10.6 saved tables. Does not import the environment or run models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE106_CSV = ROOT / "src" / "wt" / "htw_phase106" / "outputs" / "untouched" / "phase106_final_results.csv"
PHASE106_BOOT = ROOT / "src" / "wt" / "htw_phase106" / "outputs" / "untouched" / "phase106_paired_bootstrap.json"
OUT = ROOT / "final_artifacts" / "figures"


def _read_methods(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot frozen Phase-10.6 tables. No environment. No untouched re-run.")
    parser.add_argument("--csv", type=Path, default=PHASE106_CSV)
    parser.add_argument("--bootstrap", type=Path, default=PHASE106_BOOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    rows = _read_methods(args.csv)
    names = [row["method"] for row in rows]
    payload = [float(row["payload"]) for row in rows]
    blocks = [float(row["completed_blocks"]) for row in rows]
    queue = [float(row["residual_queue"]) for row in rows]
    drop = [float(row["dropped"]) for row in rows]
    flight = [float(row["flight_distance"]) for row in rows]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    note = "FINAL UNTOUCHED TEST (saved tables only)"
    args.out.mkdir(parents=True, exist_ok=True)
    colors = ["#6c757d", "#84a98c", "#bc4749", "#2a6f97"]

    def bar(values, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, values, color=colors[: len(names)])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{note}")
        fig.tight_layout()
        fig.savefig(args.out / fname, dpi=120)
        plt.close(fig)

    bar(payload, "payload bits", "final_payload.png", "Final payload comparison")
    bar(blocks, "completed blocks", "final_completed_blocks.png", "Completed blocks")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, queue, color=colors)
    axes[0].set_title("Residual queue")
    axes[1].bar(names, drop, color=colors)
    axes[1].set_title("Drop")
    fig.suptitle(note)
    fig.tight_layout()
    fig.savefig(args.out / "final_queue_drop.png", dpi=120)
    plt.close(fig)
    bar(flight, "flight distance (m)", "final_flight.png", "Flight distance")
    boot = json.loads(args.bootstrap.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["M3-M0", "M3-M1", "M3-M2"]
    means = [boot["M3_minus_M0"]["mean_delta"], boot["M3_minus_M1"]["mean_delta"], boot["M3_minus_M2"]["mean_delta"]]
    ax.bar(labels, means, color="#2a6f97")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("mean paired payload delta")
    ax.set_title(f"Final paired payload delta (saved bootstrap means)\n{note}")
    fig.tight_layout()
    fig.savefig(args.out / "final_paired_delta.png", dpi=120)
    plt.close(fig)
    print(str(args.out))


if __name__ == "__main__":
    main()
