"""Paper-asset post-processing. Reads frozen CSVs/JSON only. No environment. No Phase-10.6 re-run."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE106 = ROOT / "src" / "wt" / "htw_phase106" / "outputs" / "untouched"
PHASE945 = ROOT / "src" / "wt" / "htw_phase945" / "outputs" / "utility_aligned"
PHASE943 = ROOT / "src" / "wt" / "htw_phase943" / "outputs" / "order_free_decoder"
FIG = ROOT / "paper_assets" / "figures"
TAB = ROOT / "paper_assets" / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")


def copy_tables() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    src_rows = _read_csv(PHASE106 / "phase106_final_results.csv")
    fields = ["method_id", "method", "payload", "completed_blocks", "residual_queue", "dropped", "flight_distance", "legality"]
    main = TAB / "PAPER_TABLE_MAIN_RESULTS.csv"
    with main.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in src_rows:
            writer.writerow({key: row[key] for key in fields})
    shutil.copy2(main, ROOT / "PAPER_TABLE_MAIN_RESULTS.csv")
    boot = json.loads((PHASE106 / "phase106_paired_bootstrap.json").read_text(encoding="utf-8"))
    names = {
        "M3_minus_M0": "FINAL-HIER vs Myopic",
        "M3_minus_M1": "FINAL-HIER vs Exclusive",
        "M3_minus_M2": "FINAL-HIER vs DeepSets-old",
    }
    pair = TAB / "PAPER_TABLE_PAIRWISE_STATS.csv"
    with pair.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["comparison", "mean_delta", "median_delta", "positive_scenario_fraction", "ci95_low", "ci95_high", "relative_gain"],
        )
        writer.writeheader()
        for key, label in names.items():
            item = boot[key]
            writer.writerow(
                {
                    "comparison": label,
                    "mean_delta": item["mean_delta"],
                    "median_delta": item["median_delta"],
                    "positive_scenario_fraction": item["positive_scenario_frac"],
                    "ci95_low": item["ci95"]["low"],
                    "ci95_high": item["ci95"]["high"],
                    "relative_gain": item["relative_gain"],
                }
            )
    shutil.copy2(pair, ROOT / "PAPER_TABLE_PAIRWISE_STATS.csv")


def plot_main(plt) -> None:
    rows = _read_csv(PHASE106 / "phase106_final_results.csv")
    names = [row["method"] for row in rows]
    payload = [float(row["payload"]) for row in rows]
    blocks = [float(row["completed_blocks"]) for row in rows]
    queue = [float(row["residual_queue"]) for row in rows]
    drop = [float(row["dropped"]) for row in rows]
    flight = [float(row["flight_distance"]) for row in rows]
    colors = ["#6c757d", "#84a98c", "#bc4749", "#2a6f97"]
    note = "Phase-10.6 Final Untouched (saved tables)"

    def bar(values, ylabel, stem, title):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(names, values, color=colors[: len(names)])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{note}")
        _save(fig, stem)
        plt.close(fig)

    bar(payload, "Payload (bits)", "fig02_final_payload", "Figure 2. Final payload comparison")
    bar(blocks, "Completed blocks", "fig04_completed_blocks", "Figure 4. Completed blocks")
    bar(flight, "Flight distance (m)", "fig06_flight_distance", "Figure 6. Flight distance")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(names, queue, color=colors)
    axes[0].set_title("Residual queue")
    axes[1].bar(names, drop, color=colors)
    axes[1].set_title("Drop")
    fig.suptitle(f"Figure 5. Residual queue / drop\n{note}")
    fig.tight_layout()
    _save(fig, "fig05_queue_drop")
    plt.close(fig)
    boot = json.loads((PHASE106 / "phase106_paired_bootstrap.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["vs Myopic", "vs Exclusive", "vs DeepSets-old"]
    means = [boot["M3_minus_M0"]["mean_delta"], boot["M3_minus_M1"]["mean_delta"], boot["M3_minus_M2"]["mean_delta"]]
    lows = [boot["M3_minus_M0"]["ci95"]["low"], boot["M3_minus_M1"]["ci95"]["low"], boot["M3_minus_M2"]["ci95"]["low"]]
    highs = [boot["M3_minus_M0"]["ci95"]["high"], boot["M3_minus_M1"]["ci95"]["high"], boot["M3_minus_M2"]["ci95"]["high"]]
    yerr = [[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]]
    ax.bar(labels, means, color="#2a6f97", yerr=yerr, capsize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean paired payload delta (bits)")
    ax.set_title(f"Figure 3. Paired payload delta (Phase-10.6 bootstrap)\n{note}")
    _save(fig, "fig03_paired_delta")
    plt.close(fig)


def plot_ablations(plt) -> None:
    summary = json.loads((PHASE945 / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["DeepSets-old", "DeepSets-util", "HGNN-old", "HGNN-util"]
    values = [summary["d_old"], summary["d_util"], summary["g_old"], summary["g_util"]]
    ax.bar(labels, values, color=["#6c757d", "#2a6f97", "#bc4749", "#84a98c"])
    ax.set_ylabel("Confirmation payload (bits)")
    ax.set_title("Figure 7. Utility-aligned objective (Phase-9.4.5 saved summary)")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "fig07_utility_aligned")
    plt.close(fig)
    seeds = _read_csv(PHASE945 / "phase945_per_seed_metrics.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = [int(row["model_seed"]) for row in seeds]
    ax.plot(xs, [float(row["d_util"]) for row in seeds], marker="o", label="DeepSets-util")
    ax.plot(xs, [float(row["g_util"]) for row in seeds], marker="s", label="HGNN-util")
    ax.set_xlabel("Model seed")
    ax.set_ylabel("Confirmation payload (bits)")
    ax.set_title("Figure 8a. DeepSets vs HGNN under utility-aligned training (Phase-9.4.5)")
    ax.legend()
    _save(fig, "fig08a_deepsets_vs_hgnn")
    plt.close(fig)
    perm = _read_csv(PHASE943 / "phase943_permutation_sensitivity.csv")
    ppv = [float(row["ppv"]) for row in perm]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(ppv, bins=[-0.05, 0.05, 0.5, 1.0], color="#2a6f97")
    ax.set_xlabel("Permutation payload variation (PPV)")
    ax.set_ylabel("Count (encoder-scenario rows)")
    ax.set_title("Figure 8b. Order-free decoder permutation sensitivity (Phase-9.4.3)")
    _save(fig, "fig08b_order_free_ppv")
    plt.close(fig)


def plot_architecture(plt) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.set_title("Figure 1. FINAL-HIER architecture (HGNN / PPO / Transformer not in the main stack)")
    boxes = [
        (0.2, 1.4, 1.8, 1.8, "Environment\nUAV set\nBuoy set\nqueue / SINR"),
        (2.3, 1.4, 1.9, 1.8, "DeepSets\nset encoder"),
        (4.5, 1.4, 2.2, 1.8, "Utility-aligned\nOrder-Free\nassignment decoder"),
        (7.0, 1.4, 1.9, 1.8, "Task Option\nhold=15\nbuoy / idle"),
        (9.2, 1.4, 2.9, 1.8, "fly-toward +\nInRangeZeroMotionGate\nmovement + service"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#e9eef3", edgecolor="#1d3557", linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ymid = boxes[i][1] + boxes[i][3] / 2
        ax.annotate("", xy=(x1, ymid), xytext=(x0, ymid), arrowprops={"arrowstyle": "->", "color": "#1d3557"})
    ax.text(6.25, 0.45, "Assignment  →  Task Option  →  Movement  →  Payload", ha="center", fontsize=10)
    _save(fig, "fig01_architecture")
    plt.close(fig)


def main() -> None:
    copy_tables()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_main(plt)
    plot_ablations(plt)
    plot_architecture(plt)
    print(str(FIG))
    print(str(TAB))


if __name__ == "__main__":
    main()
