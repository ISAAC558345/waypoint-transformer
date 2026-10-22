"""Plot stored Phase-6.x JSON means. Does not train or re-evaluate."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
J62 = ROOT / "src" / "wt" / "htw_phase62" / "outputs" / "goal_execution" / "summary.json"
J64 = ROOT / "src" / "wt" / "htw_phase64" / "outputs" / "high_level_improvement" / "summary.json"


def _cjk_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if name in names:
            return name
    return "DejaVu Sans"


def _style(font: str) -> None:
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style(_cjk_font())
    p62 = json.loads(J62.read_text(encoding="utf-8"))
    p64 = json.loads(J64.read_text(encoding="utf-8"))
    flat = float(p62["state_only"]["mean_payload_bits"])
    raw = next(float(r["mean_payload_bits"]) for r in p62["option_runs"] if r["variant"] == "A_r_comm")
    best_ll = float(p62["best_option"]["mean_payload_bits"])
    shadow = float(p62["oracle_shadow"]["mean_payload_bits"])
    learned63 = float(p64["phase63_learned"]["mean_payload_bits"])
    holds = []
    pay = []
    acc = []
    comp = []
    serv = []
    for name in ("A_hold5", "A_hold10", "A_hold15"):
        row = next(r for r in p64["runs"] if r["name"] == name)
        holds.append(int(row["hold"]))
        pay.append(float(row["mean_payload_bits"]))
        acc.append(float(row["option_selection_accuracy"]))
        comp.append(float(row["goal_completion_rate"]))
        serv.append(float(row["service_success_rate"]))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(holds, [v / 1e6 for v in pay], marker="o", color="#1f4e79", linewidth=2, label="学习高层 π_H（hold）")
    ax.axhline(flat / 1e6, color="#c0392b", linestyle="--", label=f"Flat PPO  {flat/1e6:.3f}e6")
    ax.axhline(learned63 / 1e6, color="#e67e22", linestyle=":", label=f"6.3 无 hold 学习高层  {learned63/1e6:.3f}e6")
    ax.axhline(best_ll / 1e6, color="#1e8449", linestyle="-.", label=f"6.2 Oracle+改进低层  {best_ll/1e6:.2f}e6")
    ax.set_xticks(holds)
    ax.set_xlabel("Option hold（步）")
    ax.set_ylabel("评估均值 payload（x10^6 bits）")
    ax.set_title("Level-2-A：学习高层 hold 与磁盘基线（单 seed=0，eval 8–15）")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_hold_payload_line.png", dpi=160)
    plt.close(fig)

    labels = [
        "Flat PPO",
        "Oracle+原始低层",
        "6.3 无 hold",
        "hold=5",
        "hold=10",
        "hold=15",
        "Oracle+改进低层",
        "fly-toward 影子",
    ]
    values = [flat, raw, learned63, pay[0], pay[1], pay[2], best_ll, shadow]
    colors = ["#c0392b", "#922b21", "#e67e22", "#5dade2", "#2e86c1", "#1f4e79", "#1e8449", "#7f8c8d"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(range(len(labels)), [v / 1e6 for v in values], color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("评估均值 payload（x10^6 bits）")
    ax.set_title("磁盘存档对照：层次 Task Option 相对 Flat PPO")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{val/1e6:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_method_payload_bars.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(holds, acc, marker="o", color="#1f4e79", label="option accuracy（相对逐步 Oracle）")
    ax.plot(holds, comp, marker="s", color="#1e8449", label="completion")
    ax.plot(holds, serv, marker="^", color="#8e44ad", label="service")
    ax.set_xticks(holds)
    ax.set_xlabel("Option hold（步）")
    ax.set_ylabel("比率")
    ax.set_ylim(0, 1)
    ax.set_title("Phase-6.4 学习高层：hold 与辅助指标（磁盘 runs[]）")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_hold_aux_metrics_line.png", dpi=160)
    plt.close(fig)

    names = [r["variant"] for r in p62["option_runs"]]
    payloads = [float(r["mean_payload_bits"]) / 1e6 for r in p62["option_runs"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(range(len(names)), payloads, color="#5dade2")
    ax.axhline(flat / 1e6, color="#c0392b", linestyle="--", label=f"Flat PPO  {flat/1e6:.3f}e6")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("评估均值 payload（x10^6 bits）")
    ax.set_title("Phase-6.2：Oracle z + 低层奖励网格（高层仍为 Oracle）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig4_phase62_reward_grid.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"wrote": [str(p) for p in sorted(OUT.glob("fig*.png"))]}, indent=2))


if __name__ == "__main__":
    main()
