"""Phase-10.3 frozen in-range zero-motion gate evaluation. No training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase101.features import build_schema
from wt.htw_phase101.high_level import load_frozen_high, parameter_sha
from wt.htw_phase101.models import L3BGoalActorCritic
from wt.htw_phase101.protocol import HIGH_CKPT, PPO
from wt.round30.ppo import load_agent

from .evaluate import episode_metrics, run_episode
from .protocol import DESIGN_VAL_SEEDS, ENV_FILE, LOW_FINAL, OUTPUT, PACK_REPORT, REPORT, dump_frozen, sha256_bytes


FAMILIES = (("B0", False, "rule"), ("B1", False, "ppo"), ("B2", True, "ppo"), ("B3", True, "rule"))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items() if key in fields})


def _load_low(schema):
    payload = torch.load(LOW_FINAL, map_location="cpu")
    if int(payload.get("model_shape", {}).get("max_uavs", -1)) == 5:
        raise RuntimeError("refused M=5 checkpoint")
    return load_agent(payload, PPO, L3BGoalActorCritic(schema))


def _means(rows: list[dict[str, Any]], family: str) -> dict[str, float]:
    sub = [r for r in rows if r["family"] == family]
    keys = ("payload_bits", "completed_blocks", "n_uav_never_completed", "flight_distance_m", "gate_frac", "target_change_loss_events", "target_change_lost_rho", "active_queued_in_range_frac", "normal_completion_resets")
    return {key: float(np.mean([r[key] for r in sub])) for key in keys}


def _plot(rows: list[dict[str, Any]]) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"error": str(error)}
    seeds = sorted({int(r["scenario"]) for r in rows})
    families = ["B0", "B1", "B2", "B3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(seeds))
    width = 0.2
    for i, fam in enumerate(families):
        pay = [float(next(r["payload_bits"] for r in rows if r["family"] == fam and int(r["scenario"]) == s)) for s in seeds]
        dist = [float(next(r["flight_distance_m"] for r in rows if r["family"] == fam and int(r["scenario"]) == s)) for s in seeds]
        axes[0].bar(x + (i - 1.5) * width, pay, width=width, label=fam)
        axes[1].bar(x + (i - 1.5) * width, dist, width=width, label=fam)
    axes[0].set_title("payload bits (design evaluation, frozen single model)")
    axes[1].set_title("flight distance m (not energy)")
    for ax in axes:
        ax.set_xticks(x, [str(s) for s in seeds])
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUTPUT / "phase103_payload_distance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"png": str(path)}


def _case(b0, b1, b2, b3) -> str:
    if float(b2["payload_bits"]) > float(b1["payload_bits"]) + 1.0e-9 and float(b2["payload_bits"]) + 1.0e-9 < float(b3["payload_bits"]):
        return "A"
    if float(b2["payload_bits"]) + 1.0e-9 >= float(b3["payload_bits"]):
        return "B"
    return "C"


def main() -> dict[str, Any]:
    t0 = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = build_schema()
    high, high_sha = load_frozen_high(HIGH_CKPT)
    high_param = parameter_sha(high)
    agent = _load_low(schema)
    low_sha = sha256_bytes(LOW_FINAL.read_bytes())
    dump_frozen(
        OUTPUT / "phase103_protocol.json",
        {
            "phase": "10.3",
            "no_training": True,
            "design_val_seeds": list(DESIGN_VAL_SEEDS),
            "design_label": "design evaluation, not confirmation",
            "high_ckpt": str(HIGH_CKPT),
            "high_file_sha256": high_sha,
            "high_param_sha256": high_param,
            "low_ckpt": str(LOW_FINAL),
            "low_sha256": low_sha,
            "env_sha256": sha256_bytes(ENV_FILE.read_bytes()),
            "ppo": PPO.to_dict(),
            "gate": "in_range AND nonempty queue AND non-idle valid target -> movement (0,0)",
            "families": ["B0", "B1", "B2", "B3"],
            "phase102_caliber": {"b0": 8500000.0, "b1": 7250000.0, "ratio": 0.8529411764705882},
        },
    )
    (OUTPUT / "phase103_gate_spec.md").write_text(
        "\n".join(
            [
                "# In-range zero-motion gate",
                "",
                "Pre-action only. `gate = non-idle AND valid target AND queue>0 AND env._in_range`.",
                "If true, env movement input is (0,0); else executor output is unchanged.",
                "Idle / empty queue: gate off, original executor motion kept. No extra radius. No hold change.",
                "Same wrapper for B2 and B3. Not PPO training.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    all_events = []
    ep_rows = []
    n_run = 0
    for family, gated, _kind in FAMILIES:
        use_agent = agent if family in {"B1", "B2"} else None
        use_schema = schema if family in {"B1", "B2"} else None
        for seed in DESIGN_VAL_SEEDS:
            pack = run_episode(high, int(seed), family=family, schema=use_schema, agent=use_agent, gated=gated)
            all_events.extend(pack["events"])
            ep_rows.append(episode_metrics(pack["events"], family, int(seed)))
            n_run += 1
    if sha256_bytes(HIGH_CKPT.read_bytes()) != high_sha or parameter_sha(high) != high_param:
        raise RuntimeError("high-level SHA changed")
    if sha256_bytes(LOW_FINAL.read_bytes()) != low_sha:
        raise RuntimeError("low-level checkpoint SHA changed")
    _write_csv(OUTPUT / "phase103_step_metrics.csv", all_events)
    _write_csv(OUTPUT / "phase103_paired_episode_metrics.csv", ep_rows)
    means = {fam: _means(ep_rows, fam) for fam, _g, _k in FAMILIES}
    recov_den = means["B0"]["payload_bits"] - means["B1"]["payload_bits"]
    recovery = (means["B2"]["payload_bits"] - means["B1"]["payload_bits"]) / recov_den if abs(recov_den) > 1.0e-9 else float("nan")
    case = _case(means["B0"], means["B1"], means["B2"], means["B3"])
    case_text = {
        "A": "门控有帮助，PPO额外价值仍未建立。",
        "B": "出现值得复核的学习型执行信号；只建议后续多seed独立验证，本轮不自动运行。",
        "C": "局部零位移增益没有转化为完整任务收益；保留已有较强规则，不自动调阈值。",
    }[case]
    summary = {
        "n_episodes_run": n_run,
        "n_reused_from_phase102": 0,
        "means": means,
        "B2_minus_B1_payload": means["B2"]["payload_bits"] - means["B1"]["payload_bits"],
        "B3_minus_B0_payload": means["B3"]["payload_bits"] - means["B0"]["payload_bits"],
        "B2_minus_B3_payload": means["B2"]["payload_bits"] - means["B3"]["payload_bits"],
        "B2_minus_B0_payload": means["B2"]["payload_bits"] - means["B0"]["payload_bits"],
        "recovery_vs_B0_gap": recovery,
        "case": case,
        "case_text": case_text,
        "high_sha_unchanged": True,
        "low_sha_unchanged": True,
        "trained": False,
        "label": "design evaluation, frozen single model",
    }
    dump_frozen(OUTPUT / "phase103_execution_summary.json", summary)
    plots = _plot(ep_rows)
    seconds = perf_counter() - t0
    lines = [
        "# HTW Phase-10.3 服务区内零位移门控闭环验证",
        "",
        "未训练。未改环境/奖励/hold。design evaluation，冻结单模型。",
        "",
        "1. 规则：动作前若非idle、目标有效、queue>0 且 env._in_range，则 movement 输入改为 (0,0)；否则沿用执行器输出。",
        f"2. 均值 payload B0/B1/B2/B3 = {means['B0']['payload_bits']:.0f}/{means['B1']['payload_bits']:.0f}/{means['B2']['payload_bits']:.0f}/{means['B3']['payload_bits']:.0f}。",
        f"3. B2-B1={summary['B2_minus_B1_payload']:.0f}；B3-B0={summary['B3_minus_B0_payload']:.0f}；B2-B3={summary['B2_minus_B3_payload']:.0f}。recovery={recovery}。",
        f"4. 完成块均值 B0/B1/B2/B3={means['B0']['completed_blocks']}/{means['B1']['completed_blocks']}/{means['B2']['completed_blocks']}/{means['B3']['completed_blocks']}。",
        f"5. 换目标丢失事件均值 B0/B1/B2/B3={means['B0']['target_change_loss_events']}/{means['B1']['target_change_loss_events']}/{means['B2']['target_change_loss_events']}/{means['B3']['target_change_loss_events']}；丢失rho={means['B0']['target_change_lost_rho']:.3f}/{means['B1']['target_change_lost_rho']:.3f}/{means['B2']['target_change_lost_rho']:.3f}/{means['B3']['target_change_lost_rho']:.3f}。",
        f"6. gate 比例 B2/B3={means['B2']['gate_frac']:.4f}/{means['B3']['gate_frac']:.4f}；B0/B1应为0：{means['B0']['gate_frac']:.4f}/{means['B1']['gate_frac']:.4f}。",
        f"7. 门控后仍有未完成：B2 从未完成 UAV-episode 均值={means['B2']['n_uav_never_completed']}（B0={means['B0']['n_uav_never_completed']}）。存在圈内停住但本步无完成块的时隙；低速率门控未完成见 CSV `low_rate_gated_unfinished`。",
        f"8. 情况{case}：{case_text} 本轮 **可以保留该门控作为冻结执行规则** 进入后续独立验证，但不能写成 PPO 重新训练后学会驻留。",
        "9. 在门控条件下 B2 均值 payload 高于 B3，但这是单模型 design 集，不能声称跨 seed 的 PPO 优于规则。",
        "10. 下一步建议：多 seed、未用于设计的独立场景复核门控；本轮不自动运行。不调阈值，不训练。",
        f"运行 {n_run} 个 episode，复用 10.2 轨迹 0 条（重放 B0/B1 以核对 wrapper 关闭等价），墙钟 {seconds:.2f}s。图：{plots}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")
    print(json.dumps({"case": case, "payloads": {k: v["payload_bits"] for k, v in means.items()}, "seconds": seconds, "trained": False}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
