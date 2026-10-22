"""Phase-10.2 diagnosis runner. Replay + branches only; no training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase101.features import build_schema
from wt.htw_phase101.high_level import load_frozen_high
from wt.htw_phase101.models import L3BGoalActorCritic
from wt.htw_phase101.protocol import HIGH_CKPT, PPO
from wt.l3b.env import L3BEnv
from wt.round30.ppo import load_agent

from .analyze import payload_caliber, reset_breakdown, reward_audit, slot_metrics, windows
from .branches import run_branch
from .protocol import (
    DESIGN_VAL_SEEDS,
    LOW_FINAL,
    MAX_EPISODES,
    MAX_PER_SCENARIO,
    MAX_SNAPSHOTS,
    OUTPUT,
    PACK_REPORT,
    PHASE101_OUTPUT,
    REPORT,
    SNAPSHOT_RULE,
    dump_frozen,
    sha256_bytes,
)
from .replay import replay_episode


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


def _load_low(schema: dict[str, Any]):
    payload = torch.load(LOW_FINAL, map_location="cpu")
    if int(payload.get("model_shape", {}).get("max_uavs", -1)) == 5:
        raise RuntimeError("refused M=5 checkpoint")
    return load_agent(payload, PPO, L3BGoalActorCritic(schema))


def _disk_payloads() -> tuple[list[float], list[float]]:
    path = PHASE101_OUTPUT / "phase101_design_paired_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [float(r["b0_payload_bits"]) for r in rows], [float(r["b1_payload_bits"]) for r in rows]


def select_snapshots(b1: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    probe = L3BEnv()
    radius = float(probe.config.service_radius_m)
    buoys = np.asarray(probe.buoy_positions, dtype=np.float64)
    chosen: list[dict[str, Any]] = []
    per_scene: dict[int, int] = {}
    for seed in DESIGN_VAL_SEEDS:
        pack = b1[int(seed)]
        for index, start in enumerate(pack["start_snaps"]):
            if per_scene.get(int(seed), 0) >= MAX_PER_SCENARIO:
                break
            pending = int(start["ctrl"]["pending"])
            if pending <= 0:
                continue
            option = start["ctrl"]["option"]
            pos = np.asarray(start["env"]["uav_positions"], dtype=np.float64)
            queue = np.asarray(start["env"]["queue"], dtype=np.float64)
            for uav in range(int(probe.config.num_uavs)):
                if per_scene.get(int(seed), 0) >= MAX_PER_SCENARIO or len(chosen) >= MAX_SNAPSHOTS:
                    break
                tgt = int(option.targets[uav])
                if tgt < 0 or tgt >= int(probe.config.num_buoys):
                    continue
                if float(queue[tgt]) <= 1.0e-9:
                    continue
                if float(np.linalg.norm(pos[uav] - buoys[tgt])) > radius:
                    continue
                chosen.append(
                    {
                        "scenario": int(seed),
                        "index": int(index),
                        "uav": int(uav),
                        "target": int(tgt),
                        "t": int(start["t"]),
                        "pending": int(pending),
                        "snap": start,
                        "recorded": pack["trace"][index : index + pending],
                    }
                )
                per_scene[int(seed)] = per_scene.get(int(seed), 0) + 1
            if len(chosen) >= MAX_SNAPSHOTS:
                return chosen
    return chosen


def _window_tag_table(wins: list[dict[str, Any]]) -> dict[str, Any]:
    tags = ["A_late_or_short_hold", "B_in_range_low_rate", "C_switch_or_leave_range", "D_queue_empty", "F_unexplained", "F_never_in_range_with_queue", "completed"]
    out = {}
    for tag in tags:
        subset = [w for w in wins if tag in str(w["tags"]).split("|")]
        out[tag] = {
            "n_windows": len(subset),
            "n_in_range_queued_slots": int(sum(w["n_in_range_queued"] for w in subset)),
            "n_completed_blocks": int(sum(w["n_completed_blocks"] for w in subset)),
            "n_unfinished": int(sum(1 for w in subset if int(w["n_completed_blocks"]) == 0)),
        }
    out["note"] = "tag counts may overlap; do not add across tags as a payload decomposition"
    return out


def write_docs(caliber, metrics, resets, reward, branches, diagnosis, n_ep, seconds) -> None:
    (OUTPUT / "phase102_metric_definitions.md").write_text(
        "\n".join(
            [
                "# Phase-10.2 指标口径",
                "",
                "## Phase-10.1 in_range_active_frac（历史实现，未改）",
                "- 分母：assignment >= 0 的 UAV-时隙（排除 idle）。",
                "- 不排除无效目标（目标越界会在 env 中改成 idle）。",
                "- **不排除空队列**。空队列仍可计入 in-range。",
                "- 距离在 **env.step 之后** 计算。",
                "- r_aux 在 step 之后、针对本步已执行 option 计算。",
                "- progress_reset：任意 rho>0 变为 ~0，**包含正常完成清零**。",
                "",
                "## 本阶段补充指标",
                "- raw_in_range_frac：所有 UAV-时隙，含 idle（idle 的 in-range 恒为 False）。",
                "- active_in_range_frac：非 idle。",
                "- active_queued_in_range_frac：非 idle 且 step 开始时目标 queue>0。",
                "",
                "## Payload ratio",
                f"- mean(B1)/mean(B0) = {caliber['mean_b1_over_mean_b0']}",
                f"- mean((B1-B0))/mean(B0) = {caliber['mean_delta_over_mean_b0']}",
                f"- mean_i(B1_i/B0_i) = {caliber['mean_of_per_scene_ratio']}（Phase-10.1 `mean_ratio`）",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rec = diagnosis.get("recommendation", {})
    lines = [
        "# HTW Phase-10.2 进圈后服务完成差距诊断",
        "",
        "未训练。未改网络/reward/L3-B/hold。未访问 confirmation。未开新 Round。",
        "",
        f"1. 磁盘 B0/B1 均值 payload={caliber['mean_b0']}/{caliber['mean_b1']}；mean(B1)/mean(B0)={caliber['mean_b1_over_mean_b0']:.6f}；相对缺口={caliber['gap_frac']:.6%}；逐场景 ratio 平均={caliber['mean_of_per_scene_ratio']:.6f}。",
        f"2. 进圈率 B0/B1：raw={metrics['B0']['raw_in_range_frac']:.4f}/{metrics['B1']['raw_in_range_frac']:.4f}；active={metrics['B0']['active_in_range_frac']:.4f}/{metrics['B1']['active_in_range_frac']:.4f}；active+queue={metrics['B0']['active_queued_in_range_frac']:.4f}/{metrics['B1']['active_queued_in_range_frac']:.4f}。",
        f"3. 进度清零 B0 正常完成={resets['B0'].get('NORMAL_COMPLETION', 0)}，目标切换丢失={resets['B0'].get('TARGET_CHANGE', 0)}；B1 完成={resets['B1'].get('NORMAL_COMPLETION', 0)}，切换丢失={resets['B1'].get('TARGET_CHANGE', 0)}。10.1 的 reset 计数把完成清零算进去了。",
        f"4. 主要损失见 diagnosis JSON。完成 UAV-episode B0/B1={metrics['B0']['n_uav_completed_service']}/{metrics['B1']['n_uav_completed_service']}；从未完成={metrics['B0']['n_uav_never_completed']}/{metrics['B1']['n_uav_never_completed']}。",
        f"5. 零位移 C1-C0 完成块差均值={diagnosis.get('c1_minus_c0_blocks')}",
        f"6. fly-toward C2-C0 完成块差均值={diagnosis.get('c2_minus_c0_blocks')}",
        f"7. r_aux 空队列仍可进圈计分；idle 不计。空队列进圈次数 B1={reward['B1']['uav_empty_queue_in_range']}。",
        f"8. {diagnosis.get('fact_vs_hypothesis')}",
        f"9. 下一步只建议：{rec.get('item')}。依据：{rec.get('reason')}",
        f"10. 重放 episode={n_ep}，分支快照={diagnosis.get('n_snapshots')}，墙钟={seconds:.2f}s。",
        "",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    t0 = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    b0_disk, b1_disk = _disk_payloads()
    caliber = payload_caliber(b0_disk, b1_disk)
    dump_frozen(
        OUTPUT / "phase102_protocol.json",
        {
            "phase": "10.2",
            "no_training": True,
            "design_val_seeds": list(DESIGN_VAL_SEEDS),
            "high_ckpt": str(HIGH_CKPT),
            "low_ckpt": str(LOW_FINAL),
            "low_sha256": sha256_bytes(LOW_FINAL.read_bytes()),
            "max_episodes": MAX_EPISODES,
            "snapshot_rule": SNAPSHOT_RULE,
            "max_snapshots": MAX_SNAPSHOTS,
            "max_per_scenario": MAX_PER_SCENARIO,
            "caliber_from_disk": caliber,
            "do_not_reselect_checkpoint": True,
        },
    )
    schema = build_schema()
    high, high_sha = load_frozen_high(HIGH_CKPT)
    agent = _load_low(schema)
    assert sha256_bytes(HIGH_CKPT.read_bytes()) == high_sha
    events: list[dict[str, Any]] = []
    b0_runs = {}
    b1_runs = {}
    n_ep = 0
    for seed in DESIGN_VAL_SEEDS:
        b0 = replay_episode(high, int(seed), mode="B0", schema=schema, agent=None)
        b1 = replay_episode(high, int(seed), mode="B1", schema=schema, agent=agent)
        b0_runs[int(seed)] = b0
        b1_runs[int(seed)] = b1
        events.extend(b0["events"])
        events.extend(b1["events"])
        n_ep += 2
        if n_ep >= MAX_EPISODES:
            break
        if abs(b0["payload"] - b0_disk[list(DESIGN_VAL_SEEDS).index(int(seed))]) > 1.0e-6:
            raise RuntimeError(f"B0 payload replay mismatch seed={seed}")
        if abs(b1["payload"] - b1_disk[list(DESIGN_VAL_SEEDS).index(int(seed))]) > 1.0e-6:
            raise RuntimeError(f"B1 payload replay mismatch seed={seed}")
    csv_events = [{k: v for k, v in e.items()} for e in events]
    _write_csv(OUTPUT / "phase102_service_event_log.csv", csv_events)
    metrics = {"B0": slot_metrics(events, "B0"), "B1": slot_metrics(events, "B1")}
    resets = {"B0": reset_breakdown(events, "B0"), "B1": reset_breakdown(events, "B1")}
    wins0 = windows(events, "B0")
    wins1 = windows(events, "B1")
    _write_csv(OUTPUT / "phase102_service_window_summary.csv", wins0 + wins1)
    reward = {"B0": reward_audit(events, "B0"), "B1": reward_audit(events, "B1")}
    _write_csv(OUTPUT / "phase102_reward_component_audit.csv", [reward["B0"], reward["B1"]])
    chosen = select_snapshots(b1_runs)
    branch_rows = []
    c0_ok = 0
    c1_blocks = []
    c2_blocks = []
    c0_blocks = []
    aborted = 0
    for item in chosen:
        rec = item["recorded"]
        c0 = run_branch(item["snap"], kind="C0", focus_uav=int(item["uav"]), schema=schema, agent=agent, recorded=rec)
        row = {"scenario": item["scenario"], "t": item["t"], "uav": item["uav"], "target": item["target"], "pending": item["pending"]}
        if not c0["c0_match_ok"]:
            aborted += 1
            row.update({"aborted": True, "c0_match_fail": c0["c0_match_fail"]})
            branch_rows.append(row)
            continue
        c0_ok += 1
        c1 = run_branch(item["snap"], kind="C1", focus_uav=int(item["uav"]), schema=schema, agent=agent, recorded=None)
        c2 = run_branch(item["snap"], kind="C2", focus_uav=int(item["uav"]), schema=schema, agent=agent, recorded=None)
        c0_blocks.append(c0["focus_blocks"])
        c1_blocks.append(c1["focus_blocks"])
        c2_blocks.append(c2["focus_blocks"])
        row.update({"aborted": False})
        for name, res in (("c0", c0), ("c1", c1), ("c2", c2)):
            for key, value in res.items():
                row[f"{name}_{key}"] = value
        row["c1_minus_c0_blocks"] = int(c1["focus_blocks"]) - int(c0["focus_blocks"])
        row["c2_minus_c0_blocks"] = int(c2["focus_blocks"]) - int(c0["focus_blocks"])
        row["c1_minus_c0_payload"] = float(c1["team_payload_delta"]) - float(c0["team_payload_delta"])
        row["c2_minus_c0_payload"] = float(c2["team_payload_delta"]) - float(c0["team_payload_delta"])
        branch_rows.append(row)
    _write_csv(OUTPUT / "phase102_snapshot_branch_results.csv", branch_rows)
    mean_c1 = float(np.mean(np.asarray(c1_blocks) - np.asarray(c0_blocks))) if c0_blocks else None
    mean_c2 = float(np.mean(np.asarray(c2_blocks) - np.asarray(c0_blocks))) if c0_blocks else None
    if mean_c2 is not None and mean_c2 > 0.05:
        rec = {"item": "研究圈内真实服务位置/进度完成量训练信号", "reason": "同快照下仅焦点 UAV 改 fly-toward 提高了完成块，说明圈内位置/速率而非“不会飞”。"}
    elif mean_c1 is not None and mean_c1 > 0.05:
        rec = {"item": "研究圈内动作稳定性（接近零位移）", "reason": "同快照零位移减少丢失或提高完成块。"}
    elif reward["B1"]["uav_empty_queue_in_range"] > 0 and metrics["B1"]["active_queued_in_range_frac"] + 0.02 < metrics["B1"]["active_in_range_frac"]:
        rec = {"item": "研究真实服务进度/完成量的训练信号", "reason": "进圈含空队列；r_aux 不要求 queue>0。这是可复现的辅助奖励实现问题，但未授权改 lambda。完成缺口仍需完成量信号。"}
    else:
        rec = {"item": "保留规则执行器，停止无依据修改", "reason": "局部分支未证明可稳定补回完成量，且不得因未超过规则改高层/HGNN。"}
    diagnosis = {
        "caliber": caliber,
        "slot_metrics": metrics,
        "resets": resets,
        "window_tags_b0": _window_tag_table(wins0),
        "window_tags_b1": _window_tag_table(wins1),
        "reward": reward,
        "n_snapshots": len(chosen),
        "c0_match_ok": c0_ok,
        "c0_aborted": aborted,
        "c1_minus_c0_blocks": mean_c1,
        "c2_minus_c0_blocks": mean_c2,
        "env_note": "L3-B does not zero progress on leaving range; only target change and block completion zero rho",
        "phase101_in_range_includes_empty_queue": True,
        "recommendation": rec,
        "fact_vs_hypothesis": "磁盘 payload/重放一致性、指标分母、完成清零混入 reset、r_aux 空队列计分为日志事实；分支增益不可外推整 episode，也不可填回全部 1.25e6。",
        "trained": False,
    }
    dump_frozen(OUTPUT / "phase102_diagnosis.json", diagnosis)
    seconds = perf_counter() - t0
    write_docs(caliber, metrics, resets, reward, branch_rows, diagnosis, n_ep, seconds)
    print(json.dumps({"episodes": n_ep, "seconds": seconds, "ratio": caliber["mean_b1_over_mean_b0"], "c2_minus_c0": mean_c2, "trained": False}, indent=2), flush=True)
    return diagnosis


if __name__ == "__main__":
    main()
