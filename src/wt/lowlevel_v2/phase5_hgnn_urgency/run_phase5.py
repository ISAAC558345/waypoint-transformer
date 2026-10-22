"""Run the four Phase-5 ablations on an isolated seed band."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.observation import COMM_DIM, SELF_DIM, TASK_DIM

from .actor import ORIGINAL_OBS_DIM, URGENCY_INDEX, URGENCY_OBS_DIM
from .encoder import Phase5OrderFreeModel
from .evaluate import (
    high_level_changes_when_urgency_zeroed,
    judge,
    mean_alignment,
    rollout_metrics,
    urgency_sensitivity,
)
from .planner import Phase5Planner, load_frozen_deepsets, new_hgnn
from .protocol import (
    DEPLOYMENT_VERIFIED,
    EVAL_SEEDS,
    HL_EPOCHS,
    MAPPO_UPDATES,
    OUTPUT,
    PLANNER_READY,
    REPORT,
    TRAIN_SEEDS,
    assert_pool,
)
from .train import collect_observable_samples, train_hgnn, train_mappo
from .worker import Phase5Worker

VARIANTS = (
    {"name": "Baseline DeepSets+MAPPO", "high": "deepsets", "high_urgency": False, "low_urgency": False},
    {"name": "HGNN+MAPPO", "high": "hgnn", "high_urgency": False, "low_urgency": False},
    {"name": "DeepSets+UrgencyMAPPO", "high": "deepsets", "high_urgency": False, "low_urgency": True},
    {"name": "HGNN+UrgencyMAPPO", "high": "hgnn", "high_urgency": True, "low_urgency": True},
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _mean(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _report(summary: dict) -> str:
    lines = [
        "# HTW Phase-5 HGNN + Shared Urgency",
        "",
        f"**phase_status = {summary['phase_status']}**",
        "",
        f"**deployment_verified = {str(summary['deployment_verified'])}**",
        "",
        f"**planner_ready = {str(summary['planner_ready'])}**",
        "",
        "本阶段按隔离种子执行了四组消融。没有修改 reward，没有修改 `high_level.decide()`，没有使用 230000–230127，没有中间评价层，没有 HAC，也没有高低层联合更新。",
        "",
        "## 实现",
        "",
        "- 高层：HGNN 只替换 encoder。UAV–UAV 与 UAV–Buoy 做消息传递。解码器仍是 `OrderFreeSetDecoder` 的 joint greedy。",
        "- 共享紧急程度：`u_k(t) = q_k(t) / (max_j q_j(t) + 1e-8)`，范围 [0, 1]。只读当前队列。一个 option 的 15 步内不重算。",
        "- 低层：urgency-aware Residual MAPPO 在原观测后追加该无人机所分配浮标的 urgency。动作仍是二维位移残差。",
        "- 环境没有能耗状态。表中的路径长度是位移积分，不是新的能耗模型。",
        "",
        f"训练种子 {summary['train_seeds']}，评估种子 {summary['eval_seeds']}。高层 {summary['hl_epochs']} 个 epoch，低层 {summary['mappo_updates']} 次 MAPPO 更新。这是有界探针，不是收敛后的部署模型。",
        "",
        "## 消融",
        "",
        "| 实验 | payload | residual queue | completion | path length | trajectory reward | assignment match | conflict | entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["ablation"]:
        lines.append(
            "| {name} | {payload} | {residual_queue} | {completion_time} | {path_length_m} | {trajectory_reward} | {assignment_match} | {assignment_conflict} | {assignment_entropy} |".format(
                name=row["experiment"],
                payload=_fmt(row["payload"]),
                residual_queue=_fmt(row["residual_queue"]),
                completion_time=_fmt(row["completion_time"]),
                path_length_m=_fmt(row["path_length_m"]),
                trajectory_reward=_fmt(row["trajectory_reward"]),
                assignment_match=_fmt(row["assignment_match"]),
                assignment_conflict=_fmt(row["assignment_conflict"]),
                assignment_entropy=_fmt(row["assignment_entropy"]),
            )
        )
    lines.extend(
        [
            "",
            "## 高低层一致性",
            "",
            "Q_H 是 order-free decoder 在其他无人机已分配条件下对候选浮标的 logit。U_LL 是把该候选锁定 15 步后，当前低层策略的累计 reward。",
            "",
            "| 实验 | Spearman | Kendall | Top-1 | normalized regret | n |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["alignment"]:
        lines.append(
            f"| {row['experiment']} | {_fmt(row['spearman'])} | {_fmt(row['kendall'])} | {_fmt(row['top1_agreement'])} | {_fmt(row['normalized_regret'])} | {row['n']} |"
        )
    lines.extend(["", "## Urgency 是否被使用", ""])
    for row in summary["urgency_effect"]:
        lines.append(
            f"- {row['experiment']}: low-level |Δμ|={_fmt(row['low_level_action_delta'])} ({row['low_level_flag']}); high-level zero-urgency changes assignment={row['high_level_changes']} ({row['high_level_flag']}); assigned-minus-rest urgency={_fmt(row['urgency_gap'])}."
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            summary["judgment_text"],
            "",
            "下一阶段没有启动。`planner_ready` 保持 false。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    assert_pool(TRAIN_SEEDS)
    assert_pool(EVAL_SEEDS)
    if set(TRAIN_SEEDS) & set(EVAL_SEEDS):
        raise RuntimeError("train and eval seeds overlap")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print("collect observable high-level labels", flush=True)
    samples = collect_observable_samples(TRAIN_SEEDS)
    deepsets = load_frozen_deepsets()
    deepsets_planner = Phase5Planner("deepsets", high_urgency=False, model=deepsets)
    hgnn = new_hgnn(high_urgency=False)
    hgnn_u = new_hgnn(high_urgency=True)
    hgnn_log = train_hgnn(hgnn, samples, high_urgency=False, epochs=HL_EPOCHS)
    hgnn_u_log = train_hgnn(hgnn_u, samples, high_urgency=True, epochs=HL_EPOCHS)
    torch.save(hgnn.state_dict(), checkpoint_dir / "hgnn_orderfree.pt")
    torch.save(hgnn_u.state_dict(), checkpoint_dir / "hgnn_urgency_orderfree.pt")
    hgnn_planner = Phase5Planner("hgnn", high_urgency=False, model=hgnn)
    hgnn_u_planner = Phase5Planner("hgnn", high_urgency=True, model=hgnn_u)
    print("train original MAPPO under frozen DeepSets", flush=True)
    actor_base, critic_base, mappo_base = train_mappo(deepsets_planner, use_urgency=False, seeds=TRAIN_SEEDS, updates=MAPPO_UPDATES)
    print("train urgency MAPPO under frozen DeepSets", flush=True)
    actor_u, critic_u, mappo_u = train_mappo(deepsets_planner, use_urgency=True, seeds=TRAIN_SEEDS, updates=MAPPO_UPDATES)
    print("train urgency MAPPO under HGNN urgency", flush=True)
    actor_full, critic_full, mappo_full = train_mappo(hgnn_u_planner, use_urgency=True, seeds=TRAIN_SEEDS, updates=MAPPO_UPDATES)
    torch.save(actor_base.state_dict(), checkpoint_dir / "mappo_original.pt")
    torch.save(actor_u.state_dict(), checkpoint_dir / "mappo_urgency_deepsets.pt")
    torch.save(actor_full.state_dict(), checkpoint_dir / "mappo_urgency_hgnn.pt")
    bundles = {
        "Baseline DeepSets+MAPPO": (deepsets_planner, actor_base, critic_base, False),
        "HGNN+MAPPO": (hgnn_planner, actor_base, critic_base, False),
        "DeepSets+UrgencyMAPPO": (deepsets_planner, actor_u, critic_u, True),
        "HGNN+UrgencyMAPPO": (hgnn_u_planner, actor_full, critic_full, True),
    }
    ablation = []
    alignment_rows = []
    urgency_rows = []
    per_seed = []
    for spec in VARIANTS:
        planner, actor, critic, use_urgency = bundles[spec["name"]]
        seed_rows = []
        align = []
        for seed in EVAL_SEEDS:
            print(f"eval {spec['name']} seed={seed}", flush=True)
            metrics = rollout_metrics(planner, actor, critic, use_urgency=use_urgency, seed=int(seed))
            align.extend(metrics.pop("alignment"))
            seed_rows.append(metrics)
            per_seed.append({"experiment": spec["name"], "seed": int(seed), **metrics})
        summary_row = {
            "experiment": spec["name"],
            "status": "RUN",
            "payload": _mean(seed_rows, "payload"),
            "residual_queue": _mean(seed_rows, "residual_queue"),
            "completion_time": _mean(seed_rows, "completion_time"),
            "path_length_m": _mean(seed_rows, "path_length_m"),
            "energy": "not_exposed",
            "trajectory_reward": _mean(seed_rows, "trajectory_reward"),
            "assignment_match": _mean(seed_rows, "assignment_match"),
            "assignment_conflict": _mean(seed_rows, "assignment_conflict"),
            "assignment_entropy": _mean(seed_rows, "assignment_entropy"),
            "urgency_gap": _mean(seed_rows, "urgency_gap"),
        }
        ablation.append(summary_row)
        agree = mean_alignment(align)
        alignment_rows.append({"experiment": spec["name"], **agree})
        probe = Phase5Worker(planner, actor, critic, use_urgency=use_urgency, seeds=(int(EVAL_SEEDS[0]),))
        info = probe.reset(int(EVAL_SEEDS[0]))
        _, rec = probe.step(info, deterministic=True)
        delta = urgency_sensitivity(actor, rec["obs"], rec["urgency"])
        if not use_urgency:
            low_flag = "NOT_APPLICABLE"
        elif not np.isfinite(delta) or delta < 1.0e-5:
            low_flag = "URGENCY_IGNORED_BY_LOW_LEVEL"
        else:
            low_flag = "USED"
        if spec["high_urgency"]:
            changed = any(high_level_changes_when_urgency_zeroed(planner, int(seed)) for seed in EVAL_SEEDS)
            high_flag = "USED" if changed else "URGENCY_IGNORED_BY_HIGH_LEVEL"
        else:
            changed = False
            high_flag = "NOT_IN_ENCODER"
        urgency_rows.append(
            {
                "experiment": spec["name"],
                "low_level_action_delta": delta,
                "low_level_flag": low_flag,
                "high_level_changes": changed,
                "high_level_flag": high_flag,
                "urgency_gap": summary_row["urgency_gap"],
            }
        )
    baseline = next(row for row in ablation if row["experiment"] == "Baseline DeepSets+MAPPO")
    full = next(row for row in ablation if row["experiment"] == "HGNN+UrgencyMAPPO")
    base_align = next(row for row in alignment_rows if row["experiment"] == "Baseline DeepSets+MAPPO")
    full_align = next(row for row in alignment_rows if row["experiment"] == "HGNN+UrgencyMAPPO")
    situation = judge(baseline["payload"], full["payload"], base_align, full_align)
    judgment = {
        "A": "情况 A：完整方案的 payload 和 Q_H/U_LL 排序一致性都高于基线，并且完整方案的 Spearman 不低于 0.3、Top-1 不低于 0.5。这一有界探针没有显示必须立刻做高低层联合优化。",
        "B": "情况 B：HGNN 或 urgency 带来了 payload 或排序改善，但 Q_H 与 15 步低层回报的排序仍然明显不一致。高低层目标不一致还在。本阶段停止，不进入联合训练。",
        "INCONCLUSIVE_BOUNDED_BUDGET": "有界探针没有同时满足情况 A 或情况 B 的门槛。不能据此宣布 HGNN 和 urgency 已经解决目标不一致，也不能据此宣布必须进入联合优化。",
    }[situation]
    summary = {
        "phase_status": situation,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "planner_ready": PLANNER_READY,
        "reward_modified": False,
        "deployment_path_modified": False,
        "joint_training": False,
        "intermediate_evaluator": False,
        "hac": False,
        "holdout_230000_used": False,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "hl_epochs": HL_EPOCHS,
        "mappo_updates": MAPPO_UPDATES,
        "hl_samples": len(samples),
        "hgnn": hgnn_log,
        "hgnn_urgency": hgnn_u_log,
        "mappo_original": mappo_base,
        "mappo_urgency_deepsets": mappo_u,
        "mappo_urgency_hgnn": mappo_full,
        "observation_dim_original": int(ORIGINAL_OBS_DIM),
        "observation_dim_urgency": int(URGENCY_OBS_DIM),
        "urgency_index": int(URGENCY_INDEX),
        "packed_check": int(SELF_DIM + TASK_DIM + COMM_DIM + 16 + 1),
        "ablation": ablation,
        "alignment": alignment_rows,
        "urgency_effect": urgency_rows,
        "per_seed": per_seed,
        "judgment_text": judgment,
        "decoder_class": "OrderFreeSetDecoder",
        "encoder_class": Phase5OrderFreeModel.__name__,
    }
    (OUTPUT / "phase5_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(
        OUTPUT / "results" / "ablation_results.csv",
        ablation,
        ["experiment", "status", "payload", "residual_queue", "completion_time", "path_length_m", "energy", "trajectory_reward", "assignment_match", "assignment_conflict", "assignment_entropy"],
    )
    _write_csv(
        OUTPUT / "results" / "high_low_alignment.csv",
        alignment_rows,
        ["experiment", "spearman", "kendall", "top1_agreement", "normalized_regret", "n"],
    )
    _write_csv(
        OUTPUT / "results" / "urgency_effect_analysis.csv",
        urgency_rows,
        ["experiment", "low_level_action_delta", "low_level_flag", "high_level_changes", "high_level_flag", "urgency_gap"],
    )
    _write_contracts(summary)
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(judgment, flush=True)
    return summary


def _write_contracts(summary: dict) -> None:
    gate = {
        "status": "BOUNDED_PROBE_COMPLETE",
        "eligible_development_data": len(summary["train_seeds"]),
        "human_specification": "USER_PHASE5_INSTRUCTION",
        "human_approved": False,
        "training_started": True,
        "hgnn_trained": True,
        "urgency_injected": True,
        "reward_modified": False,
        "holdout_used": False,
        "deployment_verified": False,
        "planner_ready": False,
        "phase_status": summary["phase_status"],
        "entered_phase4f": False,
        "cross_layer_alignment_started": True,
        "isolated_train_seeds": summary["train_seeds"],
        "isolated_eval_seeds": summary["eval_seeds"],
    }
    (OUTPUT / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "phase5_shared_urgency_definition.json").write_text(
        json.dumps(
            {
                "status": "ACTIVE_IN_PHASE5_PROBE",
                "formula": "u_k(t) = q_k(t) / (max_j q_j(t) + epsilon)",
                "input_variables": ["queue q_k(t)"],
                "epsilon": 1e-8,
                "range": "[0,1]",
                "future_leakage": False,
                "used_by_high_level": "HGNN buoy channel in the full variant only",
                "used_by_low_level": "appended assigned-target urgency in urgency MAPPO",
                "hold_semantics": "frozen for the 15-step TaskOption",
                "reward_term": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "phase5_lowlevel_interface.json").write_text(
        json.dumps(
            {
                "status": "URGENCY_APPENDED",
                "task_option_extended": True,
                "old_observation": "self(5)+task(8)+communication(4)+neighbor(16)+nearest(1)",
                "urgency_appended": True,
                "old_obs_dim": summary["observation_dim_original"],
                "new_obs_dim": summary["observation_dim_urgency"],
                "urgency_index": summary["urgency_index"],
                "reward_modified": False,
                "critic_urgency_added": True,
                "action_dim": 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "phase5_hgnn_architecture.json").write_text(
        json.dumps(
            {
                "status": "TRAINED_BOUNDED_PROBE",
                "encoder_replaced": True,
                "decoder_unchanged": "OrderFreeSetDecoder class and joint greedy decode_assignment",
                "official_decide_replaced": False,
                "nodes": ["UAV", "Buoy"],
                "edges": ["UAV-UAV", "UAV-Buoy"],
                "buoy_buoy": False,
                "buoy_feature_extension": "append shared u_k for the full variant",
                "forbidden_inputs": ["future_trajectory", "future_reward", "oracle", "U_traj", "low_level_rollout"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
