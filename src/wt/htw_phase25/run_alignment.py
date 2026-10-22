"""Phase-2.5 lambda / tracking-loss / waypoint-comm alignment. No Round, no HGNN."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wt.htw_phase2.experiment import (
    BaselineSpec,
    evaluate_controller,
    oracle_low_level_conditioner,
    state_only_conditioner,
)
from wt.htw_phase2.vector import HierarchicalTrackingVector
from wt.htwc_phase1.conditioned_actor import GoalConditionedActorCritic
from wt.htwc_phase1.validation import MINI_UPDATES, SEED, SMOKE_CONFIG, eval_slice, train_slice
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent, seed_everything
from wt.round30.runner import collect

from .alignment import alignment_rows, summarize_alignment
from .tracking_loss import TrackingLossAgent, collect_tracked

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "alignment"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase2_5_alignment_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase2_5_alignment_report.md"
LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)
BETAS = (0.01, 0.05, 0.1)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if key != "agent"}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _metrics_row(kind: str, weight: float, evaluated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "weight": float(weight),
        "mean_payload_bits": evaluated["mean_payload_bits"],
        "mean_return": evaluated["mean_return"],
        "mean_residual_queue_bits": evaluated["mean_residual_queue_bits"],
        "mean_waypoint_tracking_error": evaluated["mean_waypoint_tracking_error"],
        "action_waypoint_cosine": evaluated["action_waypoint"].get("mean_rho"),
        "environment_steps": evaluated.get("environment_steps"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty csv: {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_reward_penalty(tracking_lambda: float, updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    spec = BaselineSpec("HTW-Oracle-PPO", "htw_oracle", 16, GoalConditionedActorCritic(), oracle_low_level_conditioner)
    agent = PPOAgent(spec.model, SMOKE_CONFIG)
    vector = HierarchicalTrackingVector(train_slice(), agent.config.num_envs, 73000 + SEED, spec.conditioner, tracking_lambda)
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"lambda={tracking_lambda} produced NaN at update {index}")
    evaluated = evaluate_controller(spec, agent, eval_slice())
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["tracking_lambda"] = float(tracking_lambda)
    return evaluated


def train_state_only(updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    spec = BaselineSpec("State-only PPO", "state_only", 8, DeepSetsActorCritic(), state_only_conditioner)
    agent = PPOAgent(spec.model, SMOKE_CONFIG)
    from wt.htwc_phase1.validation import ConditionedVector

    vector = ConditionedVector(train_slice(), agent.config.num_envs, 73000 + SEED, spec.conditioner)
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"state-only produced NaN at update {index}")
    evaluated = evaluate_controller(spec, agent, eval_slice())
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["tracking_lambda"] = 0.0
    return evaluated


def train_tracking_loss(beta: float, updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    spec = BaselineSpec("HTW-Oracle-loss", "htw_oracle", 16, GoalConditionedActorCritic(), oracle_low_level_conditioner)
    agent = TrackingLossAgent(spec.model, SMOKE_CONFIG, track_beta=beta)
    vector = HierarchicalTrackingVector(train_slice(), agent.config.num_envs, 73000 + SEED, spec.conditioner, tracking_lambda=0.0)
    for index in range(int(updates)):
        storage, events, health = collect_tracked(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"beta={beta} produced NaN at update {index}")
    evaluated = evaluate_controller(spec, agent, eval_slice())
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["track_beta"] = float(beta)
    return evaluated


def write_report(payload: Mapping[str, Any]) -> None:
    state_pay = payload["state_only"]["mean_payload_bits"]
    lambda_rows = payload["lambda_sensitivity"]
    penalty_rows = [row for row in lambda_rows if row["kind"] == "reward_penalty"]
    best_lambda = max(penalty_rows, key=lambda row: row["mean_payload_bits"])
    any_better = any(row["mean_payload_bits"] > state_pay + 1.0 for row in penalty_rows)
    loss_rows = payload["tracking_loss"]
    best_loss = max(loss_rows, key=lambda row: row["mean_payload_bits"]) if loss_rows else None
    loss_better = any(row["mean_payload_bits"] > state_pay + 1.0 for row in loss_rows)
    align = payload["alignment_summary"]
    waypoint_helps = align["frac_waypoint_higher_rate_to_target"] >= 0.55 and align["mean_rate_delta_target_bps"] > 0.0
    if any_better:
        verdict = "A"
        verdict_text = "调整lambda即可继续HTW：至少一个 λ 使 Oracle HTW payload 超过 State-only。"
    elif not waypoint_helps:
        verdict = "B"
        verdict_text = "需要重新生成 communication-aware waypoint：当前 W 对目标 buoy 的速率/增益提升不足或为负。"
    else:
        verdict = "C"
        verdict_text = "需要重新设计 hierarchical controller：waypoint 有通信指向，但 reward penalty 与 loss 均未把 payload 抬过 State-only。"
    if (not any_better) and waypoint_helps and (best_lambda["mean_payload_bits"] < state_pay) and loss_better is False:
        verdict = "C"
        verdict_text = "需要重新设计 hierarchical controller：W 偏向靠近目标、提高对目标速率，但 λ 扫描与 tracking loss 都没有通信收益。"
    lines = [
        "# HTW Phase-2.5 Waypoint-Communication Alignment Report",
        "",
        "诊断 tracking 与通信的冲突。未改 HGNN / Transformer / TW 定义 / 环境 reward / 未开 Round。",
        f"预算：与 Phase-2 相同（{payload['updates']} updates，seed {SEED}）。State-only payload={state_pay:.3f}。",
        "",
        "## 实验1：λ 敏感性（reward penalty）",
        "",
        "| λ | payload | env return | residual queue | tracking error | action-W cosine |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in lambda_rows:
        if row["kind"] != "reward_penalty":
            continue
        lines.append(
            f"| {row['weight']} | {row['mean_payload_bits']:.3f} | {row['mean_return']:.6f} | "
            f"{row['mean_residual_queue_bits']:.3f} | {row['mean_waypoint_tracking_error']:.6f} | {row['action_waypoint_cosine']} |"
        )
    lines.extend(
        [
            "",
            f"State-only：payload={state_pay:.3f}, return={payload['state_only']['mean_return']:.6f}, "
            f"queue={payload['state_only']['mean_residual_queue_bits']:.3f}, "
            f"track={payload['state_only']['mean_waypoint_tracking_error']:.6f}, "
            f"ρ={payload['state_only']['action_waypoint'].get('mean_rho')}",
            "",
            f"最佳 λ={best_lambda['weight']}，payload={best_lambda['mean_payload_bits']:.3f}。",
            "",
            "## 实验2：soft tracking loss（L_PPO + β mean||p-W||，reward 仍为 r_comm）",
            "",
            "| β | payload | env return | residual queue | tracking error | action-W cosine |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in loss_rows:
        lines.append(
            f"| {row['weight']} | {row['mean_payload_bits']:.3f} | {row['mean_return']:.6f} | "
            f"{row['mean_residual_queue_bits']:.3f} | {row['mean_waypoint_tracking_error']:.6f} | {row['action_waypoint_cosine']} |"
        )
    if best_loss is not None:
        lines.append("")
        lines.append(f"最佳 β={best_loss['weight']}，payload={best_loss['mean_payload_bits']:.3f}。")
    lines.extend(
        [
            "",
            "## 实验3：waypoint 通信对齐",
            "",
            json.dumps(align, ensure_ascii=False, indent=2),
            "",
            f"- 更靠近 queue-rate 目标 buoy 的比例：{align['frac_waypoint_closer_to_target']:.3f}",
            f"- 对目标 buoy 速率更高的比例：{align['frac_waypoint_higher_rate_to_target']:.3f}",
            f"- 对目标 buoy 增益更高的比例：{align['frac_waypoint_higher_gain_to_target']:.3f}",
            f"- 目标即最大队列的比例：{align['frac_target_is_max_queue']:.3f}",
            f"- 平均速率差（W−当前）：{align['mean_rate_delta_target_bps']:.3f} bps",
            "",
            "## 三个问题",
            "",
            f"1. 是否存在合适 λ 使 HTW payload > State-only？ **{'是' if any_better else '否'}**",
            "2. tracking 约束应该作为 reward 还是 loss？ "
            + (
                "本预算下 **loss 更好**。"
                if best_loss is not None and best_loss["mean_payload_bits"] > best_lambda["mean_payload_bits"] + 1.0
                else "本预算下 **reward penalty 不劣于 loss**；两者都未稳定超过 State-only 时，形式本身不是主因。"
            ),
            f"3. 当前 waypoint 是否具有通信价值？ **{'有指向目标、提高对目标速率/增益的统计倾向' if waypoint_helps else '弱/不足：对目标通信量提升不明显'}**",
            "",
            "## 最终判断",
            "",
            f"**判定：{verdict}**",
            "",
            verdict_text,
            "",
            f"`lambda_sensitivity.csv`：`{(OUTPUT / 'lambda_sensitivity.csv').as_posix()}`",
            f"`waypoint_communication_alignment.csv`：`{(OUTPUT / 'waypoint_communication_alignment.csv').as_posix()}`",
            "",
        ]
    )
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    updates = MINI_UPDATES
    state = train_state_only(updates)
    lambda_rows = [_metrics_row("state_only", 0.0, state)]
    for value in LAMBDAS:
        evaluated = train_reward_penalty(value, updates)
        lambda_rows.append(_metrics_row("reward_penalty", value, evaluated))
    loss_rows = []
    for beta in BETAS:
        evaluated = train_tracking_loss(beta, updates)
        loss_rows.append(_metrics_row("tracking_loss", beta, evaluated))
    rows = alignment_rows(train_slice())
    summary = summarize_alignment(rows)
    _write_csv(OUTPUT / "lambda_sensitivity.csv", lambda_rows)
    _write_csv(OUTPUT / "tracking_loss.csv", loss_rows)
    _write_csv(OUTPUT / "waypoint_communication_alignment.csv", rows)
    _write_csv(ROOT / "results/root_exports/lambda_sensitivity.csv", lambda_rows)
    _write_csv(ROOT / "results/root_exports/waypoint_communication_alignment.csv", rows)
    payload = {
        "updates": updates,
        "seed": SEED,
        "fresh_access": 0,
        "untouched_access": 0,
        "state_only": state,
        "lambda_sensitivity": lambda_rows,
        "tracking_loss": loss_rows,
        "alignment_summary": summary,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps({"state_payload": result["state_only"]["mean_payload_bits"], "lambda": result["lambda_sensitivity"], "loss": result["tracking_loss"], "alignment": result["alignment_summary"]}, indent=2))
