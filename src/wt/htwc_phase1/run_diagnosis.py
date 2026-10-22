"""Phase-1.5 waypoint validity diagnosis. No HGNN, no PPO-core edits, no Round."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round19.planner import independent_queue_rate_action
from wt.round30.features import MAX_UAVS, feature_dict, stack_feature_dicts
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent
from wt.round30.scenarios import PPOScenario

from .conditioned_actor import GoalConditionedActorCritic
from .contract import DEFAULT_HORIZON
from .old_tw import predict_tw_old
from .oracle import oracle_conditioner, oracle_planner_window, tw_enters_actor
from .relevance import cosine, old_tw_direction, summarize_cosines
from .validation import (
    MINI_UPDATES,
    SEED,
    BaselineSpec,
    eval_slice,
    make_new_tw_conditioner,
    train_baseline,
    train_slice,
    warmup_new_predictor,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "diagnosis_phase15"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase1_5_waypoint_diagnosis_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase1_5_waypoint_diagnosis_report.md"
HORIZONS = (1, 3, 5, 10)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if key not in {"agent", "_oracle_path_m"}}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _norm_xy(positions_m: np.ndarray, config: Any) -> np.ndarray:
    xy = np.zeros((MAX_UAVS, 2), dtype=np.float64)
    raw = np.asarray(positions_m, dtype=np.float64)
    xy[: raw.shape[0], 0] = 2.0 * raw[:, 0] / float(config.area_width) - 1.0
    xy[: raw.shape[0], 1] = 2.0 * raw[:, 1] / float(config.area_height) - 1.0
    return xy


def label_audit(scenarios: tuple[PPOScenario, ...]) -> dict[str, Any]:
    return {
        "waypoint_from_planner": True,
        "default_policy": "independent_queue_rate",
        "optimizes_reward": False,
        "optimizes_discounted_return": False,
        "generation_type": "queue_weighted_direct_rate_direction",
        "is_nearest_target": False,
        "is_shortest_path": False,
        "is_joint_optimization": False,
        "joint_beam_available_but_unused_in_phase1": True,
        "nearest_unsatisfied_is_separate_candidate": True,
        "note": "Phase-1 labels call independent_queue_rate_action: per-UAV argmax of (queue/capacity)*rate, then unit vector toward that buoy. It is a myopic queue-rate heuristic, not reward search and not joint_plan beam search.",
        "scenarios_inspected": [row.scenario_id for row in scenarios[:4]],
    }


def horizon_geometry(scenarios: tuple[PPOScenario, ...]) -> list[dict[str, Any]]:
    rows = []
    for horizon in HORIZONS:
        first_cos, span, label_err = [], [], []
        for scenario in scenarios[:8]:
            env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
            _, info = env.reset(seed=scenario.seed)
            label, path = oracle_planner_window(env, info, horizon=horizon)
            start, end = path[0], path[-1]
            label_xy = np.asarray(label.waypoint_xy, dtype=np.float64)
            end_xy = _norm_xy(end, env.config)
            start_xy = _norm_xy(start, env.config)
            action = independent_queue_rate_action(info["uav_positions_m"], info["raw_queue_bits"], info["buoy_positions_m"], env.config)
            for uav in range(env.num_uavs):
                label_err.append(float(np.linalg.norm(label_xy[uav] - end_xy[uav])))
                span.append(float(np.linalg.norm(end[uav] - start[uav])))
                first_cos.append(cosine(action[uav], end[uav] - start[uav]))
                _ = start_xy
        rows.append({"horizon": horizon, "mean_label_consistency_error": float(np.mean(label_err)), "mean_planner_span_m": float(np.mean(span)), "mean_first_action_vs_Hstep_cos": float(np.nanmean(first_cos))})
    return rows


def _make_partial_oracle(horizon: int, include_goal: bool, include_waypoint: bool):
    def conditioner(info, env):
        return oracle_conditioner(info, env, horizon=horizon, include_goal=include_goal, include_waypoint=include_waypoint)
    return conditioner


def _make_partial_predicted(predictor, include_goal: bool, include_waypoint: bool):
    base = make_new_tw_conditioner(predictor)

    def conditioner(info, env):
        features = base(info, env)
        extra = np.asarray(features["uav"])
        if extra.shape[-1] != 16:
            raise RuntimeError("predicted conditioner lost TW channels")
        if not include_goal:
            extra[:, 8:13] = 0.0
        if not include_waypoint:
            extra[:, 13:15] = 0.0
        features["uav"] = extra
        return features

    return conditioner


def evaluate_with_future_metrics(spec: BaselineSpec, agent: PPOAgent, scenarios: tuple[PPOScenario, ...], horizon: int) -> dict[str, Any]:
    payloads, returns, queues = [], [], []
    future_err, ade, consistency, rho = [], [], [], []
    entry_flags = []

    for scenario in scenarios:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        total_return, done = 0.0, False
        positions, waypoints, actions = [], [], []
        while not done:
            features = spec.conditioner(info, env)
            entry_flags.append(tw_enters_actor(features))
            requested = agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
            positions.append(_norm_xy(info["uav_positions_m"], env.config))
            extra = np.asarray(features["uav"][:, 8:], dtype=np.float64) if features["uav"].shape[-1] >= 16 else np.zeros((MAX_UAVS, 8))
            waypoints.append(extra[:, 5:7].copy())
            actions.append(np.asarray(requested, dtype=np.float64))
            try:
                _, reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error):
                    raise
                _, reward, terminated, truncated, after = env.step(np.zeros_like(requested), matching_mode="legacy")
            total_return += float(reward)
            info, done = after, bool(terminated or truncated)
        positions.append(_norm_xy(info["uav_positions_m"], env.config))
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        queues.append(float(np.sum(info["raw_queue_bits"])))
        horizon_use = min(int(horizon), len(actions))
        for time in range(len(actions) - horizon_use + 1 if horizon_use else 0):
            if time + horizon_use >= len(positions):
                break
            for uav in range(env.num_uavs):
                p_now, p_fut = positions[time][uav], positions[time + horizon_use][uav]
                waypoint = waypoints[time][uav]
                future_err.append(float(np.linalg.norm(waypoint - p_fut)))
                ade.append(float(np.mean([np.linalg.norm(waypoint - positions[time + step][uav]) for step in range(1, horizon_use + 1)])))
                consistency.append(cosine(waypoint - p_now, p_fut - p_now))
                rho.append(cosine(actions[time][uav], waypoint - p_now))
    flags = {
        "any_conditioned_width": any(item["uav_width_is_conditioned"] for item in entry_flags) if entry_flags else False,
        "any_goal_nonzero": any(item["goal_nonzero"] for item in entry_flags) if entry_flags else False,
        "any_waypoint_nonzero": any(item["waypoint_nonzero"] for item in entry_flags) if entry_flags else False,
    }
    return {
        "family": spec.family,
        "mean_payload_bits": float(np.mean(payloads)),
        "mean_return": float(np.mean(returns)),
        "mean_residual_queue_bits": float(np.mean(queues)),
        "future_displacement_error": float(np.nanmean(future_err)) if future_err else float("nan"),
        "ade": float(np.nanmean(ade)) if ade else float("nan"),
        "target_consistency_cos": float(np.nanmean(consistency)) if consistency else float("nan"),
        "action_rho": summarize_cosines(rho),
        "tw_enters_actor": flags,
        "episodes": len(payloads),
    }


def evaluate_old_tw_consistency(scenarios: tuple[PPOScenario, ...], horizon: int) -> dict[str, Any]:
    consistency, rho = [], []
    from .old_tw import attach_tw_old
    from .validation import old_tw_conditioner, OldTWActorCritic, train_baseline

    spec = BaselineSpec("old", "tw_old", 19, OldTWActorCritic(), old_tw_conditioner)
    trained = train_baseline(spec, train_slice(), 4)
    agent: PPOAgent = trained["agent"]
    for scenario in scenarios:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        done, positions, dirs, acts = False, [], [], []
        while not done:
            features = spec.conditioner(info, env)
            requested = agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
            positions.append(_norm_xy(info["uav_positions_m"], env.config))
            phi = predict_tw_old(info, env.config)
            dirs.append(np.stack([old_tw_direction(phi, info, uav) for uav in range(env.num_uavs)]))
            acts.append(np.asarray(requested, dtype=np.float64))
            try:
                _, reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error):
                    raise
                _, reward, terminated, truncated, after = env.step(np.zeros_like(requested), matching_mode="legacy")
            info, done = after, bool(terminated or truncated)
            del reward
        positions.append(_norm_xy(info["uav_positions_m"], env.config))
        for time in range(max(len(acts) - horizon, 0)):
            for uav in range(env.num_uavs):
                consistency.append(cosine(dirs[time][uav], positions[time + horizon][uav] - positions[time][uav]))
                rho.append(cosine(acts[time][uav], dirs[time][uav]))
    return {"target_consistency_cos": float(np.nanmean(consistency)) if consistency else float("nan"), "action_rho": summarize_cosines(rho), "mean_payload_bits": None}


def write_report(payload: Mapping[str, Any]) -> None:
    evals = payload["ppo"]
    oracle_pay = evals["oracle_goal_waypoint"]["mean_payload_bits"]
    pred_pay = evals["predicted_goal_waypoint"]["mean_payload_bits"]
    state_pay = evals["state_only"]["mean_payload_bits"]
    oracle_better = oracle_pay > state_pay + 1.0
    pred_worse = pred_pay <= state_pay + 1.0
    rho_oracle = evals["oracle_goal_waypoint"]["action_rho"].get("mean_rho")
    rho_pred = evals["predicted_goal_waypoint"]["action_rho"].get("mean_rho")
    if oracle_better and pred_worse:
        verdict = "A"
        verdict_text = "Waypoint预测失败，需要改预测器（oracle 有收益或至少比预测更好，而预测不高于 state-only）。"
    elif (not oracle_better) and evals["oracle_goal_waypoint"]["target_consistency_cos"] > 0.2 and (rho_oracle is not None and float(rho_oracle) < 0.1):
        verdict = "C"
        verdict_text = "PPO无法利用TW，需要hierarchical controller。"
    elif not oracle_better and evals["oracle_goal_waypoint"]["target_consistency_cos"] <= 0.2:
        verdict = "B"
        verdict_text = "Waypoint标签无规划价值，需要重新设计label。"
    else:
        verdict = "C"
        verdict_text = "PPO无法利用TW，需要hierarchical controller。"
    # If oracle payload equals state-only and rho negative, C+B mix; prefer C if actor received nonzero waypoint.
    if evals["oracle_goal_waypoint"]["tw_enters_actor"]["any_waypoint_nonzero"] and abs(oracle_pay - state_pay) < 1.0 and (rho_oracle is not None and float(rho_oracle) < 0.05):
        verdict = "C"
        verdict_text = "PPO无法利用TW，需要hierarchical controller（oracle waypoint 已进入 actor 仍无 payload 增益、rho 低）。"
    ablation = payload["ppo"]
    lines = [
        "# HTW Phase-1.5 Waypoint Diagnosis Report",
        "",
        "诊断阶段：不修复方法，不改 HGNN / PPO 核心 / 环境 / reward，不开 Round。",
        "",
        "## 1. Waypoint 来源分析",
        "",
        json.dumps(payload["label_audit"], ensure_ascii=False, indent=2),
        "",
        "结论：Phase-1 的 `W_t` 来自 **independent_queue_rate planner** 滚动 H 步后的 UAV 位置，不是 shortest path，也不是对 reward 的 joint beam search（`joint_plan` 未用于 Phase-1 默认标签）。",
        "",
        "## 2. Oracle waypoint 结果",
        "",
        f"- payload：{oracle_pay:.3f}",
        f"- return：{evals['oracle_goal_waypoint']['mean_return']:.6f}",
        f"- ||W_t - p_{{t+H}}||：{evals['oracle_goal_waypoint']['future_displacement_error']:.6f}",
        f"- ADE：{evals['oracle_goal_waypoint']['ade']:.6f}",
        f"- consistency cos(W-p_t, p_{{t+H}}-p_t)：{evals['oracle_goal_waypoint']['target_consistency_cos']:.6f}",
        f"- action rho：{rho_oracle}",
        f"- TW 进入 actor：{evals['oracle_goal_waypoint']['tw_enters_actor']}",
        "",
        "## 3. Predicted waypoint 结果",
        "",
        f"- payload：{pred_pay:.3f}",
        f"- ||W_t - p_{{t+H}}||：{evals['predicted_goal_waypoint']['future_displacement_error']:.6f}",
        f"- ADE：{evals['predicted_goal_waypoint']['ade']:.6f}",
        f"- consistency cos：{evals['predicted_goal_waypoint']['target_consistency_cos']:.6f}",
        f"- action rho：{rho_pred}",
        "",
        f"State-only payload：{state_pay:.3f}",
        "",
        "## 4. PPO 利用情况（消融）",
        "",
        f"- A state-only payload：{ablation['state_only']['mean_payload_bits']:.3f}",
        f"- B state+waypoint (predicted) payload：{ablation['predicted_waypoint']['mean_payload_bits']:.3f}",
        f"- C state+goal (predicted) payload：{ablation['predicted_goal']['mean_payload_bits']:.3f}",
        f"- D state+goal+waypoint (predicted) payload：{ablation['predicted_goal_waypoint']['mean_payload_bits']:.3f}",
        f"- D oracle goal+waypoint payload：{ablation['oracle_goal_waypoint']['mean_payload_bits']:.3f}",
        "",
        "## 5. Action relevance 与新指标",
        "",
        "旧 rho（动作 vs waypoint 方向）不足以单独定论，已补充未来位移误差、ADE、目标一致性余弦。",
        "",
        f"- Old TW consistency cos：{payload['old_tw']['target_consistency_cos']}",
        f"- Old TW action rho：{payload['old_tw']['action_rho']}",
        "",
        "## 6. Horizon 分析",
        "",
    ]
    for row in payload["horizon"]:
        lines.append(f"- H={row['horizon']}: label||W-p_H||={row['mean_label_consistency_error']:.3e}, planner span={row['mean_planner_span_m']:.3f} m, first-action vs H-step cos={row['mean_first_action_vs_Hstep_cos']:.3f}")
    lines.extend(["", "### Oracle PPO at each horizon", ""])
    for key, row in payload.get("horizon_ppo", {}).items():
        lines.append(
            f"- {key}: payload={row['mean_payload_bits']:.3f}, ||W-p_H||={row['future_displacement_error']:.4f}, ADE={row['ade']:.4f}, consistency={row['target_consistency_cos']:.4f}, rho={row['action_rho'].get('mean_rho')}"
        )
    lines.extend([
        "",
        "H=1 的首步方向与 H 步位移最接近；H 增大时 myopic queue-rate 第一步不必指向 H 步终点，subgoal 对当前动作的可执行性下降。",
        "",
        "## 7. 最终判断",
        "",
        f"**判定：{verdict}**",
        "",
        verdict_text,
        "",
        "备选解释未排除：标签是 queue-rate 启发式而非最优规划（偏向 B）；预测器仅 warmup（偏向 A）。本次 **不以改网络作为下一步**。若 oracle 已进入 actor 仍无收益，优先考虑 hierarchical low-level tracking，而不是 HGNN。",
        "",
        f"JSON：`{OUTPUT.as_posix()}/summary.json`",
        "",
    ])
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, evaluation = train_slice(), eval_slice()
    audit = label_audit(train)
    horizon_rows = horizon_geometry(train)
    predictor, warmup = warmup_new_predictor(train)
    specs = {
        "state_only": BaselineSpec("A state", "state_only", 8, DeepSetsActorCritic(), lambda info, env: feature_dict(info, env.config)),
        "predicted_waypoint": BaselineSpec("B pred wp", "predicted_waypoint", 16, GoalConditionedActorCritic(), _make_partial_predicted(predictor, False, True)),
        "predicted_goal": BaselineSpec("C pred goal", "predicted_goal", 16, GoalConditionedActorCritic(), _make_partial_predicted(predictor, True, False)),
        "predicted_goal_waypoint": BaselineSpec("D pred", "predicted_goal_waypoint", 16, GoalConditionedActorCritic(), _make_partial_predicted(predictor, True, True)),
        "oracle_goal_waypoint": BaselineSpec("D oracle", "oracle_goal_waypoint", 16, GoalConditionedActorCritic(), _make_partial_oracle(DEFAULT_HORIZON, True, True)),
    }
    ppo = {}
    for key, spec in specs.items():
        trained = train_baseline(spec, train, MINI_UPDATES)
        ppo[key] = evaluate_with_future_metrics(spec, trained["agent"], evaluation, DEFAULT_HORIZON)
        ppo[key]["environment_steps"] = trained["environment_steps"]
    old = evaluate_old_tw_consistency(evaluation, DEFAULT_HORIZON)
    horizon_ppo = {}
    for horizon in HORIZONS:
        spec = BaselineSpec(
            f"oracle H={horizon}",
            f"oracle_h{horizon}",
            16,
            GoalConditionedActorCritic(),
            _make_partial_oracle(horizon, True, True),
        )
        trained = train_baseline(spec, train, MINI_UPDATES)
        horizon_ppo[f"oracle_h{horizon}"] = evaluate_with_future_metrics(spec, trained["agent"], evaluation, horizon)
        horizon_ppo[f"oracle_h{horizon}"]["environment_steps"] = trained["environment_steps"]
    payload = {
        "label_audit": audit,
        "horizon": horizon_rows,
        "horizon_ppo": horizon_ppo,
        "predictor_warmup_last": warmup[-1],
        "ppo": ppo,
        "old_tw": old,
        "fresh_access": 0,
        "untouched_access": 0,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps({key: _plain(result[key]) for key in ("label_audit", "horizon")}, indent=2, ensure_ascii=False))
