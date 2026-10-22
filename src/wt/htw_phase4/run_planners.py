"""Phase-4: swap high-level TW source, keep the Phase-3 low-level PPO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wt.htw_phase2.experiment import BaselineSpec, state_only_conditioner
from wt.htw_phase2.tracking import action_waypoint_cosine, normalize_positions_m, tracking_distance, waypoint_from_uav_features
from wt.htwc_phase1.adapter import CONDITIONED_UAV_DIM
from wt.htwc_phase1.conditioned_actor import GoalConditionedActorCritic
from wt.htwc_phase1.oracle import tw_enters_actor
from wt.htwc_phase1.relevance import summarize_cosines
from wt.htwc_phase1.validation import MINI_UPDATES, SEED, SMOKE_CONFIG, ConditionedVector, eval_slice, train_slice
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.features import UAV_DIM, stack_feature_dicts
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent, seed_everything
from wt.round30.runner import collect
from wt.round30.scenarios import PPOScenario

from .planners import PLANNER_A, PLANNER_B, PLANNER_C, planner_c_window, planner_window, planner_a_action, planner_b_action

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "planner_compare"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase4_high_level_planner_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase4_high_level_planner_report.md"
AUDIT = ROOT / "docs" / "archive" / "phase_reports" / "planner_audit_report.md"
PACK_AUDIT = Path(__file__).resolve().parent / "planner_audit_report.md"


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


def evaluate_tw_policy(spec: BaselineSpec, agent: PPOAgent, scenarios: tuple[PPOScenario, ...]) -> dict[str, Any]:
    payloads, returns, queues, distances, rhos = [], [], [], [], []
    flags = []
    for scenario in scenarios:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        total_return, done = 0.0, False
        while not done:
            policy_features = spec.conditioner(info, env)
            flags.append(tw_enters_actor(policy_features) if policy_features["uav"].shape[-1] == 16 else {"uav_width_is_conditioned": False, "goal_nonzero": False, "waypoint_nonzero": False, "horizon_nonzero": False})
            requested = agent.act(stack_feature_dicts([policy_features]), deterministic=True)["action"][0]
            if policy_features["uav"].shape[-1] == 16:
                waypoint = waypoint_from_uav_features(policy_features["uav"])
                now_xy = normalize_positions_m(info["uav_positions_m"], env.config)
                for uav in range(env.num_uavs):
                    rhos.append(action_waypoint_cosine(requested[uav], now_xy[uav], waypoint[uav]))
            try:
                _, reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error):
                    raise
                _, reward, terminated, truncated, after = env.step(np.zeros_like(requested), matching_mode="legacy")
            if policy_features["uav"].shape[-1] == 16:
                after_xy = normalize_positions_m(after["uav_positions_m"], env.config)
                distances.append(tracking_distance(after_xy, waypoint_from_uav_features(policy_features["uav"]), policy_features["uav_mask"]))
            total_return += float(reward)
            info, done = after, bool(terminated or truncated)
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        queues.append(float(np.sum(info["raw_queue_bits"])))
    return {
        "family": spec.family,
        "episodes": len(payloads),
        "mean_payload_bits": float(np.mean(payloads)),
        "mean_return": float(np.mean(returns)),
        "mean_residual_queue_bits": float(np.mean(queues)),
        "mean_waypoint_tracking_error": float(np.mean(distances)) if distances else float("nan"),
        "action_waypoint": summarize_cosines(rhos),
        "tw_enters_actor": {
            "any_conditioned_width": any(item["uav_width_is_conditioned"] for item in flags) if flags else False,
            "any_waypoint_nonzero": any(item["waypoint_nonzero"] for item in flags) if flags else False,
        },
    }


def train_policy(spec: BaselineSpec, updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    agent = PPOAgent(spec.model, SMOKE_CONFIG)
    vector = ConditionedVector(train_slice(), agent.config.num_envs, 73000 + SEED, spec.conditioner)
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"{spec.name} produced NaN at update {index}")
    evaluated = evaluate_tw_policy(spec, agent, eval_slice())
    evaluated["environment_steps"] = vector.environment_steps
    return evaluated


def tw_diversity(scenarios: tuple[PPOScenario, ...]) -> dict[str, float]:
    diffs_ab, diffs_ac, diffs_bc = [], [], []
    for scenario in scenarios[:4]:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        wa, _ = planner_window(env, info, planner_a_action)
        wb, _ = planner_window(env, info, planner_b_action)
        wc, _ = planner_c_window(env, info, horizon=5)
        mask = np.asarray(wa.uav_mask, dtype=bool)
        diffs_ab.append(float(np.mean(np.linalg.norm(wa.waypoint_xy[mask] - wb.waypoint_xy[mask], axis=-1))))
        diffs_ac.append(float(np.mean(np.linalg.norm(wa.waypoint_xy[mask] - wc.waypoint_xy[mask], axis=-1))))
        diffs_bc.append(float(np.mean(np.linalg.norm(wb.waypoint_xy[mask] - wc.waypoint_xy[mask], axis=-1))))
    return {
        "mean_waypoint_l2_A_vs_B": float(np.mean(diffs_ab)),
        "mean_waypoint_l2_A_vs_C": float(np.mean(diffs_ac)),
        "mean_waypoint_l2_B_vs_C": float(np.mean(diffs_bc)),
    }


def write_audit() -> None:
    text = """# Planner Audit Report

范围：Round-19 `src/wt/round19/planner.py`，以及 Phase-1/3 用来生成 TW 的 `independent_queue_rate`。未改环境、reward、PPO、HGNN。

## 1. 当前 planner 优化目标

`independent_queue_rate_action` **不优化**环境 reward，也 **不优化**折扣累计通信收益。

对每个 UAV 独立取

`k* = argmax_k (queue_k / capacity) * rate_{uav,k}`

其中 `rate` 来自 `rate_matrix`（单 UAV 静态大尺度信道，**不含**共享频谱干扰与联合 matching）。动作是指向 `k*` 的单位向量。

`joint_plan` 才对 **预测 payload 之和** 做有限 beam search，并在 shadow 里用 `joint_aware` matching。Phase-1/3 默认 TW **没有**用它。

## 2. 输入状态

仅当前可观测量：

- UAV 位置
- buoy 位置
- 当前队列 `raw_queue_bits`
- 由位置算出的当前速率/距离

不输入：其他 UAV 的未来意图、未来到达、累计 return、干扰后的 SINR（独立策略路径上）。

## 3. 输出 TW 的方式

滚动该启发式 H=5 步（shadow，不改 live env）：

- `goal` = 窗口内各 UAV 对 buoy 的累计 payload 份额
- `waypoint` = 第 H 步位置（归一化）
- `horizon` = 5

这是 **myopic 方向的积分终点**，不是最优控制子目标。

## 4. 为什么不能提供长期价值

1. 目标是瞬时 queue×正交速率，不是 `sum_t r_comm`。
2. 忽略干扰：两机可能冲向同一高队列 buoy，共享频谱下互相压速率。
3. H 步滚动仍每步 myopic，终点不必是累计 payload 最优位置。
4. 这些量（队列、位置、近似速率）**已经在低层 state 里**。TW_A 近似是 state 的确定性函数，所以 Phase-3 条件化没有新信息。

Phase-4 用同一 TW 槽位接入：

- TW_A：当前 queue-rate
- TW_B：`joint_plan(H=1)`，最大化一步干扰感知 payload
- TW_C：`joint_plan(H=5)` 逐步重规划，最大化窗口累计预测 payload
"""
    AUDIT.write_text(text, encoding="utf-8")
    PACK_AUDIT.write_text(text, encoding="utf-8")


def write_report(payload: Mapping[str, Any]) -> None:
    results = payload["results"]
    state_pay = results["state_only"]["mean_payload_bits"]
    best_name, best_row = max(
        ((name, row) for name, row in results.items() if name != "state_only"),
        key=lambda item: item[1]["mean_payload_bits"],
    )
    any_better = any(row["mean_payload_bits"] > state_pay + 1.0 for name, row in results.items() if name != "state_only")
    continue_htw = any_better
    lines = [
        "# HTW Phase-4 High-level Planner Report",
        "",
        "同一低层 Goal-conditioned PPO（原始 `r_comm`，无 tracking）。只替换 Oracle TW 来源。未改 HGNN / PPO 核心 / 环境 / reward；未训练 Transformer；未开 Round。",
        "",
        "## Planner 设计",
        "",
        "- TW 槽位不变：`(goal, waypoint, horizon)`。",
        "- A：independent queue-rate（当前）。",
        "- B：`joint_plan` H=1，一步最大化干扰下 payload（queue pressure × 共享速率）。",
        "- C：每步 `joint_plan` H=5，窗口累计预测 payload。",
        "",
        f"Waypoint 差异（归一化 L2）：{json.dumps(payload['diversity'], ensure_ascii=False)}",
        "",
        "## 结果",
        "",
        "| 方法 | payload | return | residual queue | waypoint dist | action–W cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = ("state_only", "tw_a", "tw_b", "tw_c")
    labels = {"state_only": "State-only PPO", "tw_a": "TW_A conditioned", "tw_b": "TW_B conditioned", "tw_c": "TW_C conditioned"}
    for key in order:
        row = results[key]
        rho = row["action_waypoint"].get("mean_rho") if isinstance(row.get("action_waypoint"), dict) else None
        lines.append(
            f"| {labels[key]} | {row['mean_payload_bits']:.3f} | {row['mean_return']:.6f} | "
            f"{row['mean_residual_queue_bits']:.3f} | {row['mean_waypoint_tracking_error']:.6f} | {rho} |"
        )
    lines.extend(
        [
            "",
            "## 四个问题",
            "",
            "1. 当前 planner 为什么无收益？TW_A 是当前 queue/rate 的确定性滚动，信息已在 state 中，且不优化累计通信。详见 `planner_audit_report.md`。",
            f"2. 什么类型 TW 可以提供额外信息？相对 A 的 waypoint L2 更大、且 payload 更高的来源才算有新信息。本次最佳非 state 来源：**{best_name}**（payload={best_row['mean_payload_bits']:.3f}）。",
            f"3. 是否值得继续 HTW-C？ **{'值得' if continue_htw else '仍不充分'}**：{'已有 TW 源超过 State-only，层次结构有通信价值。' if continue_htw else 'A/B/C 均未超过 State-only，仅换启发式 planner 还不够，或需要更长训练 / 可执行子目标，而不是先加 HGNN。'}",
            "4. HGNN 未来应该放在哪个位置？若继续 HTW-C：**只放在 high-level planner**（对多 UAV/多 buoy 关系建模以生成更好的 `(G,W,H)`），不要放进低层 PPO，也不要替代环境 matching。当前 **尚未证明** 该位置的 HGNN 有必要。",
            "",
            f"**相对 State-only 是否有 TW 增益：{'是' if any_better else '否'}**",
            "",
            f"JSON：`{OUTPUT.as_posix()}/summary.json`",
            "",
        ]
    )
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_audit()
    updates = MINI_UPDATES
    train = train_slice()
    print("diversity", flush=True)
    diversity = tw_diversity(train)
    print(diversity, flush=True)
    specs = {
        "state_only": BaselineSpec("State-only", "state_only", UAV_DIM, DeepSetsActorCritic(), state_only_conditioner),
        "tw_a": BaselineSpec("TW_A", "tw_a", CONDITIONED_UAV_DIM, GoalConditionedActorCritic(), PLANNER_A),
        "tw_b": BaselineSpec("TW_B", "tw_b", CONDITIONED_UAV_DIM, GoalConditionedActorCritic(), PLANNER_B),
        "tw_c": BaselineSpec("TW_C", "tw_c", CONDITIONED_UAV_DIM, GoalConditionedActorCritic(), PLANNER_C),
    }
    results = {}
    for key, spec in specs.items():
        print(f"training {key}", flush=True)
        results[key] = train_policy(spec, updates)
        print(f"done {key} payload={results[key]['mean_payload_bits']}", flush=True)
    payload = {
        "updates": updates,
        "seed": SEED,
        "fresh_access": 0,
        "untouched_access": 0,
        "hgnn": False,
        "diversity": diversity,
        "results": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps({"diversity": result["diversity"], "payloads": {key: row["mean_payload_bits"] for key, row in result["results"].items()}}, indent=2))
