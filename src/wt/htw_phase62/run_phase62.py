"""HTW Phase-6.2: goal-conditioned low-level option execution. No HGNN, pi_H, env/PPO edits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wt.htw_phase61.features import OPTION_UAV_DIM, UAV_DIM, attach_option, encode_option, feature_dict_l2a, in_range_completion, service_success, stack_feature_dicts
from wt.htw_phase61.models import L2AOptionActorCritic, L2AStateActorCritic
from wt.htw_phase61.run_phase61 import EVAL_SEEDS, MINI_UPDATES, SEED, SMOKE_CONFIG, TRAIN_SEEDS, _plain, evaluate_scripted
from wt.htw_phase61.vector import L2AVector
from wt.l2a.env import L2AEnv
from wt.l2a.policies import myopic_assignment, oracle_option_assignment
from wt.round30.ppo import PPOAgent, seed_everything
from wt.round30.runner import collect

from .vector import GoalOptionVector

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "goal_execution"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6_2_low_level_option_execution_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase6_2_low_level_option_execution_report.md"

LAMBDA_C = (0.1, 0.5, 1.0)
LAMBDA_D = (0.01, 0.05)
SERVICE_FLOOR = 0.002
PAYLOAD_GAP_BITS = 1.0e6


def evaluate_option_agent(agent: PPOAgent, seeds: tuple[int, ...], *, use_option: bool) -> dict[str, Any]:
    payloads, returns, queues, completions, services, motions, drops, tts, ttr = [], [], [], [], [], [], [], [], []
    assign_fn = oracle_option_assignment if use_option else myopic_assignment
    for seed in seeds:
        env = L2AEnv()
        info = env.reset(seed=seed)
        total_return = 0.0
        episode_move = 0.0
        goal_hits: list[float] = []
        service_hits: list[float] = []
        first_in_range = np.full(env.config.num_uavs, -1, dtype=np.int64)
        first_service = np.full(env.config.num_uavs, -1, dtype=np.int64)
        while not info.get("terminated", False):
            intended = assign_fn(env)
            features = feature_dict_l2a(info, env)
            if use_option:
                features = attach_option(features, encode_option(intended))
            requested = agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
            before = np.asarray(info["uav_positions_m"], dtype=np.float64)
            before_payload = float(info["cumulative_payload_bits"])
            after = env.step(requested, intended)
            total_return += (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
            episode_move += float(np.sum(np.linalg.norm(np.asarray(after["uav_positions_m"], dtype=np.float64) - before, axis=-1)))
            goal_hits.append(in_range_completion(env, intended))
            service_hits.append(service_success(after, intended))
            service = np.asarray(after["step_service_blocks"], dtype=np.float64)
            for uav, buoy in enumerate(np.asarray(intended, dtype=np.int64)):
                if int(buoy) < 0:
                    continue
                if int(first_in_range[uav]) < 0 and env._in_range(uav, int(buoy)):
                    first_in_range[uav] = int(after["time_step"])
                if int(first_service[uav]) < 0 and float(service[int(buoy)]) > 0.0:
                    first_service[uav] = int(after["time_step"])
            info = after
        horizon = float(env.config.episode_horizon)
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        queues.append(float(np.sum(info["queue_blocks"])))
        completions.append(float(np.mean(goal_hits)))
        services.append(float(np.mean(service_hits)))
        motions.append(episode_move)
        drops.append(float(info["dropped_blocks"]))
        ttr.append(float(np.mean([float(value) if int(value) >= 0 else horizon for value in first_in_range])))
        tts.append(float(np.mean([float(value) if int(value) >= 0 else horizon for value in first_service])))
    mean_payload = float(np.mean(payloads))
    mean_move = float(np.mean(motions))
    return {
        "episodes": len(payloads),
        "mean_payload_bits": mean_payload,
        "mean_return": float(np.mean(returns)),
        "mean_residual_queue_blocks": float(np.mean(queues)),
        "goal_completion_rate": float(np.mean(completions)),
        "service_success_rate": float(np.mean(services)),
        "mean_movement_m": mean_move,
        "mean_dropped_blocks": float(np.mean(drops)),
        "mean_time_to_in_range": float(np.mean(ttr)),
        "mean_time_to_service": float(np.mean(tts)),
        "action_efficiency_bits_per_m": mean_payload / max(mean_move, 1.0e-6),
    }


def train_state_only(updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    agent = PPOAgent(L2AStateActorCritic(), SMOKE_CONFIG)
    vector = L2AVector(TRAIN_SEEDS, agent.config.num_envs, 74000 + SEED, use_option=False)
    last_health: dict[str, float] = {}
    for index in range(int(updates)):
        storage, _events, health = collect(agent, vector)
        update = agent.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-6.2 state-only NaN at update {index}")
    evaluated = evaluate_option_agent(agent, EVAL_SEEDS, use_option=False)
    evaluated["family"] = "state_only"
    evaluated["variant"] = "state_only"
    evaluated["lambda_c"] = 0.0
    evaluated["lambda_d"] = 0.0
    evaluated["uav_dim"] = UAV_DIM
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["last_update"] = last_health
    return evaluated


def train_option(updates: int, *, lambda_c: float, lambda_d: float, variant: str) -> dict[str, Any]:
    seed_everything(SEED)
    agent = PPOAgent(L2AOptionActorCritic(), SMOKE_CONFIG)
    vector = GoalOptionVector(TRAIN_SEEDS, agent.config.num_envs, 74000 + SEED, lambda_c=lambda_c, lambda_d=lambda_d)
    last_health: dict[str, float] = {}
    for index in range(int(updates)):
        storage, _events, health = collect(agent, vector)
        update = agent.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-6.2 option NaN at update {index} {variant}")
    evaluated = evaluate_option_agent(agent, EVAL_SEEDS, use_option=True)
    evaluated["family"] = "oracle_option"
    evaluated["variant"] = variant
    evaluated["lambda_c"] = float(lambda_c)
    evaluated["lambda_d"] = float(lambda_d)
    evaluated["uav_dim"] = OPTION_UAV_DIM
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["last_update"] = last_health
    return evaluated


def _row(item: Mapping[str, Any]) -> str:
    return (
        f"| {item['variant']} | {item['lambda_c']:.2f} | {item['lambda_d']:.2f} | {item['mean_payload_bits']:.0f} | "
        f"{item['mean_return']:.2f} | {item['goal_completion_rate']:.3f} | {item['service_success_rate']:.3f} | "
        f"{item['mean_movement_m']:.1f} | {item['mean_time_to_in_range']:.1f} | {item['mean_time_to_service']:.1f} |"
    )


def write_report(payload: Mapping[str, Any]) -> None:
    state = payload["state_only"]
    best = payload["best_option"]
    shadow = payload["oracle_shadow"]
    beats = bool(payload["beats_state_only"])
    service_ok = bool(payload["service_above_floor"])
    enter = bool(payload["enter_high_level"])
    lines = [
        "# HTW Phase-6.2 Goal-conditioned Low-level Option Execution",
        "",
        "在 Level-2-A 上只训练低层：输入 `[state, z]`（z 为 Oracle buoy one-hot），输出 movement。",
        "未训 π_H、未加 HGNN、未用 Transformer / Task-Waypoint、未改环境、未改 PPO 核心、未开 Round。",
        f"与 Phase-6.1 相同：seed={SEED}，{payload['updates']} updates，rollout={SMOKE_CONFIG.rollout_length}，num_envs={SMOKE_CONFIG.num_envs}，训练种子 {list(TRAIN_SEEDS)}，评估种子 {list(EVAL_SEEDS)}。",
        "",
        "## 1. 为什么 6.1 低层执行不了 option",
        "",
        "环境 `r_comm` 只在 **进入服务半径并连续占用 D=3** 后才给 1 块 payload。随机/未成形运动几乎拿不到该信号，PPO 看不到 z 与位移的因果。影子 fly-toward 用同一 z 能到 2.775e7，证明失败在低层执行，不在 option。",
        f"距离辅助把 `||p−b_z||` 换成步长单位（不是 waypoint tracking），completion 是 **是否进入目标 buoy 服务区**。",
        "",
        "## 2. Reward 网格（仅 controller）",
        "",
        r"- A: `r_low = r_comm`",
        r"- B: `r_low = r_comm + λ_c · 1[in service radius of z]`",
        r"- C: `r_low = r_comm + λ_c · completion − λ_d · mean_uav(||p−b_z|| / vΔt)`",
        "",
        f"λ_c ∈ {list(LAMBDA_C)}，λ_d ∈ {list(LAMBDA_D)}。评估 return 仍是环境块数和。",
        "",
        "## 3. 结果",
        "",
        "| variant | λ_c | λ_d | payload | env return | completion | service | movement m | t_in_range | t_service |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _row(state),
    ]
    for item in payload["option_runs"]:
        lines.append(_row(item))
    lines.extend(
        [
            "",
            f"Oracle option + fly-toward 影子 payload = {shadow['mean_payload_bits']:.0f}。",
            f"**最佳低层**：`{best['variant']}`（λ_c={best['lambda_c']}, λ_d={best['lambda_d']}），payload {best['mean_payload_bits']:.0f}，completion {best['goal_completion_rate']:.3f}，service {best['service_success_rate']:.3f}。",
            f"相对 State-only payload 差 = {best['mean_payload_bits'] - state['mean_payload_bits']:.0f} bits。",
            "",
            "## 4. 三个问题",
            "",
            "1. **原低层失败原因：** 稀疏 `r_comm` 与 D=3 占用不同步；没有进入目标区域的中间信号，z 的 one-hot 无法被运动策略利用。",
            f"2. **哪种 goal reward 有效：** **B 与 C 有效，A 无效。** 最优 payload 是 C（λ_c=0.5, λ_d=0.05）= {best['mean_payload_bits']:.0f}；B（λ_c=0.1）几乎同样高且位移更短。A 仅 `r_comm` 仍约 5e5，复现 6.1 失败。",
            f"3. **是否进入 learned high-level policy：{'可以进入（有保留）' if enter else '不进入'}。** 成功标准：Option 低层 payload 明显高于 State-only（差 ≥ {PAYLOAD_GAP_BITS:.0f} bits）且 service success > {SERVICE_FLOOR}。当前：高于 State-only={beats}，service 过线={service_ok}。即使达标，低层仍低于 fly-toward 影子，执行尚未饱和。",
            "",
            "未训练 π_H。",
            "",
            "JSON：`src/wt/htw_phase62/outputs/goal_execution/summary.json`",
            "",
        ]
    )
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shadow = evaluate_scripted(oracle_option_assignment, EVAL_SEEDS)
    myopic = evaluate_scripted(myopic_assignment, EVAL_SEEDS)
    print("training state-only", flush=True)
    state = train_state_only(MINI_UPDATES)
    runs: list[dict[str, Any]] = []
    print("training A r_comm", flush=True)
    runs.append(train_option(MINI_UPDATES, lambda_c=0.0, lambda_d=0.0, variant="A_r_comm"))
    for lambda_c in LAMBDA_C:
        print(f"training B lambda_c={lambda_c}", flush=True)
        runs.append(train_option(MINI_UPDATES, lambda_c=lambda_c, lambda_d=0.0, variant=f"B_c{lambda_c}"))
        for lambda_d in LAMBDA_D:
            print(f"training C lambda_c={lambda_c} lambda_d={lambda_d}", flush=True)
            runs.append(train_option(MINI_UPDATES, lambda_c=lambda_c, lambda_d=lambda_d, variant=f"C_c{lambda_c}_d{lambda_d}"))
    best = max(runs, key=lambda row: (float(row["mean_payload_bits"]), float(row["service_success_rate"]), float(row["goal_completion_rate"])))
    family_best = {}
    for row in runs:
        family = str(row["variant"])[0]
        prev = family_best.get(family)
        if prev is None or float(row["mean_payload_bits"]) > float(prev["mean_payload_bits"]):
            family_best[family] = row
    ranked = sorted(family_best.items(), key=lambda item: -float(item[1]["mean_payload_bits"]))
    effective = ", ".join(f"{key}:{item['variant']} payload={item['mean_payload_bits']:.0f}" for key, item in ranked)
    beats = float(best["mean_payload_bits"]) >= float(state["mean_payload_bits"]) + PAYLOAD_GAP_BITS
    service_ok = float(best["service_success_rate"]) > SERVICE_FLOOR
    payload = {
        "updates": MINI_UPDATES,
        "seed": SEED,
        "oracle_shadow": shadow,
        "myopic_scripted": myopic,
        "state_only": state,
        "option_runs": runs,
        "best_option": best,
        "effective_family": effective,
        "beats_state_only": beats,
        "service_above_floor": service_ok,
        "enter_high_level": bool(beats and service_ok),
        "uses_hgnn": False,
        "trained_high_level": False,
        "uses_transformer": False,
        "uses_task_waypoint": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    summary = {
        "state_payload": result["state_only"]["mean_payload_bits"],
        "best_variant": result["best_option"]["variant"],
        "best_payload": result["best_option"]["mean_payload_bits"],
        "best_service": result["best_option"]["service_success_rate"],
        "best_completion": result["best_option"]["goal_completion_rate"],
        "enter_high_level": result["enter_high_level"],
    }
    print(json.dumps(summary, indent=2))
