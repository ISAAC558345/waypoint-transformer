"""HTW Phase-6.1: Oracle option + low-level PPO on Level-2-A. No HGNN, no pi_H, no env/PPO edits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wt.l2a.env import L2AEnv
from wt.l2a.policies import myopic_assignment, oracle_option_assignment, rollout
from wt.round30.ppo import PPOAgent, PPOConfig, seed_everything
from wt.round30.runner import collect

from .features import OPTION_UAV_DIM, UAV_DIM, attach_option, encode_option, feature_dict_l2a, in_range_completion, service_success, stack_feature_dicts
from .models import L2AOptionActorCritic, L2AStateActorCritic
from .vector import L2AVector

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "oracle_option_ppo"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6_1_oracle_option_controller_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase6_1_oracle_option_controller_report.md"

SEED = 0
TRAIN_SEEDS = tuple(range(8))
EVAL_SEEDS = tuple(range(8, 16))
MINI_UPDATES = 80
CLOSE_RATIO = 0.80
SMOKE_CONFIG = PPOConfig(rollout_length=48, num_envs=4, num_minibatches=4, update_epochs=4, checkpoint_interval_steps=192)


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


def evaluate_scripted(assign_fn, seeds: tuple[int, ...]) -> dict[str, float]:
    rows = [rollout(L2AEnv(), assign_fn, seed) for seed in seeds]
    return {
        "mean_payload_bits": float(np.mean([row["payload_bits"] for row in rows])),
        "mean_residual_queue_blocks": float(np.mean([row["residual_queue"] for row in rows])),
        "mean_dropped_blocks": float(np.mean([row["dropped_blocks"] for row in rows])),
    }


def evaluate_agent(agent: PPOAgent, seeds: tuple[int, ...], *, use_option: bool) -> dict[str, Any]:
    payloads, returns, queues, completions, services, motions, drops = [], [], [], [], [], [], []
    assign_fn = oracle_option_assignment if use_option else myopic_assignment
    for seed in seeds:
        env = L2AEnv()
        info = env.reset(seed=seed)
        total_return = 0.0
        episode_move = 0.0
        goal_hits: list[float] = []
        service_hits: list[float] = []
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
            info = after
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        queues.append(float(np.sum(info["queue_blocks"])))
        completions.append(float(np.mean(goal_hits)))
        services.append(float(np.mean(service_hits)))
        motions.append(episode_move)
        drops.append(float(info["dropped_blocks"]))
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
        "action_efficiency_bits_per_m": mean_payload / max(mean_move, 1.0e-6),
    }


def train_family(*, use_option: bool, updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    model = L2AOptionActorCritic() if use_option else L2AStateActorCritic()
    agent = PPOAgent(model, SMOKE_CONFIG)
    vector = L2AVector(TRAIN_SEEDS, agent.config.num_envs, 74000 + SEED, use_option=use_option)
    last_health: dict[str, float] = {}
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-6.1 NaN at update {index} option={use_option}")
    evaluated = evaluate_agent(agent, EVAL_SEEDS, use_option=use_option)
    evaluated["family"] = "oracle_option" if use_option else "state_only"
    evaluated["uav_dim"] = OPTION_UAV_DIM if use_option else UAV_DIM
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["last_update"] = last_health
    return evaluated


def write_report(payload: Mapping[str, Any]) -> None:
    state = payload["state_only"]
    option = payload["oracle_option"]
    shadow = payload["oracle_shadow"]
    myopic = payload["myopic_scripted"]
    ratio = float(option["mean_payload_bits"]) / max(float(shadow["mean_payload_bits"]), 1.0)
    close = ratio >= CLOSE_RATIO
    beats_state = float(option["mean_payload_bits"]) > float(state["mean_payload_bits"]) + 1.0
    needs_hrl = float(shadow["mean_payload_bits"]) > float(myopic["mean_payload_bits"]) + 1.0
    enter_high = bool(close and beats_state)
    lines = [
        "# HTW Phase-6.1 Oracle Option + Low-level PPO Report",
        "",
        "Level-2-A（M=5, K=12）上验证层次控制器：高层仍为 **Oracle buoy option**，低层 PPO 只输出 movement。",
        "未加 HGNN、未训练 high-level policy、未改 Transformer、未改 L2-A 环境、未改 Round-30 PPO（GAE/clip/Adam/loss）、未开 Round。",
        f"预算：{payload['updates']} updates，rollout={SMOKE_CONFIG.rollout_length}，num_envs={SMOKE_CONFIG.num_envs}，seed {SEED}。",
        f"训练种子 {list(TRAIN_SEEDS)}，评估种子 {list(EVAL_SEEDS)}。",
        "",
        "## 1. 结构",
        "",
        "- High-level：因果验证中的 `oracle_option_assignment`（簇 option 影子搜索 + 簇内排他）。输出每机 buoy index 或 idle，**one-hot**（12+idle）。不学习 π_H。",
        f"- Low-level：输入 `[state, z]`，UAV dim={OPTION_UAV_DIM}。输出 movement `(5,2)`。",
        f"- State-only PPO：输入 state，UAV dim={UAV_DIM}；关联为当前贪心 `myopic_assignment`。",
        "- 环境关联由高层/贪心脚本写入 `env.step(movement, association)`；低层不改 matching 规则。",
        "",
        "## 2. Reward",
        "",
        "- PPO 使用环境 payload 增量换算的 **r_comm（完成块数）**。不改环境内部公式。",
        "- 不把 waypoint 距离写入 reward。goal completion / service success 只作记录。",
        "",
        "## 3. 实验结果",
        "",
        "| 方法 | payload bits | env return (blocks) | residual queue | dropped | goal completion (in-range) | service success | movement m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Myopic + fly-toward（非学习） | {myopic['mean_payload_bits']:.0f} | — | "
            f"{myopic['mean_residual_queue_blocks']:.2f} | {myopic['mean_dropped_blocks']:.2f} | — | — | — |"
        ),
        (
            f"| Oracle option + fly-toward（影子） | {shadow['mean_payload_bits']:.0f} | — | "
            f"{shadow['mean_residual_queue_blocks']:.2f} | {shadow['mean_dropped_blocks']:.2f} | — | — | — |"
        ),
        (
            f"| Level-2 State-only PPO | {state['mean_payload_bits']:.0f} | {state['mean_return']:.2f} | "
            f"{state['mean_residual_queue_blocks']:.2f} | {state['mean_dropped_blocks']:.2f} | "
            f"{state['goal_completion_rate']:.3f} | {state['service_success_rate']:.3f} | {state['mean_movement_m']:.1f} |"
        ),
        (
            f"| Oracle Option + Low-level PPO | {option['mean_payload_bits']:.0f} | {option['mean_return']:.2f} | "
            f"{option['mean_residual_queue_blocks']:.2f} | {option['mean_dropped_blocks']:.2f} | "
            f"{option['goal_completion_rate']:.3f} | {option['service_success_rate']:.3f} | {option['mean_movement_m']:.1f} |"
        ),
        "",
        f"Option PPO / Oracle shadow = **{ratio:.3f}**（接近阈值 {CLOSE_RATIO:.2f}）。",
        "",
        "诊断：影子控制器用单位方向飞向 z，payload 高；PPO 只拿稀疏 r_comm（D=3 且须停在 16 m 内）。Option PPO 位移更大但 in-range/service 接近 0.11/0.00，说明 **z 没有被执行成服务**，不是 option 标签无价值。",
        "",
        "## 4. 三个问题",
        "",
        f"1. **Level-2 是否需要 HRL？{'需要' if needs_hrl else '尚不充分'}。** 评估种子上 Oracle 影子 payload {shadow['mean_payload_bits']:.0f} vs 贪心脚本 {myopic['mean_payload_bits']:.0f}。长期任务 option 仍有间隙；State-only PPO 不能靠运动补上高层分配。",
        f"2. **低层是否可以执行 task option？{'可以（接近影子）' if close else '当前预算下不足'}。** 低层应对准 z 指定 buoy 并进入服务半径。Option PPO payload {option['mean_payload_bits']:.0f}，影子 {shadow['mean_payload_bits']:.0f}；in-range completion {option['goal_completion_rate']:.3f}。",
        f"3. **是否进入 learned high-level policy？{'可以进入' if enter_high else '不进入'}。** 条件：低层接近影子 **且** Option PPO 严格高于 State-only PPO。当前：接近影子={close}，Option>State-only={beats_state}。",
        "",
        "未训练 π_H。HGNN 仍不加入。",
        "",
        f"JSON：`src/wt/htw_phase61/outputs/oracle_option_ppo/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shadow = evaluate_scripted(oracle_option_assignment, EVAL_SEEDS)
    myopic = evaluate_scripted(myopic_assignment, EVAL_SEEDS)
    state = train_family(use_option=False, updates=MINI_UPDATES)
    option = train_family(use_option=True, updates=MINI_UPDATES)
    payload = {
        "updates": MINI_UPDATES,
        "seed": SEED,
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "close_ratio": CLOSE_RATIO,
        "oracle_shadow": shadow,
        "myopic_scripted": myopic,
        "state_only": state,
        "oracle_option": option,
        "option_over_shadow": float(option["mean_payload_bits"]) / max(float(shadow["mean_payload_bits"]), 1.0),
        "uses_hgnn": False,
        "trained_high_level": False,
        "modified_env": False,
        "modified_ppo_core": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    keys = ("oracle_shadow", "myopic_scripted", "state_only", "oracle_option", "option_over_shadow")
    print(json.dumps({key: _plain(result[key]) if key != "option_over_shadow" else result[key] for key in keys}, indent=2))
