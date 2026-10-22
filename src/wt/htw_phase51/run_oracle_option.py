"""Phase-5.1: oracle buoy option + low-level PPO. No HGNN, TW, env/reward edits, or Round."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wt.htw_phase2.experiment import BaselineSpec, state_only_conditioner
from wt.htwc_phase1.validation import MINI_UPDATES, SEED, SMOKE_CONFIG, ConditionedVector, eval_slice, train_slice
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv
from wt.round30.features import UAV_DIM, stack_feature_dicts
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent, seed_everything
from wt.round30.runner import collect
from wt.round30.scenarios import PPOScenario

from .models import OptionActorCritic
from .option import COMPLETION_COEF, OPTION_DIM, OPTION_HOLD, OPTION_UAV_DIM, completion_score, oracle_assignment
from .vector import OptionVector

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "oracle_option"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5_1_oracle_option_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase5_1_oracle_option_report.md"


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


def evaluate_policy(agent: PPOAgent, scenarios: tuple[PPOScenario, ...], *, use_option: bool, hold: int = OPTION_HOLD) -> dict[str, Any]:
    payloads, returns, queues, completions, motions, payloads_step = [], [], [], [], [], []
    for scenario in scenarios:
        env = SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=np.asarray(scenario.buoy_positions_m), options=scenario.options())
        _, info = env.reset(seed=scenario.seed)
        total_return, done = 0.0, False
        ttl, intended = 0, None
        episode_move = 0.0
        while not done:
            if use_option:
                if ttl <= 0:
                    intended = oracle_assignment(info, env)
                    ttl = int(hold)
                from .option import attach_option, encode_option
                from wt.round30.features import feature_dict

                features = attach_option(feature_dict(info, env.config), encode_option(intended))
            else:
                from wt.round30.features import feature_dict

                features = feature_dict(info, env.config)
            requested = agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
            before = np.asarray(info["uav_positions_m"], dtype=np.float64)
            try:
                _, reward, terminated, truncated, after = env.step(requested, matching_mode="legacy")
            except RuntimeError as error:
                if "hard UAV separation projection did not converge" not in str(error):
                    raise
                _, reward, terminated, truncated, after = env.step(np.zeros_like(requested), matching_mode="legacy")
            episode_move += float(np.sum(np.linalg.norm(np.asarray(after["uav_positions_m"], dtype=np.float64) - before, axis=-1)))
            if use_option:
                completions.append(completion_score(after["uav_to_buoy_assignment"], intended, env.num_uavs))
                ttl -= 1
            else:
                intended_now = oracle_assignment(info, env)
                completions.append(completion_score(after["uav_to_buoy_assignment"], intended_now, env.num_uavs))
            payloads_step.append(float(np.sum(after["transmitted_payload_bits"])))
            total_return += float(reward)
            info, done = after, bool(terminated or truncated)
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        queues.append(float(np.sum(info["raw_queue_bits"])))
        motions.append(episode_move)
    mean_payload = float(np.mean(payloads))
    mean_move = float(np.mean(motions))
    return {
        "episodes": len(payloads),
        "mean_payload_bits": mean_payload,
        "mean_return": float(np.mean(returns)),
        "mean_residual_queue_bits": float(np.mean(queues)),
        "goal_completion_rate": float(np.mean(completions)) if completions else float("nan"),
        "mean_movement_m": mean_move,
        "action_efficiency_bits_per_m": mean_payload / max(mean_move, 1.0e-6),
    }


def train_state_only(updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    spec = BaselineSpec("State-only", "state_only", UAV_DIM, DeepSetsActorCritic(), state_only_conditioner)
    agent = PPOAgent(spec.model, SMOKE_CONFIG)
    vector = ConditionedVector(train_slice(), agent.config.num_envs, 73000 + SEED, spec.conditioner)
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"state-only NaN at update {index}")
    evaluated = evaluate_policy(agent, eval_slice(), use_option=False)
    evaluated["family"] = "state_only"
    evaluated["uav_dim"] = UAV_DIM
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["completion_coef"] = 0.0
    return evaluated


def train_option(updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    model = OptionActorCritic()
    agent = PPOAgent(model, SMOKE_CONFIG)
    vector = OptionVector(train_slice(), agent.config.num_envs, 73000 + SEED)
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"option PPO NaN at update {index}")
    evaluated = evaluate_policy(agent, eval_slice(), use_option=True)
    evaluated["family"] = "oracle_option"
    evaluated["uav_dim"] = OPTION_UAV_DIM
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["completion_coef"] = COMPLETION_COEF
    evaluated["option_hold"] = OPTION_HOLD
    return evaluated


def write_report(payload: Mapping[str, Any]) -> None:
    state = payload["state_only"]
    option = payload["oracle_option"]
    better = float(option["mean_payload_bits"]) > float(state["mean_payload_bits"]) + 1.0
    lines = [
        "# HTW Phase-5.1 Oracle Task Option Report",
        "",
        "Oracle buoy option + 低层 PPO。无 HGNN / Transformer / 环境 / 原始 reward 修改；不开 Round；不使用 Task-Waypoint。",
        f"预算：{payload['updates']} updates，seed {SEED}，option hold={OPTION_HOLD}，completion coef={COMPLETION_COEF}。",
        "",
        "## 1. 网络结构",
        "",
        f"- State-only：UAV dim={UAV_DIM}，`π(a|s)`，`V(s)`。",
        f"- Oracle option：UAV dim={OPTION_UAV_DIM}=state({UAV_DIM})+option({OPTION_DIM}，含 idle)。actor/critic 共用该宽度。",
        "- PPO：Round-30 GAE/clip/Adam/loss 未改。",
        "",
        "## 2. 输入输出",
        "",
        "- High-level（规则 oracle）：每 hold 步用当前 `joint_aware_matching` 得到每机 buoy 或 idle，编码 one-hot。不改环境 matching。",
        "- Low-level 输入：`[state, z]`。输出：movement `(M,2)`。",
        "",
        "## 3. Reward 设计",
        "",
        "- 环境仍只产生 `r_comm`。评估 return 是环境回合和。",
        f"- Controller 内部：`r_ppo = r_comm + {COMPLETION_COEF} * 1[assignment==z]`（UAV 平均）。无 waypoint 距离。",
        "",
        "## 4. 实验结果",
        "",
        "| 方法 | payload | env return | residual queue | goal completion | action efficiency (bits/m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| State-only PPO | {state['mean_payload_bits']:.3f} | {state['mean_return']:.6f} | "
            f"{state['mean_residual_queue_bits']:.3f} | {state['goal_completion_rate']:.4f} | {state['action_efficiency_bits_per_m']:.3f} |"
        ),
        (
            f"| Oracle Task Option PPO | {option['mean_payload_bits']:.3f} | {option['mean_return']:.6f} | "
            f"{option['mean_residual_queue_bits']:.3f} | {option['goal_completion_rate']:.4f} | {option['action_efficiency_bits_per_m']:.3f} |"
        ),
        "",
        "## 5. 是否进入 High-level policy 训练",
        "",
        f"**{'进入' if better else '不进入'}**：Oracle option payload {'>' if better else '≤'} State-only。",
        "",
        "成功标准：仅当 payload 严格更高时认为 task option 方向成立，可进入可学习 π_H。否则 task decomposition（离散 buoy option）在当前环境/预算下不成立，不是 TW 表示问题。",
        "",
        f"JSON：`{OUTPUT.as_posix()}/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    updates = MINI_UPDATES
    state = train_state_only(updates)
    option = train_option(updates)
    payload = {
        "updates": updates,
        "seed": SEED,
        "fresh_access": 0,
        "untouched_access": 0,
        "hgnn": False,
        "task_waypoint": False,
        "state_only": state,
        "oracle_option": option,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps(_plain({"state": result["state_only"], "option": result["oracle_option"]}), indent=2))
