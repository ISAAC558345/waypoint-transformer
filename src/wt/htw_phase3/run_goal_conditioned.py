"""Phase-3 goal-conditioned PPO: TW is input only, env reward is unmodified."""
from __future__ import annotations

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
from wt.htwc_phase1.adapter import CONDITIONED_UAV_DIM
from wt.htwc_phase1.conditioned_actor import GoalConditionedActorCritic
from wt.htwc_phase1.validation import MINI_UPDATES, SEED, SMOKE_CONFIG, ConditionedVector, eval_slice, train_slice
from wt.round30.features import UAV_DIM
from wt.round30.models import DeepSetsActorCritic
from wt.round30.ppo import PPOAgent, seed_everything
from wt.round30.runner import collect

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "goal_conditioned"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase3_goal_conditioned_controller_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase3_goal_conditioned_controller_report.md"
PHASE2_JSON = ROOT / "src" / "wt" / "htw_phase2" / "outputs" / "oracle_controller" / "summary.json"


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


def train_comm_reward(spec: BaselineSpec, updates: int) -> dict[str, Any]:
    seed_everything(SEED)
    agent = PPOAgent(spec.model, SMOKE_CONFIG)
    vector = ConditionedVector(train_slice(), agent.config.num_envs, 73000 + SEED, spec.conditioner)
    curve = []
    for index in range(int(updates)):
        storage, events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {"update": float(index), **health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if events:
            row["completed_payload_bits"] = float(np.mean([item["payload_bits"] for item in events]))
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"{spec.name} produced NaN at update {index}")
        curve.append(row)
    evaluated = evaluate_controller(spec, agent, eval_slice())
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["tracking_penalty"] = False
    evaluated["ppo_reward"] = "r_comm"
    evaluated["actor_uav_dim"] = spec.uav_dim
    evaluated["critic_uav_dim"] = int(spec.model.critic.shape.uav_dim)
    evaluated["mean_ppo_reward_last"] = curve[-1]["mean_raw_reward"]
    return evaluated


def phase2_tracking_row() -> dict[str, Any]:
    payload = json.loads(PHASE2_JSON.read_text(encoding="utf-8"))
    htw = payload["results"]["htw_oracle"]
    state = payload["results"]["state_only"]
    return {
        "phase2_state_only": {
            "mean_payload_bits": state["mean_payload_bits"],
            "mean_return": state["mean_return"],
            "mean_residual_queue_bits": state["mean_residual_queue_bits"],
            "mean_waypoint_tracking_error": state["mean_waypoint_tracking_error"],
            "action_waypoint_cosine": state["action_waypoint"]["mean_rho"],
        },
        "phase2_tracking": {
            "mean_payload_bits": htw["mean_payload_bits"],
            "mean_return": htw["mean_return"],
            "mean_residual_queue_bits": htw["mean_residual_queue_bits"],
            "mean_waypoint_tracking_error": htw["mean_waypoint_tracking_error"],
            "action_waypoint_cosine": htw["action_waypoint"]["mean_rho"],
            "tracking_lambda": 1.0,
        },
    }


def write_report(payload: Mapping[str, Any]) -> None:
    state = payload["state_only"]
    goal = payload["goal_conditioned"]
    track = payload["phase2"]["phase2_tracking"]
    p2_state = payload["phase2"]["phase2_state_only"]
    closed = float(goal["mean_payload_bits"]) > float(state["mean_payload_bits"]) + 1.0
    lines = [
        "# HTW Phase-3 Goal-conditioned Low-level Controller Report",
        "",
        "Oracle TW 作为 actor/critic 输入；PPO 使用未修改的环境 `r_comm`。无 tracking penalty、无 HGNN、无 predictor 训练、不开 Round。",
        f"预算：{payload['updates']} updates，seed {SEED}，与 Phase-2 相同。",
        "",
        "## 1. 网络结构变化",
        "",
        f"- State-only：actor/critic UAV dim = {UAV_DIM}，`V(s)`。",
        f"- Goal-conditioned：UAV dim = {CONDITIONED_UAV_DIM} = state(8)+goal(5)+waypoint(2)+horizon(1)。",
        "- Actor：`π(a | s, TW)`，`SharedDeepSetsActor`，未改 GAE/clip/Adam/loss。",
        "- Critic：`V(s, TW)`，同一拼接后的 UAV token 进入 `CentralDeepSetsCritic`（不是只进 actor）。",
        "- High-level：仍用 oracle `independent_queue_rate` 窗口，不训练 Transformer。",
        "",
        "## 2. Reward 变化",
        "",
        "- Phase-2：controller 内部 `r_total = r_comm − λ d(p, W)`，λ=1。",
        "- Phase-3：**删除该辅助项**。`ConditionedVector.step` 把 `env.step` 的 `r_comm` 原样交给 PPO。",
        "- 环境 reward 公式未改。",
        "",
        "## 3. 实验结果",
        "",
        "| 方法 | payload | env return | residual queue | waypoint distance | action–W cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| State-only PPO | {state['mean_payload_bits']:.3f} | {state['mean_return']:.6f} | "
            f"{state['mean_residual_queue_bits']:.3f} | {state['mean_waypoint_tracking_error']:.6f} | "
            f"{state['action_waypoint'].get('mean_rho')} |"
        ),
        (
            f"| HTW tracking PPO (Phase-2, λ=1) | {track['mean_payload_bits']:.3f} | {track['mean_return']:.6f} | "
            f"{track['mean_residual_queue_bits']:.3f} | {track['mean_waypoint_tracking_error']:.6f} | "
            f"{track['action_waypoint_cosine']} |"
        ),
        (
            f"| HTW goal-conditioned PPO | {goal['mean_payload_bits']:.3f} | {goal['mean_return']:.6f} | "
            f"{goal['mean_residual_queue_bits']:.3f} | {goal['mean_waypoint_tracking_error']:.6f} | "
            f"{goal['action_waypoint'].get('mean_rho')} |"
        ),
        "",
        f"Goal-conditioned 的 critic UAV dim={goal['critic_uav_dim']}，TW 进入 actor：{goal['tw_enters_actor']}。",
        "",
        "## 4. 与 Phase-2 比较",
        "",
        f"- Phase-2 当时 State-only payload={p2_state['mean_payload_bits']:.3f}，ρ={p2_state['action_waypoint_cosine']:.3f}。",
        "- Tracking 提高了 ρ、降低了距离，但 payload 下降。",
        "- Phase-3 去掉强制跟踪后，低层用原始通信奖励，TW 仅作条件信息。",
        "",
        "## 5. HTW-C 是否成立",
        "",
        f"**{'成立' if closed else '不成立'}**：Goal-conditioned payload {'>' if closed else '≤'} State-only。",
        "",
        "判定规则：仅当 goal-conditioned payload 严格高于本次 State-only 时认为闭环成立；否则 high-level planner / 条件化方式仍不足，而不是再加 tracking。",
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
    state_spec = BaselineSpec("State-only PPO", "state_only", UAV_DIM, DeepSetsActorCritic(), state_only_conditioner)
    goal_spec = BaselineSpec("HTW goal-conditioned PPO", "htw_oracle", CONDITIONED_UAV_DIM, GoalConditionedActorCritic(), oracle_low_level_conditioner)
    state = train_comm_reward(state_spec, updates)
    goal = train_comm_reward(goal_spec, updates)
    payload = {
        "updates": updates,
        "seed": SEED,
        "fresh_access": 0,
        "untouched_access": 0,
        "hgnn": False,
        "predictor_trained": False,
        "tracking_penalty": False,
        "state_only": state,
        "goal_conditioned": goal,
        "phase2": phase2_tracking_row(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps(_plain({"state": result["state_only"], "goal": result["goal_conditioned"]}), indent=2))
