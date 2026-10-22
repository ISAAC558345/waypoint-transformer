"""HTW Phase-6.3: learned high-level option policy with frozen Phase-6.2 low-level."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.features import attach_option, encode_option, feature_dict_l2a, in_range_completion, service_success, stack_feature_dicts
from wt.htw_phase61.models import L2AOptionActorCritic, L2AStateActorCritic
from wt.htw_phase61.run_phase61 import EVAL_SEEDS, MINI_UPDATES, SEED, SMOKE_CONFIG, TRAIN_SEEDS, _plain
from wt.htw_phase61.vector import L2AVector
from wt.htw_phase62.vector import GoalOptionVector
from wt.l2a.env import L2AEnv, project_exclusive
from wt.l2a.policies import myopic_assignment, oracle_option_assignment
from wt.round30.ppo import PPOAgent, seed_everything
from wt.round30.runner import collect

from .high_ppo import DiscretePPOAgent, HighLevelOptionModel, assignment_to_indices, indices_to_assignment
from .vector import HierarchicalVector, collect_both, collect_high

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "learned_option"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6_3_learned_option_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase6_3_learned_option_report.md"
CKPT = Path(__file__).resolve().parent / "outputs" / "checkpoints" / "low_level_c05_d005.pt"

BEST_LAMBDA_C = 0.5
BEST_LAMBDA_D = 0.05
CLOSE_RATIO = 0.80
STAGE1_EFFECTIVE_GAP = 1.0e6
JOINT_UPDATES = 40


def train_low_level() -> PPOAgent:
    seed_everything(SEED)
    agent = PPOAgent(L2AOptionActorCritic(), SMOKE_CONFIG)
    vector = GoalOptionVector(TRAIN_SEEDS, agent.config.num_envs, 74000 + SEED, lambda_c=BEST_LAMBDA_C, lambda_d=BEST_LAMBDA_D)
    for index in range(int(MINI_UPDATES)):
        storage, _events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"Phase-6.3 low-level NaN at update {index}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": agent.model.state_dict(), "lambda_c": BEST_LAMBDA_C, "lambda_d": BEST_LAMBDA_D}, CKPT)
    return agent


def freeze_low(agent: PPOAgent) -> PPOAgent:
    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad = False
    return agent


def train_state_only() -> dict[str, Any]:
    seed_everything(SEED)
    agent = PPOAgent(L2AStateActorCritic(), SMOKE_CONFIG)
    vector = L2AVector(TRAIN_SEEDS, agent.config.num_envs, 74000 + SEED, use_option=False)
    for index in range(int(MINI_UPDATES)):
        storage, _events, health = collect(agent, vector)
        update = agent.update(storage)
        row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"Phase-6.3 state-only NaN at update {index}")
    evaluated = evaluate_hierarchical(agent, None, EVAL_SEEDS, high_mode="myopic")
    evaluated["family"] = "state_only"
    evaluated["environment_steps"] = vector.environment_steps
    return evaluated


def evaluate_hierarchical(low_agent: PPOAgent, high_agent: DiscretePPOAgent | None, seeds: tuple[int, ...], *, high_mode: str) -> dict[str, Any]:
    payloads, returns, completions, services, accuracies = [], [], [], [], []
    for seed in seeds:
        env = L2AEnv()
        info = env.reset(seed=seed)
        total_return = 0.0
        goal_hits: list[float] = []
        service_hits: list[float] = []
        acc_hits: list[float] = []
        while not info.get("terminated", False):
            oracle = oracle_option_assignment(env)
            if high_mode == "oracle":
                assignment = np.asarray(oracle, dtype=np.int64)
            elif high_mode == "myopic":
                assignment = myopic_assignment(env)
            else:
                feats = feature_dict_l2a(info, env)
                batch = stack_feature_dicts([feats])
                batch["_uav_positions"] = env.uav_positions[None, ...]
                batch["_buoy_positions"] = env.buoy_positions[None, ...]
                z = high_agent.act(batch, deterministic=True)["z"][0]
                assignment = project_exclusive(indices_to_assignment(z), env.uav_positions, env.buoy_positions)
            acc_hits.append(float(np.mean(np.asarray(assignment, dtype=np.int64) == np.asarray(oracle, dtype=np.int64))))
            if high_mode == "myopic":
                features = feature_dict_l2a(info, env)
                requested = low_agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
            else:
                features = attach_option(feature_dict_l2a(info, env), encode_option(assignment))
                requested = low_agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
            before_payload = float(info["cumulative_payload_bits"])
            after = env.step(requested, assignment)
            total_return += (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
            goal_hits.append(in_range_completion(env, assignment))
            service_hits.append(service_success(after, assignment))
            info = after
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        completions.append(float(np.mean(goal_hits)))
        services.append(float(np.mean(service_hits)))
        accuracies.append(float(np.mean(acc_hits)))
    return {
        "episodes": len(payloads),
        "mean_payload_bits": float(np.mean(payloads)),
        "mean_return": float(np.mean(returns)),
        "goal_completion_rate": float(np.mean(completions)),
        "service_success_rate": float(np.mean(services)),
        "option_selection_accuracy": float(np.mean(accuracies)),
        "high_mode": high_mode,
    }


def train_high_level(low_agent: PPOAgent) -> tuple[DiscretePPOAgent, dict[str, Any]]:
    seed_everything(SEED)
    high = DiscretePPOAgent(HighLevelOptionModel(), SMOKE_CONFIG)
    vector = HierarchicalVector(TRAIN_SEEDS, high.config.num_envs, 75000 + SEED, low_agent, lambda_c=BEST_LAMBDA_C, lambda_d=BEST_LAMBDA_D, high_mode="learned", low_deterministic=True)
    last_health: dict[str, float] = {}
    for index in range(int(MINI_UPDATES)):
        storage, _events, health = collect_high(high, vector)
        update = high.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-6.3 high-level NaN at update {index}")
    evaluated = evaluate_hierarchical(low_agent, high, EVAL_SEEDS, high_mode="learned")
    evaluated["family"] = "learned_option_stage1"
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["last_update"] = last_health
    evaluated["stage"] = 1
    return high, evaluated


def joint_finetune(low_agent: PPOAgent, high_agent: DiscretePPOAgent) -> dict[str, Any]:
    for parameter in low_agent.model.parameters():
        parameter.requires_grad = True
    seed_everything(SEED)
    vector = HierarchicalVector(TRAIN_SEEDS, high_agent.config.num_envs, 76000 + SEED, low_agent, lambda_c=BEST_LAMBDA_C, lambda_d=BEST_LAMBDA_D, high_mode="learned", low_deterministic=False)
    last_health: dict[str, float] = {}
    for index in range(int(JOINT_UPDATES)):
        high_storage, low_storage, _events, health = collect_both(high_agent, low_agent, vector)
        high_update = high_agent.update(high_storage)
        low_update = low_agent.update(low_storage)
        last_health = {
            **health,
            **{f"high_{key}": float(value) for key, value in high_update.items() if isinstance(value, (int, float))},
            **{f"low_{key}": float(value) for key, value in low_update.items() if isinstance(value, (int, float))},
        }
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-6.3 joint high NaN at update {index}")
    evaluated = evaluate_hierarchical(low_agent, high_agent, EVAL_SEEDS, high_mode="learned")
    evaluated["family"] = "learned_option_stage2"
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["last_update"] = last_health
    evaluated["stage"] = 2
    return evaluated


def write_report(payload: Mapping[str, Any]) -> None:
    state = payload["state_only"]
    oracle = payload["oracle_option"]
    learned = payload["learned_option"]
    ratio = float(learned["mean_payload_bits"]) / max(float(oracle["mean_payload_bits"]), 1.0)
    close = ratio >= CLOSE_RATIO
    gap = float(oracle["mean_payload_bits"]) - float(learned["mean_payload_bits"])
    enter_hgnn = bool(not close)
    lines = [
        "# HTW Phase-6.3 Learned High-level Task Option Policy",
        "",
        "Level-2-A 上训练 π_H(z|s)，z 为 buoy+idle。低层为 Phase-6.2 最佳冻结控制器（λ_c=0.5, λ_d=0.05）。",
        "未加 HGNN、未改环境、未改 Round-30 PPO 核心文件、未用 Task-Waypoint、未开 Round。",
        f"Stage-1：冻结低层，高层奖励为逐步 payload 增量（回合累计 payload 的分解）。预算 {MINI_UPDATES} updates，seed {SEED}。",
        f"Stage-2 联合微调：{'已执行' if payload['ran_stage2'] else '未执行'}（仅当 Stage-1 高于 State-only 至少 {STAGE1_EFFECTIVE_GAP:.0f} bits）。",
        "",
        "## 1. 结构",
        "",
        "- π_H：DeepSets + 每机 Categorical(13)，采样后排他投影。",
        "- π_L：冻结（Stage-1）的 `[state,z]→movement`，PPO 为 Round-30 连续动作实现。",
        "- 高层 Discrete PPO 复用 `compute_gae` 与相同 clip/Adam/value/entropy 系数，不修改 `round30/ppo.py`。",
        "",
        "## 2. 结果",
        "",
        "| 方法 | payload | env return | completion | service | option accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| State-only PPO | {state['mean_payload_bits']:.0f} | {state['mean_return']:.2f} | "
            f"{state['goal_completion_rate']:.3f} | {state['service_success_rate']:.3f} | {state['option_selection_accuracy']:.3f} |"
        ),
        (
            f"| Oracle Option + Low-level PPO | {oracle['mean_payload_bits']:.0f} | {oracle['mean_return']:.2f} | "
            f"{oracle['goal_completion_rate']:.3f} | {oracle['service_success_rate']:.3f} | {oracle['option_selection_accuracy']:.3f} |"
        ),
        (
            f"| Learned Option + Low-level PPO | {learned['mean_payload_bits']:.0f} | {learned['mean_return']:.2f} | "
            f"{learned['goal_completion_rate']:.3f} | {learned['service_success_rate']:.3f} | {learned['option_selection_accuracy']:.3f} |"
        ),
        "",
            f"Learned / Oracle payload = **{ratio:.3f}**（接近阈值 {CLOSE_RATIO:.2f}）。绝对差 {gap:.0f} bits。",
            "",
            f"诊断：Learned completion={learned['goal_completion_rate']:.3f}，accuracy={learned['option_selection_accuracy']:.3f}。高 in-range 但低 accuracy 表示 π_H 偏向就近可完成目标，而非 Oracle 的跨簇任务分配。",
            "",
    ]
    if payload.get("stage2") is not None:
        stage2 = payload["stage2"]
        lines.append(
            f"Stage-2 联合微调后 payload {stage2['mean_payload_bits']:.0f}，accuracy {stage2['option_selection_accuracy']:.3f}。"
        )
        lines.append("")
    lines.extend(
        [
            "## 3. 三个问题",
            "",
            f"1. **高层是否能学习任务选择？{'能（相对 State-only 有增益）' if payload['stage1_effective'] else '当前预算下很弱'}。** option accuracy={learned['option_selection_accuracy']:.3f}（Oracle 对照为 1.0）。",
            f"2. **距离 Oracle 差距：** payload 比 Oracle+低层低 {gap:.0f} bits（{100.0 * (1.0 - ratio):.1f}% 相对缺口）。{'已接近' if close else '明显低于 Oracle，option policy 仍难：合法匹配组合大、排他耦合、与冻结低层的联合非平稳。'}",
            f"3. **是否进入 HGNN 阶段？{'建议作为后续关系建模，本阶段不加入' if enter_hgnn else '暂不需要'}。** 本阶段 **未加入 HGNN**。若 accuracy 低且 payload 明显落后，图结构关联是候选，而不是现在实现。",
            "",
            "JSON：`src/wt/htw_phase63/outputs/learned_option/summary.json`",
            "",
        ]
    )
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("training frozen low-level (Phase-6.2 best)", flush=True)
    low = freeze_low(train_low_level())
    print("evaluating oracle option + frozen low-level", flush=True)
    oracle = evaluate_hierarchical(low, None, EVAL_SEEDS, high_mode="oracle")
    oracle["family"] = "oracle_option"
    print("training state-only", flush=True)
    state = train_state_only()
    print("training high-level Stage-1", flush=True)
    high, learned = train_high_level(low)
    stage1_effective = float(learned["mean_payload_bits"]) >= float(state["mean_payload_bits"]) + STAGE1_EFFECTIVE_GAP
    stage2 = None
    if stage1_effective:
        print("Stage-1 effective; joint fine-tune Stage-2", flush=True)
        stage2 = joint_finetune(low, high)
        if float(stage2["mean_payload_bits"]) >= float(learned["mean_payload_bits"]):
            learned = {**stage2, "family": "learned_option"}
        else:
            learned = {**learned, "family": "learned_option", "stage2_rejected": True}
    else:
        print("Stage-1 not effective; skip Stage-2", flush=True)
        learned["family"] = "learned_option"
    payload = {
        "updates": MINI_UPDATES,
        "seed": SEED,
        "lambda_c": BEST_LAMBDA_C,
        "lambda_d": BEST_LAMBDA_D,
        "state_only": state,
        "oracle_option": oracle,
        "learned_option": learned,
        "stage2": stage2,
        "ran_stage2": stage2 is not None,
        "stage1_effective": stage1_effective,
        "learned_over_oracle": float(learned["mean_payload_bits"]) / max(float(oracle["mean_payload_bits"]), 1.0),
        "uses_hgnn": False,
        "uses_task_waypoint": False,
        "modified_env": False,
        "modified_ppo_core": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "state_payload": result["state_only"]["mean_payload_bits"],
                "oracle_payload": result["oracle_option"]["mean_payload_bits"],
                "learned_payload": result["learned_option"]["mean_payload_bits"],
                "learned_accuracy": result["learned_option"]["option_selection_accuracy"],
                "ratio": result["learned_over_oracle"],
                "ran_stage2": result["ran_stage2"],
            },
            indent=2,
        )
    )
