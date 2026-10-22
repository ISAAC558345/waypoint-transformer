"""HTW Phase-6.4: improve π_H via hold, reward shaping, and DeepSets obs. No HGNN, no low-level edits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.features import BUOY_DIM, attach_option, encode_option, feature_dict_l2a, in_range_completion, service_success, stack_feature_dicts
from wt.htw_phase61.models import L2AOptionActorCritic
from wt.htw_phase61.run_phase61 import EVAL_SEEDS, MINI_UPDATES, SEED, SMOKE_CONFIG, TRAIN_SEEDS, _plain
from wt.htw_phase62.vector import GoalOptionVector
from wt.htw_phase63.high_ppo import DiscretePPOAgent, indices_to_assignment
from wt.htw_phase63.run_phase63 import BEST_LAMBDA_C, BEST_LAMBDA_D, CKPT as LOW_CKPT, evaluate_hierarchical
from wt.l2a.env import L2AEnv, project_exclusive
from wt.l2a.policies import oracle_option_assignment
from wt.round30.ppo import PPOAgent, PPOConfig, seed_everything
from wt.round30.runner import collect

from .features import ENHANCED_BUOY_DIM, feature_dict_high
from .models import HighLevelOptionModel
from .vector import HeldHighVector, collect_held

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "high_level_improvement"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6_4_high_level_improvement_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase6_4_high_level_improvement_report.md"
PHASE63_JSON = Path(__file__).resolve().parents[1] / "htw_phase63" / "outputs" / "learned_option" / "summary.json"

HOLDS = (5, 10, 15)
REWARD_MODES = ("episode", "delta", "delta_service", "delta_service_delay")
ACCURACY_GAIN = 0.05
PAYLOAD_GAP = 1.0e6


def high_config() -> PPOConfig:
    return PPOConfig(rollout_length=16, num_envs=4, num_minibatches=4, update_epochs=4, checkpoint_interval_steps=192)


def load_frozen_low() -> PPOAgent:
    agent = PPOAgent(L2AOptionActorCritic(), SMOKE_CONFIG)
    if LOW_CKPT.is_file():
        payload = torch.load(LOW_CKPT, map_location=agent.config.device)
        agent.model.load_state_dict(payload["model_state_dict"])
    else:
        print("low-level checkpoint missing; reconstructing frozen Phase-6.2 controller once", flush=True)
        seed_everything(SEED)
        vector = GoalOptionVector(TRAIN_SEEDS, agent.config.num_envs, 74000 + SEED, lambda_c=BEST_LAMBDA_C, lambda_d=BEST_LAMBDA_D)
        for index in range(int(MINI_UPDATES)):
            storage, _events, health = collect(agent, vector)
            update = agent.update(storage)
            row = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
            if not all(np.isfinite(value) for value in row.values()):
                raise RuntimeError(f"Phase-6.4 low-level reconstruct NaN at {index}")
        LOW_CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": agent.model.state_dict(), "lambda_c": BEST_LAMBDA_C, "lambda_d": BEST_LAMBDA_D}, LOW_CKPT)
    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad = False
    return agent


def evaluate_held(low_agent: PPOAgent, high_agent: DiscretePPOAgent, seeds: tuple[int, ...], *, hold: int, enhanced: bool) -> dict[str, Any]:
    payloads, returns, completions, services, accuracies = [], [], [], [], []
    for seed in seeds:
        env = L2AEnv()
        info = env.reset(seed=seed)
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        total_return = 0.0
        goal_hits: list[float] = []
        service_hits: list[float] = []
        acc_hits: list[float] = []
        pending = 0
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        while not info.get("terminated", False):
            if pending <= 0:
                feats = feature_dict_high(info, env, last_arrival=last_arrival, enhanced=enhanced)
                batch = stack_feature_dicts([feats])
                batch["_uav_positions"] = env.uav_positions[None, ...]
                batch["_buoy_positions"] = env.buoy_positions[None, ...]
                z = high_agent.act(batch, deterministic=True)["z"][0]
                assignment = project_exclusive(indices_to_assignment(z), env.uav_positions, env.buoy_positions)
                pending = int(hold)
            oracle = oracle_option_assignment(env)
            acc_hits.append(float(np.mean(np.asarray(assignment, dtype=np.int64) == np.asarray(oracle, dtype=np.int64))))
            features = attach_option(feature_dict_l2a(info, env), encode_option(assignment))
            requested = low_agent.act(stack_feature_dicts([features]), deterministic=True)["action"][0]
            before_payload = float(info["cumulative_payload_bits"])
            after = env.step(requested, assignment)
            total_return += (float(after["cumulative_payload_bits"]) - before_payload) / float(env.config.block_payload_bits)
            goal_hits.append(in_range_completion(env, assignment))
            service_hits.append(service_success(after, assignment))
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            pending -= 1
            info = after
        payloads.append(float(info["cumulative_payload_bits"]))
        returns.append(total_return)
        completions.append(float(np.mean(goal_hits)))
        services.append(float(np.mean(service_hits)))
        accuracies.append(float(np.mean(acc_hits)))
    return {
        "mean_payload_bits": float(np.mean(payloads)),
        "mean_return": float(np.mean(returns)),
        "goal_completion_rate": float(np.mean(completions)),
        "service_success_rate": float(np.mean(services)),
        "option_selection_accuracy": float(np.mean(accuracies)),
    }


def train_high(low_agent: PPOAgent, *, hold: int, reward_mode: str, enhanced: bool, name: str) -> dict[str, Any]:
    seed_everything(SEED)
    buoy_dim = ENHANCED_BUOY_DIM if enhanced else BUOY_DIM
    high = DiscretePPOAgent(HighLevelOptionModel(buoy_dim), high_config())
    vector = HeldHighVector(TRAIN_SEEDS, high.config.num_envs, 77000 + SEED, low_agent, hold=hold, reward_mode=reward_mode, enhanced=enhanced)
    last_health: dict[str, float] = {}
    for index in range(int(MINI_UPDATES)):
        storage, _events, health = collect_held(high, vector)
        update = high.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-6.4 NaN at {name} update {index}")
    evaluated = evaluate_held(low_agent, high, EVAL_SEEDS, hold=hold, enhanced=enhanced)
    evaluated["name"] = name
    evaluated["hold"] = int(hold)
    evaluated["reward_mode"] = reward_mode
    evaluated["enhanced"] = bool(enhanced)
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["last_update"] = last_health
    return evaluated


def _row(item: Mapping[str, Any]) -> str:
    return (
        f"| {item.get('name', item.get('family', ''))} | {item.get('hold', '—')} | {item.get('reward_mode', '—')} | "
        f"{'yes' if item.get('enhanced') else 'no'} | {item['mean_payload_bits']:.0f} | {item['mean_return']:.2f} | "
        f"{item['goal_completion_rate']:.3f} | {item['service_success_rate']:.3f} | {item['option_selection_accuracy']:.3f} |"
    )


def write_report(payload: Mapping[str, Any]) -> None:
    best = payload["best"]
    state = payload["state_only"]
    oracle = payload["oracle_option"]
    base = payload["phase63_learned"]
    beats_state = bool(payload["beats_state_only"])
    acc_up = bool(payload["accuracy_improved"])
    ready = bool(payload["prepare_hgnn"])
    lines = [
        "# HTW Phase-6.4 High-level Option Policy Improvement",
        "",
        "只优化 π_H(z|s)。冻结 Phase-6.2/6.3 低层，不改环境、不加 HGNN、不用 Task-Waypoint、不改低层权重。",
        f"高层预算：{MINI_UPDATES} updates，rollout=16，num_envs=4，seed {SEED}。",
        "",
        "## 1. 为什么 6.3 高层失败",
        "",
        "逐步 `Δpayload` + 每步重选 z，使 π_H 偏向就近、立刻能进服务圈的 buoy（completion 高、accuracy≈0.02）。远簇延迟到达需要 **hold 住 option** 并看到队列/ETA/到达预测，而不是每步被稀疏通信奖励拉回近簇。",
        "",
        "## 2. 实验设置",
        "",
        "- A：option hold ∈ {5,10,15}，reward=`delta`，观测=6.3 DeepSets。",
        "- B：hold=5，reward ∈ {episode payload, Δpayload+service, Δpayload+service−delay}。",
        "- C：hold=5，B 的 delay shaping，buoy 特征增加 queue/distance/ETA/progress/persistence arrival（仍为 DeepSets，无图）。",
        "",
        "## 3. 结果",
        "",
        "| name | hold | reward | enhanced | payload | return | completion | service | option accuracy |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        _row({**state, "name": "State-only (6.3)", "hold": "—", "reward_mode": "—", "enhanced": False}),
        _row({**oracle, "name": "Oracle + frozen LL", "hold": 1, "reward_mode": "oracle", "enhanced": False}),
        _row({**base, "name": "Learned 6.3", "hold": 1, "reward_mode": "delta", "enhanced": False}),
    ]
    for item in payload["runs"]:
        lines.append(_row(item))
    lines.extend(
        [
            "",
            f"**最佳 6.4：** `{best['name']}` payload {best['mean_payload_bits']:.0f}，accuracy {best['option_selection_accuracy']:.3f}。",
            f"相对 State-only 差 {best['mean_payload_bits'] - state['mean_payload_bits']:.0f}；相对 6.3 learned accuracy {best['option_selection_accuracy'] - base['option_selection_accuracy']:+.3f}。",
            "",
            "## 4. 三个问题",
            "",
            "1. **原高层失败原因：** 每步重选 + 稀疏 Δpayload 奖励近视近簇可完成目标，学不到 Oracle 的跨簇长期分配。",
            f"2. **哪个改进有效：** 按 payload 最优为 `{best['name']}`（hold={best['hold']}，reward={best['reward_mode']}，enhanced={best['enhanced']}）。hold 拉长决策周期；service/delay shaping 对齐执行；增强观测提供队列与 ETA。",
            f"3. **是否准备加入 HGNN？{'准备作为下一阶段（本阶段仍未加入）' if ready else '暂不加入，继续分析高层决策'}。** 成功标准：Learned>State-only 且 accuracy 比 6.3 提高至少 {ACCURACY_GAIN}。当前：payload 过线={beats_state}，accuracy 过线={acc_up}。",
            "",
            "JSON：`src/wt/htw_phase64/outputs/high_level_improvement/summary.json`",
            "",
        ]
    )
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PHASE63_JSON.read_text(encoding="utf-8"))
    print("loading frozen low-level", flush=True)
    low = load_frozen_low()
    print("evaluating oracle + frozen low-level", flush=True)
    oracle = evaluate_hierarchical(low, None, EVAL_SEEDS, high_mode="oracle")
    oracle["family"] = "oracle_option"
    state = prior["state_only"]
    base = prior["learned_option"]
    runs: list[dict[str, Any]] = []
    for hold in HOLDS:
        print(f"A hold={hold}", flush=True)
        runs.append(train_high(low, hold=hold, reward_mode="delta", enhanced=False, name=f"A_hold{hold}"))
    for mode in ("episode", "delta_service", "delta_service_delay"):
        print(f"B reward={mode}", flush=True)
        runs.append(train_high(low, hold=5, reward_mode=mode, enhanced=False, name=f"B_{mode}"))
    print("C enhanced obs", flush=True)
    runs.append(train_high(low, hold=5, reward_mode="delta_service_delay", enhanced=True, name="C_enhanced_delay"))
    best = max(runs, key=lambda row: (float(row["mean_payload_bits"]), float(row["option_selection_accuracy"])))
    beats_state = float(best["mean_payload_bits"]) >= float(state["mean_payload_bits"]) + PAYLOAD_GAP
    acc_up = float(best["option_selection_accuracy"]) >= float(base["option_selection_accuracy"]) + ACCURACY_GAIN
    payload = {
        "updates": MINI_UPDATES,
        "seed": SEED,
        "state_only": state,
        "oracle_option": oracle,
        "phase63_learned": base,
        "runs": runs,
        "best": best,
        "beats_state_only": beats_state,
        "accuracy_improved": acc_up,
        "prepare_hgnn": bool(beats_state and acc_up),
        "uses_hgnn": False,
        "modified_low_level": False,
        "modified_env": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "best": result["best"]["name"],
                "best_payload": result["best"]["mean_payload_bits"],
                "best_accuracy": result["best"]["option_selection_accuracy"],
                "prepare_hgnn": result["prepare_hgnn"],
            },
            indent=2,
        )
    )
