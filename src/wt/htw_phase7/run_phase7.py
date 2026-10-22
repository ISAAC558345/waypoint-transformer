"""HTW Phase-7: HGNN high-level option policy. Frozen low-level, hold=15, delta reward."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wt.htw_phase61.features import BUOY_DIM, attach_option, encode_option, feature_dict_l2a, in_range_completion, service_success, stack_feature_dicts
from wt.htw_phase61.run_phase61 import EVAL_SEEDS, MINI_UPDATES, SEED, TRAIN_SEEDS, _plain
from wt.htw_phase63.high_ppo import indices_to_assignment
from wt.htw_phase64.run_phase64 import high_config, load_frozen_low
from wt.l2a.env import L2AEnv, project_exclusive
from wt.l2a.policies import oracle_option_assignment
from wt.round30.ppo import seed_everything

from .agent import GraphDiscretePPOAgent
from .graph import build_edge_features
from .models import DeepSetsOptionModel, HeteroOptionHGNN, MLPOptionModel
from .vector import GraphHeldVector, collect_graph_held

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "hgnn_option"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase7_HGNN_option_policy_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase7_HGNN_option_policy_report.md"
PHASE64_JSON = Path(__file__).resolve().parents[1] / "htw_phase64" / "outputs" / "high_level_improvement" / "summary.json"
HOLD = 15
REWARD_MODE = "delta"


def evaluate_encoder(low_agent, high_agent, seeds: tuple[int, ...]) -> dict[str, Any]:
    payloads, returns, completions, services, accuracies, entropies = [], [], [], [], [], []
    for seed in seeds:
        env = L2AEnv()
        info = env.reset(seed=seed)
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        total_return = 0.0
        goal_hits, service_hits, acc_hits, ent_hits = [], [], [], []
        pending = 0
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        while not info.get("terminated", False):
            if pending <= 0:
                from wt.htw_phase64.features import feature_dict_high

                feats = feature_dict_high(info, env, last_arrival=last_arrival, enhanced=False)
                batch = stack_feature_dicts([feats])
                batch["_uav_positions"] = env.uav_positions[None, ...]
                batch["_buoy_positions"] = env.buoy_positions[None, ...]
                batch["edge"] = build_edge_features(info, env)[None, ...]
                decision = high_agent.act(batch, deterministic=True)
                assignment = project_exclusive(indices_to_assignment(decision["z"][0]), env.uav_positions, env.buoy_positions)
                ent_hits.append(float(decision["entropy"][0]))
                pending = int(HOLD)
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
        entropies.append(float(np.mean(ent_hits)))
    return {
        "mean_payload_bits": float(np.mean(payloads)),
        "mean_return": float(np.mean(returns)),
        "goal_completion_rate": float(np.mean(completions)),
        "service_success_rate": float(np.mean(services)),
        "option_selection_accuracy": float(np.mean(accuracies)),
        "high_level_entropy": float(np.mean(entropies)),
    }


def train_encoder(low_agent, model, name: str) -> dict[str, Any]:
    seed_everything(SEED)
    high = GraphDiscretePPOAgent(model, high_config())
    vector = GraphHeldVector(TRAIN_SEEDS, high.config.num_envs, 78000 + SEED, low_agent, hold=HOLD, reward_mode=REWARD_MODE, enhanced=False)
    last_health: dict[str, float] = {}
    for index in range(int(MINI_UPDATES)):
        storage, _events, health = collect_graph_held(high, vector)
        update = high.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-7 NaN at {name} update {index}")
        print(f"{name} update {index + 1}/{MINI_UPDATES} reward={health['mean_raw_reward']:.3f} entropy={health['mean_entropy']:.3f}", flush=True)
    evaluated = evaluate_encoder(low_agent, high, EVAL_SEEDS)
    evaluated["name"] = name
    evaluated["hold"] = HOLD
    evaluated["reward_mode"] = REWARD_MODE
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["train_entropy"] = float(last_health.get("entropy", last_health.get("mean_entropy", float("nan"))))
    evaluated["last_update"] = last_health
    return evaluated


def write_report(payload: Mapping[str, Any]) -> None:
    mlp, deep, hgnn = payload["mlp"], payload["deepsets"], payload["hgnn"]
    state = payload["state_only"]
    oracle = payload["oracle_option"]
    hold15 = payload["phase64_hold15"]
    better_payload = float(hgnn["mean_payload_bits"]) > float(deep["mean_payload_bits"]) + 1.0
    better_acc = float(hgnn["option_selection_accuracy"]) > float(deep["option_selection_accuracy"]) + 1.0e-4
    paper = bool(better_payload and better_acc)
    lines = [
        "# HTW Phase-7 HGNN High-level Option Policy",
        "",
        "HGNN 只编码高层 option。低层仍是冻结的 `[state,z]→movement` PPO。hold=15，reward=`delta`，与 Phase-6.4 最佳设置相同。未改环境、未改低层、未用 Task-Waypoint。",
        f"相同 seed={SEED}，{MINI_UPDATES} updates，rollout=16，num_envs=4。",
        "",
        "## 1. 结构",
        "",
        "- MLP：把 padded UAV/buoy 拼成向量再出每机 13 维 logits。",
        "- DeepSets：与 6.4 相同的置换不变集合编码器。",
        "- HGNN：UAV–Buoy 二部图两层消息传递。边特征为 distance / ETA / rate 代理 / 邻机干扰。",
        "",
        "## 2. 结果",
        "",
        "| 方法 | payload | completion | service | option accuracy | eval entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| State-only (6.3) | {state['mean_payload_bits']:.0f} | {state['goal_completion_rate']:.3f} | {state['service_success_rate']:.3f} | {state['option_selection_accuracy']:.3f} | — |",
        f"| Oracle + frozen LL | {oracle['mean_payload_bits']:.0f} | {oracle['goal_completion_rate']:.3f} | {oracle['service_success_rate']:.3f} | {oracle['option_selection_accuracy']:.3f} | — |",
        f"| DeepSets hold=15 (6.4) | {hold15['mean_payload_bits']:.0f} | {hold15['goal_completion_rate']:.3f} | {hold15['service_success_rate']:.3f} | {hold15['option_selection_accuracy']:.3f} | — |",
        f"| MLP High-level | {mlp['mean_payload_bits']:.0f} | {mlp['goal_completion_rate']:.3f} | {mlp['service_success_rate']:.3f} | {mlp['option_selection_accuracy']:.3f} | {mlp['high_level_entropy']:.3f} |",
        f"| DeepSets High-level | {deep['mean_payload_bits']:.0f} | {deep['goal_completion_rate']:.3f} | {deep['service_success_rate']:.3f} | {deep['option_selection_accuracy']:.3f} | {deep['high_level_entropy']:.3f} |",
        f"| HGNN High-level | {hgnn['mean_payload_bits']:.0f} | {hgnn['goal_completion_rate']:.3f} | {hgnn['service_success_rate']:.3f} | {hgnn['option_selection_accuracy']:.3f} | {hgnn['high_level_entropy']:.3f} |",
        "",
        f"HGNN − DeepSets payload = {hgnn['mean_payload_bits'] - deep['mean_payload_bits']:.0f}；accuracy 差 = {hgnn['option_selection_accuracy'] - deep['option_selection_accuracy']:+.3f}。",
        "",
        "## 3. 三个问题",
        "",
        f"1. **HGNN 是否提升高层任务选择？{'是' if paper else '否（本预算下未同时超过 DeepSets 的 payload 与 accuracy）'}。** 成功标准要求 payload>DeepSets 且 option accuracy 提高。",
        "2. **图结构优势来自哪里？** 若成立，应来自 UAV–Buoy 边上的 distance/ETA/rate/interference 消息，而不是节点 DeepSets 均值；MLP 则把集合顺序当成固定 ID。",
        f"3. **是否形成最终论文模型？{'可以作为高层编码器候选写入论文设定' if paper else '还不能定为最终论文模型'}。** 低层保持 6.2 冻结 PPO；HGNN 未接入 movement。",
        "",
        "JSON：`src/wt/htw_phase7/outputs/hgnn_option/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PHASE64_JSON.read_text(encoding="utf-8"))
    hold15 = next(item for item in prior["runs"] if item["name"] == "A_hold15")
    print("loading frozen low-level", flush=True)
    low = load_frozen_low()
    print("training MLP", flush=True)
    mlp = train_encoder(low, MLPOptionModel(BUOY_DIM), "mlp")
    print("training DeepSets", flush=True)
    deep = train_encoder(low, DeepSetsOptionModel(BUOY_DIM), "deepsets")
    print("training HGNN", flush=True)
    hgnn = train_encoder(low, HeteroOptionHGNN(BUOY_DIM), "hgnn")
    payload = {
        "hold": HOLD,
        "reward_mode": REWARD_MODE,
        "updates": MINI_UPDATES,
        "seed": SEED,
        "state_only": prior["state_only"],
        "oracle_option": prior["oracle_option"],
        "phase64_hold15": hold15,
        "mlp": mlp,
        "deepsets": deep,
        "hgnn": hgnn,
        "hgnn_beats_deepsets_payload": float(hgnn["mean_payload_bits"]) > float(deep["mean_payload_bits"]),
        "hgnn_beats_deepsets_accuracy": float(hgnn["option_selection_accuracy"]) > float(deep["option_selection_accuracy"]),
        "uses_hgnn_high_level": True,
        "uses_hgnn_low_level": False,
        "modified_env": False,
        "modified_low_level": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "mlp_payload": result["mlp"]["mean_payload_bits"],
                "deepsets_payload": result["deepsets"]["mean_payload_bits"],
                "hgnn_payload": result["hgnn"]["mean_payload_bits"],
                "mlp_acc": result["mlp"]["option_selection_accuracy"],
                "deepsets_acc": result["deepsets"]["option_selection_accuracy"],
                "hgnn_acc": result["hgnn"]["option_selection_accuracy"],
            },
            indent=2,
        )
    )
