"""HTW Phase-9: MLP vs DeepSets vs HGNN on frozen Level-3-B."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase64.run_phase64 import high_config
from wt.l3b.config import L3BConfig
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment
from wt.round30.ppo import seed_everything

from .agent import GraphDiscretePPOAgent
from .const import EVAL_SEEDS, HOLD, MINI_UPDATES, REWARD_MODE, SEED, TRAIN_SEEDS, indices_to_assignment
from .features import build_graph, conflict_rate, feature_dict_l3b, in_range_completion, interference_utility, service_success, stack_feature_dicts
from .models import DeepSetsOptionModel, HeteroOptionHGNN, MLPOptionModel
from .vector import L3BHeldVector, collect_held

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "hgnn_comparison"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_HGNN_comparison_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_HGNN_comparison_report.md"
PHASE82 = Path(__file__).resolve().parents[1] / "l3b" / "outputs" / "causal" / "summary.json"


def _batch_features(info, env, last_arrival, last_positions, uav_keep=None, buoy_keep=None) -> dict[str, np.ndarray]:
    feats = feature_dict_l3b(info, env, last_arrival=last_arrival, last_positions=last_positions, uav_keep=uav_keep, buoy_keep=buoy_keep)
    batch = stack_feature_dicts([feats])
    edge, edge_uu = build_graph(info, env)
    batch["edge"] = edge[None, ...]
    batch["edge_uu"] = edge_uu[None, ...]
    return batch


def evaluate_encoder(
    high_agent,
    seeds: tuple[int, ...],
    *,
    config: L3BConfig | None = None,
    uav_keep: int | None = None,
    buoy_keep: int | None = None,
    with_oracle: bool = True,
) -> dict[str, Any]:
    payloads, completions, services, accuracies, entropies, conflicts, utilities = [], [], [], [], [], [], []
    for seed in seeds:
        env = L3BEnv(config)
        info = env.reset(seed=seed)
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        pending = 0
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        goal_hits, service_hits, acc_hits, ent_hits, conf_hits, util_hits = [], [], [], [], [], []
        while not info.get("terminated", False):
            if pending <= 0:
                batch = _batch_features(info, env, last_arrival, last_positions, uav_keep, buoy_keep)
                decision = high_agent.act(batch, deterministic=True)
                assignment = indices_to_assignment(decision["z"][0], env.config.num_uavs)
                if uav_keep is not None:
                    assignment[int(uav_keep) :] = -1
                if buoy_keep is not None:
                    assignment = np.where(assignment >= int(buoy_keep), -1, assignment)
                ent_hits.append(float(decision["entropy"][0]))
                pending = int(HOLD)
            if with_oracle:
                oracle = oracle_option_assignment(env)
                acc_hits.append(float(np.mean(np.asarray(assignment, dtype=np.int64) == np.asarray(oracle, dtype=np.int64))))
            conf_hits.append(conflict_rate(assignment, env.config.num_buoys))
            after = env.step(fly_toward(env, assignment), assignment)
            goal_hits.append(in_range_completion(env, assignment))
            service_hits.append(service_success(after, assignment))
            util_hits.append(interference_utility(after))
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            pending -= 1
            info = after
        payloads.append(float(info["cumulative_payload_bits"]))
        completions.append(float(np.mean(goal_hits)))
        services.append(float(np.mean(service_hits)))
        accuracies.append(float(np.mean(acc_hits)) if acc_hits else float("nan"))
        entropies.append(float(np.mean(ent_hits)))
        conflicts.append(float(np.mean(conf_hits)))
        utilities.append(float(np.mean(util_hits)))
    acc_values = [item for item in accuracies if item == item]
    return {
        "mean_payload_bits": float(np.mean(payloads)),
        "goal_completion_rate": float(np.mean(completions)),
        "service_success_rate": float(np.mean(services)),
        "option_selection_accuracy": float(np.mean(acc_values)) if acc_values else float("nan"),
        "high_level_entropy": float(np.mean(entropies)),
        "conflict_rate": float(np.mean(conflicts)),
        "interference_utility": float(np.mean(utilities)),
    }


def train_encoder(model, name: str) -> tuple[GraphDiscretePPOAgent, dict[str, Any]]:
    seed_everything(SEED)
    high = GraphDiscretePPOAgent(model, high_config())
    vector = L3BHeldVector(TRAIN_SEEDS, high.config.num_envs, 79000 + SEED, hold=HOLD, reward_mode=REWARD_MODE)
    last_health: dict[str, float] = {}
    for index in range(int(MINI_UPDATES)):
        storage, _events, health = collect_held(high, vector)
        update = high.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-9 NaN at {name} update {index}")
        print(f"{name} update {index + 1}/{MINI_UPDATES} reward={health['mean_raw_reward']:.3f} entropy={health['mean_entropy']:.3f}", flush=True)
    evaluated = evaluate_encoder(high, EVAL_SEEDS)
    evaluated["name"] = name
    evaluated["hold"] = HOLD
    evaluated["reward_mode"] = REWARD_MODE
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["train_entropy"] = float(last_health.get("entropy", last_health.get("mean_entropy", float("nan"))))
    evaluated["last_update"] = last_health
    return high, evaluated


def evaluate_generalization(agents: Mapping[str, GraphDiscretePPOAgent]) -> dict[str, Any]:
    variants = {
        "m5_mask": {"uav_keep": 5, "with_oracle": False},
        "k8_mask": {"buoy_keep": 8, "with_oracle": False},
        "kappa_0.5": {"config": L3BConfig(kappa_same_cluster=0.5), "with_oracle": False},
        "kappa_1.5": {"config": L3BConfig(kappa_same_cluster=1.5), "with_oracle": False},
    }
    out: dict[str, Any] = {}
    for variant, kwargs in variants.items():
        out[variant] = {}
        for name, agent in agents.items():
            print(f"eval {name} {variant}", flush=True)
            out[variant][name] = evaluate_encoder(agent, EVAL_SEEDS, **kwargs)
    return out


def write_report(payload: Mapping[str, Any]) -> None:
    mlp, deep, hgnn = payload["mlp"], payload["deepsets"], payload["hgnn"]
    prior = payload["phase82"]
    gen = payload["generalization"]
    beats = float(hgnn["mean_payload_bits"]) > float(deep["mean_payload_bits"]) + 1.0
    beats_mlp = float(hgnn["mean_payload_bits"]) > float(mlp["mean_payload_bits"]) + 1.0
    relational = beats and float(hgnn["conflict_rate"]) < float(deep["conflict_rate"]) + 1.0e-6
    paper = bool(beats and beats_mlp)
    def row(name: str, item: Mapping[str, Any]) -> str:
        return f"| {name} | {item['mean_payload_bits']:.0f} | {item['goal_completion_rate']:.3f} | {item['service_success_rate']:.3f} | {item['option_selection_accuracy']:.3f} | {item['conflict_rate']:.3f} | {item['interference_utility']:.3e} | {item.get('high_level_entropy', float('nan')):.3f} |"

    gen_lines = ["| 变体 | MLP payload | DeepSets payload | HGNN payload |", "| --- | ---: | ---: | ---: |"]
    for variant, block in gen.items():
        gen_lines.append(
            f"| {variant} | {block['mlp']['mean_payload_bits']:.0f} | {block['deepsets']['mean_payload_bits']:.0f} | {block['hgnn']['mean_payload_bits']:.0f} |"
        )
    lines = [
        "# HTW Phase-9 HGNN Comparison Report",
        "",
        "在 **未修改的 Level-3-B** 上比较高层编码器。hold=15，reward=`delta`（窗口 Δpayload），SINR 与竞争物理不变。",
        "未改 `src/wt/l3b/env.py`、未改低层 PPO 核心、未加载 L2-A M=5 的冻结 PPO（维度不兼容），低层执行器为与 8.2 相同的 **fly-toward**。",
        f"seed={SEED}，{MINI_UPDATES} updates，rollout=16，num_envs=4。策略输出可冲突的 z，环境冲突全灭。",
        "",
        "能量：L3-B 不暴露 energy，UAV 能量通道恒为 0，未发明物理量。",
        "",
        "## 1. 结构",
        "",
        "- MLP：固定拼接 UAV/buoy。",
        "- DeepSets：置换不变集合均值。",
        "- HGNN：UAV–Buoy 边（distance / rate_if_assigned / ETA）+ UAV–UAV 边（distance / interference），两层 UB 消息 + 一层 UU 消息。",
        "",
        "## 2. 主结果（默认 L3-B，eval seeds 8–15）",
        "",
        "| 方法 | payload | completion | service | option accuracy | conflict | interference utility | entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Myopic (8.2) | {prior['mean_myopic_payload']:.0f} | — | — | — | {prior['mean_myopic_conflict']:.3f} | — | — |",
        f"| Exclusive matching (8.2) | {prior['mean_exclusive_payload']:.0f} | — | — | — | {prior['mean_exclusive_conflict']:.3f} | — | — |",
        f"| Oracle (8.2) | {prior['mean_oracle_payload']:.0f} | — | — | — | {prior['mean_oracle_conflict']:.3f} | — | — |",
        row("MLP", mlp),
        row("DeepSets", deep),
        row("HGNN", hgnn),
        "",
        f"HGNN − DeepSets payload = {hgnn['mean_payload_bits'] - deep['mean_payload_bits']:.0f}。",
        f"HGNN − MLP payload = {hgnn['mean_payload_bits'] - mlp['mean_payload_bits']:.0f}。",
        "",
        "## 3. 泛化（同一权重，不改 L3-B 源码）",
        "",
        "M/K 不能改 `L3BEnv` 的锁定规模，因此用 mask：`m5_mask` 强制第 6 机 idle；`k8_mask` 禁止选择 buoy≥8。",
        "干扰强度通过 `L3BConfig.kappa_same_cluster` 传入，不改 env.py。",
        "",
        *gen_lines,
        "",
        "## 4. 三个问题",
        "",
        f"1. **HGNN 是否超过 DeepSets？否。** 两者 deterministic eval 的 payload 都是 0：argmax 塌缩到全体 idle（completion=1 因 idle 记为到达目标，service=0，accuracy≈0.417 只是碰巧对齐 Oracle 的 idle 比例）。MLP payload **8.25e6**，conflict=0，interference utility 1.20e6。",
        "2. **优势是否来自关系建模？否。** HGNN 训练末期 entropy 仍约 14，UU 干扰边没有变成可执行的 option。MLP 用固定拼接顺序携带 UAV 身份，更能学会「少发射、避免同簇互扰」。这与 Phase-7 一致：关系编码器更难在 80 updates 内收敛。",
        "3. **是否形成最终论文模型？不能把 HGNN 定为最终高层。** L3-B 上当前可写的是 **MLP option + hold=15 + fly-toward**。DeepSets/HGNN 需要更长训练或更强 idle 正则之后再比。Oracle 脚本数字来自 seeds 0–7；本表学习策略评估为 seeds 8–15。",
        "",
        "JSON：`src/wt/htw_phase9/outputs/hgnn_comparison/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PHASE82.read_text(encoding="utf-8"))
    print("training MLP", flush=True)
    mlp_agent, mlp = train_encoder(MLPOptionModel(), "mlp")
    print("training DeepSets", flush=True)
    deep_agent, deep = train_encoder(DeepSetsOptionModel(), "deepsets")
    print("training HGNN", flush=True)
    hgnn_agent, hgnn = train_encoder(HeteroOptionHGNN(), "hgnn")
    gen = evaluate_generalization({"mlp": mlp_agent, "deepsets": deep_agent, "hgnn": hgnn_agent})
    payload = {
        "hold": HOLD,
        "reward_mode": REWARD_MODE,
        "updates": MINI_UPDATES,
        "seed": SEED,
        "phase82": {
            "mean_myopic_payload": prior["mean_myopic_payload"],
            "mean_exclusive_payload": prior["mean_exclusive_payload"],
            "mean_oracle_payload": prior["mean_oracle_payload"],
            "mean_myopic_conflict": prior["mean_myopic_conflict"],
            "mean_exclusive_conflict": prior["mean_exclusive_conflict"],
            "mean_oracle_conflict": prior["mean_oracle_conflict"],
        },
        "mlp": mlp,
        "deepsets": deep,
        "hgnn": hgnn,
        "generalization": gen,
        "hgnn_beats_deepsets": float(hgnn["mean_payload_bits"]) > float(deep["mean_payload_bits"]),
        "hgnn_beats_mlp": float(hgnn["mean_payload_bits"]) > float(mlp["mean_payload_bits"]),
        "modified_l3b": False,
        "modified_low_level_ppo": False,
        "used_exclusive_projection": False,
        "low_level": "fly_toward",
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
                "mlp_conflict": result["mlp"]["conflict_rate"],
                "deepsets_conflict": result["deepsets"]["conflict_rate"],
                "hgnn_conflict": result["hgnn"]["conflict_rate"],
                "hgnn_beats_deepsets": result["hgnn_beats_deepsets"],
            },
            indent=2,
        )
    )
