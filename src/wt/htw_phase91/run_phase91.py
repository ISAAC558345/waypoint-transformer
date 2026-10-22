"""Phase-9.1: remove idle, 300 updates, MLP→HGNN imitation warm start."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase64.run_phase64 import high_config
from wt.htw_phase9.agent import GraphDiscretePPOAgent
from wt.htw_phase9.const import EVAL_SEEDS, HOLD, REWARD_MODE, SEED, TRAIN_SEEDS
from wt.htw_phase9.run_phase9 import evaluate_encoder
from wt.htw_phase9.vector import L3BHeldVector, collect_held
from wt.round30.ppo import seed_everything

from .models import HeteroOptionHGNN, MLPOptionModel, OPTION_DIM

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "hgnn_stabilization"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_1_HGNN_stabilization_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_1_HGNN_stabilization_report.md"
MINI_UPDATES = 300
IMITATE_STEPS = 512
IMITATE_EPOCHS = 40
IMITATE_BATCH = 64


def train_ppo(model, name: str, *, updates: int = MINI_UPDATES, agent: GraphDiscretePPOAgent | None = None) -> tuple[GraphDiscretePPOAgent, dict[str, Any]]:
    if agent is None:
        seed_everything(SEED)
        agent = GraphDiscretePPOAgent(model, high_config())
    vector = L3BHeldVector(TRAIN_SEEDS, agent.config.num_envs, 79100 + SEED, hold=HOLD, reward_mode=REWARD_MODE)
    last_health: dict[str, float] = {}
    for index in range(int(updates)):
        storage, _events, health = collect_held(agent, vector)
        update = agent.update(storage)
        last_health = {**health, **{key: float(value) for key, value in update.items() if isinstance(value, (int, float))}}
        if not all(np.isfinite(value) for value in last_health.values()):
            raise RuntimeError(f"Phase-9.1 NaN at {name} update {index}")
        print(f"{name} update {index + 1}/{updates} reward={health['mean_raw_reward']:.3f} entropy={health['mean_entropy']:.3f}", flush=True)
    evaluated = evaluate_encoder(agent, EVAL_SEEDS)
    evaluated["name"] = name
    evaluated["hold"] = HOLD
    evaluated["updates"] = updates
    evaluated["option_dim"] = OPTION_DIM
    evaluated["idle_allowed"] = False
    evaluated["environment_steps"] = vector.environment_steps
    evaluated["train_entropy"] = float(last_health.get("entropy", last_health.get("mean_entropy", float("nan"))))
    evaluated["last_update"] = last_health
    return agent, evaluated


def collect_teacher_batch(teacher: GraphDiscretePPOAgent, steps: int = IMITATE_STEPS) -> dict[str, np.ndarray]:
    vector = L3BHeldVector(TRAIN_SEEDS, teacher.config.num_envs, 79200 + SEED, hold=HOLD, reward_mode=REWARD_MODE)
    keys = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
    rows = {key: [] for key in keys}
    labels = []
    features = vector.features()
    for _ in range(int(steps)):
        decision = teacher.act(features, deterministic=True)
        for key in keys:
            rows[key].append(features[key])
        labels.append(decision["z"])
        features, _rewards, _dones, _done = vector.step(decision["z"])
    packed = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    packed["z"] = np.concatenate(labels, axis=0)
    return packed


def imitate_hgnn(teacher: GraphDiscretePPOAgent, student: GraphDiscretePPOAgent) -> dict[str, float]:
    data = collect_teacher_batch(teacher)
    count = int(data["z"].shape[0])
    device = student.config.device
    losses, matches = [], []
    student.model.train()
    for epoch in range(IMITATE_EPOCHS):
        order = np.random.permutation(count)
        for start in range(0, count, IMITATE_BATCH):
            index = order[start : start + IMITATE_BATCH]
            batch = {}
            for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
                dtype = torch.bool if key.endswith("mask") else torch.float32
                batch[key] = torch.as_tensor(data[key][index], dtype=dtype, device=device)
            z = torch.as_tensor(data["z"][index], dtype=torch.int64, device=device)
            logits, _value = student.model(
                batch["uav"],
                batch["buoy"],
                batch["uav_mask"],
                batch["buoy_mask"],
                batch["global"],
                batch["edge"],
                batch["edge_uu"],
            )
            logp = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(logp.reshape(-1, OPTION_DIM), z.reshape(-1), reduction="none")
            weights = batch["uav_mask"].reshape(-1).to(nll.dtype)
            loss = (nll * weights).sum() / weights.sum().clamp_min(1.0)
            student.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.model.parameters(), student.config.max_grad_norm)
            student.optimizer.step()
            pred = torch.argmax(logits, dim=-1)
            match = ((pred == z) & batch["uav_mask"]).sum() / batch["uav_mask"].sum().clamp_min(1)
            losses.append(float(loss.detach().cpu()))
            matches.append(float(match.detach().cpu()))
        print(f"imitation epoch {epoch + 1}/{IMITATE_EPOCHS} loss={np.mean(losses[-max(1, count // IMITATE_BATCH):]):.4f}", flush=True)
    return {"imitation_loss": float(np.mean(losses)), "imitation_accuracy": float(np.mean(matches[-max(1, count // IMITATE_BATCH) :]))}


def write_report(payload: Mapping[str, Any]) -> None:
    mlp, rand, warm = payload["mlp"], payload["hgnn_random"], payload["hgnn_warm"]
    ratio = float(warm["mean_payload_bits"]) / max(float(mlp["mean_payload_bits"]), 1.0)
    success = bool(float(warm["mean_payload_bits"]) >= 0.80 * float(mlp["mean_payload_bits"]))
    lines = [
        "# HTW Phase-9.1 HGNN Training Stabilization Report",
        "",
        "修正 Phase-9 的全体 idle 崩溃。未改 L3-B、SINR、低层 PPO、hold=15。未改 Phase-9 代码与结果。",
        "低层仍为 fly-toward。高层动作 **去掉 idle**：`z ∈ {0,…,11}`，每机必须选一个 buoy。",
        f"seed={SEED}，PPO updates={MINI_UPDATES}，reward=`delta`。",
        "",
        "## 1. 失败诊断（Phase-9）",
        "",
        "DeepSets/HGNN greedy eval payload=0，argmax 逃到 idle。不能据此判 HGNN 无效。",
        "",
        "## 2. 稳定化设置",
        "",
        "1. 动作空间：无 idle。",
        "2. 训练预算：300 updates（不是 80）。",
        "3. Warm start：冻结 MLP 当教师，交叉熵监督 HGNN option，再 PPO fine-tune。",
        "",
        "## 3. 结果（eval seeds 8–15）",
        "",
        "| 方法 | payload | option accuracy | entropy | conflict | interference utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| MLP baseline | {mlp['mean_payload_bits']:.0f} | {mlp['option_selection_accuracy']:.3f} | {mlp['high_level_entropy']:.3f} | {mlp['conflict_rate']:.3f} | {mlp['interference_utility']:.3e} |",
        f"| HGNN random init | {rand['mean_payload_bits']:.0f} | {rand['option_selection_accuracy']:.3f} | {rand['high_level_entropy']:.3f} | {rand['conflict_rate']:.3f} | {rand['interference_utility']:.3e} |",
        f"| HGNN imitation warm start | {warm['mean_payload_bits']:.0f} | {warm['option_selection_accuracy']:.3f} | {warm['high_level_entropy']:.3f} | {warm['conflict_rate']:.3f} | {warm['interference_utility']:.3e} |",
        "",
        f"模仿阶段准确率 = {payload['imitation']['imitation_accuracy']:.3f}。",
        f"Warm / MLP payload 比 = {ratio:.2%}。",
        "",
        "## 4. 成功标准",
        "",
        f"**HGNN warm start 接近或超过 MLP（≥80% payload）：{'成立' if success else '不成立'}**",
        "",
        f"- 若成立：HGNN 具备关系建模潜力，失败主要是优化/idle 而不是图无用。",
        f"- 若仍失败：需要重新设计 HGNN 输出（例如排他 matching 头、匈牙利层、或显式冲突惩罚），而不是继续加 updates。",
        "",
        "补充：去掉 idle 后 MLP 可学。随机 HGNN 仍高熵、高冲突。模仿准确率高说明 HGNN 能拟合 MLP option；随后 PPO 易冲掉 teacher。下一刀应改 HGNN 输出/优化接口，而不是再加 updates。",
        "",
        "JSON：`src/wt/htw_phase91/outputs/hgnn_stabilization/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("training MLP 300", flush=True)
    mlp_agent, mlp = train_ppo(MLPOptionModel(), "mlp")
    print("training HGNN random 300", flush=True)
    _rand_agent, rand = train_ppo(HeteroOptionHGNN(), "hgnn_random")
    print("HGNN imitation warm start", flush=True)
    seed_everything(SEED)
    warm_agent = GraphDiscretePPOAgent(HeteroOptionHGNN(), high_config())
    imitation = imitate_hgnn(mlp_agent, warm_agent)
    print("PPO fine-tune warm HGNN 300", flush=True)
    _warm_agent, warm = train_ppo(warm_agent.model, "hgnn_warm", agent=warm_agent)
    warm["imitation"] = imitation
    payload = {
        "hold": HOLD,
        "updates": MINI_UPDATES,
        "seed": SEED,
        "idle_allowed": False,
        "option_dim": OPTION_DIM,
        "mlp": mlp,
        "hgnn_random": rand,
        "hgnn_warm": warm,
        "imitation": imitation,
        "warm_over_mlp": float(warm["mean_payload_bits"]) / max(float(mlp["mean_payload_bits"]), 1.0),
        "success_warm_near_mlp": bool(float(warm["mean_payload_bits"]) >= 0.80 * float(mlp["mean_payload_bits"])),
        "modified_l3b": False,
        "modified_low_level_ppo": False,
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
                "hgnn_random_payload": result["hgnn_random"]["mean_payload_bits"],
                "hgnn_warm_payload": result["hgnn_warm"]["mean_payload_bits"],
                "imitation_accuracy": result["imitation"]["imitation_accuracy"],
                "success_warm_near_mlp": result["success_warm_near_mlp"],
            },
            indent=2,
        )
    )
