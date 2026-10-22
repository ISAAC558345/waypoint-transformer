"""Phase-9.2: supervised assignment heads. No direct PPO. Does not edit L3-B."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase9.const import EVAL_SEEDS, HOLD, SEED, TRAIN_SEEDS
from wt.htw_phase9.features import build_graph, conflict_rate, feature_dict_l3b, interference_utility, stack_feature_dicts
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment
from wt.round30.ppo import seed_everything

from .matching import match_uav_buoy
from .models import DeepSetsAssignmentHead, HGNNAssignmentHead, IDLE_INDEX, MLPOptionClassifier

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "assignment_head"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_2_HGNN_assignment_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_2_HGNN_assignment_report.md"
COLLECT_SEEDS = tuple(list(TRAIN_SEEDS) + list(range(16, 24)))
EPOCHS = 80
BATCH = 32
LR = 1.0e-3


def _batch_from_env(info, env, last_arrival, last_positions) -> dict[str, np.ndarray]:
    feats = feature_dict_l3b(info, env, last_arrival=last_arrival, last_positions=last_positions)
    batch = stack_feature_dicts([feats])
    edge, edge_uu = build_graph(info, env)
    batch["edge"] = edge[None, ...]
    batch["edge_uu"] = edge_uu[None, ...]
    return batch


def collect_oracle_dataset(seeds: tuple[int, ...]) -> dict[str, np.ndarray]:
    keys = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
    rows = {key: [] for key in keys}
    labels = []
    for seed in seeds:
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        while not info.get("terminated", False):
            oracle = oracle_option_assignment(env)
            batch = _batch_from_env(info, env, last_arrival, last_positions)
            for key in keys:
                rows[key].append(batch[key][0])
            labels.append(np.asarray(oracle, dtype=np.int64))
            after = env.step(fly_toward(env, oracle), oracle)
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            info = after
    packed = {key: np.stack(value, axis=0) for key, value in rows.items()}
    packed["assignment"] = np.stack(labels, axis=0)
    return packed


def _targets(assignment: np.ndarray) -> np.ndarray:
    target = np.full(assignment.shape, IDLE_INDEX, dtype=np.int64)
    target = np.where(assignment >= 0, assignment, target)
    return target


def supervised_loss(model, batch: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    scores, idle = model(
        batch["uav"],
        batch["buoy"],
        batch["uav_mask"],
        batch["buoy_mask"],
        batch["global"],
        batch["edge"],
        batch["edge_uu"],
    )
    logits = torch.cat((scores, idle.unsqueeze(-1)), dim=-1)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    weights = batch["uav_mask"].reshape(-1).to(nll.dtype)
    return (nll * weights).sum() / weights.sum().clamp_min(1.0)


def train_model(model: torch.nn.Module, data: dict[str, np.ndarray], name: str) -> torch.nn.Module:
    seed_everything(SEED)
    device = "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    count = int(data["assignment"].shape[0])
    target_np = _targets(data["assignment"])
    model.train()
    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        losses = []
        for start in range(0, count, BATCH):
            index = order[start : start + BATCH]
            batch = {}
            for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
                dtype = torch.bool if key.endswith("mask") else torch.float32
                batch[key] = torch.as_tensor(data[key][index], dtype=dtype, device=device)
            target = torch.as_tensor(target_np[index], dtype=torch.int64, device=device)
            loss = supervised_loss(model, batch, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"{name} epoch {epoch + 1}/{EPOCHS} loss={float(np.mean(losses)):.4f}", flush=True)
    return model


def decode_assignment(name: str, scores: np.ndarray, idle: np.ndarray, uav_mask: np.ndarray, buoy_mask: np.ndarray) -> np.ndarray:
    if name == "mlp":
        logits = np.concatenate((scores, idle[:, None]), axis=-1)
        logits = np.where(uav_mask[:, None], logits, -1.0e9)
        choice = np.argmax(logits, axis=-1)
        assignment = np.where(choice >= IDLE_INDEX, -1, choice)
        assignment = np.where(uav_mask, assignment, -1)
        return assignment.astype(np.int64)
    return match_uav_buoy(scores, idle, buoy_mask, uav_mask)


def evaluate_model(model: torch.nn.Module, name: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    model.eval()
    payloads, accs, conflicts, utilities, services = [], [], [], [], []
    for seed in seeds:
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        pending = 0
        assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
        acc_hits, conf_hits, util_hits, svc_hits = [], [], [], []
        while not info.get("terminated", False):
            if pending <= 0:
                batch_np = _batch_from_env(info, env, last_arrival, last_positions)
                tensors = {}
                for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
                    dtype = torch.bool if key.endswith("mask") else torch.float32
                    tensors[key] = torch.as_tensor(batch_np[key], dtype=dtype)
                with torch.no_grad():
                    scores, idle = model(
                        tensors["uav"],
                        tensors["buoy"],
                        tensors["uav_mask"],
                        tensors["buoy_mask"],
                        tensors["global"],
                        tensors["edge"],
                        tensors["edge_uu"],
                    )
                assignment = decode_assignment(
                    name,
                    scores[0].cpu().numpy(),
                    idle[0].cpu().numpy(),
                    batch_np["uav_mask"][0],
                    batch_np["buoy_mask"][0],
                )
                pending = int(HOLD)
            oracle = oracle_option_assignment(env)
            acc_hits.append(float(np.mean(np.asarray(assignment) == np.asarray(oracle))))
            conf_hits.append(conflict_rate(assignment, env.config.num_buoys))
            after = env.step(fly_toward(env, assignment), assignment)
            from wt.htw_phase9.features import service_success

            svc_hits.append(service_success(after, assignment))
            util_hits.append(interference_utility(after))
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            pending -= 1
            info = after
        payloads.append(float(info["cumulative_payload_bits"]))
        accs.append(float(np.mean(acc_hits)))
        conflicts.append(float(np.mean(conf_hits)))
        utilities.append(float(np.mean(util_hits)))
        services.append(float(np.mean(svc_hits)))
    return {
        "mean_payload_bits": float(np.mean(payloads)),
        "assignment_accuracy": float(np.mean(accs)),
        "conflict_rate": float(np.mean(conflicts)),
        "interference_utility": float(np.mean(utilities)),
        "service_success_rate": float(np.mean(services)),
        "hold": HOLD,
        "name": name,
    }


def write_report(payload: Mapping[str, Any]) -> None:
    mlp, deep, hgnn = payload["mlp"], payload["deepsets"], payload["hgnn"]
    beats = float(hgnn["mean_payload_bits"]) > float(deep["mean_payload_bits"]) + 1.0
    lines = [
        "# HTW Phase-9.2 HGNN Assignment Head Report",
        "",
        "HGNN 不再做独立 buoy 分类。边分 `S(i,j)` 经 **Hungarian matching** 得到排他 assignment。",
        "未改 L3-B、SINR、低层 PPO、hold=15。未直接 PPO；标签为 Oracle assignment。低层 fly-toward。",
        f"监督：collect seeds {payload['collect_seeds']}，{EPOCHS} epochs。Eval seeds 8–15。",
        "Buoy 容量=1（L3-B 竞争）。允许 idle（每机一条 dummy 列）。",
        "",
        "## 1. 结构",
        "",
        "- MLP：独立 categorical（buoy+idle），argmax，可冲突。",
        "- DeepSets：UAV/Buoy/边编码 → `S(i,j)` + idle → Hungarian。",
        "- HGNN：UB 消息 + UU 干扰边 → `S(i,j)` + idle → Hungarian。",
        "",
        "## 2. 结果",
        "",
        "| 方法 | payload | assignment accuracy | conflict | SINR utility (mean rate) |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| MLP option classification | {mlp['mean_payload_bits']:.0f} | {mlp['assignment_accuracy']:.3f} | {mlp['conflict_rate']:.3f} | {mlp['interference_utility']:.3e} |",
        f"| DeepSets assignment head | {deep['mean_payload_bits']:.0f} | {deep['assignment_accuracy']:.3f} | {deep['conflict_rate']:.3f} | {deep['interference_utility']:.3e} |",
        f"| HGNN edge scoring + matching | {hgnn['mean_payload_bits']:.0f} | {hgnn['assignment_accuracy']:.3f} | {hgnn['conflict_rate']:.3f} | {hgnn['interference_utility']:.3e} |",
        "",
        f"HGNN − DeepSets payload = {hgnn['mean_payload_bits'] - deep['mean_payload_bits']:.0f}。",
        "",
        "## 3. 成功标准",
        "",
        f"**HGNN assignment > DeepSets assignment：{'成立' if beats else '不成立'}**",
        "",
        "若成立：HGNN 适合关系匹配，问题出在独立 option 分类而不是图编码器。",
        "若不成：即使有 matching 层，边分仍不够；需要改 score 设计或标签，而不是再对独立 categorical 做 PPO。",
        "",
        "未做 PPO fine-tune（Hungarian 不可微；本阶段只验证匹配头）。",
        "",
        "JSON：`src/wt/htw_phase92/outputs/assignment_head/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    print("collecting oracle labels", flush=True)
    data = collect_oracle_dataset(COLLECT_SEEDS)
    print(f"dataset size {data['assignment'].shape[0]}", flush=True)
    print("train MLP", flush=True)
    mlp = train_model(MLPOptionClassifier(), data, "mlp")
    print("train DeepSets", flush=True)
    deep = train_model(DeepSetsAssignmentHead(), data, "deepsets")
    print("train HGNN", flush=True)
    hgnn = train_model(HGNNAssignmentHead(), data, "hgnn")
    print("eval MLP", flush=True)
    mlp_eval = evaluate_model(mlp, "mlp", EVAL_SEEDS)
    print("eval DeepSets", flush=True)
    deep_eval = evaluate_model(deep, "deepsets", EVAL_SEEDS)
    print("eval HGNN", flush=True)
    hgnn_eval = evaluate_model(hgnn, "hgnn", EVAL_SEEDS)
    payload = {
        "hold": HOLD,
        "epochs": EPOCHS,
        "collect_seeds": list(COLLECT_SEEDS),
        "dataset_size": int(data["assignment"].shape[0]),
        "direct_ppo": False,
        "mlp": mlp_eval,
        "deepsets": deep_eval,
        "hgnn": hgnn_eval,
        "hgnn_beats_deepsets": float(hgnn_eval["mean_payload_bits"]) > float(deep_eval["mean_payload_bits"]),
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
                "deepsets_payload": result["deepsets"]["mean_payload_bits"],
                "hgnn_payload": result["hgnn"]["mean_payload_bits"],
                "hgnn_beats_deepsets": result["hgnn_beats_deepsets"],
                "mlp_conflict": result["mlp"]["conflict_rate"],
                "hgnn_conflict": result["hgnn"]["conflict_rate"],
            },
            indent=2,
        )
    )
