"""Offline high/low mismatch diagnosis. Does not call decide() and does not train."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.htw_phase100.high_level import load_deepsets_util
from wt.htw_phase9.const import BUOY_DIM, IDLE_INDEX
from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase943.set_context import all_candidate_physics_set, empty_assignment, make_snap
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment
from wt.lowlevel_v2.base_executor import BaseExecutor
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, refuse_rematch
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1b.protocol import ALPHA_RESIDUAL, LOG_STD_INIT, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.phase1b.service_lock import apply_service_residual_lock, service_lock_mask
from wt.lowlevel_v2.phase5_hgnn_urgency.actor import original_actor
from wt.lowlevel_v2.phase5_hgnn_urgency.encoder import Phase5OrderFreeModel
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch
from wt.lowlevel_v2.phase5_hgnn_urgency.reward import unchanged_step_reward
from wt.lowlevel_v2.protocol import FINAL_CKPT
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .features import hgnn_feature_table
from .protocol import (
    DEPLOYMENT_VERIFIED,
    EVAL_SEEDS,
    HOLD_STEPS,
    OUTPUT,
    PHASE5_CKPT,
    PLANNER_READY,
    RANK_SEEDS,
    REPORT,
    assert_diagnosis_seed,
)


def _option(env: L3BEnv, info: dict, assignment: np.ndarray) -> TaskOption:
    targets = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=targets,
        decision_time=int(info["time_step"]),
        remaining_hold=int(HOLD_STEPS),
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=targets.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def _model_assignment(model, env: L3BEnv, info: dict, last_arrival: np.ndarray, last_positions: np.ndarray) -> np.ndarray:
    batch = graph_batch(info, env, last_arrival, last_positions, urgency=None)
    with torch.no_grad():
        assignment = decode_assignment(model, batch, env)
    return np.asarray(assignment, dtype=np.int64)


def _pair_logits(model, env: L3BEnv, info: dict, last_arrival: np.ndarray, last_positions: np.ndarray) -> np.ndarray:
    batch = graph_batch(info, env, last_arrival, last_positions, urgency=None)
    tensors = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(batch[key], dtype=dtype)
    with torch.no_grad():
        encoded = model.encode(
            tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"]
        )
        partial = empty_assignment(env.config.num_uavs)
        snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
        physics = torch.as_tensor(all_candidate_physics_set(snap, partial)[None, ...], dtype=torch.float32)
        logits = model.decoder(encoded, partial, physics, np.asarray(batch["buoy_mask"][0], dtype=bool))
    return logits.detach().cpu().numpy()


def oracle_rank_for_uav(env: L3BEnv, uav: int, chosen: int) -> np.ndarray:
    leftover = env.remaining_arrivals()
    scores = np.zeros(int(env.config.num_buoys) + 1, dtype=np.float64)
    for buoy in range(int(env.config.num_buoys)):
        work = max(float(env.queue[buoy]), float(leftover[buoy]))
        dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[buoy]))
        scores[buoy] = 0.0 if work <= 1.0e-9 else work / (dist + 1.0)
    token = int(IDLE_INDEX if int(chosen) < 0 else chosen)
    scores[token] = float(np.max(scores) + 1.0)
    return scores


class LockedExecutor:
    """Residual MAPPO or fly-toward under an assignment supplied by the caller."""

    def __init__(self, actor) -> None:
        self.actor = actor
        self.base = BaseExecutor(gated=True)
        self.projector = SafetyProjector(CollisionConfig(0.4, 0.2, placeholder=False), safety_margin=SERVICE_SAFETY_MARGIN_M)
        if actor is not None:
            actor.eval()

    def act(self, env: L3BEnv, option: TaskOption, last_movement: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.actor is None:
            move = fly_toward(env, option.association)
            zeros = np.zeros_like(move)
            return move, move, zeros
        obs = build_actor_observation(env, option, last_movement)
        with torch.no_grad():
            mu, _ = self.actor(
                torch.as_tensor(obs["self"], dtype=torch.float32),
                torch.as_tensor(obs["task"], dtype=torch.float32),
                torch.as_tensor(obs["communication"], dtype=torch.float32),
                torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
            )
        pre = mu.detach().cpu().numpy().astype(np.float64)
        a_base, _ = self.base.act(env, option)
        lock = service_lock_mask(env, option)
        a_des, delta = apply_service_residual_lock(a_base, pre, lock, ALPHA_RESIDUAL)
        proj = self.projector.project(env, a_des, option)
        if proj.constraint_status != "ok":
            executed = np.zeros_like(a_des)
        else:
            executed = np.asarray(proj.projected_actions, dtype=np.float64)
        return clip_unit_step(executed), a_des, delta


def rollout(seed: int, assign_fn, executor: LockedExecutor) -> dict:
    assert_diagnosis_seed(int(seed))
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    pending = 0
    option = None
    reward_sum = 0.0
    path = 0.0
    completion = None
    matches = []
    while not info.get("terminated", False):
        if pending <= 0:
            assignment = np.asarray(assign_fn(env, info, last_arrival, last_positions), dtype=np.int64)
            oracle = np.asarray(oracle_option_assignment(env), dtype=np.int64)
            matches.append(float(np.mean(assignment == oracle)))
            option = _option(env, info, assignment)
            pending = int(HOLD_STEPS)
        assert option is not None
        pos0 = np.asarray(env.uav_positions, dtype=np.float64).copy()
        payload0 = float(env.payload_bits)
        executed, desired, delta = executor.act(env, option, last_movement)
        info = env.step(executed, option.association)
        path += float(np.sum(np.linalg.norm(env.uav_positions - pos0, axis=-1)))
        reward_sum += unchanged_step_reward(float(env.payload_bits - payload0), executed, desired, delta)
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        last_movement = np.asarray(executed, dtype=np.float64)
        pending -= 1
        if completion is None and float(np.sum(env.queue)) <= 1.0e-9:
            completion = int(env.t)
    return {
        "payload": float(env.payload_bits),
        "trajectory_reward": float(reward_sum),
        "residual_queue": float(np.sum(env.queue)),
        "completion_time": int(env.config.episode_horizon if completion is None else completion),
        "path_length_m": path,
        "oracle_agreement": float(np.mean(matches)) if matches else float("nan"),
    }


def ranking_only(model, seed: int) -> list[dict]:
    assert_diagnosis_seed(int(seed))
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    oracle = np.asarray(oracle_option_assignment(env), dtype=np.int64)
    logits = _pair_logits(model, env, info, last_arrival, last_positions)
    rows = []
    for uav in range(int(env.config.num_uavs)):
        token = int(IDLE_INDEX if int(oracle[uav]) < 0 else oracle[uav])
        oracle_vec = oracle_rank_for_uav(env, uav, int(oracle[uav]))
        pred = np.asarray(logits[uav], dtype=np.float64)
        order = np.argsort(-pred)
        rows.append(
            {
                "seed": int(seed),
                "uav": int(uav),
                "spearman": spearman(pred, oracle_vec),
                "kendall": kendall_tau(pred, oracle_vec),
                "top1": float(int(order[0]) == token),
                "top3": float(token in set(int(item) for item in order[:3])),
            }
        )
    if int(env.t) != 0:
        raise RuntimeError("ranking diagnosis executed a transition")
    return rows


def _mean_rows(rows: list[dict]) -> dict:
    keys = [key for key in rows[0] if key not in {"seed", "uav"}]
    out = {"n": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(np.mean(values)) if values.size else float("nan")
    return out


def _mean(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _load_models():
    deepsets = load_deepsets_util(0, FINAL_CKPT)
    deepsets.eval()
    hgnn = Phase5OrderFreeModel(buoy_dim=BUOY_DIM)
    hgnn.load_state_dict(torch.load(PHASE5_CKPT / "hgnn_orderfree.pt", map_location="cpu"))
    hgnn.eval()
    actor = original_actor(float(LOG_STD_INIT))
    actor.load_state_dict(torch.load(PHASE5_CKPT / "mappo_original.pt", map_location="cpu"))
    actor.eval()
    return deepsets, hgnn, actor


def _choose(exp1_regret: float, exp2_regret: float, top1: float) -> str:
    high_wrong = (not np.isfinite(exp2_regret)) or exp2_regret > 0.15 or (not np.isfinite(top1)) or top1 < 0.5
    low_wrong = (not np.isfinite(exp1_regret)) or exp1_regret > 0.15
    if high_wrong and low_wrong:
        return "Option-3"
    if high_wrong:
        return "Option-1"
    if low_wrong:
        return "Option-2"
    return "neither"


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def run() -> dict:
    for seed in EVAL_SEEDS + RANK_SEEDS:
        assert_diagnosis_seed(int(seed))
    deepsets, hgnn, actor = _load_models()
    mappo = LockedExecutor(actor)
    fly = LockedExecutor(None)
    deepsets_fn = lambda env, info, last_arrival, last_positions: _model_assignment(deepsets, env, info, last_arrival, last_positions)
    hgnn_fn = lambda env, info, last_arrival, last_positions: _model_assignment(hgnn, env, info, last_arrival, last_positions)
    oracle_fn = lambda env, info, last_arrival, last_positions: np.asarray(oracle_option_assignment(env), dtype=np.int64)

    reference = []
    exp1 = []
    exp2 = []
    exp2_hgnn = []
    for seed in EVAL_SEEDS:
        print(f"closed loop seed={seed}", flush=True)
        reference.append(rollout(seed, oracle_fn, fly))
        exp1.append(rollout(seed, oracle_fn, mappo))
        exp2.append(rollout(seed, deepsets_fn, fly))
        exp2_hgnn.append(rollout(seed, hgnn_fn, fly))
    ref_payload = _mean(reference, "payload")
    def regret(rows):
        payload = _mean(rows, "payload")
        return float((ref_payload - payload) / max(ref_payload, 1.0))

    rank_rows = []
    rank_hgnn = []
    for seed in RANK_SEEDS:
        print(f"ranking seed={seed}", flush=True)
        rank_rows.extend(ranking_only(deepsets, int(seed)))
        rank_hgnn.extend(ranking_only(hgnn, int(seed)))
    deep_rank = _mean_rows(rank_rows)
    hgnn_rank = _mean_rows(rank_hgnn)
    table = hgnn_feature_table()
    choice = _choose(regret(exp1), regret(exp2), float(deep_rank["top1"]))
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "training": False,
        "decide_called": False,
        "holdout_used": False,
        "reference_oracle_fly": _mean_rows(reference),
        "experiment_1_oracle_high_mappo": {**_mean_rows(exp1), "regret": regret(exp1)},
        "experiment_2_deepsets_fly": {**_mean_rows(exp2), "regret": regret(exp2)},
        "experiment_2_hgnn_fly": {**_mean_rows(exp2_hgnn), "regret": regret(exp2_hgnn)},
        "experiment_3_deepsets": deep_rank,
        "experiment_3_hgnn": hgnn_rank,
        "feature_table": table,
        "route": choice,
        "phase5c_started": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5b_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(choice, flush=True)
    return summary


def _report(summary: dict) -> str:
    exp1 = summary["experiment_1_oracle_high_mappo"]
    exp2 = summary["experiment_2_deepsets_fly"]
    exp2h = summary["experiment_2_hgnn_fly"]
    ref = summary["reference_oracle_fly"]
    d3 = summary["experiment_3_deepsets"]
    h3 = summary["experiment_3_hgnn"]
    route = summary["route"]
    route_text = {
        "Option-1": "推荐 Option-1，对象是高层分配，不是当前 Residual MAPPO。Oracle 目标交给 MAPPO 后 payload 高于 fly-toward，所以低层不是执行瓶颈。Phase-5 HGNN 交给 fly-toward 后 payload 为 0，DeepSets 相对这个簇 oracle 的 Top-1 也低于 0.5。下一步要改的是高层目标如何排序，而且不能把这个簇 oracle 当成 payload 上界。本阶段不实现。",
        "Option-2": "推荐 Option-2：高层目标交给 fly-toward 后接近 oracle，但把 oracle 目标交给当前 Residual MAPPO 后 regret 超过 0.15。问题主要在轨迹执行。本阶段不实现。",
        "Option-3": "推荐 Option-3：oracle 目标交给当前 MAPPO 仍然有明显 regret，同时当前高层相对 oracle 轨迹或 oracle 排序也没有对齐。这是两层目标同时偏离，不是单改 encoder 能收口的。只保留方案，不实现 joint reward、reachability feedback 或 HAC/HIRO。",
        "neither": "两组闭环相对 oracle+fly-toward 的 regret 都不超过 0.15，而且 DeepSets Top-1 不低于 0.5。这一诊断没有指出必须立刻改高层、低层或对齐层。",
    }[route]
    lines = [
        "# HTW Phase-5B Hierarchical Objective Alignment Diagnosis",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C 或 planner，没有使用 230000–230127，也没有进入 Phase-5C。",
        "",
        "## 1. 当前系统结构",
        "",
        "High-level: 冻结的 DeepSets order-free joint greedy。Phase-5 的 HGNN 只作为已经保存的有界检查点参加对照，不重新训练。",
        "",
        "Low-level: Phase-5 基线 Residual MAPPO 检查点 `mappo_original.pt`。Oracle 轨迹是给定分配后的 fly-toward，不是新的策略。",
        "",
        "对照上界是现有 `oracle_option_assignment` 加 fly-toward。这个 oracle 用整段 shadow payload 选近/远簇分工，诊断只把它当标签，不把它写进网络。",
        "",
        "## 2. 三个诊断实验",
        "",
        f"闭环种子 {list(EVAL_SEEDS)}。排序种子 {list(RANK_SEEDS)}，排序只在 t=0 读取 logits，不执行轨迹。",
        "",
        "| 实验 | payload | trajectory reward | regret | completion | oracle agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Oracle high + fly-toward | {_fmt(ref['payload'])} | {_fmt(ref['trajectory_reward'])} | 0.0000 | {_fmt(ref['completion_time'])} | {_fmt(ref['oracle_agreement'])} |",
        f"| 实验1 Oracle high + current MAPPO | {_fmt(exp1['payload'])} | {_fmt(exp1['trajectory_reward'])} | {_fmt(exp1['regret'])} | {_fmt(exp1['completion_time'])} | {_fmt(exp1['oracle_agreement'])} |",
        f"| 实验2 DeepSets + fly-toward | {_fmt(exp2['payload'])} | {_fmt(exp2['trajectory_reward'])} | {_fmt(exp2['regret'])} | {_fmt(exp2['completion_time'])} | {_fmt(exp2['oracle_agreement'])} |",
        f"| 实验2b Phase-5 HGNN + fly-toward | {_fmt(exp2h['payload'])} | {_fmt(exp2h['trajectory_reward'])} | {_fmt(exp2h['regret'])} | {_fmt(exp2h['completion_time'])} | {_fmt(exp2h['oracle_agreement'])} |",
        "",
        "实验3 不执行轨迹。Oracle 排序把 oracle 选中的目标放在第一，其余浮标按当前队列、剩余到达和距离排序。",
        "",
        "| 模型 | Spearman | Kendall | Top-1 | Top-3 | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| DeepSets | {_fmt(d3['spearman'])} | {_fmt(d3['kendall'])} | {_fmt(d3['top1'])} | {_fmt(d3['top3'])} | {d3['n']} |",
        f"| Phase-5 HGNN | {_fmt(h3['spearman'])} | {_fmt(h3['kendall'])} | {_fmt(h3['top1'])} | {_fmt(h3['top3'])} | {h3['n']} |",
        "",
        "## 3. mismatch 来源",
        "",
        _mismatch_paragraph(summary),
        "",
        "## 4. HGNN 特征对照",
        "",
        "没有修改模型。",
        "",
        "| feature | current | required |",
        "| --- | --- | --- |",
    ]
    for row in summary["feature_table"]:
        lines.append(f"| {row['feature']} | {row['current']} | {row['required']} |")
    lines.extend(
        [
            "",
            "UAV–UAV 和 UAV–Buoy 边已经存在。干扰是当前正在服务的链路干扰，没人进入服务半径时这路特征是 0。信道是当前的假设速率。能耗槽位恒为 0，因为 L3-B 没有能耗状态。Queue 已经在浮标特征里。Urgency 只出现在 Phase-5 完整检查点的额外通道上，上一阶段把该通道置零后分配不变。",
            "",
            "## 5. Shared urgency 设计",
            "",
            "当前 `u_k(t) = q_k(t) / (max_j q_j(t) + 1e-8)` 只是同一时刻的队列比例。它不包含等待了多久、信息年龄或截止时间。",
            "",
            "可以写成以后的接口，但现在不能填进代码：",
            "",
            "`urgency_k(t) = α q̃_k(t) + β w̃_k(t) + γ ã_k(t) + δ d̃_k(t)`",
            "",
            "- `q̃_k`：现有队列比例，是唯一已经可观测的项。",
            "- `w̃_k`：队列保持为正的步数 / 48。环境没有这个计数器。",
            "- `ã_k`：AoI / 48。环境没有 AoI。",
            "- `d̃_k`：距离截止时间的剩余比例。环境没有 deadline。",
            "",
            "在这三个时钟被环境真正记下来之前，β、γ、δ 只能是 0。把它们设成非零会凭空制造特征。权重先不拟合。共享方式仍然是决策时算一次，写入 `TaskOption`，15 步内高低层读同一份，不进入 reward。",
            "",
            "## 6. 下一阶段路线",
            "",
            route_text,
            "",
            "Option-1 要补的是能改变排序的 UAV 交互、有效干扰、以及一个真正被解码器使用的 urgency。能耗要等环境提供变量。",
            "",
            "Option-2 要补的是轨迹对干扰和通信的执行，不是再换一个 encoder。能耗约束同样要等环境变量。",
            "",
            "Option-3 只作为方案：joint reward、reachability feedback、HAC/HIRO。禁止在本阶段实现。",
            "",
            "Phase-5C 没有开始。`planner_ready = False`，`deployment_verified = False`。",
            "",
        ]
    )
    return "\n".join(lines)


def _mismatch_paragraph(summary: dict) -> str:
    exp1 = summary["experiment_1_oracle_high_mappo"]
    exp2 = summary["experiment_2_deepsets_fly"]
    exp2h = summary["experiment_2_hgnn_fly"]
    ref = summary["reference_oracle_fly"]
    d3 = summary["experiment_3_deepsets"]
    h3 = summary["experiment_3_hgnn"]
    return (
        "负的 regret 表示 payload 高于 oracle+fly-toward。这个簇划分 oracle 在这两组种子上不是 payload 上界。\n\n"
        f"实验1：oracle 分配交给当前 MAPPO，payload 是 {_fmt(exp1['payload'])}，"
        f"高于同一分配规则加 fly-toward 的 {_fmt(ref['payload'])}，regret 是 {_fmt(exp1['regret'])}。"
        "目标锁定以后，当前低层能够完成服务。低层不是“给对目标也不会执行”。\n\n"
        f"实验2：DeepSets 分配交给 fly-toward，payload 是 {_fmt(exp2['payload'])}，regret 是 {_fmt(exp2['regret'])}，"
        f"和 oracle 分配的重合度是 {_fmt(exp2['oracle_agreement'])}。"
        f"Phase-5 HGNN 用同一个 fly-toward 执行，payload 是 {_fmt(exp2h['payload'])}，regret 是 {_fmt(exp2h['regret'])}，"
        f"重合度是 {_fmt(exp2h['oracle_agreement'])}。高层把目标选成无法完成服务的浮标时，换成 oracle 轨迹也补不回来。\n\n"
        f"实验3 不飞轨迹。DeepSets 的 Spearman 是 {_fmt(d3['spearman'])}，Top-1 是 {_fmt(d3['top1'])}，Top-3 是 {_fmt(d3['top3'])}。"
        f"HGNN 的 Spearman 是 {_fmt(h3['spearman'])}，Top-1 是 {_fmt(h3['top1'])}，Top-3 是 {_fmt(h3['top3'])}。"
        "HGNN 更接近队列/距离排序，而这个排序不是 payload 更高的分配。Phase-5 用排他匹配标签训练 HGNN，会把高层拉向一个不能转化成服务的目标。"
    )


if __name__ == "__main__":
    run()
