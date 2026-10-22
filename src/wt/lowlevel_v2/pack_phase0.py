"""Dump LL-V2.0 artifacts from live interfaces. No training. No Phase-10.6 eval."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase100.contract import TaskOption
from wt.l3b.config import L3BConfig
from wt.l3b.env import L3BEnv, active_links, interference_matrix, sinr_rates_for_links

from .base_executor import BaseExecutor
from .contracts import CollisionConfig, clip_unit_step
from .critic_state import build_critic_state, expected_critic_dim
from .data_registry import build_registry, refuse_untouched
from .metrics import collision_diagnostics
from .neighbor_encoder import NeighborSetEncoder, permutation_invariance_error
from .observation import COMM_DIM, NEIGHBOR_EDGE_DIM, SELF_DIM, TASK_DIM, build_actor_observation, communication_features, neighbor_edges
from .protocol import (
    ALPHA_RESIDUAL_PLACEHOLDER,
    COLLISION_REQUIRES_PRETRAINING_FREEZE,
    COLLISION_MARGIN_PLACEHOLDER_M,
    D_SAFE_PLACEHOLDER_M,
    EXPECTED_HIGH_SHA,
    FINAL_CKPT,
    HOLD_STEPS,
    OUTPUT,
    PHASE106_UNTOUCHED,
    ROOT,
    SERVICE_SAFETY_MARGIN_PLACEHOLDER_M,
    SUGGESTED_COLLISION_MARGIN_RANGE_M,
    SUGGESTED_UAV_RADIUS_RANGE_M,
    UAV_PHYSICAL_RADIUS_PLACEHOLDER_M,
    sha256_bytes,
)
from .residual_action import residual_desired
from .residual_policy import ResidualActor
from .reward_spec import reward_draft_dict
from .safety_projector import SafetyProjector
from .service_safe_envelope import envelope_active, inner_radius


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _option(env: L3BEnv, assoc: np.ndarray, hold: int = HOLD_STEPS) -> TaskOption:
    assoc = np.asarray(assoc, dtype=np.int64)
    return TaskOption(assoc, int(env.t), int(hold), np.ones(6, dtype=bool), np.ones(12, dtype=bool), assoc.copy())


def communication_sensitivity() -> dict:
    env = L3BEnv()
    env.reset(seed=8)
    env.uav_positions[0] = env.buoy_positions[0].copy()
    env.uav_positions[1] = env.buoy_positions[1].copy()
    env.assignment = np.array([0, 1, -1, -1, -1, -1], dtype=np.int64)
    env.queue[:] = 4.0
    links = active_links(env)
    sinr0, rate0 = sinr_rates_for_links(env, links)
    inter0 = interference_matrix(env)
    env.uav_positions[1] = env.uav_positions[1] + np.array([0.75, 0.0])
    sinr1, rate1 = sinr_rates_for_links(env, links)
    inter1 = interference_matrix(env)
    changed = bool(
        np.max(np.abs(sinr1 - sinr0)) > 0.0
        or np.max(np.abs(rate1 - rate0)) > 0.0
        or np.max(np.abs(inter1 - inter0)) > 0.0
    )
    return {
        "used_existing_l3b_formulas": True,
        "second_sinr_formula": False,
        "n_active_links": len(links),
        "sinr_delta_max": float(np.max(np.abs(sinr1 - sinr0))),
        "rate_delta_max": float(np.max(np.abs(rate1 - rate0))),
        "interference_delta_max": float(np.max(np.abs(inter1 - inter0))),
        "responds_to_position_perturbation": changed,
        "scientific_result": False,
        "design_probe_only": True,
    }


def pack() -> dict:
    refuse_untouched(8)
    digest = sha256_bytes(FINAL_CKPT.read_bytes())
    if digest != EXPECTED_HIGH_SHA:
        raise RuntimeError("frozen high-level SHA mismatch; refusing to proceed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = L3BEnv()
    env.reset(seed=8)
    option = _option(env, np.array([0, 1, 2, 3, 4, -1], dtype=np.int64))
    base = BaseExecutor(gated=True)
    a_base, _ = base.act(env, option)
    a_zero = residual_desired(a_base, np.zeros_like(a_base))
    zero_ok = bool(np.allclose(a_zero, clip_unit_step(a_base)))
    obs = build_actor_observation(env, option, np.zeros((6, 2)))
    critic = build_critic_state(env, option, np.zeros((6, 2)))
    encoder = NeighborSetEncoder()
    edges = torch.as_tensor(neighbor_edges(env, option, np.zeros((6, 2)), 0), dtype=torch.float32).unsqueeze(0)
    perm = torch.randperm(edges.shape[-2])
    perm_err = permutation_invariance_error(encoder, edges, perm)
    actor = ResidualActor()
    with torch.no_grad():
        mu, _ = actor(
            torch.as_tensor(obs["self"], dtype=torch.float32),
            torch.as_tensor(obs["task"], dtype=torch.float32),
            torch.as_tensor(obs["communication"], dtype=torch.float32),
            torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
        )
    mu_max = float(torch.max(torch.abs(mu)))
    projector = SafetyProjector()
    proj = projector.project(env, a_base, option)
    collision = CollisionConfig(UAV_PHYSICAL_RADIUS_PLACEHOLDER_M, COLLISION_MARGIN_PLACEHOLDER_M, True)
    comm_sens = communication_sensitivity()
    registry = build_registry()
    cfg = L3BConfig()
    protocol = {
        "name": "LL-V2.0",
        "method_future": "ICAR-MAPPO",
        "trained": False,
        "modified_final_hier": False,
        "visited_phase106_untouched": False,
        "high_level": "frozen DeepSets-util + Order-Free + Task Option hold=15",
        "checkpoint": str(FINAL_CKPT),
        "checkpoint_sha": digest,
        "hold": HOLD_STEPS,
        "low_level_controls": "movement",
        "high_level_controls": "association",
        "l0_baseline": "fly-toward + InRangeZeroMotionGate",
        "collision_requires_pretraining_freeze": COLLISION_REQUIRES_PRETRAINING_FREEZE,
        "d_safe_placeholder_m": D_SAFE_PLACEHOLDER_M,
        "historical_maritime_min_separation_m": 15.0,
        "historical_separation_not_used_as_l3b_truth": True,
        "l3b_spawn_spacing_m_approx": 1.2,
        "suggested_uav_radius_range_m": list(SUGGESTED_UAV_RADIUS_RANGE_M),
        "suggested_margin_range_m": list(SUGGESTED_COLLISION_MARGIN_RANGE_M),
        "alpha_residual_placeholder": ALPHA_RESIDUAL_PLACEHOLDER,
        "service_safety_margin_placeholder_m": SERVICE_SAFETY_MARGIN_PLACEHOLDER_M,
        "M": cfg.num_uavs,
        "K": cfg.num_buoys,
        "T": cfg.episode_horizon,
        "zero_residual_equals_base": zero_ok,
        "neighbor_perm_error": perm_err,
        "actor_zero_init_max_abs_mu": mu_max,
        "projector_status_on_design_seed8": proj.constraint_status,
        "critic_dim": critic["dim"],
        "expected_critic_dim": expected_critic_dim(6, 12),
    }
    schema = {
        "self_dim": SELF_DIM,
        "task_dim": TASK_DIM,
        "communication_dim": COMM_DIM,
        "neighbor_edge_dim": NEIGHBOR_EDGE_DIM,
        "self_fields": ["x", "y", "last_dx", "last_dy", "mask"],
        "task_fields": ["valid", "rel_x", "rel_y", "distance", "queue", "progress", "remaining_hold", "remaining_episode_time"],
        "communication_fields": ["target_sinr", "target_rate", "total_interference", "interference_to_noise"],
        "neighbor_fields": ["rel_x", "rel_y", "rel_dx", "rel_dy", "distance", "active", "idle", "neighbor_target_rel_x", "neighbor_target_rel_y", "neighbor_progress", "neighbor_in_range"],
        "no_velocity_invented": True,
        "idle_never_indexes_last_buoy": True,
        "no_future_arrivals": True,
        "example_shapes": {
            "self": list(obs["self"].shape),
            "task": list(obs["task"].shape),
            "communication": list(obs["communication"].shape),
            "neighbor_set": list(obs["neighbor_set"].shape),
        },
    }
    action_contract = {
        "space": "unit-disk 2D movement, same as L3-B step input",
        "a_base": "gated fly-toward",
        "delta_rl": "tanh-normalized residual",
        "a_des": "clip(a_base + alpha_residual * tanh(delta_rl))",
        "alpha_residual_placeholder": ALPHA_RESIDUAL_PLACEHOLDER,
        "zero_residual_equals_base": zero_ok,
        "low_level_cannot_change_association": True,
    }
    collision_model = {
        "geometry": "horizontal safety disc",
        "uav_physical_radius_placeholder_m": UAV_PHYSICAL_RADIUS_PLACEHOLDER_M,
        "collision_margin_placeholder_m": COLLISION_MARGIN_PLACEHOLDER_M,
        "d_safe_placeholder_m": D_SAFE_PLACEHOLDER_M,
        "formula": "d_safe = 2*uav_physical_radius + collision_margin",
        "requires_pretraining_freeze": True,
        "scientific_result": False,
        "l3b_has_native_radius": False,
        "metrics": collision_diagnostics(env.uav_positions, collision),
    }
    baselines = {
        "L0": "gated fly-toward (FINAL-HIER low-level)",
        "L1": "non-residual MAPPO",
        "L2": "Residual MAPPO without neighbor/interference features",
        "L3": "Collision-aware Residual MAPPO + neighbor encoder + SafetyProjector",
        "L4": "ICAR-MAPPO = task-conditioned + residual MAPPO + neighbor-aware + interference-aware + SafetyProjector + ServiceSafeEnvelope",
        "ran": False,
    }
    success = {
        "payload_above_L0": True,
        "multi_seed_stable": True,
        "ci95_supports": True,
        "collision_count_zero": True,
        "min_separation_ge_d_safe": True,
        "completed_blocks_not_worse": True,
        "projector_intervention_not_near_100pct": True,
        "do_not_lower_bar_to_keep_rl": True,
    }
    files = {
        "protocol": ROOT / "docs" / "archive" / "phase_reports" / "LLV2_PROTOCOL.md",
        "schema": ROOT / "results/root_exports/LLV2_OBSERVATION_SCHEMA.json",
        "action": ROOT / "results/root_exports/LLV2_ACTION_CONTRACT.json",
        "collision": ROOT / "results/root_exports/LLV2_COLLISION_MODEL.json",
        "projector": ROOT / "docs" / "archive" / "phase_reports" / "LLV2_SAFETY_PROJECTOR_REPORT.md",
        "comm": ROOT / "results/root_exports/LLV2_COMMUNICATION_SENSITIVITY.json",
        "registry": ROOT / "results/root_exports/LLV2_DATA_REGISTRY.json",
        "registry_alias": ROOT / "results/root_exports/lowlevel_v2_data_registry.json",
        "baselines": ROOT / "results/root_exports/LLV2_BASELINE_MANIFEST.json",
        "reward": ROOT / "docs" / "archive" / "phase_reports" / "LLV2_REWARD_DRAFT.md",
        "report": ROOT / "docs/archive/phase_reports/HTW_LowLevelV2_Phase0_Report_CN.md",
    }
    _dump(files["schema"], schema)
    _dump(files["action"], action_contract)
    _dump(files["collision"], collision_model)
    _dump(files["comm"], comm_sens)
    _dump(files["registry"], registry)
    _dump(files["registry_alias"], registry)
    _dump(files["baselines"], baselines)
    _dump(OUTPUT / "llv2_phase0_status.json", protocol)
    _dump(
        files["protocol"],
        "\n".join(
            [
                "# LL-V2.0 Protocol",
                "",
                "Independent branch. FINAL-HIER is read-only. No RL training.",
                "",
                f"- frozen checkpoint SHA: `{digest}`",
                f"- hold={HOLD_STEPS}",
                "- low-level: movement only; high-level: association / Task Option",
                "- L0: fly-toward + InRangeZeroMotionGate",
                "- future method: ICAR-MAPPO",
                "- collision d_safe is a **design placeholder**; `requires_pretraining_freeze=true`",
                "- Phase-10.6 seeds 106000–106063 are permanently blocked",
                "- confirmation / new final untouched pools are not created in this phase",
                "",
            ]
        ),
    )
    _dump(
        files["projector"],
        "\n".join(
            [
                "# SafetyProjector report (interface only)",
                "",
                "Deterministic iterative projection on predicted next positions.",
                "Constraints: unit-step, pairwise predicted distance >= d_safe (placeholder), ServiceSafeEnvelope inner disk.",
                "If infeasible: `constraint_status=failure`. Not silently labeled safe.",
                "No QP dependency in the current environment; iterative projection is the v1 implementation.",
                "",
                f"- design seed 8 status: `{proj.constraint_status}`",
                f"- intervention_norm: {proj.intervention_norm}",
                f"- min predicted distance: {proj.minimum_predicted_distance}",
                f"- envelope_ok: {proj.envelope_ok}",
                f"- finite: {proj.finite}",
                f"- inner service radius: {inner_radius(env)} (service_radius - placeholder margin {SERVICE_SAFETY_MARGIN_PLACEHOLDER_M})",
                f"- envelope active count on seed 8 after reset: {int(np.sum(envelope_active(env, option)))}",
                "",
                "Not a scientific benchmark result.",
                "",
            ]
        ),
    )
    _dump(
        files["reward"],
        "\n".join(
            [
                "# LL-V2 reward draft",
                "",
                "Coefficients are **not** frozen. Auto-search is forbidden. One design freeze is required before any training.",
                "",
                json.dumps(reward_draft_dict(), indent=2),
                "",
                "Evaluation never uses the training reward. Flight distance is not energy.",
                "",
            ]
        ),
    )
    report = f"""# HTW Low-Level V2 Phase-0 报告

本阶段只建设接口。未训练。未修改 FINAL-HIER。未访问 Phase-10.6 untouched。

1. 新低层完整结构：冻结高层 Task Option → L0 base (fly-toward+gate) → residual desired → SafetyProjector / ServiceSafeEnvelope → L3-B movement。未来算法名 ICAR-MAPPO。
2. 来自现有 final 系统：DeepSets-util checkpoint、Order-Free 高层接口、Task Option、hold=15、fly-toward、InRangeZeroMotionGate、L3-B SINR/rate/interference 公式。
3. 新加入：neighbor set encoder schema、residual action 合同、水平安全盘 placeholder、SafetyProjector、ServiceSafeEnvelope、critic schema、新 seed registry、MAPPO skeleton（零残差初始化）。
4. Actor 看：self、task、communication（现有 SINR/rate/interference）、neighbor_set、d_nearest。无未来到达。
5. Critic 看：全体 UAV self/task、association、queue/progress、SINR/rate、两两距离、干扰矩阵、剩余 hold 与剩余时间。
6. Residual：a_des = clip(a_base + alpha * tanh(delta_rl))。Zero residual 时 a_des == a_base：{zero_ok}。
7. 碰撞体积：二维水平安全盘。L3-B 无原生 UAV 半径。历史 maritime `multi_uav_min_separation_m=15` 不能移植（起飞间距约 1.2 m）。placeholder d_safe={D_SAFE_PLACEHOLDER_M} m，requires_pretraining_freeze=true。
8. SafetyProjector：迭代推开预测位置，再还原为步长受限动作；不可行则 failure，不称为安全。
9. 通信干扰：经现有 SINR→rate→payload 进入未来主奖励；Actor/Critic 观察现有干扰特征。
10. 第一版不加额外 interference penalty：避免第二套人工通信损失；真实 payload 已含干扰。
11. ServiceSafeEnvelope：在有效服务区内时，预测下一位置须留在 service_radius - safety_margin 内。旧 hard-zero gate 仍是 L0。
12. 旧 final untouched 是否访问？**否。**
13. 本阶段是否训练？**否。**
14. 是否修改旧 FINAL-HIER？**否。**
15. 新 train/design 是否与历史隔离？train/design 新池与扫描到的历史种子及 106000–106063 无交集。confirmation/final untouched 未创建。
16. 是否具备进入 LL-V2.1 单 seed Residual MAPPO pilot 的工程条件？接口与测试具备后可以进入 **pilot 工程准备**，但必须先由用户冻结 d_safe 与 alpha_residual；本阶段 **不自动训练**。

checkpoint SHA: `{digest}`
neighbor perm error: {perm_err}
actor init |mu|_max: {mu_max}
critic dim: {critic['dim']} (expected {expected_critic_dim(6, 12)})
通信位置扰动响应: {comm_sens['responds_to_position_perturbation']}
"""
    _dump(files["report"], report)
    for src in files.values():
        if src.suffix in {".json", ".md"}:
            _dump(OUTPUT / src.name, src.read_text(encoding="utf-8") if src.suffix == ".md" else json.loads(src.read_text(encoding="utf-8")))
    return protocol


if __name__ == "__main__":
    print(json.dumps(pack(), indent=2))
