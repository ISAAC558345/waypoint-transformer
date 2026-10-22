"""Phase-6.2 low-level PPO vs L3-B compatibility. No weight loading, no training."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase61.features import MAX_BUOYS as L2_K
from wt.htw_phase61.features import MAX_UAVS as L2_M
from wt.htw_phase61.features import OPTION_UAV_DIM, UAV_DIM
from wt.htw_phase61.models import L2AOptionActorCritic
from wt.htw_phase63.run_phase63 import CKPT as PHASE62_CKPT
from wt.htw_phase9.const import MAX_BUOYS as L3_K
from wt.htw_phase9.const import MAX_UAVS as L3_M
from wt.l3b.config import L3BConfig
from wt.round30.models import DeepSetsPPOShape


def audit() -> dict:
    l2_shape = DeepSetsPPOShape(uav_dim=OPTION_UAV_DIM, buoy_dim=7, global_dim=3, max_uavs=L2_M, action_dim=2)
    probe = L2AOptionActorCritic()
    l3 = L3BConfig()
    ckpt_exists = Path(PHASE62_CKPT).is_file()
    issues = [
        {"item": "num_uavs", "phase62": int(L2_M), "l3b": int(l3.num_uavs), "ok": int(L2_M) == int(l3.num_uavs)},
        {"item": "num_buoys", "phase62": int(L2_K), "l3b": int(l3.num_buoys), "ok": int(L2_K) == int(l3.num_buoys)},
        {"item": "option_uav_dim", "phase62": int(OPTION_UAV_DIM), "l3b_needed": int(UAV_DIM) + int(L3_K) + 1, "ok": int(OPTION_UAV_DIM) == int(UAV_DIM) + int(L3_K) + 1},
        {"item": "actor_max_uavs", "phase62": int(l2_shape.max_uavs), "l3b": int(L3_M), "ok": int(l2_shape.max_uavs) == int(L3_M)},
        {"item": "sinr_in_low_level_reward", "phase62": False, "l3b": True, "ok": False},
        {"item": "checkpoint_on_disk", "path": str(PHASE62_CKPT), "ok": bool(ckpt_exists)},
        {"item": "hardcoded_MAX_UAVS_in_phase61_features", "value": int(L2_M), "ok": False},
        {"item": "trained_env", "phase62": "L2AEnv M=5 no SINR", "l3b": "L3BEnv M=6 SINR", "ok": False},
        {"item": "phase62_association_source", "phase62": "oracle_option_assignment inside GoalOptionVector", "l3b_required": "frozen high-level association; low-level movement only", "ok": False},
        {"item": "goal_reward", "phase62": "r_comm + lambda_c * in_range_completion - lambda_d * dist_slots on L2-A", "l3b": "SINR-scaled service; D=3 not a fixed three-step finish", "ok": False},
    ]
    compatible_load = all(
        row.get("ok")
        for row in issues
        if row["item"] in {"num_uavs", "num_buoys", "option_uav_dim", "actor_max_uavs", "checkpoint_on_disk", "trained_env"}
    )
    if compatible_load:
        verdict = "direct_load"
        verdict_zh = "可直接加载且接口兼容"
    elif int(L2_M) != int(l3.num_uavs) or int(OPTION_UAV_DIM) != int(UAV_DIM) + int(L3_K) + 1:
        verdict = "needs_independent_adapter"
        verdict_zh = "结构/输入需要独立适配后才能训练"
    else:
        verdict = "retrain_on_l3b"
        verdict_zh = "结构可复用，但需要在 L3-B 重新训练"
    return {
        "verdict": verdict,
        "verdict_zh": verdict_zh,
        "phase62_ckpt_exists": bool(ckpt_exists),
        "phase62_ckpt": str(PHASE62_CKPT),
        "probe_class": type(probe).__name__,
        "action_dim": 2,
        "action_range": "unit disk then env clips by step_length_m; same as L3-B env.step",
        "idle_encoding_phase62": "one-hot index 12 on OPTION_DIM=13; assignment -1",
        "issues": issues,
        "reusable": [
            "Round-30 PPO core (GAE/clip/Adam) as an algorithm, not as loaded weights",
            "goal-conditioned idea: concatenate option one-hot onto UAV features",
            "completion + distance auxiliary reward template from Phase-6.2",
        ],
        "must_adapt": [
            "MAX_UAVS 5 -> L3-B M=6 (hard-coded pads and actor sharing)",
            "option feature attachment currently rejects non-(5,21) UAV stacks",
            "SINR-aware service: D=3 is not a guaranteed three-step completion",
            "Phase-6.2 checkpoint exists on disk but must not be loaded into an L3-B M=6 actor",
            "refuse silent fly-toward fallback labeled as PPO",
            "utility labels in 9.4.5 were generated under fly-toward, not a learned executor",
        ],
        "trained_this_phase": False,
        "random_init_eval_forbidden": True,
    }


__all__ = ["audit"]
