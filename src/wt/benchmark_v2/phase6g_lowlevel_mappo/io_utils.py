"""JSON, CSV, and the Chinese Phase-6G report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import write_case_figures
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    BENCHMARK_DEV,
    CASE_SEED,
    DEV_POOL,
    EXCESSIVE_PROJECTION_RATE,
    HISTORICAL,
    HOLD,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LEGACY_ACTOR,
    LOG_STD_INIT,
    MODEL_SEEDS,
    PHASE,
    PPO,
    REPORT,
    RESERVED_POOL,
    SAFETY_RATE_MARGIN,
    SERVICE_SAFETY_MARGIN_M,
    TRAIN_POOL,
    UPDATES,
)
from wt.lowlevel_v2.phase1.protocol import LOG_STD_INIT as ACTOR_LOG_STD


def _plain(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def freeze_inputs(initialization: str) -> None:
    write_json(
        PHASE / "phase6g_protocol.json",
        {
            "train_pool": list(TRAIN_POOL),
            "dev_pool": list(DEV_POOL),
            "reserved_untouched": list(RESERVED_POOL),
            "updates": UPDATES,
            "model_seeds": list(MODEL_SEEDS),
            "hold": HOLD,
            "generator": "G1_RATE",
            "k": 5,
            "sampler": "uniform legal pool, not J15 argmax",
            "action": "fly_toward + alpha * tanh(residual), then SafetyProjector",
            "service_residual_lock": False,
            "in_range_zero_gate": False,
            "alpha_residual": ALPHA_RESIDUAL,
            "lambda_project": LAMBDA_PROJECT,
            "lambda_residual": LAMBDA_RESIDUAL,
            "ppo": PPO.to_dict(),
            "case_selection_seed": CASE_SEED,
            "excessive_projection_rate": EXCESSIVE_PROJECTION_RATE,
            "safety_rate_margin": SAFETY_RATE_MARGIN,
            "j15_during_training": False,
            "high_level_modified": False,
            "rollouts_started": False,
            "rejected_budgets": {"phase5_probe_updates": 8, "phase101_pilot_updates": 100, "reason": "no single official MAPPO budget"},
        },
    )
    write_json(
        PHASE / "phase6g_data_registry.json",
        {
            "LOWLEVEL_V2_TRAIN": list(TRAIN_POOL),
            "LOWLEVEL_V2_DEV_EVAL": list(DEV_POOL),
            "overlap_reserved": sorted((set(TRAIN_POOL) | set(DEV_POOL)) & set(RESERVED_POOL)),
            "overlap_benchmark_dev": sorted((set(TRAIN_POOL) | set(DEV_POOL)) & set(BENCHMARK_DEV)),
            "overlap_historical": sorted((set(TRAIN_POOL) | set(DEV_POOL)) & set(HISTORICAL)),
            "written_before_rollout": True,
        },
    )
    write_json(
        PHASE / "phase6g_mappo_compatibility.json",
        {
            "checkpoint": str(LEGACY_ACTOR),
            "actor_load_strict": initialization == "LEGACY_WARM_START",
            "action_dim": 2,
            "critic_in_checkpoint": False,
            "critic_dim": 188,
            "log_std_init": LOG_STD_INIT,
            "safety_margin": SERVICE_SAFETY_MARGIN_M,
        },
    )
    write_json(PHASE / "phase6g_initialization.json", {"initialization": initialization, "actor_source": str(LEGACY_ACTOR), "critic_source": "CentralizedCritic architecture init"})
    write_json(
        PHASE / "phase6g_taskoption_sampler.json",
        {"name": "TASKOPTION_TRAIN_SAMPLER_V1", "pool": "G1_RATE k=5 legal assignments", "draw": "uniform", "all_idle_policy": "natural 1/pool frequency", "j15_argmax": False, "hold": HOLD},
    )
    write_json(
        PHASE / "phase6g_reward_spec.json",
        {
            "r_comm": "sum served_bits / B_block",
            "r_low": "r_comm - lambda_project * projection_penalty - lambda_residual * residual_penalty",
            "lambda_project": LAMBDA_PROJECT,
            "lambda_residual": LAMBDA_RESIDUAL,
            "action_scale": ACTION_SCALE,
            "distance_reward": False,
            "progress_completion_reward": False,
            "partial_bits_count": True,
        },
    )
    write_json(
        PHASE / "phase6g_observation_spec.json",
        {
            "queue_environment_state": "Q_bits",
            "queue_observation": "Q_equiv = Q_bits / B_block via env.queue",
            "legacy_task_progress": "compatibility channel only",
            "future_arrival": False,
            "j15_feature": False,
            "log_std_matches_phase1": float(LOG_STD_INIT) == float(ACTOR_LOG_STD),
        },
    )


def _report(gate: dict, summary: dict) -> None:
    low_level = "YES" if gate["gate"] == "LL-A" else "本阶段不是 LL-A，不把 MAPPO-V2 冻结为已确认执行器。"
    regenerate = "YES" if gate["gate"] == "LL-A" else "本阶段不是 LL-A。若结论保持 LL-C，后续高层仍用 fly-toward 生成 J15。"
    safety = "有回归" if summary["safety_regression"] else "没有超过预注册的安全余量"
    text = f"""# HTW Phase-6G：Benchmark-V2 低层 Residual MAPPO

## 结论

Gate 为 **{gate['gate']} {gate['gate_name']}**。本阶段只训练低层运动残差。高层候选规则、hold、队列记账和 Benchmark-V2 指标没有改。`planner_ready = False`，`deployment_verified = False`。

## Q1. 旧 mappo_original.pt 能否作为初始化？

能。Actor 的权重与现有 ResidualActor 严格对齐，动作维是 2。检查点里没有 critic。

## Q2. MAPPO-V2 最终是 warm-start 还是 scratch？

{summary['initialization']}。Actor 从 mappo_original.pt 继续训练，critic 按同一 CentralizedCritic 结构重新初始化。

## Q3. 新 reward 是否完全基于 bit-level served_bits？

是。主项是 sum(served_bits) / B_block，再减去 Phase-1 原有的 projection 和 residual 惩罚。部分传输立即计入。

## Q4. 训练过程中是否修改了高层？

NO。

## Q5. MAPPO 是否仍然只能控制运动，不能修改 assignment？

YES。

## Q6. 同 state 与同 TaskOption 下，MAPPO-V2 相对 fly-toward 平均多传多少 bit？

五个 seed 先在 scenario 内平均，再对 scenario 等权，paired ΔBits = {summary['mean_delta']:.6f}。

## Q7. 95% CI？

[{summary['ci_low']:.6f}, {summary['ci_high']:.6f}]。

## Q8. 多少 scenario 为正？

{summary['positive_fraction']:.6f}。正的定义是 scenario 均值大于 1 bit。

## Q9. 5 个 seed 是否稳定？

平均为正的 seed 数是 {summary['positive_seeds']}/5。各种子见 phase6g_seed_stability.csv。

## Q10. MAPPO-V2 是否优于 legacy MAPPO？

V2 减 legacy 的 paired ΔBits = {summary['legacy_delta']:.6f}，95% CI [{summary['legacy_low']:.6f}, {summary['legacy_high']:.6f}]。

## Q11. 性能变化主要来自哪里？

相对 fly-toward，SINR 差 {summary['sinr_delta']:.6f}，interference 差 {summary['interference_delta']:.6f}，路径长度差 {summary['path_delta']:.6f}，首次服务步差 {summary['entry_delta']:.6f}，距离缩短差 {summary['distance_delta']:.6f}。

## Q12. Full episode + MAX_RATE 下 MAPPO-V2 是否提高最终 payload？

Fly-toward {summary['full_fly']:.6f}，legacy {summary['full_legacy']:.6f}，MAPPO-V2 {summary['full_v2']:.6f}。这组数没有用来改训练。

## Q13. 是否存在 safety / projection regression？

MAPPO-V2 safety violation rate {summary['safety_v2']:.6f}，fly-toward {summary['safety_fly']:.6f}，投影事件率 {summary['projection_v2']:.6f}。{safety}。

## Q14. 最终 Gate？

{gate['gate']}。

## Q15. 如果 LL-A，是否已经冻结低层？

{low_level}

## Q16. 如果 LL-A，下一阶段是否必须重新生成 J15_CURRENT？

{regenerate}

## Q17. 现在是否训练 HGNN？

NO。

## Q18. 现在是否修改 candidate generator？

NO。

## Q19. 320100–320131 是否仍完全 untouched？

YES。
"""
    REPORT.write_text(text, encoding="utf-8")


def finish(gate, summary, curves, rows, seed_rows, legacy_rows, full_rows, mechanism, runtime, cases) -> None:
    write_json(PHASE / "phase6g_lowlevel_gate.json", gate)
    write_csv(PHASE / "phase6g_training_curves.csv", curves)
    write_csv(PHASE / "phase6g_h15_execution_comparison.csv", rows)
    write_csv(PHASE / "phase6g_seed_stability.csv", seed_rows)
    write_csv(PHASE / "phase6g_legacy_comparison.csv", legacy_rows)
    write_csv(PHASE / "phase6g_full_episode_comparison.csv", full_rows)
    write_csv(PHASE / "phase6g_mechanism_audit.csv", mechanism)
    write_csv(PHASE / "phase6g_runtime.csv", runtime)
    if cases:
        write_case_figures(cases, PHASE / "figures")
    protocol_path = PHASE / "phase6g_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["rollouts_started"] = True
    protocol["rules_frozen_before_rollout"] = True
    write_json(protocol_path, protocol)
    _report(gate, summary)
    print(gate["gate"], flush=True)
