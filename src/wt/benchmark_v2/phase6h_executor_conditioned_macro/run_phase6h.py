"""Run the frozen executor-conditioned macro comparison."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import checkpoint_sha, primary_checkpoint
from wt.benchmark_v2.phase6h_executor_conditioned_macro.jobs import (
    closed_loop_scenario,
    label_shift_scenario,
    mechanism_scenario,
    robustness_scenario,
)
from wt.benchmark_v2.phase6h_executor_conditioned_macro.metrics import bootstrap_mean, median, positive_fraction, spearman
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import (
    ALIGNMENT_SPEARMAN_GAIN,
    ASSIGNMENT_DISAGREE_MATERIAL,
    CHECKPOINT_DIR,
    CLOSED_LOOP_POOL,
    LABEL_SHIFT_POOL,
    PHASE,
    POSITIVE_BIT,
    PRIMARY_SEED,
    REPORT,
    RESERVED_POOL,
    ROBUSTNESS_POOL,
    ROBUSTNESS_SEEDS,
    SEED_CV_LIMIT,
    TOP1_DISAGREEMENT_MATERIAL,
    choose_gate,
)


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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _write_csv(path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def _scenario_mean(rows: list[dict], key: str) -> np.ndarray:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["scenario_id"]), []).append(float(row[key]))
    return np.asarray([float(np.mean(grouped[scenario])) for scenario in sorted(grouped)], dtype=np.float64)


def _pair(results: list[dict], left: str, right: str) -> np.ndarray:
    payloads = {row["controller"]: {} for row in results}
    for row in results:
        payloads[row["controller"]][int(row["scenario_id"])] = float(row["payload_bits"])
    return np.asarray([payloads[left][scenario] - payloads[right][scenario] for scenario in CLOSED_LOOP_POOL], dtype=np.float64)


def _alignment(decisions: list[dict], controller: str) -> dict:
    rows = [row for row in decisions if row["controller"] == controller and int(row["truncated_window"]) == 0 and row["predicted_j15"] is not None]
    predicted = np.asarray([row["predicted_j15"] for row in rows], dtype=np.float64)
    realized = np.asarray([row["served_bits"] for row in rows], dtype=np.float64)
    error = predicted - realized
    return {
        "controller": controller,
        "n": int(predicted.size),
        "spearman": spearman(predicted, realized) if predicted.size else 0.0,
        "mae": float(np.mean(np.abs(error))) if predicted.size else 0.0,
        "bias": float(np.mean(error)) if predicted.size else 0.0,
    }


def _mechanism(rows: list[dict]) -> dict:
    paired = {}
    for row in rows:
        paired.setdefault((row["scenario_id"], row["decision_time"]), {})[row["executor"]] = row
    disagree = []
    matched = []
    for key, item in paired.items():
        if "fly_toward" not in item or "mappo_v2_seed0" not in item:
            continue
        same = item["fly_toward"]["assignment"] == item["mappo_v2_seed0"]["assignment"]
        disagree.append(0.0 if same else 1.0)
        if same:
            matched.append(float(item["mappo_v2_seed0"]["served_bits"]) - float(item["fly_toward"]["served_bits"]))
        _ = key
    disagree_fraction = float(np.mean(disagree)) if disagree else 0.0
    matched_delta = float(np.mean(matched)) if matched else 0.0
    if disagree_fraction >= ASSIGNMENT_DISAGREE_MATERIAL and matched_delta >= -POSITIVE_BIT:
        cause = "HIGH_LEVEL_POLICY_NOT_EXECUTOR_ADAPTED"
    elif matched_delta < -POSITIVE_BIT and disagree_fraction < ASSIGNMENT_DISAGREE_MATERIAL:
        cause = "LOW_LEVEL_EXECUTION_FAILURE"
    elif disagree_fraction >= ASSIGNMENT_DISAGREE_MATERIAL and matched_delta < -POSITIVE_BIT:
        cause = "MIXED"
    else:
        cause = "NO_CLEAR_SHIFT"
    return {"assignment_disagreement": disagree_fraction, "same_assignment_delta_bits": matched_delta, "cause": cause, "matched_windows": len(matched)}


def _boundary(rows: list[dict]) -> dict:
    groups: dict[tuple, list[int]] = {}
    for row in rows:
        groups.setdefault((row["split"], int(row["scenario_id"]), int(row["decision_time"])), []).append(int(row["edge_rank"]))
    summary = {}
    for split in sorted({key[0] for key in groups}):
        chosen = [ranks for key, ranks in groups.items() if key[0] == split]
        flat = [rank for ranks in chosen for rank in ranks]
        summary[split] = {
            "decisions": len(chosen),
            "rank5_assignment_fraction": float(np.mean([1.0 if 5 in ranks else 0.0 for ranks in chosen])) if chosen else 0.0,
            **{f"rank_{rank}_edge_fraction": float(np.mean([item == rank for item in flat])) if flat else 0.0 for rank in range(1, 6)},
        }
    return summary


def _report(gate: dict, summary: dict) -> None:
    dataset = "YES" if gate["gate"] == "ALIGN-A" else "本阶段不是 ALIGN-A，不开始高层数据集。"
    text = f"""# HTW Phase-6H：按冻结低层执行器重算宏动作价值

## 结论

Gate 为 **{gate['gate']} {gate['gate_name']}**。主低层检查点固定为 seed {PRIMARY_SEED}，没有按开发表现挑选。没有训练，没有改 MAPPO-V2，没有改候选规则。`planner_ready = False`，`deployment_verified = False`。

## Q1. 为什么同任务 H15 上 MAPPO 更好，MAX_RATE 全 episode 却没有提高？

在 320000–320015 上，MAX_RATE 分配的决策不一致比例是 {summary['assignment_disagreement']:.6f}。同一分配下 MAPPO 减 fly-toward 的窗口服务差是 {summary['same_assignment_delta_bits']:.6f} bit。按预先写好的规则，原因归类为 {summary['cause']}。

## Q2. 两套 J15 的 assignment ranking 差多少？

320000–320015 上，scenario 等权的 Spearman 是 {summary['spearman']:.6f}，Kendall 是 {summary['kendall']:.6f}，Top-3 overlap 是 {summary['top3']:.6f}，Top-5 overlap 是 {summary['top5']:.6f}。

## Q3. Top-1 disagreement 是多少？

{summary['top1_disagreement']:.6f}。MAPPO top-1 在 fly ranking 中的平均名次是 {summary['mappo_rank_in_fly']:.3f}。

## Q4. matched planner 是否好于 mismatched planner？

C2 减 C1 的平均 payload 差是 {summary['c2_c1_mean']:.6f} bit，95% CI [{summary['c2_c1_low']:.6f}, {summary['c2_c1_high']:.6f}]。

## Q5. matched planner 是否稳定超过同一个 MAPPO 下的 MAX_RATE？

{"是" if summary['c2_c0_pass'] else "否"}。

## Q6. C2 vs C0 的 mean、CI、positive fraction？

mean {summary['c2_c0_mean']:.6f}，median {summary['c2_c0_median']:.6f}，95% CI [{summary['c2_c0_low']:.6f}, {summary['c2_c0_high']:.6f}]，positive fraction {summary['c2_c0_positive']:.6f}。

## Q7. C2 vs C1 的 mean 和 CI？

mean {summary['c2_c1_mean']:.6f}，95% CI [{summary['c2_c1_low']:.6f}, {summary['c2_c1_high']:.6f}]。

## Q8. 预测的宏价值是否更接近真实 H15 return？

完整 15 步窗口上，C1 Spearman {summary['align_c1']:.6f}、Kendall {summary['kendall_c1']:.6f}、MAE {summary['mae_c1']:.6f}、bias {summary['bias_c1']:.6f}。C2 Spearman {summary['align_c2']:.6f}、Kendall {summary['kendall_c2']:.6f}、MAE {summary['mae_c2']:.6f}、bias {summary['bias_c2']:.6f}。alignment improved = {summary['alignment_improved']}。t=45 的截断窗口没有放进这个相关。

## Q9. MAX_RATE + MAPPO-V2 下降的原因是什么？

{summary['cause']}。分配是否跟着轨迹改变，以及同一分配下服务是否下降，都记在 phase6h_maxrate_executor_shift.csv。

## Q10. k=5 最优是否经常压在 rank-5 边界？

closed-loop C2 中，含 rank-5 edge 的最优分配比例是 {summary['rank5_fraction']:.6f}。本阶段没有把 k 扩到 6。

## Q11. 不同 MAPPO seed 对同一 assignment 的 H15 return 是否稳定？

320048–320063 上，固定 assignment 的平均变异系数是 {summary['seed_cv']:.6f}。seed0 选出的分配在其他 seed 上进入前两名的比例是 {summary['seed_top2']:.6f}。没有为每个 seed 重新做穷举搜索。

## Q12. 高层 macro label 是否必须由实际冻结的低层执行器重新生成？

{summary['must_regenerate']}。标签漂移达到预注册门槛：{summary['label_shift_material']}。C2 相对 C1 的比较见 Q7。

## Q13. 最终 Gate？

{gate['gate']}。

## Q14. 如果 ALIGN-A，下一阶段是否可以开始构建 Benchmark-V2 learned high-level dataset？

{dataset}

## Q15. 现在是否直接训练 HGNN？

NO。

## Q16. 现在是否重新训练 MAPPO？

NO。

## Q17. 320100–320131 是否仍完全 untouched？

YES。
"""
    REPORT.write_text(text, encoding="utf-8")


def _fail(message: str) -> None:
    gate = choose_gate(failures=1, seed_cv=0.0, c2_c0_pass=False, c2_c1_pass=False, label_shift_material=False, alignment_improved=False)
    gate["error"] = message
    _write_json(PHASE / "phase6h_executor_conditioned_macro_gate.json", gate)
    print(gate["gate"], flush=True)


def _pool_map(function, jobs):
    with ProcessPoolExecutor(max_workers=8) as pool:
        return list(pool.map(function, jobs, chunksize=1))


def main() -> None:
    hashes = {path.name: checkpoint_sha(path) for path in sorted(CHECKPOINT_DIR.glob("phase6g_seed*.pt"))}
    if checkpoint_sha(primary_checkpoint()) != hashes["phase6g_seed0.pt"]:
        raise RuntimeError("primary checkpoint hash mismatch")
    _write_json(
        PHASE / "phase6h_protocol.json",
        {
            "primary_seed_rule": "MINIMUM_SEED_ID",
            "primary_seed": PRIMARY_SEED,
            "label_shift_pool": list(LABEL_SHIFT_POOL),
            "closed_loop_pool": list(CLOSED_LOOP_POOL),
            "robustness_pool": list(ROBUSTNESS_POOL),
            "reserved_untouched": list(RESERVED_POOL),
            "label_shift_state_source": "MAX_RATE_INDEPENDENT + fly-toward decision states",
            "j15_horizon": "min(15, episode_horizon - t)",
            "future_arrivals_in_j15": 0,
            "robustness_execution_uses_realized_arrivals": True,
            "top1_disagreement_material": TOP1_DISAGREEMENT_MATERIAL,
            "alignment_spearman_gain": ALIGNMENT_SPEARMAN_GAIN,
            "seed_cv_limit": SEED_CV_LIMIT,
            "actor_stateless_expected": True,
            "rollouts_started": False,
            "retrained_mappo": False,
            "trained_hgnn": False,
        },
    )
    _write_json(PHASE / "phase6h_lowlevel_freeze_manifest.json", {"checkpoints": hashes, "primary": "phase6g_seed0.pt", "robustness_seeds": list(ROBUSTNESS_SEEDS)})
    try:
        label = _pool_map(label_shift_scenario, list(LABEL_SHIFT_POOL))
        closed_jobs = [(scenario, controller) for scenario in CLOSED_LOOP_POOL for controller in ("c2", "c1", "c0", "c3")]
        closed = _pool_map(closed_loop_scenario, closed_jobs)
        mechanism = _pool_map(mechanism_scenario, list(LABEL_SHIFT_POOL))
        robust = _pool_map(robustness_scenario, list(ROBUSTNESS_POOL))
    except Exception as exc:
        _fail(str(exc))
        raise
    if any(checkpoint_sha(CHECKPOINT_DIR / name) != digest for name, digest in hashes.items()):
        _fail("checkpoint bytes changed")
        raise RuntimeError("checkpoint bytes changed")
    label_rows = [row for item in label for row in item["rows"]]
    shift_mean = {
        "spearman": float(np.mean(_scenario_mean(label_rows, "spearman"))),
        "kendall": float(np.mean(_scenario_mean(label_rows, "kendall"))),
        "top3": float(np.mean(_scenario_mean(label_rows, "top3_overlap"))),
        "top5": float(np.mean(_scenario_mean(label_rows, "top5_overlap"))),
        "top1_disagreement": float(np.mean(_scenario_mean(label_rows, "argmax_disagreement"))),
        "mappo_rank_in_fly": float(np.mean(_scenario_mean(label_rows, "mappo_top1_fly_rank"))),
    }
    decisions = [row for item in closed for row in item["decisions"]]
    align_c1 = _alignment(decisions, "c1")
    align_c2 = _alignment(decisions, "c2")
    from wt.benchmark_v2.phase6h_executor_conditioned_macro.metrics import kendall

    align_rows = []
    for name, align in (("c1", align_c1), ("c2", align_c2)):
        rows = [row for row in decisions if row["controller"] == name and int(row["truncated_window"]) == 0 and row["predicted_j15"] is not None]
        tau = kendall(np.asarray([row["predicted_j15"] for row in rows], dtype=np.float64), np.asarray([row["served_bits"] for row in rows], dtype=np.float64)) if rows else 0.0
        align_rows.append({**align, "kendall": tau})
    c2_c0 = _pair(closed, "c2", "c0")
    c2_c1 = _pair(closed, "c2", "c1")
    c2_c0_mean, c2_c0_low, c2_c0_high = bootstrap_mean(c2_c0)
    c2_c1_mean, c2_c1_low, c2_c1_high = bootstrap_mean(c2_c1)
    c2_c0_pass = c2_c0_mean > 0.0 and c2_c0_low > 0.0 and positive_fraction(c2_c0) >= 0.60
    c2_c1_pass = c2_c1_mean > 0.0 and c2_c1_low > 0.0
    alignment_improved = float(align_c2["spearman"] - align_c1["spearman"]) >= ALIGNMENT_SPEARMAN_GAIN
    label_shift_material = shift_mean["top1_disagreement"] >= TOP1_DISAGREEMENT_MATERIAL
    robust_rows = [row for item in robust for row in item["rows"]]
    unique_cv = {(row["scenario_id"], row["decision_time"]): float(row["slot_cv_mean"]) for row in robust_rows}
    seed_cv = float(np.mean(list(unique_cv.values()))) if unique_cv else 0.0
    seed_top2 = float(np.mean([row["top2_for_other_seed"] for row in robust_rows])) if robust_rows else 0.0
    mechanism_rows = [row for item in mechanism for episode in (item["fly"], item["mappo"]) for row in episode["decisions"]]
    cause = _mechanism(mechanism_rows)
    boundary_rows = [row for item in label for row in item["boundary"]] + [row for item in closed for row in item["boundary"]]
    boundary = _boundary(boundary_rows)
    failures = 0
    gate = choose_gate(
        failures=failures,
        seed_cv=seed_cv,
        c2_c0_pass=c2_c0_pass,
        c2_c1_pass=c2_c1_pass,
        label_shift_material=label_shift_material,
        alignment_improved=alignment_improved,
    )
    summary = {
        **shift_mean,
        **cause,
        "c2_c0_mean": c2_c0_mean,
        "c2_c0_low": c2_c0_low,
        "c2_c0_high": c2_c0_high,
        "c2_c0_median": median(c2_c0),
        "c2_c0_positive": positive_fraction(c2_c0),
        "c2_c0_pass": c2_c0_pass,
        "c2_c1_mean": c2_c1_mean,
        "c2_c1_low": c2_c1_low,
        "c2_c1_high": c2_c1_high,
        "align_c1": align_c1["spearman"],
        "align_c2": align_c2["spearman"],
        "kendall_c1": float(next(row["kendall"] for row in align_rows if row["controller"] == "c1")),
        "kendall_c2": float(next(row["kendall"] for row in align_rows if row["controller"] == "c2")),
        "mae_c1": align_c1["mae"],
        "mae_c2": align_c2["mae"],
        "bias_c1": align_c1["bias"],
        "bias_c2": align_c2["bias"],
        "alignment_improved": alignment_improved,
        "label_shift_material": label_shift_material,
        "must_regenerate": "YES" if gate["gate"] in {"ALIGN-A", "ALIGN-B"} else "测得的差异见 Q2、Q3、Q7，本阶段没有把不一致的结果改写成必须重生成。",
        "rank5_fraction": boundary.get("closed_loop_c2", {}).get("rank5_assignment_fraction", 0.0),
        "seed_cv": seed_cv,
        "seed_top2": seed_top2,
    }
    _write_csv(PHASE / "phase6h_shadow_isolation.csv", [row for item in label for row in item["isolation"]] + [row for item in closed for row in item["isolation"]])
    _write_csv(PHASE / "phase6h_label_shift.csv", label_rows)
    _write_csv(PHASE / "phase6h_assignment_ranking_shift.csv", label_rows)
    _write_csv(PHASE / "phase6h_closed_loop_results.csv", [{"scenario_id": row["scenario_id"], "controller": row["controller"], "payload_bits": row["payload_bits"], "residual_queue_bits": row["residual_queue_bits"], "dropped_bits": row["dropped_bits"]} for row in closed])
    _write_csv(PHASE / "phase6h_c2_vs_c0.csv", [{"scenario_id": scenario, "delta_bits": float(c2_c0[index])} for index, scenario in enumerate(CLOSED_LOOP_POOL)])
    _write_csv(PHASE / "phase6h_c2_vs_c1.csv", [{"scenario_id": scenario, "delta_bits": float(c2_c1[index])} for index, scenario in enumerate(CLOSED_LOOP_POOL)])
    _write_csv(PHASE / "phase6h_alignment_metrics.csv", align_rows)
    _write_csv(PHASE / "phase6h_maxrate_executor_shift.csv", mechanism_rows)
    _write_csv(PHASE / "phase6h_candidate_boundary.csv", boundary_rows)
    _write_csv(PHASE / "phase6h_seed_robustness.csv", robust_rows)
    runtime_rows = [row for item in label for row in item["runtime"]] + [row for item in closed for row in item["runtime"]]
    runtime_rows.extend(
        {"scenario_id": int(item["fly"]["decisions"][0]["scenario_id"]), "stage": "mechanism_pair_episode", "episode_seconds": float(item["episode_seconds"])}
        for item in mechanism
    )
    runtime_rows.extend(
        {"scenario_id": int(item["rows"][0]["scenario_id"]) if item["rows"] else -1, "stage": "robustness_episode", "episode_seconds": float(item["episode_seconds"])}
        for item in robust
    )
    _write_csv(PHASE / "phase6h_runtime.csv", runtime_rows)
    _write_json(PHASE / "phase6h_executor_conditioned_macro_gate.json", gate)
    protocol = json.loads((PHASE / "phase6h_protocol.json").read_text(encoding="utf-8"))
    protocol["rollouts_started"] = True
    protocol["rules_frozen_before_mappo_j15"] = True
    _write_json(PHASE / "phase6h_protocol.json", protocol)
    _report(gate, summary)
    print(gate["gate"], flush=True)


if __name__ == "__main__":
    main()
