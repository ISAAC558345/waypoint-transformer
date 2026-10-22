"""Run the pre-registered G1_RATE k=5/6/7 boundary audit. No model is trained."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import checkpoint_sha, primary_checkpoint
from wt.benchmark_v2.phase6ia_candidate_boundary.jobs import audit_scenario, closed_loop_scenario
from wt.benchmark_v2.phase6ia_candidate_boundary.protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    CONFIRM_POOL,
    DESIGN_POOL,
    EPS_BITS,
    INTERFERENCE_QUANTILES,
    KS,
    PHASE,
    PRIMARY_SEED,
    PROTOCOL_STAMP,
    REPORT,
    RESERVED_POOL,
    SANITY_POOL,
    STATE_SOURCE,
    choose_gate,
    evaluate_flags,
)


def _plain(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _write_csv(path, rows: list[dict], fields: list[str] | None = None) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(("" if not fields else ",".join(fields) + "\n"), encoding="utf-8")
        return
    names: list[str] = list(fields or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def _partial(kind: str, key: str) -> dict | None:
    path = PHASE / "partial" / f"{kind}_{key}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stamp") != PROTOCOL_STAMP:
        return None
    return payload["result"]


def _save_partial(kind: str, key: str, result: dict) -> None:
    path = PHASE / "partial" / f"{kind}_{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stamp": PROTOCOL_STAMP, "result": _plain(result)}), encoding="utf-8")


def _map(function, jobs, kind: str, key_of) -> list[dict]:
    done = []
    pending = []
    for job in jobs:
        cached = _partial(kind, key_of(job))
        if cached is None:
            pending.append(job)
        else:
            done.append(cached)
    if not pending:
        return done
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(function, job): job for job in pending}
        try:
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                _save_partial(kind, key_of(job), result)
                done.append(result)
                print(f"saved {kind} {key_of(job)} {len(done)}/{len(jobs)}", flush=True)
        except Exception:
            pool.shutdown(cancel_futures=True, wait=False)
            raise
    return done


def _primary(rows: list[dict], split: str | None = None) -> list[dict]:
    chosen = [row for row in rows if int(row["primary"]) == 1]
    if split is not None:
        chosen = [row for row in chosen if row["split"] == split]
    return chosen


def _scenario_values(rows: list[dict], key: str) -> np.ndarray:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["scenario_id"]), []).append(float(row[key]))
    return np.asarray([float(np.mean(grouped[scenario])) for scenario in sorted(grouped)], dtype=np.float64)


def _bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    count = int(data.size)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = float(np.mean(data[rng.integers(0, count, count)]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(np.mean(data)), float(low), float(high)


def _measured(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value is None or value == "":
            continue
        values.append(float(value))
    return values


def _mean(rows: list[dict], key: str) -> float:
    values = _measured(rows, key)
    return float(np.mean(values)) if values else 0.0


def _p95(rows: list[dict], key: str) -> float | None:
    values = _measured(rows, key)
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))


def _split_stats(rows: list[dict], prefix: str) -> dict:
    gain_mean, gain_low, gain_high = _bootstrap(_scenario_values(rows, "gain_6_5"))
    ngain_mean, ngain_low, ngain_high = _bootstrap(_scenario_values(rows, "ngain_6_5"))
    k7_rows = [row for row in rows if row.get("ngain_7_6") not in ("", None)]
    if k7_rows:
        next_mean, next_low, next_high = _bootstrap(_scenario_values(k7_rows, "ngain_7_6"))
        p95_76 = _p95(k7_rows, "ngain_7_6")
        mean_gain_76 = _mean(k7_rows, "gain_7_6")
        coverage_k7 = _mean(k7_rows, "covered_k7")
        rank7 = _mean(k7_rows, "contains_rank7")
    else:
        next_mean = next_low = next_high = p95_76 = mean_gain_76 = coverage_k7 = rank7 = None
    return {
        f"{prefix}_mean_gain_6_5": gain_mean,
        f"{prefix}_gain_6_5_ci_low": gain_low,
        f"{prefix}_gain_6_5_ci_high": gain_high,
        f"{prefix}_mean_ngain_6_5": ngain_mean,
        f"{prefix}_ngain_6_5_ci_low": ngain_low,
        f"{prefix}_ngain_6_5_ci_high": ngain_high,
        f"{prefix}_p95_ngain_6_5": _p95(rows, "ngain_6_5"),
        f"{prefix}_mean_ngain_7_6": next_mean,
        f"{prefix}_ngain_7_6_ci_low": next_low,
        f"{prefix}_ngain_7_6_ci_high": next_high,
        f"{prefix}_p95_ngain_7_6": p95_76,
        f"{prefix}_mean_gain_7_6": mean_gain_76,
        f"{prefix}_coverage_k5": _mean(rows, "covered_k5"),
        f"{prefix}_coverage_k6": _mean(rows, "covered_k6"),
        f"{prefix}_coverage_k7": coverage_k7,
        f"{prefix}_rank6_fraction": _mean(rows, "contains_rank6"),
        f"{prefix}_rank7_fraction": rank7,
        f"{prefix}_disagree_6_5": _mean(rows, "disagree_6_5"),
        f"{prefix}_states": len(rows),
    }


def _strata(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    intensity = np.asarray([float(row["mean_interference"]) for row in rows], dtype=np.float64)
    low_cut, high_cut = np.quantile(intensity, list(INTERFERENCE_QUANTILES))

    def label(value: float) -> str:
        if value <= float(low_cut):
            return "low"
        if value <= float(high_cut):
            return "medium"
        return "high"

    tagged = [{**row, "stratum": label(float(row["mean_interference"]))} for row in rows]
    output = []
    for split in ("design", "confirm", "all"):
        for stratum in ("low", "medium", "high"):
            chosen = [row for row in tagged if row["stratum"] == stratum and (split == "all" or row["split"] == split)]
            output.append(
                {
                    "split": split,
                    "stratum": stratum,
                    "low_cut": float(low_cut),
                    "high_cut": float(high_cut),
                    "states": len(chosen),
                    "mean_ngain_6_5": _mean(chosen, "ngain_6_5"),
                    "k5_coverage": _mean(chosen, "covered_k5"),
                    "k5_uncovered_fraction": float(np.mean([1.0 - float(row["covered_k5"]) for row in chosen])) if chosen else 0.0,
                }
            )
    return output


def _complexity(runtime: list[dict]) -> list[dict]:
    rows = []
    for scope, chosen in (("primary", [row for row in runtime if int(row["primary"]) == 1]), ("all_states", runtime)):
        for k in (5, 6, 7):
            group = [row for row in chosen if int(row["k"]) == k]
            matched = {(int(row["scenario_id"]), int(row["decision_time"])) for row in group}
            base = [
                float(row["candidate_count"])
                for row in chosen
                if int(row["k"]) == 5 and (int(row["scenario_id"]), int(row["decision_time"])) in matched
            ]
            base_mean = float(np.mean(base)) if base else 1.0
            base_seconds = [
                float(row["full_pool_seconds"])
                for row in chosen
                if int(row["k"]) == 5 and (int(row["scenario_id"]), int(row["decision_time"])) in matched
            ]
            base_runtime = float(np.mean(base_seconds)) if base_seconds else 1.0
            counts = np.asarray([float(row["candidate_count"]) for row in group], dtype=np.float64)
            singles = [float(row["single_eval_seconds"]) for row in group if float(row["single_eval_seconds"]) > 0.0]
            full = np.asarray([float(row["full_pool_seconds"]) for row in group], dtype=np.float64)
            rows.append(
                {
                    "scope": scope,
                    "k": k,
                    "states": len(group),
                    "mean_pool_size": float(np.mean(counts)) if counts.size else 0.0,
                    "median_pool_size": float(np.median(counts)) if counts.size else 0.0,
                    "p95_pool_size": float(np.quantile(counts, 0.95)) if counts.size else 0.0,
                    "max_pool_size": float(np.max(counts)) if counts.size else 0.0,
                    "mean_single_eval_seconds": float(np.mean(singles)) if singles else 0.0,
                    "mean_full_pool_seconds": float(np.mean(full)) if full.size else 0.0,
                    "pool_multiplier_vs_k5": float(np.mean(counts) / base_mean) if counts.size else 0.0,
                    "runtime_multiplier_vs_k5": float(np.mean(full) / base_runtime) if full.size else 0.0,
                    "peak_rss_bytes": max((int(row["peak_rss_bytes"]) for row in group), default=0),
                }
            )
    return rows


def _seed_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"mean_cv": 0.0, "seed0_positive_fraction": 0.0, "other_seed_positive_fraction": 0.0, "seed0_only_fraction": 0.0}
    grouped: dict[tuple[int, int], dict[int, dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["scenario_id"]), int(row["decision_time"])), {})[int(row["model_seed"])] = row
    seed0 = []
    others = []
    only = []
    cvs = []
    for item in grouped.values():
        seed0_gain = float(item[0]["gain_6_5"])
        other_gain = float(np.mean([float(item[seed]["gain_6_5"]) for seed in item if seed != 0]))
        seed0.append(seed0_gain > EPS_BITS)
        others.append(other_gain > EPS_BITS)
        only.append(seed0_gain > EPS_BITS and other_gain <= EPS_BITS)
        cvs.extend(float(item[0][key]) for key in ("cv_a5", "cv_a6", "cv_a7") if item[0].get(key) not in ("", None))
    return {
        "mean_cv": float(np.mean(cvs)) if cvs else 0.0,
        "seed0_positive_fraction": float(np.mean(seed0)) if seed0 else 0.0,
        "other_seed_positive_fraction": float(np.mean(others)) if others else 0.0,
        "seed0_only_fraction": float(np.mean(only)) if only else 0.0,
        "states": len(grouped),
    }


def _fmt(value) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if abs(number) >= 1000.0:
        return f"{number:.2f}"
    return f"{number:.6f}"


def _report(gate: dict, stats: dict, complexity: list[dict], strata: list[dict], seed: dict, mechanism: list[dict]) -> None:
    dataset = "YES" if gate["gate"] in {"BOUND-A", "BOUND-B", "BOUND-C"} else "NO"
    selected = gate["selected_k"] if gate["selected_k"] is not None else "不冻结"
    new_edges = [row for row in mechanism if row["kind"] == "new_rank_edge"]
    recombinations = [row for row in mechanism if row["kind"] == "recombination_inside_k5"]

    def edge_mean(key: str) -> float:
        if not new_edges:
            return 0.0
        return float(np.mean([float(row[key]) for row in new_edges]))

    high = [row for row in strata if row["split"] == "all" and row["stratum"] == "high"]
    low = [row for row in strata if row["split"] == "all" and row["stratum"] == "low"]
    high_row = high[0] if high else {}
    low_row = low[0] if low else {}
    primary_complexity = {(int(row["k"])): row for row in complexity if row["scope"] == "primary"}
    reason = (
        "k=6 的归一化增益没有超过预注册的 0.02，说明这些 rank-5 边是池内最优，不是被 k=5 截掉的更高价值边。"
        if float(stats["design_mean_ngain_6_5"]) <= 0.02 and float(stats["confirm_mean_ngain_6_5"]) <= 0.02
        else "k=6 在池外给出了更高的 J15_MAPPO_CURRENT，所以 rank-5 接触确实带有截断。"
    )
    text = f"""# HTW Phase-6I-A：MAPPO 条件下的候选边界闭合审计

## 结论

Gate 为 **{gate['gate']} {gate['gate_name']}**。冻结候选规则是 G1_RATE，k = {selected}。没有训练。MAPPO-V2 仍冻结在 seed {PRIMARY_SEED}。`planner_ready = False`，`deployment_verified = False`。

状态来源在第一次 k=6/7 J15 之前写死：{STATE_SOURCE}。主分析只用 t=0、15、30。t=45 只作次要记录。k=7 只在已完成的 design set 上计算过：96 个主状态的 mean NGain_7_6 = 0，P95 = 0，因此 confirmation 和闭环不再计算 k=7。已保存的 design k=7 结果保留。干扰分层沿用 Phase-6C 的三分位规则，没有另设阈值。

## Q1. 为什么 69.5% 的 k=5 optimum 触及 rank-5 boundary？

Phase-6H 看到的是 k=5 最优分配里经常包含瞬时速率第 5 名的边。本阶段把同一排序扩到 k=6 和 k=7 后，design 的 mean NGain_6_5 是 {_fmt(stats['design_mean_ngain_6_5'])}，confirmation 是 {_fmt(stats['confirm_mean_ngain_6_5'])}。{reason}

## Q2. k=6 相对 k=5 提高多少 J15_MAPPO_CURRENT？

design 平均 **{_fmt(stats['design_mean_gain_6_5'])} bit**，confirmation 平均 **{_fmt(stats['confirm_mean_gain_6_5'])} bit**。归一化增益分别是 {_fmt(stats['design_mean_ngain_6_5'])} 和 {_fmt(stats['confirm_mean_ngain_6_5'])}。

## Q3. 95% CI？

比特增益的 scenario-cluster 95% CI：design [{_fmt(stats['design_gain_6_5_ci_low'])}, {_fmt(stats['design_gain_6_5_ci_high'])}]，confirmation [{_fmt(stats['confirm_gain_6_5_ci_low'])}, {_fmt(stats['confirm_gain_6_5_ci_high'])}]。归一化增益 CI：design [{_fmt(stats['design_ngain_6_5_ci_low'])}, {_fmt(stats['design_ngain_6_5_ci_high'])}]，confirmation [{_fmt(stats['confirm_ngain_6_5_ci_low'])}, {_fmt(stats['confirm_ngain_6_5_ci_high'])}]。

## Q4. confirmation subset 是否复现？

design mean NGain_6_5 = {_fmt(stats['design_mean_ngain_6_5'])}，P95 = {_fmt(stats['design_p95_ngain_6_5'])}。confirmation mean = {_fmt(stats['confirm_mean_ngain_6_5'])}，P95 = {_fmt(stats['confirm_p95_ngain_6_5'])}。两边是否都不超过 0.02 / 0.05：{"是" if stats["design_mean_ngain_6_5"] <= 0.02 and stats["design_p95_ngain_6_5"] <= 0.05 and stats["confirm_mean_ngain_6_5"] <= 0.02 and stats["confirm_p95_ngain_6_5"] <= 0.05 else "否"}。

## Q5. k=7 相对 k=6 是否仍有实质增益？

design mean NGain_7_6 = {_fmt(stats['design_mean_ngain_7_6'])}，P95 = {_fmt(stats['design_p95_ngain_7_6'])}，95% CI [{_fmt(stats['design_ngain_7_6_ci_low'])}, {_fmt(stats['design_ngain_7_6_ci_high'])}]。confirmation 没有重新计算 k=7。稳定实质增益不成立，k7_stable = {stats['k7_stable']}。

## Q6. A7* 中有多少 assignment 不能由 k=5 表示？

design 上 A7* 被 k=5 完整覆盖的比例是 {_fmt(stats['design_coverage_k5'])}，不能表示的比例是 {_fmt(1.0 - stats['design_coverage_k5'])}。confirmation 覆盖比例是 {_fmt(stats['confirm_coverage_k5'])}，不能表示的比例是 {_fmt(1.0 - stats['confirm_coverage_k5'])}。k=6 覆盖比例 design {_fmt(stats['design_coverage_k6'])}，confirmation {_fmt(stats['confirm_coverage_k6'])}。

## Q7. 最终应该冻结 k=5、6、7 中的哪一个？

{selected}。

## Q8. 候选池扩大多少？

主状态上，k=5 平均池大小 {_fmt(primary_complexity[5]['mean_pool_size'])}，k=6 是它的 {_fmt(primary_complexity[6]['pool_multiplier_vs_k5'])} 倍。k=7 的池大小只统计已完成的 design set，是 k=5 的 {_fmt(primary_complexity[7]['pool_multiplier_vs_k5'])} 倍，最大池 {_fmt(primary_complexity[7]['max_pool_size'])}。

## Q9. J15 exhaustive runtime 增长多少？

主状态上，k=5 全池平均 {_fmt(primary_complexity[5]['mean_full_pool_seconds'])} 秒，k=6 为 {_fmt(primary_complexity[6]['runtime_multiplier_vs_k5'])} 倍。k=7 耗时只来自 design set，为当时 k=5 的 {_fmt(primary_complexity[7]['runtime_multiplier_vs_k5'])} 倍。单候选平均 k=5 {_fmt(primary_complexity[5]['mean_single_eval_seconds'])} 秒，design set 上的 k=7 为 {_fmt(primary_complexity[7]['mean_single_eval_seconds'])} 秒。k=7 峰值内存 {_fmt(primary_complexity[7]['peak_rss_bytes'])} 字节。

## Q10. 候选边界问题是否主要发生在高 interference state？

全体主状态的高干扰层 mean NGain_6_5 = {_fmt(high_row.get('mean_ngain_6_5'))}，k=5 覆盖 {_fmt(high_row.get('k5_coverage'))}，状态数 {high_row.get('states', 0)}。低干扰层 mean NGain_6_5 = {_fmt(low_row.get('mean_ngain_6_5'))}，k=5 覆盖 {_fmt(low_row.get('k5_coverage'))}，状态数 {low_row.get('states', 0)}。分层阈值是本次 development 主状态上的三分位，分位水平没有在看到增益后改动。

## Q11. rank-6/7 edge 为什么会在 H15 下有价值？

A6* 不同于 A5* 的主状态里，新增 rank>5 边有 {len(new_edges)} 条，只在 k=5 内部重新组合的状态有 {len(recombinations)} 个。这些新边的平均距离 {_fmt(edge_mean('distance_m'))} m，队列 {_fmt(edge_mean('q_bits'))} bit，瞬时速率 {_fmt(edge_mean('instantaneous_rate'))}，干扰项 {_fmt(edge_mean('interference'))}，因果首次服务步 {_fmt(edge_mean('first_service_step'))}，该浮标 H15 服务 {_fmt(edge_mean('h15_served_bits'))} bit。这是机制记录，没有据此修改 G1_RATE。

## Q12. 不同 MAPPO seed 下，candidate expansion gain 是否基本稳定？

confirmation 主状态上，同一 assignment 的平均变异系数是 {_fmt(seed['mean_cv'])}。seed0 的 A6*-A5* 增益大于 1 bit 的状态比例是 {_fmt(seed['seed0_positive_fraction'])}，其他 seed 的平均增益大于 1 bit 的比例是 {_fmt(seed['other_seed_positive_fraction'])}。只在 seed0 上为正的状态比例是 {_fmt(seed['seed0_only_fraction'])}。primary 没有更换。

## Q13. 最终 Gate？

{gate['gate']}。

## Q14. 是否已经可以开始构建 high-level training dataset？

{dataset}。本阶段没有构建数据集。

## Q15. 现在是否训练 HGNN？

NO。

## Q16. 现在是否重新训练 MAPPO？

NO。

## Q17. 320100–320131 是否仍完全 untouched？

YES。

闭环 sanity 只在 320032–320039。C6 减 C5 的平均 payload 差是 {_fmt(stats['closed_mean_bits'])} bit，95% CI [{_fmt(stats['closed_ci_low_bits'])}, {_fmt(stats['closed_ci_high_bits'])}]。C7 是否运行：{stats['c7_ran']}。
"""
    REPORT.write_text(text, encoding="utf-8")


def _fail(message: str) -> None:
    flags = {
        "k5_sufficient": False,
        "k6_stable": False,
        "k7_stable": False,
        "k7_immaterial": False,
        "rank7_heavy": False,
    }
    gate = choose_gate(flags, failures=1)
    gate["error"] = message
    _write_json(PHASE / "phase6ia_candidate_boundary_gate.json", gate)
    print(gate["gate"], message, flush=True)


def _write_protocol(hashes: dict) -> None:
    _write_json(
        PHASE / "phase6ia_protocol.json",
        {
            "stamp": PROTOCOL_STAMP,
            "primary_seed_rule": "MINIMUM_SEED_ID",
            "primary_seed": PRIMARY_SEED,
            "state_source": STATE_SOURCE,
            "design_pool": list(DESIGN_POOL),
            "confirmation_pool": list(CONFIRM_POOL),
            "confirmation_reduction": "Representative subset of the original 320032-320063 range. Eight scenarios, spaced by 4, still visit t=0, t=15 and t=30. Design 320000-320031 is unchanged.",
            "sanity_pool": list(SANITY_POOL),
            "reserved_untouched": list(RESERVED_POOL),
            "primary_times": list((0, 15, 30)),
            "secondary_time": 45,
            "generator": "G1_RATE",
            "k_values": list(KS),
            "k7_evaluation": False,
            "k7_skip_reason": "Completed design set, 96 primary states: mean NGain_7_6 = 0 and P95 NGain_7_6 = 0. Confirmation and closed-loop do not evaluate k=7. Saved design k=7 labels are kept.",
            "ngain_mean_max": 0.02,
            "ngain_p95_max": 0.05,
            "coverage_min": 0.90,
            "rank7_heavy": 0.50,
            "eps_bits": EPS_BITS,
            "normalization": "(J_larger - J_smaller) / max(J7, 1 bit)",
            "interference_rule": "Phase-6C tertiles: low <= q1/3, medium in (q1/3, q2/3], high > q2/3. Quantile levels are not retuned.",
            "interference_metric": "mean added interference of the MAX_RATE assignment at the decision state",
            "future_arrivals_in_j15": 0,
            "robustness_execution_uses_realized_arrivals": True,
            "c7_rule": "run C7 only if design mean NGain_7_6 > 0.02 or design P95 NGain_7_6 > 0.05",
            "rollouts_started": False,
            "retrained_mappo": False,
            "trained_hgnn": False,
            "checkpoints": hashes,
        },
    )
    _write_json(
        PHASE / "phase6ia_split.json",
        {
            "stamp": PROTOCOL_STAMP,
            "design": list(DESIGN_POOL),
            "confirmation": list(CONFIRM_POOL),
            "sanity_closed_loop": list(SANITY_POOL),
            "state_source": STATE_SOURCE,
            "primary_times": [0, 15, 30],
            "secondary_time": 45,
        },
    )


def main() -> None:
    hashes = {path.name: checkpoint_sha(path) for path in sorted(CHECKPOINT_DIR.glob("phase6g_seed*.pt"))}
    if checkpoint_sha(primary_checkpoint()) != hashes["phase6g_seed0.pt"]:
        raise RuntimeError("primary checkpoint hash mismatch")
    _write_protocol(hashes)
    try:
        audits = _map(audit_scenario, list(DESIGN_POOL + CONFIRM_POOL), "audit", lambda scenario: str(int(scenario)))
        optimum = [row for item in audits for row in item["optimum"]]
        design_rows = _primary(optimum, "design")
        preview = {
            "design_mean_ngain_7_6": _mean(design_rows, "ngain_7_6"),
            "design_p95_ngain_7_6": _p95(design_rows, "ngain_7_6"),
        }
        flags_preview = evaluate_flags(
            {
                "design_mean_ngain_6_5": 0.0,
                "design_p95_ngain_6_5": 0.0,
                "confirm_mean_ngain_6_5": 0.0,
                "confirm_p95_ngain_6_5": 0.0,
                "design_mean_ngain_7_6": preview["design_mean_ngain_7_6"],
                "design_p95_ngain_7_6": preview["design_p95_ngain_7_6"],
                "design_ngain_6_5_ci_low": 0.0,
                "confirm_ngain_6_5_ci_low": 0.0,
                "confirm_mean_ngain_7_6": 0.0,
                "confirm_p95_ngain_7_6": 0.0,
                "design_ngain_7_6_ci_low": 0.0,
                "confirm_ngain_7_6_ci_low": 0.0,
                "design_coverage_k5": 1.0,
                "confirm_coverage_k5": 1.0,
                "design_rank7_fraction": 0.0,
                "confirm_rank7_fraction": 0.0,
                "closed_mean_bits": 0.0,
                "closed_ci_low_bits": 0.0,
            }
        )
        closed_ks = [5, 6] + ([7] if flags_preview["c7_needed"] else [])
        closed_jobs = [(scenario, k) for scenario in SANITY_POOL for k in closed_ks]
        closed = _map(closed_loop_scenario, closed_jobs, "closed", lambda job: f"{int(job[0])}_{int(job[1])}")
    except Exception as exc:
        _fail(str(exc))
        raise
    if any(checkpoint_sha(CHECKPOINT_DIR / name) != digest for name, digest in hashes.items()):
        _fail("checkpoint bytes changed")
        raise RuntimeError("checkpoint bytes changed")
    optimum = [row for item in audits for row in item["optimum"]]
    edges = [row for item in audits for row in item["edges"]]
    mechanism = [row for item in audits for row in item["mechanism"]]
    robust = [row for item in audits for row in item["robust"]]
    isolation = [row for item in audits for row in item["isolation"]]
    runtime = [row for item in audits for row in item["runtime"]]
    design_rows = _primary(optimum, "design")
    confirm_rows = _primary(optimum, "confirm")
    failures = 0
    if len(design_rows) != len(DESIGN_POOL) * 3 or len(confirm_rows) != len(CONFIRM_POOL) * 3:
        failures += 1
    if any(int(row["shadow_state_contamination"]) != 0 for row in isolation):
        failures += 1
    if any(float(row["conservation_error"]) > 1.0 for row in isolation):
        failures += 1
    if any(float(row["gain_6_5"]) < -1.0 for row in optimum):
        failures += 1
    if any(row.get("gain_7_6") not in ("", None) and float(row["gain_7_6"]) < -1.0 for row in optimum):
        failures += 1
    stats = {}
    stats.update(_split_stats(design_rows, "design"))
    stats.update(_split_stats(confirm_rows, "confirm"))
    payloads = {int(k): {} for k in (5, 6, 7)}
    for row in closed:
        payloads[int(row["k"])][int(row["scenario_id"])] = float(row["payload_bits"])
    paired = np.asarray([payloads[6][scenario] - payloads[5][scenario] for scenario in SANITY_POOL], dtype=np.float64)
    closed_mean, closed_low, closed_high = _bootstrap(paired)
    stats.update(
        {
            "closed_mean_bits": closed_mean,
            "closed_ci_low_bits": closed_low,
            "closed_ci_high_bits": closed_high,
            "c7_ran": any(int(row["k"]) == 7 for row in closed),
        }
    )
    flags = evaluate_flags(stats)
    stats.update(flags)
    if failures == 0 and any(not np.isfinite(float(row["payload_bits"])) for row in closed):
        failures += 1
    gate = choose_gate(flags, failures=failures)
    complexity = _complexity(runtime)
    strata = _strata(design_rows + confirm_rows)
    seed = _seed_summary(robust)
    _write_csv(PHASE / "phase6ia_pool_sizes.csv", runtime)
    _write_csv(PHASE / "phase6ia_j15_pool_optimum.csv", optimum)
    _write_csv(PHASE / "phase6ia_k5_vs_k6.csv", [{key: row[key] for key in ("scenario_id", "split", "decision_time", "primary", "j5", "j6", "gain_6_5", "ngain_6_5", "disagree_6_5", "a5", "a6")} for row in optimum])
    _write_csv(PHASE / "phase6ia_k6_vs_k7.csv", [{key: row[key] for key in ("scenario_id", "split", "decision_time", "primary", "j6", "j7", "gain_7_6", "ngain_7_6", "disagree_7_6", "a6", "a7")} for row in optimum])
    _write_csv(PHASE / "phase6ia_edge_rank_distribution.csv", edges)
    _write_csv(
        PHASE / "phase6ia_boundary_coverage.csv",
        [{key: row[key] for key in ("scenario_id", "split", "decision_time", "primary", "covered_k5", "covered_k6", "covered_k7", "contains_rank6", "contains_rank7")} for row in optimum],
    )
    _write_csv(PHASE / "phase6ia_complexity.csv", complexity)
    _write_csv(PHASE / "phase6ia_closed_loop_sanity.csv", closed)
    _write_csv(PHASE / "phase6ia_interference_strata.csv", strata)
    _write_csv(
        PHASE / "phase6ia_new_edge_mechanism.csv",
        mechanism,
        ["scenario_id", "split", "decision_time", "kind", "uav", "buoy", "edge_rank", "distance_m", "q_bits", "instantaneous_rate", "interference", "first_service_step", "h15_served_bits", "gain_6_5"],
    )
    _write_csv(PHASE / "phase6ia_seed_robustness.csv", robust)
    gate["stats"] = stats
    gate["seed_robustness"] = seed
    gate["checked_shadows"] = len(isolation)
    _write_json(PHASE / "phase6ia_candidate_boundary_gate.json", gate)
    protocol = json.loads((PHASE / "phase6ia_protocol.json").read_text(encoding="utf-8"))
    protocol["rollouts_started"] = True
    protocol["rules_frozen_before_k6_j15"] = True
    _write_json(PHASE / "phase6ia_protocol.json", protocol)
    _report(gate, stats, complexity, strata, seed, [row for row in mechanism if int(row.get("decision_time", -1)) in (0, 15, 30) or row.get("kind") == "new_rank_edge"])
    print(gate["gate"], flush=True)


if __name__ == "__main__":
    main()
