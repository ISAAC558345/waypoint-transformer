"""Score frozen Phase-5Q models on eight new scenarios. No gradient is used there."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn

from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5p_executor_joint_planner.run_phase5p import _eval_head
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features, _eval_scorer

from .metrics import choose_gate, cluster_bootstrap
from .protocol import (
    ABLATION_DISAGREEMENT,
    ABLATION_REGRET_BITS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CACHE_VERSION,
    COVERAGE_EPS,
    COVERAGE_OK_RATIO,
    DEPLOYMENT_VERIFIED,
    EXPANDED_WIDTH,
    MODEL_POOL_WIDTH,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PLANNER_READY,
    REPORT,
    ROLE,
    SCENARIOS,
    SEEDS,
    TIMES,
    WORKERS,
)
from .restore import load_frozen_models
from .rollout import _init_worker, roll_assignments
from .world import candidate_pool, replay_snapshot


ADVICE = {
    "R-A": "Model-C 在这 8 个 development-generalization scenario 上稳定优于 Model-A，候选覆盖也尚可。下一阶段可以固定这个 non-additive scorer，再在更多 development data 上训练。这仍然不是 deployment。",
    "R-B": "平均收益为正，但置信区间跨过 0，或者 scenario / seed 的方向不一致。下一步增加 development-state diversity，不增加网络复杂度。",
    "R-C": "Model-C 经常达到候选池内的 oracle，但池内 oracle 明显低于扩展参考池。下一阶段应改 Stage-1 candidate generator，不能先改 scorer。",
    "R-D": "Model-C 在 310102–310109 上没有稳定优于 Model-A。Phase-5Q 的结果属于有限场景上的泛化。下一步扩大合法 development training states，不能宣布 scorer 成功。",
}


class _ZeroInteraction(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(value.shape[0], self.width, dtype=value.dtype, device=value.device)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _std(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(np.std(array, ddof=1))


def _fmt(value, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vector(assignment) -> str:
    return ";".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64))


def _load_cache() -> dict[tuple[int, int, str], float]:
    path = OUTPUT / "phase5r_rollout_cache.pt"
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != int(CACHE_VERSION):
        return {}
    return dict(payload["values"])


def _save_cache(values: dict[tuple[int, int, str], float]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"version": int(CACHE_VERSION), "values": values}, OUTPUT / "phase5r_rollout_cache.pt")


def _score_pack(result: dict) -> dict:
    scores = np.asarray(result["scores"], dtype=np.float64)
    std = float(np.std(scores)) if np.all(np.isfinite(scores)) else float("nan")
    span = float(np.max(scores) - np.min(scores)) if np.all(np.isfinite(scores)) else float("nan")
    result = dict(result)
    result["score_std"] = std
    result["score_span"] = span
    result["collapsed"] = float(not np.isfinite(std) or std < 1.0e-6 or span < 1.0e-6)
    result["oracle_j_mappo"] = float(result["j_mappo"] + result["regret"])
    return result


def _ablated_scores(scorer, features: dict, assignments: torch.Tensor, mode: str) -> np.ndarray:
    local = dict(features)
    replaced = None
    if mode == "no_global":
        local["z_global"] = torch.zeros_like(features["z_global"])
    elif mode == "no_uu":
        local["h_uu"] = torch.zeros_like(features["h_uu"])
    elif mode == "no_interaction":
        replaced = scorer.interact
        scorer.interact = _ZeroInteraction(scorer.width)
    elif mode != "full":
        raise ValueError(mode)
    try:
        with torch.no_grad():
            return scorer(local, assignments).detach().cpu().numpy()
    finally:
        if replaced is not None:
            scorer.interact = replaced


def _prepare(frozen, matching) -> list[dict]:
    prepared = []
    for scenario in SCENARIOS:
        for moment in TIMES:
            _env, _movement, batch = replay_snapshot(int(scenario), int(moment))
            additive = _features(frozen, matching, batch)
            embedded = _edge_features(frozen, batch)
            logits = additive["logits"].detach().cpu().numpy()
            num_uavs = int(logits.shape[0])
            num_buoys = int(logits.shape[1])
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, num_uavs, num_buoys)
            expanded = candidate_pool(logits, EXPANDED_WIDTH, num_uavs, num_buoys)
            if int(expanded["assignments"].shape[0]) > 5000:
                raise RuntimeError(f"{scenario} t={moment} expanded pool is too large to audit exhaustively")
            print(
                f"pool {scenario} t={moment} model={pool['assignments'].shape[0]} expanded={expanded['assignments'].shape[0]}",
                flush=True,
            )
            prepared.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "additive_features": additive,
                    "embedded": embedded,
                    "pool": pool,
                    "expanded": expanded,
                }
            )
    return prepared


def _payloads(prepared: list[dict]) -> dict[tuple[int, int, str], float]:
    cache = _load_cache()
    jobs = []
    for state in prepared:
        vectors = []
        seen = set()
        for assignments in (state["pool"]["assignments"], state["expanded"]["assignments"]):
            for row in assignments:
                vector = _vector(row)
                key = (int(state["scenario_id"]), int(state["time_step"]), vector)
                if key in cache or vector in seen:
                    continue
                seen.add(vector)
                vectors.append(vector)
        if vectors:
            jobs.append((int(state["scenario_id"]), int(state["time_step"]), vectors))
    if jobs:
        print(f"rollouts {sum(len(item[2]) for item in jobs)} across {len(jobs)} states", flush=True)
        with ProcessPoolExecutor(max_workers=int(WORKERS), initializer=_init_worker) as pool:
            futures = [pool.submit(roll_assignments, job) for job in jobs]
            for future in as_completed(futures):
                scenario, moment, rows = future.result()
                for vector, value in rows:
                    cache[(int(scenario), int(moment), vector)] = float(value)
                print(f"rolled {scenario} t={moment} n={len(rows)}", flush=True)
                _save_cache(cache)
    return cache


def _labels(assignments: np.ndarray, scenario: int, moment: int, cache: dict) -> np.ndarray:
    values = []
    for row in assignments:
        key = (int(scenario), int(moment), _vector(row))
        if key not in cache:
            raise RuntimeError(f"missing J_MAPPO for {key}")
        values.append(cache[key])
    return np.asarray(values, dtype=np.float64)


def _state_view(state: dict, labels: np.ndarray) -> dict:
    pool = state["pool"]
    return {
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "assignments": pool["assignments"],
        "candidate_mask": pool["candidate_mask"],
        "vectors": [_vector(row) for row in pool["assignments"]],
        "j_mappo": labels,
    }


def _write_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5R Frozen Non-Additive Joint Scorer Generalization",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "310102–310109 的角色是 `PHASE5_DEVELOPMENT_GENERALIZATION`。它们不是 deployment holdout，也不是最终 untouched test。Phase-5Q 没有保存权重。本阶段用原来的 seed、310100 数据和 10 个 epoch 把 Model-A、Model-C 恢复出来，并与 Phase-5Q 在 310100/310101 上的记录对齐。对齐之后，这 8 个 scenario 只做无梯度推理。",
        "",
        "310102 和 310103 曾出现在更早的闭环审计里，但不是 Phase-5Q 的训练场景。310104–310109 没有进入先前的 Phase-5 protocol 名单。",
        "",
        "## 1. 候选池与覆盖",
        "",
        f"模型池平均大小 {_fmt(summary['pool_size'], '.1f')}，扩展参考池平均大小 {_fmt(summary['expanded_size'], '.1f')}。",
        f"平均 candidate coverage ratio {_fmt(summary['coverage_ratio'])}。平均 coverage regret {_fmt(summary['coverage_regret'], '.4e')} bit。平均 scorer regret {_fmt(summary['scorer_regret'], '.4e')} bit。",
        "扩展池只用于覆盖诊断，没有提供给 Model-A 或 Model-C。边际指标是冻结 Phase-5L matching logit 的 top-5，和 top-3 候选来自同一次前向。",
        "",
        "## 2. Paired comparison",
        "",
        "每个 scenario、时刻和 model seed 先配对，再按 scenario 做 cluster bootstrap。同一 scenario 的三个时刻不会被当成独立场景。",
        "",
        f"ΔJ_MAPPO mean {_fmt(summary['delta_j']['mean'], '.4e')}，median {_fmt(summary['delta_j_median'], '.4e')}，std {_fmt(summary['delta_j_std'], '.4e')}，positive fraction {_fmt(summary['positive_fraction'])}。",
        f"cluster bootstrap 95% CI [{_fmt(summary['delta_j']['low'], '.4e')}, {_fmt(summary['delta_j']['high'], '.4e')}]。",
        f"Δnormalized regret mean {_fmt(summary['delta_regret']['mean'])}，95% CI [{_fmt(summary['delta_regret']['low'])}, {_fmt(summary['delta_regret']['high'])}]。",
        f"8 个 scenario 中，Model-C 更高 {summary['n_scenarios_up']} 个，持平 {summary['n_scenarios_tie']} 个，更低 {summary['n_scenarios_down']} 个。",
        f"5 个 seed 里，平均 ΔJ_MAPPO 大于 0 的有 {summary['n_seeds_up']} 个。",
        "",
        "| scenario | ΔJ_MAPPO | Δregret | Model-C Top-1 | coverage ratio | coverage regret | scorer regret |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *summary["scenario_lines"],
        "",
        "## 3. Inference ablation",
        "",
        "消融只发生在推理时，没有重新训练。它测量已经训练好的 Model-C 用了哪些输入，不是公平的性能比赛。",
        "",
        "| ablation | disagreement | mean |score| change | ΔJ_MAPPO | Δregret |",
        "| --- | --- | --- | --- | --- |",
        *summary["ablation_lines"],
        "",
        "Phase-5Q 的 additive / pairwise 回归是状态无关的系数表。它们在 310101 上 R2 为负，不能用来说明状态条件下的 non-additive set interaction 没有用。",
        "",
        "## 4. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。",
        "",
        ADVICE[summary["gate"]],
        "",
        "Q1. Model-C 是否仍稳定优于 Model-A？ " + summary["answers"]["q1"],
        "",
        "Q2. 收益是否跨 scenario 稳定？ " + summary["answers"]["q2"],
        "",
        "Q3. 五个 seed 是否同向？ " + summary["answers"]["q3"],
        "",
        "Q4. 剩余误差主要来自哪里？ " + summary["answers"]["q4"],
        "",
        "Q5. Model-C 实际依赖哪些部分？ " + summary["answers"]["q5"],
        "",
        "Q6. 是否可以把 non-additive scorer 正式固定为高层架构？ " + summary["answers"]["q6"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _answers(summary: dict) -> dict[str, str]:
    stable = summary["gate"] == "R-A"
    q1 = (
        f"是。mean ΔJ_MAPPO {_fmt(summary['delta_j']['mean'], '.4e')}，95% CI 下界 {_fmt(summary['delta_j']['low'], '.4e')}。"
        if stable
        else f"没有达到预先写好的稳定标准。mean ΔJ_MAPPO {_fmt(summary['delta_j']['mean'], '.4e')}，95% CI [{_fmt(summary['delta_j']['low'], '.4e')}, {_fmt(summary['delta_j']['high'], '.4e')}]。"
    )
    q2 = f"8 个 scenario 里更高 {summary['n_scenarios_up']} 个、持平 {summary['n_scenarios_tie']} 个、更低 {summary['n_scenarios_down']} 个。"
    q3 = f"{summary['n_seeds_up']}/5 个 seed 的平均 ΔJ_MAPPO 大于 0。"
    if summary["coverage_regret"] > summary["scorer_regret"] and summary["coverage_regret"] >= PAYLOAD_BLOCK_BITS:
        q4 = f"candidate coverage。平均 coverage regret {_fmt(summary['coverage_regret'], '.4e')}，平均 scorer regret {_fmt(summary['scorer_regret'], '.4e')}。"
    else:
        q4 = f"scorer。平均 scorer regret {_fmt(summary['scorer_regret'], '.4e')}，平均 coverage regret {_fmt(summary['coverage_regret'], '.4e')}。"
    used = [name for name, flag in summary["used"].items() if flag]
    q5 = "、".join(used) if used else "这三项在推理消融里都没有达到预先设定的依赖阈值。"
    q6 = "可以进入下一阶段把它固定下来并继续用 development data 训练，但现在仍不能部署。" if stable else "不能。当前证据还不够把这个结构正式固定下来。"
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6}


def run() -> dict:
    heads, scorers, frozen = load_frozen_models()
    matching = _load_matching()
    for parameter in matching.parameters():
        parameter.requires_grad_(False)
    prepared = _prepare(frozen, matching)
    cache = _payloads(prepared)
    truth_rows = []
    coverage_rows = []
    state_rows = []
    paired = []
    ablation_rows = []
    regret_rows = []
    measured = {}
    for state in prepared:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        pool_labels = _labels(state["pool"]["assignments"], scenario, moment, cache)
        expanded_labels = _labels(state["expanded"]["assignments"], scenario, moment, cache)
        pool_oracle = float(np.max(pool_labels))
        # Top-5 maximal matchings need not contain a top-3 maximal matching, because
        # an idle UAV in the smaller pool may still have a free wider candidate.
        # The reference oracle is the union of the two pools.
        expanded_oracle = float(max(pool_oracle, float(np.max(expanded_labels))))
        coverage_regret = float(expanded_oracle - pool_oracle)
        ratio = float(pool_oracle / (expanded_oracle + COVERAGE_EPS))
        options = ";".join(",".join(str(int(buoy)) for buoy in row) for row in state["pool"]["options"])
        coverage_rows.append(
            {
                "scenario_id": scenario,
                "time_step": moment,
                "n_pool": int(state["pool"]["assignments"].shape[0]),
                "n_expanded": int(state["expanded"]["assignments"].shape[0]),
                "n_unique_pool": state["pool"]["n_unique"],
                "has_idle": state["pool"]["has_idle"],
                "candidates": options,
                "pool_oracle": pool_oracle,
                "expanded_oracle": expanded_oracle,
                "coverage_ratio": ratio,
                "coverage_regret": coverage_regret,
            }
        )
        pool_vectors = {_vector(row) for row in state["pool"]["assignments"]}
        expanded_vectors = {_vector(row) for row in state["expanded"]["assignments"]}
        for vector in sorted(pool_vectors | expanded_vectors):
            truth_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "assignment_vector": vector,
                    "j_mappo": float(cache[(scenario, moment, vector)]),
                    "in_model_pool": float(vector in pool_vectors),
                    "in_expanded_pool": float(vector in expanded_vectors),
                }
            )
        view = _state_view(state, pool_labels)
        assignments = torch.as_tensor(view["assignments"], dtype=torch.long)
        for seed in SEEDS:
            started = time.perf_counter()
            additive = _score_pack(_eval_head(heads[int(seed)], state["additive_features"], view))
            additive_seconds = time.perf_counter() - started
            started = time.perf_counter()
            full = _score_pack(_eval_scorer(scorers[int(seed)], state["embedded"], view))
            scorer_seconds = time.perf_counter() - started
            additive["inference_ms"] = float(additive_seconds) * 1.0e3
            full["inference_ms"] = float(scorer_seconds) * 1.0e3
            for model, result in (("model_a", additive), ("model_c", full)):
                measured[(model, int(seed), scenario, moment)] = result
                state_rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "scenario_id": scenario,
                        "time_step": moment,
                        "spearman": result["spearman"],
                        "kendall": result["kendall"],
                        "top1": result["top1"],
                        "top3": result["top3"],
                        "j_mappo": result["j_mappo"],
                        "pool_oracle": result["oracle_j_mappo"],
                        "regret": result["regret"],
                        "normalized_regret": result["normalized_regret"],
                        "score_std": result["score_std"],
                        "score_span": result["score_span"],
                        "collapsed": result["collapsed"],
                        "inference_ms": result["inference_ms"],
                        "assignment_vector": result["assignment_vector"],
                    }
                )
            paired.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "seed": int(seed),
                    "delta_j_mappo": float(full["j_mappo"] - additive["j_mappo"]),
                    "delta_normalized_regret": float(full["normalized_regret"] - additive["normalized_regret"]),
                    "delta_top1": float(full["top1"] - additive["top1"]),
                    "delta_spearman": float(full["spearman"] - additive["spearman"]),
                }
            )
            regret_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "seed": int(seed),
                    "scorer_regret": float(full["regret"]),
                    "coverage_regret": coverage_regret,
                    "total_regret": float(full["regret"] + coverage_regret),
                }
            )
            full_scores = np.asarray(full["scores"], dtype=np.float64)
            for mode in ("no_global", "no_uu", "no_interaction"):
                scores = _ablated_scores(scorers[int(seed)], state["embedded"], assignments, mode)
                chosen = int(np.argsort(-scores, kind="stable")[0])
                measured_mode = metrics_at(scores, pool_labels, chosen)
                ablation_rows.append(
                    {
                        "ablation": mode,
                        "seed": int(seed),
                        "scenario_id": scenario,
                        "time_step": moment,
                        "disagreement": float(view["vectors"][chosen] != full["assignment_vector"]),
                        "mean_abs_score_change": float(np.mean(np.abs(scores - full_scores))),
                        "delta_j_mappo": float(measured_mode["j_mappo"] - full["j_mappo"]),
                        "delta_regret": float(measured_mode["regret"] - full["regret"]),
                    }
                )
        print(f"scored {scenario} t={moment}", flush=True)
    delta_j = cluster_bootstrap(paired, "delta_j_mappo", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    delta_regret = cluster_bootstrap(paired, "delta_normalized_regret", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 1)
    by_scenario = defaultdict(list)
    by_seed = defaultdict(list)
    for row in paired:
        by_scenario[int(row["scenario_id"])].append(float(row["delta_j_mappo"]))
        by_seed[int(row["seed"])].append(float(row["delta_j_mappo"]))
    scenario_means = {scenario: float(np.mean(values)) for scenario, values in by_scenario.items()}
    n_up = sum(value > 1.0 for value in scenario_means.values())
    n_down = sum(value < -1.0 for value in scenario_means.values())
    n_tie = len(scenario_means) - n_up - n_down
    n_seeds_up = sum(float(np.mean(values)) > 0.0 for values in by_seed.values())
    pool_hits = [float(item["regret"]) < 1.0 for (model, _seed, _scenario, _moment), item in measured.items() if model == "model_c"]
    coverage_regret = _mean(row["coverage_regret"] for row in coverage_rows)
    coverage_ratio = _mean(row["coverage_ratio"] for row in coverage_rows)
    scorer_regret = _mean(row["scorer_regret"] for row in regret_rows)
    coverage_ok = bool(coverage_ratio >= COVERAGE_OK_RATIO and coverage_regret < PAYLOAD_BLOCK_BITS)
    gate = choose_gate(
        mean_delta_j=delta_j["mean"],
        delta_j_low=delta_j["low"],
        n_scenarios_up=n_up,
        n_scenarios=len(SCENARIOS),
        n_seeds_up=n_seeds_up,
        n_seeds=len(SEEDS),
        mean_delta_regret=delta_regret["mean"],
        regret_high=delta_regret["high"],
        coverage_ok=coverage_ok,
        pool_hit_fraction=_mean(pool_hits),
        mean_coverage_regret=coverage_regret,
    )
    scenario_lines = []
    for scenario in SCENARIOS:
        rows = [row for row in paired if int(row["scenario_id"]) == int(scenario)]
        coverage = [row for row in coverage_rows if int(row["scenario_id"]) == int(scenario)]
        regrets = [row for row in regret_rows if int(row["scenario_id"]) == int(scenario)]
        top1 = [
            float(item["top1"])
            for (model, _seed, row_scenario, _moment), item in measured.items()
            if model == "model_c" and int(row_scenario) == int(scenario)
        ]
        scenario_lines.append(
            "| "
            + " | ".join(
                (
                    str(scenario),
                    _fmt(_mean(row["delta_j_mappo"] for row in rows), ".4e"),
                    _fmt(_mean(row["delta_normalized_regret"] for row in rows)),
                    _fmt(_mean(top1)),
                    _fmt(_mean(row["coverage_ratio"] for row in coverage)),
                    _fmt(_mean(row["coverage_regret"] for row in coverage), ".4e"),
                    _fmt(_mean(row["scorer_regret"] for row in regrets), ".4e"),
                )
            )
            + " |"
        )
    ablation_lines = []
    used = {}
    for mode in ("no_global", "no_uu", "no_interaction"):
        rows = [row for row in ablation_rows if row["ablation"] == mode]
        disagreement = _mean(row["disagreement"] for row in rows)
        regret_change = _mean(row["delta_regret"] for row in rows)
        used[mode] = bool(disagreement >= ABLATION_DISAGREEMENT or regret_change >= ABLATION_REGRET_BITS)
        ablation_lines.append(
            "| "
            + " | ".join(
                (
                    mode,
                    _fmt(disagreement),
                    _fmt(_mean(row["mean_abs_score_change"] for row in rows), ".4e"),
                    _fmt(_mean(row["delta_j_mappo"] for row in rows), ".4e"),
                    _fmt(regret_change, ".4e"),
                )
            )
            + " |"
        )
    seed_rows = []
    for seed in SEEDS:
        for model in ("model_a", "model_c"):
            rows = [item for (name, row_seed, _scenario, _moment), item in measured.items() if name == model and int(row_seed) == int(seed)]
            seed_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "spearman": _mean(row["spearman"] for row in rows),
                    "top1": _mean(row["top1"] for row in rows),
                    "j_mappo": _mean(row["j_mappo"] for row in rows),
                    "regret": _mean(row["regret"] for row in rows),
                    "normalized_regret": _mean(row["normalized_regret"] for row in rows),
                    "collapsed": _mean(row["collapsed"] for row in rows),
                    "delta_j_mappo": _mean(by_seed[int(seed)]) if model == "model_c" else "",
                }
            )
    summary = {
        "gate": gate,
        "pool_size": _mean(row["n_pool"] for row in coverage_rows),
        "expanded_size": _mean(row["n_expanded"] for row in coverage_rows),
        "coverage_ratio": coverage_ratio,
        "coverage_regret": coverage_regret,
        "scorer_regret": scorer_regret,
        "delta_j": delta_j,
        "delta_j_median": _median(row["delta_j_mappo"] for row in paired),
        "delta_j_std": _std(row["delta_j_mappo"] for row in paired),
        "positive_fraction": _mean(float(row["delta_j_mappo"]) > 0.0 for row in paired),
        "delta_regret": delta_regret,
        "n_scenarios_up": n_up,
        "n_scenarios_tie": n_tie,
        "n_scenarios_down": n_down,
        "n_seeds_up": n_seeds_up,
        "scenario_lines": scenario_lines,
        "ablation_lines": ablation_lines,
        "used": used,
    }
    summary["answers"] = _answers(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = {
        "scenarios": list(SCENARIOS),
        "times": list(TIMES),
        "role": ROLE,
        "deployment_holdout": False,
        "final_untouched_test": False,
        "b2_fit_data": False,
        "hyperparameter_selection_data": False,
        "deployment_validation": False,
        "historical_note": "310102 and 310103 appeared in earlier closed-loop audits. They were not Phase-5Q training scenarios. 310104-310109 are absent from prior Phase-5 protocol lists.",
    }
    (OUTPUT / "phase5r_scenario_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    _write_csv(OUTPUT / "phase5r_ground_truth_jmappo.csv", truth_rows)
    _write_csv(OUTPUT / "phase5r_state_metrics.csv", state_rows)
    _write_csv(OUTPUT / "phase5r_seed_metrics.csv", seed_rows)
    _write_csv(OUTPUT / "phase5r_paired_differences.csv", paired)
    _write_csv(OUTPUT / "phase5r_candidate_coverage.csv", coverage_rows)
    _write_csv(OUTPUT / "phase5r_inference_ablation.csv", ablation_rows)
    _write_csv(OUTPUT / "phase5r_regret_decomposition.csv", regret_rows)
    payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "role": ROLE,
        "delta_j": delta_j,
        "delta_regret": delta_regret,
        "n_scenarios_up": n_up,
        "n_scenarios_tie": n_tie,
        "n_scenarios_down": n_down,
        "n_seeds_up": n_seeds_up,
        "coverage_ratio": coverage_ratio,
        "coverage_regret": coverage_regret,
        "scorer_regret": scorer_regret,
        "coverage_ok": coverage_ok,
        "pool_hit_fraction": _mean(pool_hits),
        "ablation_used": used,
    }
    (OUTPUT / "phase5r_generalization_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "delta_j": delta_j, "n_up": n_up, "n_seeds_up": n_seeds_up})), flush=True)
    return summary


if __name__ == "__main__":
    run()
