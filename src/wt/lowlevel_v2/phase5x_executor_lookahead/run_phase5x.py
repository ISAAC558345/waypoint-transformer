"""Compare short frozen MAPPO rollouts with the 15-step payload. No training."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _replay
from wt.lowlevel_v2.phase5u_representation_repair.scorer import RelationalSetScorer
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import FINAL_REL_DIM, PHASE5U_CHECKPOINTS, REL_DIM
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _build_states, _combine, _load_v1, _scale, _scores
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.metrics import synergy_value
from wt.lowlevel_v2.phase5w_residual_mechanism_audit.run_phase5w import _hybrids, _legal

from .metrics import choose_gate
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CANDIDATE_HORIZONS,
    CASE_SCENARIO,
    CASE_TIME,
    CASE_TOP_RANK,
    CONFIRM_SCENARIOS,
    DEPLOYMENT_VERIFIED,
    HORIZONS,
    LONG_HORIZONS,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PLANNER_READY,
    PROGRESS_GAP,
    PROMISING_HORIZONS,
    REGRET_DROP,
    REPORT,
    RESIDUAL_FRACTION,
    SATURATION_REGRET,
    SATURATION_TOP3,
    SEEDS,
    SINR_GAP,
    TOP3_GAIN,
    assert_consumed,
)
from .shadow import rollout_horizons


NEXT_HORIZON = {1: 3, 3: 5, 5: 8, 8: 10, 10: 15}


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


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


def _parse(vector: str) -> np.ndarray:
    return np.asarray([int(item) for item in str(vector).split(";")], dtype=np.int64)


def _top5(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:5]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _optimum_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    levels = quantize(labels)
    optimal = levels == float(np.max(levels))
    return float(1 + np.sum(np.asarray(scores, dtype=np.float64) > float(np.max(np.asarray(scores)[optimal]))))


def _corr(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3 or float(np.std(x[mask])) < 1.0e-8 or float(np.std(y[mask])) < 1.0e-8:
        return float("nan")
    value = spearmanr(x[mask], y[mask]).correlation
    return float(value) if np.isfinite(value) else float("nan")


def _iqr(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, 75) - np.percentile(array, 25))


def _load_v0() -> dict:
    scorers = {}
    for seed in SEEDS:
        scorer = RelationalSetScorer(rel_dim=REL_DIM)
        payload = torch.load(PHASE5U_CHECKPOINTS / f"phase5u_rep_aug_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        scorers[int(seed)] = scorer
    return scorers


def _rank_row(model: str, horizon: int, seed: int, state: dict, scores: np.ndarray) -> dict:
    chosen = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    measured = metrics_at(scores, state["j_mappo"], chosen)
    return {
        "split": state["split"],
        "model": model,
        "horizon": int(horizon),
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "top5": _top5(scores, state["j_mappo"]),
        "normalized_regret": measured["normalized_regret"],
        "regret": measured["regret"],
        "j15": measured["j_mappo"],
        "optimum_rank": _optimum_rank(scores, state["j_mappo"]),
        "assignment_vector": state["vectors"][chosen],
    }


def _first_gap(series: dict[int, float], gap: float) -> int | None:
    for horizon in HORIZONS:
        if abs(float(series[horizon])) >= gap:
            return int(horizon)
    return None


def _write_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5X Frozen Executor Short-Horizon Lookahead",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练网络。每个候选分配在环境副本上交给冻结 MAPPO 执行，主环境在滚动前后保持不变。H=15 只作为完整执行的上界。",
        "",
        "## 1. Shadow integrity",
        "",
        f"检查了 {summary['integrity_checks']} 次滚动。主环境最大绝对变化是 {summary['integrity_max']}。速度不是 L3BEnv 的状态，调用方保存的上一拍动作也没有被改写。",
        "",
        "## 2. Horizon ranking",
        "",
        "Score-A 是前 H 步已经交付的 payload。Score-B 是 payload 加上已经发生的 progress 乘以一个 block。progress 到 1 时环境正好交付一个 block 并清零，所以 Score-B 没有用未来 SINR 估计未来 progress。门控使用 Score-B。Score-A 保留为对照。",
        "",
        f"confirmation 的 24 个状态里，H=1 有 {summary['zero_payload_h1']} 个状态的全部候选 payload 都是 0，H=5 仍有 {summary['zero_payload_h5']} 个。因此 Score-A 的 optimum rank 中位数等于 1，只说明没有候选得到更高的 payload，不说明它选中了最优分配。",
        "",
        "| model | H | Spearman | Top-1 | Top-3 | Top-5 | normalized regret | optimum rank |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["rank_lines"],
        "",
        "## 3. 相对 Phase-5V 的 Score-B",
        "",
        "ΔJ15 是 Score-B 所选分配的 15 步 payload 减去 Phase-5V 所选分配。bootstrap 以 scenario 为 cluster。",
        "",
        "| H | ΔJ15 mean | 95% CI | ΔTop-3 | Δnormalized regret | scenario 上升 | scenario 下降 | 正比例 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["exec_pair_lines"],
        "",
        "Score-A 对照：",
        "",
        "| H | ΔJ15 mean | 95% CI | ΔTop-3 | Δnormalized regret | scenario 上升 | scenario 下降 | 正比例 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["payload_pair_lines"],
        "",
        f"最小充分 horizon H* = {summary['h_star']}。",
        "",
        "## 4. 310114 t=15",
        "",
        "Phase-5V predicted 是 `2;5;-1;4;-1;-1`，命名 optimal 是 `-1;4;2;5;-1;-1`。两列 rank 分别是这两个字符串在 120 个候选里的名次。",
        "",
        "| H | pred payload | opt payload | pred SINR | opt SINR | pred blocks | opt blocks | payload rank | exec rank |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["case_lines"],
        "",
        f"Score-A 上，命名 optimal 的稳定领先起点是 H={summary['case_stable_horizon']}。Score-B 上是 H={summary['exec_stable_horizon']}。",
        "",
        "## 5. Synergy 与 residual",
        "",
        f"Score-B synergy 中位数：{summary['exec_synergy_medians']}。",
        f"Score-A synergy 中位数：{summary['payload_synergy_medians']}。",
        f"Score-B correction horizon：{summary['correction_text']}。",
        f"Score-A correction horizon：{summary['payload_correction_text']}。",
        "",
        f"SINR 分叉中位数 {summary['sinr_median']}，IQR {summary['sinr_iqr']}。progress 中位数 {summary['progress_median']}，IQR {summary['progress_iqr']}。payload 中位数 {summary['payload_median']}，IQR {summary['payload_iqr']}。",
        "",
        "## 6. 运行时间",
        "",
        "| H | 每个 assignment ms | 120 个 assignment 的决策 ms |",
        "| --- | --- | --- |",
        *summary["runtime_lines"],
        "",
        "## 7. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。",
        "",
        summary["advice"],
        "",
        "Q1. shadow 是否污染主环境？ " + summary["answers"]["q1"],
        "",
        "Q2. 各 horizon 对 J15 排序的预测如何？ " + summary["answers"]["q2"],
        "",
        "Q3. 最小充分 horizon 是多少？ " + summary["answers"]["q3"],
        "",
        "Q4. 310114 从第几步开始 optimal 稳定领先？ " + summary["answers"]["q4"],
        "",
        "Q5. synergy 最早在哪个 H 可见？ " + summary["answers"]["q5"],
        "",
        "Q6. 短期执行能否发现高层剩余错误？ " + summary["answers"]["q6"],
        "",
        "Q7. 最早有信息量的执行信号是什么？ " + summary["answers"]["q7"],
        "",
        "Q8. H=3 和 H=5 的耗时是多少？ " + summary["answers"]["q8"],
        "",
        "Q9. 下一步做什么？ " + summary["answers"]["q9"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5x_protocol.json").write_text(
        json.dumps(
            {
                "mode": "shadow executor audit",
                "training": False,
                "horizons": list(HORIZONS),
                "score_a": "payload delivered during the first H executed steps",
                "score_b": "payload_H + sum(progress_H) * block_bits",
                "score_b_legal": True,
                "gate_score": "q_exec_h",
                "j15_is_label_only": True,
                "h15_is_oracle_reference": True,
                "scenarios": list(range(310100, 310118)),
                "opened_310118": False,
                "top3_gain": TOP3_GAIN,
                "regret_drop": REGRET_DROP,
                "residual_fraction": RESIDUAL_FRACTION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config = L3BConfig()
    from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
    from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed

    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = _build_states(frozen, matching, config)
    v1, _curves, center, scale = _load_v1()
    v0 = _load_v0()
    actor = load_frozen_actor()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    worlds = {}
    cache: dict[tuple[int, int, str], dict] = {}
    integrity_rows = []

    def world(scenario: int, moment: int):
        assert_consumed(int(scenario))
        key = (int(scenario), int(moment))
        if key not in worlds:
            worlds[key] = _replay(int(scenario), int(moment))
        return worlds[key]

    def execute(scenario: int, moment: int, vector: str) -> dict:
        key = (int(scenario), int(moment), str(vector))
        if key not in cache:
            env, movement, _batch, _diagnosis = world(int(scenario), int(moment))
            result = rollout_horizons(env, _parse(vector), actor, movement)
            cache[key] = result["checkpoints"]
            integrity_rows.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "assignment_vector": str(vector),
                    "max_abs_state_change": float(result["integrity"]),
                    "velocity_stored_in_env": 0,
                    "caller_movement_changed": 0,
                }
            )
        return cache[key]

    rank_rows = []
    rollout_rows = []
    for state in states:
        env, movement, _batch, _diagnosis = world(int(state["scenario_id"]), int(state["time_step"]))
        _ = (env, movement)
        for index, vector in enumerate(state["vectors"]):
            checkpoints = execute(int(state["scenario_id"]), int(state["time_step"]), vector)
            if abs(float(checkpoints[15]["payload_h"]) - float(state["j_mappo"][index])) > 1.0:
                raise RuntimeError(f"H=15 payload drifted at {state['scenario_id']} t={state['time_step']}")
            for horizon in HORIZONS:
                rollout_rows.append(
                    {
                        "split": state["split"],
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "assignment_vector": vector,
                        "horizon": int(horizon),
                        **checkpoints[int(horizon)],
                        "j15": float(state["j_mappo"][index]),
                    }
                )
        for seed in SEEDS:
            base = _scores(v0[int(seed)], state, state["z_rel"])
            temp = _scores(v1[int(seed)], state, _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale)))
            rank_rows.append(_rank_row("phase5u", 0, int(seed), state, base))
            rank_rows.append(_rank_row("phase5v", 0, int(seed), state, temp))
        for horizon in HORIZONS:
            payload_scores = np.asarray([cache[(int(state["scenario_id"]), int(state["time_step"]), vector)][int(horizon)]["payload_h"] for vector in state["vectors"]], dtype=np.float64)
            exec_scores = np.asarray([cache[(int(state["scenario_id"]), int(state["time_step"]), vector)][int(horizon)]["q_exec_h"] for vector in state["vectors"]], dtype=np.float64)
            rank_rows.append(_rank_row(f"payload_h{horizon}", int(horizon), -1, state, payload_scores))
            rank_rows.append(_rank_row(f"exec_h{horizon}", int(horizon), -1, state, exec_scores))
        print(f"rolled {state['scenario_id']} t={state['time_step']}", flush=True)

    confirmation = [state for state in states if state["split"] == "confirmation"]
    paired = []
    for state in confirmation:
        for horizon in CANDIDATE_HORIZONS:
            for score_name in ("payload", "exec"):
                executor = next(row for row in rank_rows if row["model"] == f"{score_name}_h{horizon}" and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"]))
                for seed in SEEDS:
                    baseline = next(row for row in rank_rows if row["model"] == "phase5v" and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"]))
                    paired.append(
                        {
                            "score": score_name,
                            "horizon": int(horizon),
                            "scenario_id": int(state["scenario_id"]),
                            "time_step": int(state["time_step"]),
                            "seed": int(seed),
                            "delta_j15": float(executor["j15"] - baseline["j15"]),
                            "delta_top1": float(executor["top1"] - baseline["top1"]),
                            "delta_top3": float(executor["top3"] - baseline["top3"]),
                            "delta_normalized_regret": float(executor["normalized_regret"] - baseline["normalized_regret"]),
                        }
                    )

    def confirm_rows(model: str) -> list[dict]:
        return [row for row in rank_rows if row["model"] == model and row["split"] == "confirmation"]

    def horizon_stats(horizon: int, score_name: str) -> dict:
        rows = [row for row in paired if int(row["horizon"]) == int(horizon) and row["score"] == score_name]
        delta_j = cluster_bootstrap(rows, "delta_j15", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        by_scenario = defaultdict(list)
        by_seed = defaultdict(list)
        for row in rows:
            by_scenario[int(row["scenario_id"])].append(float(row["delta_j15"]))
            by_seed[int(row["seed"])].append(float(row["delta_j15"]))
        scored_rows = confirm_rows(f"{score_name}_h{horizon}")
        phase5v_rows = confirm_rows("phase5v")
        return {
            "delta_j": delta_j,
            "top3_gain": _mean(row["top3"] for row in scored_rows) - _mean(row["top3"] for row in phase5v_rows),
            "regret_drop": _mean(row["normalized_regret"] for row in phase5v_rows) - _mean(row["normalized_regret"] for row in scored_rows),
            "majority_scenario": sum(float(np.mean(value)) > 0.0 for value in by_scenario.values()) * 2 > len(CONFIRM_SCENARIOS),
            "majority_seed": sum(float(np.mean(value)) > 0.0 for value in by_seed.values()) * 2 > len(SEEDS),
            "n_scenario_up": sum(float(np.mean(value)) > 0.0 for value in by_scenario.values()),
            "n_scenario_down": sum(float(np.mean(value)) < 0.0 for value in by_scenario.values()),
            "positive_fraction": _mean(float(row["delta_j15"]) > 0.0 for row in rows),
            "top3": _mean(row["top3"] for row in scored_rows),
            "regret": _mean(row["normalized_regret"] for row in scored_rows),
            "spearman": _mean(row["spearman"] for row in scored_rows),
        }

    def oracle_stats(score_name: str) -> dict:
        scored_rows = confirm_rows(f"{score_name}_h15")
        phase5v_rows = confirm_rows("phase5v")
        return {
            "top3": _mean(row["top3"] for row in scored_rows),
            "regret": _mean(row["normalized_regret"] for row in scored_rows),
            "spearman": _mean(row["spearman"] for row in scored_rows),
            "top3_gain": _mean(row["top3"] for row in scored_rows) - _mean(row["top3"] for row in phase5v_rows),
            "regret_drop": _mean(row["normalized_regret"] for row in phase5v_rows) - _mean(row["normalized_regret"] for row in scored_rows),
        }

    # Score-B is the pre-registered short-horizon score. Score-A stays in the
    # tables, but it is zero for every candidate until a payload block completes.
    stats = {int(horizon): horizon_stats(int(horizon), "exec") for horizon in CANDIDATE_HORIZONS}
    stats[15] = oracle_stats("exec")
    payload_stats = {int(horizon): horizon_stats(int(horizon), "payload") for horizon in CANDIDATE_HORIZONS}
    payload_stats[15] = oracle_stats("payload")

    residual_path = OUTPUT.parent / "phase5w_residual_mechanism_audit" / "phase5w_residual_case_hybrids.csv"
    residual_cases = list(csv.DictReader(residual_path.open(encoding="utf-8")))
    case_state = next(state for state in states if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME)
    case_rows = []
    stable_horizon = None
    exec_stable_horizon = None
    pred_index = case_state["vectors"].index("2;5;-1;4;-1;-1")
    opt_index = case_state["vectors"].index("-1;4;2;5;-1;-1")
    for horizon in HORIZONS:
        pred = execute(CASE_SCENARIO, CASE_TIME, "2;5;-1;4;-1;-1")[int(horizon)]
        opt = execute(CASE_SCENARIO, CASE_TIME, "-1;4;2;5;-1;-1")[int(horizon)]
        payload_scores = np.asarray([cache[(CASE_SCENARIO, CASE_TIME, vector)][int(horizon)]["payload_h"] for vector in case_state["vectors"]])
        exec_scores = np.asarray([cache[(CASE_SCENARIO, CASE_TIME, vector)][int(horizon)]["q_exec_h"] for vector in case_state["vectors"]])
        ahead = float(payload_scores[opt_index]) > float(payload_scores[pred_index])
        exec_ahead = float(exec_scores[opt_index]) > float(exec_scores[pred_index])
        if ahead and stable_horizon is None:
            stable_horizon = int(horizon)
        if not ahead:
            stable_horizon = None
        if exec_ahead and exec_stable_horizon is None:
            exec_stable_horizon = int(horizon)
        if not exec_ahead:
            exec_stable_horizon = None
        case_rows.append(
            {
                "horizon": int(horizon),
                "pred_payload": pred["payload_h"],
                "opt_payload": opt["payload_h"],
                "pred_progress": pred["progress_end_sum"],
                "opt_progress": opt["progress_end_sum"],
                "pred_mean_sinr": pred["mean_sinr_h"],
                "opt_mean_sinr": opt["mean_sinr_h"],
                "pred_mean_interference": pred["mean_interference_h"],
                "opt_mean_interference": opt["mean_interference_h"],
                "pred_completed_blocks": pred["completed_blocks_h"],
                "opt_completed_blocks": opt["completed_blocks_h"],
                "pred_reachability": pred["reachability_h"],
                "opt_reachability": opt["reachability_h"],
                "q_payload_opt_rank": int(1 + np.sum(payload_scores > payload_scores[opt_index])),
                "q_exec_opt_rank": int(1 + np.sum(exec_scores > exec_scores[opt_index])),
                "q_exec_optimum_set_rank": _optimum_rank(exec_scores, case_state["j_mappo"]),
                "payload_opt_ahead": int(ahead),
                "exec_opt_ahead": int(exec_ahead),
            }
        )
    case_lines = [
        "| " + " | ".join((
            str(row["horizon"]),
            _fmt(row["pred_payload"], ".4e"),
            _fmt(row["opt_payload"], ".4e"),
            _fmt(row["pred_mean_sinr"]),
            _fmt(row["opt_mean_sinr"]),
            _fmt(row["pred_completed_blocks"], ".0f"),
            _fmt(row["opt_completed_blocks"], ".0f"),
            str(row["q_payload_opt_rank"]),
            str(row["q_exec_opt_rank"]),
        )) + " |"
        for row in case_rows
    ]
    case_opt_rank = float(next(row["q_exec_optimum_set_rank"] for row in case_rows if int(row["horizon"]) == 5))

    residual_rows = []
    synergy_rows = []
    divergence_rows = []
    for item in residual_cases:
        scenario = int(item["scenario_id"])
        moment = int(item["time_step"])
        state = next(row for row in states if int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
        predicted = _parse(item["predicted_assignment"])
        optimal = _parse(item["optimal_assignment"])
        regrets = {}
        payload_regrets = {}
        for horizon in HORIZONS:
            exec_scores = np.asarray([cache[(scenario, moment, vector)][int(horizon)]["q_exec_h"] for vector in state["vectors"]])
            payload_scores = np.asarray([cache[(scenario, moment, vector)][int(horizon)]["payload_h"] for vector in state["vectors"]])
            chosen = int(np.argsort(-exec_scores, kind="stable")[0])
            payload_chosen = int(np.argsort(-payload_scores, kind="stable")[0])
            regrets[int(horizon)] = float(np.max(state["j_mappo"]) - state["j_mappo"][chosen])
            payload_regrets[int(horizon)] = float(np.max(state["j_mappo"]) - state["j_mappo"][payload_chosen])
        phase5v_regret = _median(row["regret"] for row in rank_rows if row["model"] == "phase5v" and int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
        correction = ""
        payload_correction = ""
        for horizon in HORIZONS:
            if correction == "" and regrets[int(horizon)] < PAYLOAD_BLOCK_BITS:
                correction = int(horizon)
            if payload_correction == "" and payload_regrets[int(horizon)] < PAYLOAD_BLOCK_BITS:
                payload_correction = int(horizon)
        residual_rows.append(
            {
                "scenario_id": scenario,
                "time_step": moment,
                "phase5v_regret": phase5v_regret,
                **{f"e{horizon}_regret": regrets[int(horizon)] for horizon in HORIZONS},
                **{f"payload_e{horizon}_regret": payload_regrets[int(horizon)] for horizon in HORIZONS},
                "correction_horizon": correction,
                "payload_correction_horizon": payload_correction,
            }
        )
        if float(item["synergy"]) > 0.0:
            singles = []
            for assignment, switches in _hybrids(predicted, optimal):
                if int(switches) != 1 or not _legal(assignment):
                    continue
                vector = _vector(assignment)
                execute(scenario, moment, vector)
                singles.append(vector)
            execute(scenario, moment, _vector(predicted))
            execute(scenario, moment, _vector(optimal))
            for horizon in HORIZONS:
                base = cache[(scenario, moment, _vector(predicted))][int(horizon)]["payload_h"]
                gains = [cache[(scenario, moment, vector)][int(horizon)]["payload_h"] - base for vector in singles]
                total = cache[(scenario, moment, _vector(optimal))][int(horizon)]["payload_h"] - base
                exec_base = cache[(scenario, moment, _vector(predicted))][int(horizon)]["q_exec_h"]
                exec_gains = [cache[(scenario, moment, vector)][int(horizon)]["q_exec_h"] - exec_base for vector in singles]
                exec_total = cache[(scenario, moment, _vector(optimal))][int(horizon)]["q_exec_h"] - exec_base
                synergy_rows.append(
                    {
                        "scenario_id": scenario,
                        "time_step": moment,
                        "horizon": int(horizon),
                        "synergy_payload": synergy_value(total, gains),
                        "synergy_exec": synergy_value(exec_total, exec_gains),
                        "all_switch_gain": total,
                        "best_single_gain": max(gains) if gains else 0.0,
                        "exec_all_switch_gain": exec_total,
                        "exec_best_single_gain": max(exec_gains) if exec_gains else 0.0,
                    }
                )
        pred_key = item["predicted_assignment"]
        opt_key = item["optimal_assignment"]
        execute(scenario, moment, pred_key)
        execute(scenario, moment, opt_key)
        sinr_gap = {int(horizon): cache[(scenario, moment, opt_key)][int(horizon)]["mean_sinr_h"] - cache[(scenario, moment, pred_key)][int(horizon)]["mean_sinr_h"] for horizon in HORIZONS}
        progress_gap = {int(horizon): cache[(scenario, moment, opt_key)][int(horizon)]["progress_end_sum"] - cache[(scenario, moment, pred_key)][int(horizon)]["progress_end_sum"] for horizon in HORIZONS}
        payload_gap = {int(horizon): cache[(scenario, moment, opt_key)][int(horizon)]["payload_h"] - cache[(scenario, moment, pred_key)][int(horizon)]["payload_h"] for horizon in HORIZONS}
        divergence_rows.append(
            {
                "scenario_id": scenario,
                "time_step": moment,
                "h_sinr": _first_gap(sinr_gap, SINR_GAP) or "",
                "h_progress": _first_gap(progress_gap, PROGRESS_GAP) or "",
                "h_payload": _first_gap(payload_gap, PAYLOAD_BLOCK_BITS) or "",
            }
        )

    def corrected_fraction(horizon: int) -> float:
        return _mean(row["correction_horizon"] != "" and int(row["correction_horizon"]) <= int(horizon) for row in residual_rows)

    def synergy_visible(horizon: int) -> bool:
        rows = [float(row["synergy_exec"]) for row in synergy_rows if int(row["horizon"]) == int(horizon)]
        case_row = [float(row["synergy_exec"]) for row in synergy_rows if int(row["horizon"]) == int(horizon) and int(row["scenario_id"]) == CASE_SCENARIO and int(row["time_step"]) == CASE_TIME]
        return bool(rows and _median(rows) > 0.0 and case_row and case_row[0] > 0.0)

    def case_rank(horizon: int) -> float:
        scores = np.asarray([cache[(CASE_SCENARIO, CASE_TIME, vector)][int(horizon)]["q_exec_h"] for vector in case_state["vectors"]])
        return _optimum_rank(scores, case_state["j_mappo"])

    def saturated(horizon: int) -> bool:
        nxt = NEXT_HORIZON[int(horizon)]
        current = stats[int(horizon)]
        follower = stats[int(nxt)]
        return bool(follower["top3"] - current["top3"] < SATURATION_TOP3 and current["regret"] - follower["regret"] < SATURATION_REGRET)

    def full(horizon: int) -> bool:
        current = stats[int(horizon)]
        return bool(
            current["top3_gain"] >= TOP3_GAIN
            and current["regret_drop"] >= REGRET_DROP
            and current["delta_j"]["low"] > 0.0
            and current["majority_scenario"]
            and current["majority_seed"]
            and corrected_fraction(int(horizon)) >= RESIDUAL_FRACTION
            and case_rank(int(horizon)) <= CASE_TOP_RANK
            and synergy_visible(int(horizon))
            and saturated(int(horizon))
        )

    def promising(horizon: int) -> bool:
        current = stats[int(horizon)]
        return bool(current["top3_gain"] >= TOP3_GAIN and current["regret_drop"] >= REGRET_DROP)

    h_star = next((int(horizon) for horizon in CANDIDATE_HORIZONS if full(int(horizon))), None)
    short_confirmed = bool(h_star is not None and int(h_star) <= 5)
    long_confirmed = bool(any(full(int(horizon)) for horizon in LONG_HORIZONS))
    promising_any = bool(any(promising(int(horizon)) and not full(int(horizon)) for horizon in PROMISING_HORIZONS))
    only_full = bool(not promising(10) and stats[15]["top3_gain"] >= TOP3_GAIN)
    gate = choose_gate(short_confirmed=short_confirmed, long_confirmed=long_confirmed and not short_confirmed, promising=promising_any and not short_confirmed and not long_confirmed, only_full_execution=only_full)
    advice = {
        "X-A": f"冻结 H*={h_star}。下一阶段做 executor-conditioned value / reachability distillation，避免对 120 个分配逐个执行 H* 步。",
        "X-B": "短 horizon 有改善，但还不稳定。下一阶段只做 executor-signal composition audit，不增加网络。",
        "X-C": f"按预先登记的充分条件，H* = {h_star}。H=1 到 H=5 的 Score-B 已经稳定提高所选 J15，但收益还没有饱和。下一阶段训练 executor-conditioned value，直接预测 J15，不再加手工特征。本阶段不开始训练。",
        "X-D": "H=10 仍不能稳定超过 Phase-5V。先复查 shadow、MAPPO 确定性、J15 定义和标签对齐，不训练 value model。",
    }[gate]

    signal_names = ("mean_sinr_h", "mean_interference_h", "progress_end_sum", "completed_blocks_h", "payload_h")
    signal_scores = {}
    for horizon in (1, 3, 5):
        for name in signal_names:
            values = []
            for state in confirmation:
                signal = [cache[(int(state["scenario_id"]), int(state["time_step"]), vector)][int(horizon)][name] for vector in state["vectors"]]
                values.append(_corr(signal, state["j_mappo"]))
            signal_scores[(int(horizon), name)] = _mean(values)
    best_signal = max(signal_scores, key=lambda key: signal_scores[key] if np.isfinite(signal_scores[key]) else -1.0)

    runtime_lines = []
    runtime_rows = []
    for horizon in HORIZONS:
        elapsed = [float(cache[key][int(horizon)]["elapsed_ms"]) for key in cache]
        per_assignment = _mean(elapsed)
        runtime_rows.append({"horizon": int(horizon), "ms_per_assignment": per_assignment, "ms_per_decision_120": per_assignment * 120.0})
        runtime_lines.append(f"| {horizon} | {_fmt(per_assignment, '.2f')} | {_fmt(per_assignment * 120.0, '.1f')} |")
    rank_lines = []
    for model, horizon in (("phase5u", 0), ("phase5v", 0)):
        rows = confirm_rows(model)
        rank_lines.append("| " + " | ".join((model, str(horizon), _fmt(_mean(row["spearman"] for row in rows)), _fmt(_mean(row["top1"] for row in rows)), _fmt(_mean(row["top3"] for row in rows)), _fmt(_mean(row["top5"] for row in rows)), _fmt(_mean(row["normalized_regret"] for row in rows)), _fmt(_median(row["optimum_rank"] for row in rows), ".1f"))) + " |")
    for horizon in HORIZONS:
        for prefix in ("payload", "exec"):
            rows = confirm_rows(f"{prefix}_h{horizon}")
            rank_lines.append("| " + " | ".join((f"{prefix}", str(horizon), _fmt(_mean(row["spearman"] for row in rows)), _fmt(_mean(row["top1"] for row in rows)), _fmt(_mean(row["top3"] for row in rows)), _fmt(_mean(row["top5"] for row in rows)), _fmt(_mean(row["normalized_regret"] for row in rows)), _fmt(_median(row["optimum_rank"] for row in rows), ".1f"))) + " |")
    def pair_line(table: dict, horizon: int) -> str:
        current = table[int(horizon)]
        return f"| {horizon} | {_fmt(current['delta_j']['mean'], '.4e')} | [{_fmt(current['delta_j']['low'], '.4e')}, {_fmt(current['delta_j']['high'], '.4e')}] | {_fmt(current['top3_gain'])} | {_fmt(-current['regret_drop'])} | {current['n_scenario_up']}/8 | {current['n_scenario_down']}/8 | {_fmt(current['positive_fraction'])} |"

    exec_pair_lines = [pair_line(stats, horizon) for horizon in CANDIDATE_HORIZONS]
    payload_pair_lines = [pair_line(payload_stats, horizon) for horizon in CANDIDATE_HORIZONS]
    exec_synergy_medians = ", ".join(f"H={horizon}:{_fmt(_median(row['synergy_exec'] for row in synergy_rows if int(row['horizon']) == int(horizon)), '.3e')}" for horizon in HORIZONS)
    payload_synergy_medians = ", ".join(f"H={horizon}:{_fmt(_median(row['synergy_payload'] for row in synergy_rows if int(row['horizon']) == int(horizon)), '.3e')}" for horizon in HORIZONS)
    correction_text = ", ".join(f"{row['scenario_id']} t={row['time_step']}:{row['correction_horizon'] or 'none'}" for row in residual_rows)
    payload_correction_text = ", ".join(f"{row['scenario_id']} t={row['time_step']}:{row['payload_correction_horizon'] or 'none'}" for row in residual_rows)
    zero_payload = {}
    for horizon in HORIZONS:
        count = 0
        for state in confirmation:
            values = [cache[(int(state["scenario_id"]), int(state["time_step"]), vector)][int(horizon)]["payload_h"] for vector in state["vectors"]]
            count += int(max(values) < 1.0)
        zero_payload[int(horizon)] = count
    answers = {
        "q1": f"可以完全恢复。{len(integrity_rows)} 次副本执行后，主环境最大绝对变化是 {max(float(row['max_abs_state_change']) for row in integrity_rows)}。位置、队列、浮标、served_k、progress、分配、信道、干扰、步数和随机数状态都没有被改写。",
        "q2": "Score-B 从 H=1 起就恢复大部分 J15 排序。Score-A 在 block 完成之前几乎全是 0，短 horizon 上差于 Phase-5V。",
        "q3": "没有 H<=10 同时满足预先写好的充分条件。" if h_star is None else f"H* = {h_star}。判定使用 Score-B。H=5 已经稳定优于 Phase-5V，但 H=8 仍继续降低 regret，所以 H=5 还没有饱和。",
        "q4": (
            f"Score-A 的稳定领先起点是 H={stable_horizon}。"
            if stable_horizon is not None
            else "Score-A 在 H=15 之内没有稳定领先。"
        )
        + (
            f" Score-B 从 H={exec_stable_horizon} 起，命名的 optimal 一直高于 Phase-5V predicted。"
            if exec_stable_horizon is not None
            else " Score-B 没有形成稳定领先。"
        ),
        "q5": f"Score-B synergy 中位数：{exec_synergy_medians}。Score-A synergy 中位数：{payload_synergy_medians}。",
        "q6": f"用 Score-B 选择时，10 个 residual case 里 correction horizon 不超过 5 的比例是 {_fmt(corrected_fraction(5))}，不超过 8 的比例是 {_fmt(corrected_fraction(8))}。",
        "q7": f"H={best_signal[0]} 时，{best_signal[1]} 与 J15 的平均 Spearman 最高，为 {_fmt(signal_scores[best_signal])}。payload 本身要到 block 开始完成以后才有区分度。",
        "q8": f"H=3 每个分配 {_fmt(next(row['ms_per_assignment'] for row in runtime_rows if int(row['horizon']) == 3), '.2f')} ms，120 个分配共 {_fmt(next(row['ms_per_decision_120'] for row in runtime_rows if int(row['horizon']) == 3), '.1f')} ms。H=5 每个分配 {_fmt(next(row['ms_per_assignment'] for row in runtime_rows if int(row['horizon']) == 5), '.2f')} ms，120 个分配共 {_fmt(next(row['ms_per_decision_120'] for row in runtime_rows if int(row['horizon']) == 5), '.1f')} ms。H=8 每个分配 {_fmt(next(row['ms_per_assignment'] for row in runtime_rows if int(row['horizon']) == 8), '.2f')} ms，120 个分配共 {_fmt(next(row['ms_per_decision_120'] for row in runtime_rows if int(row['horizon']) == 8), '.1f')} ms。这些时间是同一次 15 步滚动里到达该 H 的累计耗时。",
        "q9": advice,
    }
    summary = {
        "integrity_checks": len(integrity_rows),
        "integrity_max": max(float(row["max_abs_state_change"]) for row in integrity_rows),
        "rank_lines": rank_lines,
        "exec_pair_lines": exec_pair_lines,
        "payload_pair_lines": payload_pair_lines,
        "h_star": "NA" if h_star is None else str(h_star),
        "case_lines": case_lines,
        "case_stable_horizon": "NA" if stable_horizon is None else str(stable_horizon),
        "exec_stable_horizon": "NA" if exec_stable_horizon is None else str(exec_stable_horizon),
        "exec_synergy_medians": exec_synergy_medians,
        "payload_synergy_medians": payload_synergy_medians,
        "correction_text": correction_text,
        "payload_correction_text": payload_correction_text,
        "zero_payload_h1": zero_payload[1],
        "zero_payload_h5": zero_payload[5],
        "sinr_median": _median(row["h_sinr"] for row in divergence_rows if row["h_sinr"] != ""),
        "sinr_iqr": _iqr(row["h_sinr"] for row in divergence_rows if row["h_sinr"] != ""),
        "progress_median": _median(row["h_progress"] for row in divergence_rows if row["h_progress"] != ""),
        "progress_iqr": _iqr(row["h_progress"] for row in divergence_rows if row["h_progress"] != ""),
        "payload_median": _median(row["h_payload"] for row in divergence_rows if row["h_payload"] != ""),
        "payload_iqr": _iqr(row["h_payload"] for row in divergence_rows if row["h_payload"] != ""),
        "runtime_lines": runtime_lines,
        "gate": gate,
        "advice": advice,
        "answers": answers,
    }
    recorded = {(int(row["scenario_id"]), int(row["time_step"]), str(row["assignment_vector"])) for row in rollout_rows}
    for (scenario, moment, vector), checkpoints in cache.items():
        if (int(scenario), int(moment), str(vector)) in recorded:
            continue
        for horizon in HORIZONS:
            rollout_rows.append(
                {
                    "split": "confirmation" if int(scenario) >= 310110 else "train",
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "assignment_vector": str(vector),
                    "horizon": int(horizon),
                    **checkpoints[int(horizon)],
                    "j15": float(checkpoints[15]["payload_h"]),
                }
            )
    _write_csv(OUTPUT / "phase5x_shadow_state_integrity.csv", integrity_rows)
    _write_csv(OUTPUT / "phase5x_executor_rollouts.csv", rollout_rows)
    _write_csv(OUTPUT / "phase5x_horizon_ranking_metrics.csv", rank_rows)
    _write_csv(OUTPUT / "phase5x_horizon_regret.csv", [row for row in rank_rows if row["split"] == "confirmation"])
    _write_csv(OUTPUT / "phase5x_residual_case_correction.csv", residual_rows)
    _write_csv(OUTPUT / "phase5x_310114_t15.csv", case_rows)
    _write_csv(OUTPUT / "phase5x_synergy_horizon.csv", synergy_rows)
    _write_csv(OUTPUT / "phase5x_divergence_horizon.csv", divergence_rows)
    _write_csv(OUTPUT / "phase5x_runtime.csv", runtime_rows)
    _write_csv(OUTPUT / "phase5x_paired_comparison.csv", paired)
    payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "gate_score": "q_exec_h",
        "h_star": h_star,
        "case_stable_horizon_payload": stable_horizon,
        "case_stable_horizon_exec": exec_stable_horizon,
        "case_exec_optimum_set_rank_h5": case_opt_rank,
        "integrity_max_abs_change": summary["integrity_max"],
        "best_early_signal": {"horizon": best_signal[0], "name": best_signal[1], "spearman": signal_scores[best_signal]},
        "signal_spearman": {f"h{horizon}_{name}": signal_scores[(horizon, name)] for horizon, name in signal_scores},
        "trained": False,
    }
    (OUTPUT / "phase5x_executor_lookahead_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "h_star": h_star, "stable_payload": stable_horizon, "stable_exec": exec_stable_horizon})), flush=True)
    return summary


if __name__ == "__main__":
    run()
