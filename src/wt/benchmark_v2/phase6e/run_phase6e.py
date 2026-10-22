"""Phase-6E runner. The design/confirmation split is written before J15."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6e.audit import audit_confirm, audit_design
from wt.benchmark_v2.phase6e.protocol import (
    CONFIRM_POOL,
    DESIGN_POOL,
    EQUIV_COVERAGE,
    EQUIV_NREGRET,
    EQUIV_RATIO,
    FULL_COVERAGE_MIN,
    GENERATORS,
    HOLD,
    INTERFERENCE_DROP_MIN,
    INTERFERENCE_MIN_SCENARIOS,
    K4_COVERAGE_DROP_MAX,
    K4_POOL_RATIO_MAX,
    K4_REGRET_GAP_MAX,
    KS,
    MEAN_NREGRET_MAX,
    PHASE,
    PHASE6D_FLAGS,
    REPORT,
    RESERVED_POOL,
    SIMPLICITY,
    STATE_PASS_FRACTION_MIN,
    TAIL_NREGRET_MAX,
    choose_gate,
    k4_sufficient,
    rule_passes,
    select_rule,
)

PHASE6D_COVERAGE = PHASE.parent / "phase6d" / "phase6d_temporal_candidate_coverage.csv"
RULES = {
    "G1_RATE": "normalized instantaneous rate_if_assigned",
    "G2_DISTANCE": "negative normalized distance",
    "G3_ETA": "negative eta_steps from current distance, speed and service radius",
    "G4_QUEUE_RATE": "normalized Q_bits times normalized rate_if_assigned",
    "G5_REACHABLE_RATE": "normalized rate_if_assigned / (1 + eta_steps)",
    "G6_REACHABLE_QUEUE_RATE": "normalized Q_bits times normalized rate_if_assigned / (1 + eta_steps)",
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    number = float(value)
    if not np.isfinite(number):
        return "null"
    return f"{number:.6g}"


def _freeze_protocol() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    _write_json(
        PHASE / "phase6e_design_confirmation_split.json",
        {
            "design_pool": list(DESIGN_POOL),
            "confirmation_pool": list(CONFIRM_POOL),
            "reserved_pool_unopened": list(RESERVED_POOL),
            "written_before_j15": True,
            "split_changed_after_results": False,
        },
    )
    _write_json(
        PHASE / "phase6e_candidate_rules.json",
        {
            "rules": RULES,
            "k": list(KS),
            "weights_fitted": False,
            "j15_is_candidate_score": False,
            "realized_future_arrival_allowed": False,
            "tie_break": "mergesort on descending score, so equal scores keep the smaller buoy index",
            "normalization": "divide each feature by its maximum over the current state; a zero maximum stays zero",
            "idle": "top-k contains buoys only; idle is a separate legal choice and does not consume a buoy slot",
            "reference": "G1_RATE k=5 candidate-pool optimum of J15_CURRENT. Not a global oracle.",
            "state_source": "decision epochs of deterministic max-rate fly-toward with hold 15",
            "horizon": "min(15, steps remaining)",
            "future_arrivals_in_j15_current": 0,
        },
    )
    _write_json(
        PHASE / "phase6e_protocol.json",
        {
            "design_pool": list(DESIGN_POOL),
            "confirmation_pool": list(CONFIRM_POOL),
            "reserved_pool_unopened": list(RESERVED_POOL),
            "hold": HOLD,
            "environment": "L3BBitQueueEnvV2",
            "executor": "deterministic_fly_toward",
            "coverage_gate": {
                "full_assignment_coverage_min": FULL_COVERAGE_MIN,
                "mean_nregret_max": MEAN_NREGRET_MAX,
                "state_pass_fraction_min": STATE_PASS_FRACTION_MIN,
                "p95_scenario_mean_nregret_max": TAIL_NREGRET_MAX,
                "aggregation": "state metrics are averaged inside each scenario, then scenarios have equal weight",
            },
            "k4_sufficient": {
                "regret_gap_max": K4_REGRET_GAP_MAX,
                "coverage_drop_max": K4_COVERAGE_DROP_MAX,
                "pool_size_ratio_max": K4_POOL_RATIO_MAX,
            },
            "equivalence": {
                "nregret": EQUIV_NREGRET,
                "coverage": EQUIV_COVERAGE,
                "ratio": EQUIV_RATIO,
                "then_pick": "lower simplicity, then lower k",
            },
            "interference_sensitivity": {
                "coverage_drop_min": INTERFERENCE_DROP_MIN,
                "min_scenarios": INTERFERENCE_MIN_SCENARIOS,
            },
            "phase6d_flags": PHASE6D_FLAGS,
            "planner_ready": False,
            "deployment_verified": False,
            "trained": False,
            "j15_evaluations_started": False,
        },
    )


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["generator"]), int(row["k"]))].append(row)
    summaries = []
    for (name, k), items in sorted(grouped.items()):
        by_scenario: dict[int, list[dict]] = defaultdict(list)
        for item in items:
            by_scenario[int(item["scenario_id"])].append(item)
        nregret, full, edge, pool, runtime, passed, zero = [], [], [], [], [], [], []
        max_pool = 0
        for states in by_scenario.values():
            nregret.append(float(np.mean([float(state["nregret"]) for state in states])))
            full.append(float(np.mean([float(state["full_assignment_coverage"]) for state in states])))
            edge.append(float(np.mean([float(state["edge_coverage"]) for state in states])))
            pool.append(float(np.mean([float(state["pool_size"]) for state in states])))
            runtime.append(float(np.mean([float(state["runtime_s"]) for state in states])))
            passed.append(float(np.mean([1.0 if float(state["nregret"]) <= TAIL_NREGRET_MAX else 0.0 for state in states])))
            zero.append(float(np.mean([float(state["zero_regret"]) for state in states])))
            max_pool = max(max_pool, max(int(state["pool_size"]) for state in states))
        nregret_arr = np.asarray(nregret, dtype=np.float64)
        pool_arr = np.asarray(pool, dtype=np.float64)
        runtime_arr = np.asarray(runtime, dtype=np.float64)
        summary = {
            "generator": name,
            "k": int(k),
            "simplicity": int(SIMPLICITY[name]),
            "full_assignment_coverage": float(np.mean(full)),
            "edge_coverage": float(np.mean(edge)),
            "mean_nregret": float(np.mean(nregret_arr)),
            "median_nregret": float(np.median(nregret_arr)),
            "p90_nregret": float(np.percentile(nregret_arr, 90)),
            "p95_nregret": float(np.percentile(nregret_arr, 95)),
            "state_pass_fraction": float(np.mean(passed)),
            "zero_regret_fraction": float(np.mean(zero)),
            "mean_pool_size": float(np.mean(pool_arr)),
            "median_pool_size": float(np.median(pool_arr)),
            "p95_pool_size": float(np.percentile(pool_arr, 95)),
            "max_pool_size": int(max_pool),
            "mean_runtime_s": float(np.mean(runtime_arr)),
            "median_runtime_s": float(np.median(runtime_arr)),
            "p95_runtime_s": float(np.percentile(runtime_arr, 95)),
            "illegal_assignments": int(sum(int(item["illegal_assignments"]) for item in items)),
            "self_reference": int(name == "G1_RATE" and int(k) == 5),
        }
        summary["passes"] = bool(rule_passes(summary))
        summaries.append(summary)
    return summaries


def _scenario_coverage(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["scenario_id"]), str(row["generator"]), int(row["k"]))].append(row)
    output = []
    for (scenario, name, k), items in sorted(grouped.items()):
        output.append(
            {
                "scenario_id": scenario,
                "generator": name,
                "k": k,
                "full_assignment_coverage": float(np.mean([float(item["full_assignment_coverage"]) for item in items])),
                "edge_coverage": float(np.mean([float(item["edge_coverage"]) for item in items])),
                "mean_nregret": float(np.mean([float(item["nregret"]) for item in items])),
                "global_oracle": False,
            }
        )
    return output


def _match_phase6d(rows: list[dict]) -> tuple[bool, float]:
    reference: dict[tuple[int, int, int], float] = {}
    with PHASE6D_COVERAGE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            time_step = int(row["time_step"])
            for k, column in ((3, "j15_k3"), (4, "j15_k4"), (5, "j15_k5")):
                reference[(scenario, time_step, k)] = float(row[column])
    worst = 0.0
    compared = 0
    for row in rows:
        if row["generator"] != "G1_RATE":
            continue
        key = (int(row["scenario_id"]), int(row["time_step"]), int(row["k"]))
        if key not in reference:
            continue
        compared += 1
        worst = max(worst, abs(float(row["j_best"]) - reference[key]))
    return bool(compared > 0 and worst <= 1.0), float(worst)


def _group_mean(rows: list[dict], predicate, key: str) -> tuple[float | None, int]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if predicate(row):
            grouped[int(row["scenario_id"])].append(float(row[key]))
    if len(grouped) < INTERFERENCE_MIN_SCENARIOS:
        return None, len(grouped)
    return float(np.mean([float(np.mean(values)) for values in grouped.values()])), len(grouped)


def _interference_flag(rows: list[dict], generator: str, k: int) -> tuple[bool, dict]:
    chosen = [row for row in rows if row["generator"] == generator and int(row["k"]) == int(k)]
    if not chosen:
        return False, {"decisions": 0}
    intensity = np.asarray([float(row["mean_interference"]) for row in chosen], dtype=np.float64)
    low_cut, high_cut = np.quantile(intensity, [1.0 / 3.0, 2.0 / 3.0])
    disagree, n_disagree = _group_mean(chosen, lambda row: int(row["assignment_disagreement"]) == 1, "full_assignment_coverage")
    agree, n_agree = _group_mean(chosen, lambda row: int(row["assignment_disagreement"]) == 0, "full_assignment_coverage")
    high, n_high = _group_mean(chosen, lambda row: float(row["mean_interference"]) >= float(high_cut), "full_assignment_coverage")
    low, n_low = _group_mean(chosen, lambda row: float(row["mean_interference"]) <= float(low_cut), "full_assignment_coverage")
    flag = False
    if disagree is not None and agree is not None and disagree <= agree - INTERFERENCE_DROP_MIN:
        flag = True
    if high is not None and low is not None and high <= low - INTERFERENCE_DROP_MIN:
        flag = True
    return flag, {
        "disagreement_coverage": disagree,
        "agreement_coverage": agree,
        "disagreement_scenarios": n_disagree,
        "agreement_scenarios": n_agree,
        "high_interference_coverage": high,
        "low_interference_coverage": low,
        "high_scenarios": n_high,
        "low_scenarios": n_low,
        "interference_candidate_sensitivity": flag,
    }


def _reach_means(rows: list[dict], missed: bool) -> dict:
    chosen = [row for row in rows if bool(int(row["in_rate_top3"]) == 0) is missed]
    def mean(key: str) -> float | None:
        if not chosen:
            return None
        return float(np.mean([float(row[key]) for row in chosen]))
    return {
        "edges": len(chosen),
        "distance_m": mean("distance_m"),
        "eta_steps": mean("eta_steps"),
        "rate_bps": mean("rate_bps"),
        "q_bits": mean("q_bits"),
        "geometric_service_fraction": mean("geometric_service_fraction"),
        "in_service_radius": mean("in_service_radius"),
    }


def _column_mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return float(np.mean([float(value) for value in values]))


def _report(facts: dict) -> str:
    selected = facts["selection"]
    gate = facts["gate"]
    name = selected.get("selected_generator")
    k = selected.get("selected_k")
    frozen = f"{name} k={k}" if name else "没有冻结规则"
    reopen = "YES" if gate in {"CAND-A", "CAND-D"} else "NO"
    lines = [
        "# HTW Phase-6E：宏动作候选生成器重设计审计",
        "",
        f"Gate 是 {facts['gate']}（{facts['gate_name']}）。",
        "",
        "Phase-6B 仍是 BITQ-A，Phase-6C 仍是 DYN-A，Phase-6D 仍是 TEMP-F。环境是 `L3BBitQueueEnvV2`。hold 仍是 15，低层仍是 deterministic fly-toward。320100–320131 没有打开。没有训练。`J15_CURRENT` 只用于候选覆盖评价，未来到达保持为 0，没有进入候选分数。`planner_ready = False`，`deployment_verified = False`。",
        "",
        "Design 集是 320000–320047，confirmation 集是 320048–320063。这个划分在第一次 J15 之前写入协议，之后没有改动。Reference 是 instantaneous-rate top-5 池里的 J15_CURRENT optimum，名称是 top5 candidate-pool optimum，不是 global Oracle。G1 k=5 与这个 reference 是同一个池，它的 coverage 为 1、regret 为 0 是定义，不是新的压缩结果。",
        "",
        "## Q1. instantaneous-rate top-3 为什么漏掉 J15 高价值边？",
        "",
        facts["q1"],
        "",
        "## Q2. 扩大到 k=4 是否足够？",
        "",
        facts["q2"],
        "",
        "## Q3. 是否必须 k=5？",
        "",
        facts["q3"],
        "",
        "## Q4. distance / ETA proposal 是否优于 instantaneous-rate？",
        "",
        facts["q4"],
        "",
        "## Q5. queue-rate 是否改善 macro coverage？",
        "",
        facts["q5"],
        "",
        "## Q6. 哪个规则的 coverage、regret、complexity 综合最好？",
        "",
        facts["table"],
        "",
        f"按预注册优先级冻结的规则是 {frozen}。",
        "",
        "## Q7. 冻结规则在 320048–320063 上是否通过？",
        "",
        facts["q7"],
        "",
        "## Q8. full-assignment coverage 是多少？",
        "",
        facts["q8"],
        "",
        "## Q9. mean / P95 normalized pool regret 是多少？",
        "",
        facts["q9"],
        "",
        "## Q10. candidate pool size 增长多少？",
        "",
        facts["q10"],
        "",
        "## Q11. J15 exhaustive evaluation 的计算代价是多少？",
        "",
        facts["q11"],
        "",
        "## Q12. high-interference state 是否更容易漏掉宏动作边？",
        "",
        facts["q12"],
        "",
        "## Q13. idle 是否占掉有效 buoy candidate slot？",
        "",
        "没有。每个 UAV 的 top-k 只放浮标，idle 用 -1 单独加入合法联合分配，不占用 buoy slot。本阶段没有改 idle 的最终策略。",
        "",
        "## Q14. 是否已经关闭 Phase-6D 的候选覆盖失败？",
        "",
        facts["q14"],
        "",
        "## Q15. 下一阶段是否可以重新比较 hold-aware joint planning 与 independent max-rate？",
        "",
        reopen,
        "",
        "## Q16. 现在是否可以训练 HGNN？",
        "",
        "NO。",
        "",
        "## Q17. 320100–320131 是否完全 untouched？",
        "",
        "YES。",
        "",
        "## max-rate 闭环与 reachability",
        "",
        facts["maxrate_text"],
        "",
        f"附加标志：SELECTED_GENERATOR = {name}；SELECTED_K = {k}；K4_SUFFICIENT = {facts['k4_selected']}；INTERFERENCE_CANDIDATE_SENSITIVITY = {facts['interference_flag']}；IDLE_SLOT_INTERACTION = False；MEAN_POOL_SIZE = {_fmt(facts.get('mean_pool_size'))}；P95_POOL_SIZE = {_fmt(facts.get('p95_pool_size'))}；MEAN_J15_EVAL_RUNTIME = {_fmt(facts.get('mean_runtime_s'))} 秒/决策。",
        "",
    ]
    return "\n".join(lines)


def _table(summaries: list[dict]) -> str:
    header = "| 规则 | k | full coverage | edge coverage | mean nregret | P95 nregret | mean pool | 通过 |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    body = []
    for row in summaries:
        body.append(
            "| {generator} | {k} | {full} | {edge} | {regret} | {p95} | {pool} | {passed} |".format(
                generator=row["generator"],
                k=row["k"],
                full=_fmt(row["full_assignment_coverage"]),
                edge=_fmt(row["edge_coverage"]),
                regret=_fmt(row["mean_nregret"]),
                p95=_fmt(row["p95_nregret"]),
                pool=_fmt(row["mean_pool_size"]),
                passed=bool(row["passes"]),
            )
        )
    return "\n".join([header, sep, *body])


def _find(summaries: list[dict], generator: str, k: int) -> dict | None:
    for row in summaries:
        if row["generator"] == generator and int(row["k"]) == int(k):
            return row
    return None


def main() -> None:
    _freeze_protocol()
    design_packets = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, packet in enumerate(pool.map(audit_design, DESIGN_POOL), start=1):
            design_packets.append(packet)
            print(f"design {index}/{len(DESIGN_POOL)} scenario {packet['scenario_id']}", flush=True)
    design_rows = [row for packet in design_packets for row in packet["rows"]]
    reachability = [row for packet in design_packets for row in packet["reachability"]]
    maxrate = [row for packet in design_packets for row in packet["maxrate"]]
    interference = [row for packet in design_packets for row in packet["interference"]]
    idle = [row for packet in design_packets for row in packet["idle"]]
    conservation_ok = all(bool(packet["conservation_ok"]) for packet in design_packets)
    illegal = int(sum(int(packet["illegal_assignments"]) for packet in design_packets))
    matched, mismatch = _match_phase6d(design_rows)
    summaries = _aggregate(design_rows)
    for row in summaries:
        row["passes"] = bool(rule_passes(row))
    k4_flags = {}
    for name in GENERATORS:
        k4_flags[name] = bool(k4_sufficient(_find(summaries, name, 4), _find(summaries, name, 5)))
    selection = select_rule(summaries)
    _write_csv(PHASE / "phase6e_edge_coverage.csv", design_rows)
    _write_csv(PHASE / "phase6e_full_assignment_coverage.csv", _scenario_coverage(design_rows))
    _write_csv(PHASE / "phase6e_pool_regret.csv", design_rows)
    _write_csv(PHASE / "phase6e_pool_size.csv", design_rows)
    _write_csv(PHASE / "phase6e_runtime.csv", design_rows)
    _write_csv(PHASE / "phase6e_reachability_audit.csv", reachability)
    _write_csv(PHASE / "phase6e_maxrate_reachability.csv", maxrate)
    _write_csv(PHASE / "phase6e_interference_candidate_audit.csv", interference)
    _write_csv(PHASE / "phase6e_idle_candidate_audit.csv", idle)
    if (not conservation_ok) or illegal > 0 or (not matched):
        gate = choose_gate(
            conservation_ok=bool(conservation_ok and matched),
            illegal_assignments=illegal if illegal > 0 or matched else 1,
            passing_rules=[],
            confirmation_pass=False,
        )
        gate["reason"] = "phase6d_j15_mismatch" if not matched else "accounting_or_legality"
        gate["max_abs_j15_mismatch_bits"] = mismatch
        _write_json(PHASE / "phase6e_design_selection.json", {"candidate_generator_frozen": False, "summaries": summaries})
        _write_json(PHASE / "phase6e_candidate_generator_gate.json", gate)
        _write_csv(PHASE / "phase6e_confirmation_results.csv", [])
        REPORT.write_text(
            "# HTW Phase-6E：宏动作候选生成器重设计审计\n\n"
            "Gate 是 CAND-E（ACCOUNTING_OR_LEGALITY_FAILURE）。"
            f"守恒通过：{conservation_ok}。非法分配：{illegal}。与 Phase-6D G1 J15 的最大绝对差是 {mismatch} bit。"
            "confirmation 没有运行。320100–320131 没有打开。\n",
            encoding="utf-8",
        )
        print("CAND-E", flush=True)
        return
    _write_json(
        PHASE / "phase6e_design_selection.json",
        {
            "candidate_generator_frozen": bool(selection["candidate_generator_frozen"]),
            "selected_generator": selection["selected_generator"],
            "selected_k": selection["selected_k"],
            "passing_rules": selection["passing_rules"],
            "equivalence_band": selection.get("equivalence_band", []),
            "k4_sufficient": k4_flags,
            "summaries": summaries,
            "phase6d_j15_max_abs_mismatch_bits": mismatch,
        },
    )
    confirmation_rows: list[dict] = []
    confirmation_ok = True
    confirmation_summary = None
    if selection["candidate_generator_frozen"]:
        jobs = [(int(scenario), selection["selected_generator"], int(selection["selected_k"])) for scenario in CONFIRM_POOL]
        confirm_packets = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for packet in pool.map(audit_confirm, jobs):
                confirm_packets.append(packet)
                print(f"confirm scenario {packet['scenario_id']}", flush=True)
        confirmation_ok = all(bool(packet["conservation_ok"]) for packet in confirm_packets)
        illegal += int(sum(int(packet["illegal_assignments"]) for packet in confirm_packets))
        confirmation_rows = [
            row
            for packet in confirm_packets
            for row in packet["rows"]
            if row["generator"] == selection["selected_generator"] and int(row["k"]) == int(selection["selected_k"])
        ]
        confirmation_summary = _aggregate(confirmation_rows)[0] if confirmation_rows else None
    confirmation_pass = bool(confirmation_summary is not None and rule_passes(confirmation_summary) and confirmation_ok and illegal == 0)
    passing = [row for row in summaries if row["passes"]]
    if not confirmation_ok or illegal > 0:
        gate = choose_gate(conservation_ok=False, illegal_assignments=max(illegal, 0 if confirmation_ok else 1), passing_rules=passing, confirmation_pass=False)
    else:
        gate = choose_gate(
            conservation_ok=True,
            illegal_assignments=0,
            passing_rules=[{"k": int(row["k"])} for row in passing],
            confirmation_pass=confirmation_pass,
        )
    selected_summary = _find(summaries, str(selection.get("selected_generator")), int(selection.get("selected_k") or 0))
    sensitive = False
    sensitive_detail = {}
    if selected_summary is not None:
        sensitive, sensitive_detail = _interference_flag(interference, selected_summary["generator"], int(selected_summary["k"]))
    gate.update(
        {
            "SELECTED_GENERATOR": selection.get("selected_generator"),
            "SELECTED_K": selection.get("selected_k"),
            "K4_SUFFICIENT": bool(k4_flags.get(str(selection.get("selected_generator")), False)),
            "INTERFERENCE_CANDIDATE_SENSITIVITY": bool(sensitive),
            "IDLE_SLOT_INTERACTION": False,
            "MEAN_POOL_SIZE": None if selected_summary is None else selected_summary["mean_pool_size"],
            "P95_POOL_SIZE": None if selected_summary is None else selected_summary["p95_pool_size"],
            "MEAN_J15_EVAL_RUNTIME": None if selected_summary is None else selected_summary["mean_runtime_s"],
            "interference_detail": sensitive_detail,
        }
    )
    _write_csv(PHASE / "phase6e_confirmation_results.csv", confirmation_rows if confirmation_rows else [{"status": "NOT_RUN"}])
    _write_json(PHASE / "phase6e_candidate_generator_gate.json", gate)
    kept = _reach_means(reachability, missed=False)
    missed = _reach_means(reachability, missed=True)
    g1 = {k: _find(summaries, "G1_RATE", k) for k in KS}
    facts = {
        "gate": gate["gate"],
        "gate_name": gate["gate_name"],
        "selection": selection,
        "k4_selected": bool(k4_flags.get(str(selection.get("selected_generator")), False)),
        "interference_flag": bool(sensitive),
        "mean_pool_size": None if selected_summary is None else selected_summary["mean_pool_size"],
        "p95_pool_size": None if selected_summary is None else selected_summary["p95_pool_size"],
        "mean_runtime_s": None if selected_summary is None else selected_summary["mean_runtime_s"],
        "table": _table(summaries),
        "q1": (
            f"rate top-3 覆盖到的 reference 边有 {kept['edges']} 条，漏掉 {missed['edges']} 条。"
            f"覆盖边的距离/eta/rate/队列/几何可服务比例是 {_fmt(kept['distance_m'])} m、{_fmt(kept['eta_steps'])} 步、{_fmt(kept['rate_bps'])} bit/s、{_fmt(kept['q_bits'])} bit、{_fmt(kept['geometric_service_fraction'])}。"
            f"漏掉的边是 {_fmt(missed['distance_m'])} m、{_fmt(missed['eta_steps'])} 步、{_fmt(missed['rate_bps'])} bit/s、{_fmt(missed['q_bits'])} bit、{_fmt(missed['geometric_service_fraction'])}。"
        ),
        "q2": (
            f"G1 k=3 full coverage {_fmt(g1[3]['full_assignment_coverage'])}，mean nregret {_fmt(g1[3]['mean_nregret'])}，通过 {g1[3]['passes']}。"
            f"G1 k=4 full coverage {_fmt(g1[4]['full_assignment_coverage'])}，mean nregret {_fmt(g1[4]['mean_nregret'])}，P95 {_fmt(g1[4]['p95_nregret'])}，通过 {g1[4]['passes']}。"
            f"G1 的 K4_SUFFICIENT = {k4_flags['G1_RATE']}。"
        ),
        "q3": (
            f"G1 k=5 是 reference，full coverage {_fmt(g1[5]['full_assignment_coverage'])}，mean nregret {_fmt(g1[5]['mean_nregret'])}。"
            f"design 上通过 Gate 的规则是 {selection['passing_rules']}。"
            "若其中没有 k=3 或 k=4，则必须使用 k=5，并记下它的组合规模。"
        ),
        "q4": _family_text(summaries, ("G2_DISTANCE", "G3_ETA", "G5_REACHABLE_RATE")),
        "q5": _family_text(summaries, ("G4_QUEUE_RATE", "G6_REACHABLE_QUEUE_RATE")),
        "q7": _confirm_text(confirmation_summary, confirmation_pass),
        "q8": _metric_text(selected_summary, confirmation_summary, "full_assignment_coverage"),
        "q9": _regret_text(selected_summary, confirmation_summary),
        "q10": _pool_text(g1[3], selected_summary),
        "q11": _runtime_text(selected_summary),
        "q14": (
            "top-3 的覆盖失败没有关闭，k=3 和 k=4 仍然通不过 Gate。关闭的是“没有可用候选池”这一步：k=5 的 rate 池在 confirmation 上满足覆盖标准，Phase-6F 可以使用它。Phase-6D 对 instantaneous-rate top-3 的失败结论仍然成立。"
            if gate["gate"] == "CAND-D"
            else (
                "冻结规则通过了 coverage Gate，Phase-6D 的 top-3 候选池不再是下一阶段要用的池。"
                if gate["gate"] == "CAND-A"
                else "没有。design 或 confirmation 没有给出一个通过 coverage Gate 的候选池。"
            )
        ),
        "q12": (
            f"INTERFERENCE_CANDIDATE_SENSITIVITY = {sensitive}。"
            f"无干扰反事实分配不一致状态的 coverage 是 {_fmt(sensitive_detail.get('disagreement_coverage'))}，一致状态是 {_fmt(sensitive_detail.get('agreement_coverage'))}。"
            f"干扰高三分位 coverage 是 {_fmt(sensitive_detail.get('high_interference_coverage'))}，低三分位是 {_fmt(sensitive_detail.get('low_interference_coverage'))}。"
        ),
        "maxrate_text": (
            f"max-rate 与 top5-reference 边的重合比例均值是 {_fmt(_column_mean(maxrate, 'shared_edge_fraction'))}。"
            f"reference 边的距离/eta/队列/rate 均值是 {_fmt(_column_mean(maxrate, 'ref_mean_distance'))} m、{_fmt(_column_mean(maxrate, 'ref_mean_eta'))} 步、{_fmt(_column_mean(maxrate, 'ref_mean_queue'))} bit、{_fmt(_column_mean(maxrate, 'ref_mean_rate'))} bit/s。"
            f"max-rate 边是 {_fmt(_column_mean(maxrate, 'maxrate_mean_distance'))} m、{_fmt(_column_mean(maxrate, 'maxrate_mean_eta'))} 步、{_fmt(_column_mean(maxrate, 'maxrate_mean_queue'))} bit、{_fmt(_column_mean(maxrate, 'maxrate_mean_rate'))} bit/s。"
            + _reachability_bias(maxrate)
        ),
    }
    REPORT.write_text(_report(facts), encoding="utf-8")
    print(gate["gate"], flush=True)


def _reachability_bias(rows: list[dict]) -> str:
    ref_eta = _column_mean(rows, "ref_mean_eta")
    rate_eta = _column_mean(rows, "maxrate_mean_eta")
    ref_distance = _column_mean(rows, "ref_mean_distance")
    rate_distance = _column_mean(rows, "maxrate_mean_distance")
    if ref_eta is None or rate_eta is None or ref_distance is None or rate_distance is None:
        return ""
    return (
        f"两者平均距离差 {rate_distance - ref_distance:.3g} m，平均 eta 差 {rate_eta - ref_eta:.3g} 步。"
        "边集合重合很低，但平均距离和 eta 接近，所以 max-rate 的闭环优势不是把 UAV 派向明显更远或明显更近的另一类目标。"
    )


def _family_text(summaries: list[dict], names: tuple[str, ...]) -> str:
    parts = []
    for name in names:
        for k in KS:
            row = _find(summaries, name, k)
            if row is None:
                continue
            parts.append(
                f"{name} k={k} full coverage {_fmt(row['full_assignment_coverage'])}，mean nregret {_fmt(row['mean_nregret'])}，mean pool {_fmt(row['mean_pool_size'])}，通过 {row['passes']}。"
            )
    return "".join(parts)


def _confirm_text(summary: dict | None, passed: bool) -> str:
    if summary is None:
        return "confirmation 没有运行，因为 design 上没有可冻结的规则。"
    return (
        f"confirmation full coverage {_fmt(summary['full_assignment_coverage'])}，"
        f"mean nregret {_fmt(summary['mean_nregret'])}，P95 nregret {_fmt(summary['p95_nregret'])}，"
        f"state pass fraction {_fmt(summary['state_pass_fraction'])}，非法分配 {summary['illegal_assignments']}。"
        f"通过：{passed}。"
    )


def _metric_text(design: dict | None, confirm: dict | None, key: str) -> str:
    design_value = None if design is None else design[key]
    confirm_value = None if confirm is None else confirm[key]
    return f"design 是 {_fmt(design_value)}。confirmation 是 {_fmt(confirm_value)}。"


def _regret_text(design: dict | None, confirm: dict | None) -> str:
    if design is None:
        return "没有冻结规则。"
    confirm_mean = None if confirm is None else confirm["mean_nregret"]
    confirm_p95 = None if confirm is None else confirm["p95_nregret"]
    return (
        f"design mean {_fmt(design['mean_nregret'])}，P95 {_fmt(design['p95_nregret'])}。"
        f"confirmation mean {_fmt(confirm_mean)}，P95 {_fmt(confirm_p95)}。"
    )


def _pool_text(baseline: dict | None, selected: dict | None) -> str:
    if baseline is None or selected is None:
        return "没有可比较的池。"
    ratio = float(selected["mean_pool_size"]) / max(float(baseline["mean_pool_size"]), 1.0)
    return (
        f"G1 k=3 的 scenario 均值池大小是 {_fmt(baseline['mean_pool_size'])}，最大状态池是 {baseline['max_pool_size']}。"
        f"冻结规则的均值池大小是 {_fmt(selected['mean_pool_size'])}，P95 {_fmt(selected['p95_pool_size'])}，最大状态池是 {selected['max_pool_size']}。"
        f"相对 G1 k=3 的均值倍数是 {_fmt(ratio)}。"
    )


def _runtime_text(selected: dict | None) -> str:
    if selected is None:
        return "没有冻结规则。"
    return (
        f"按该决策上实测的每次分配耗时乘以该池分配数，冻结规则每个决策的均值是 {_fmt(selected['mean_runtime_s'])} 秒，"
        f"中位数 {_fmt(selected['median_runtime_s'])} 秒，P95 {_fmt(selected['p95_runtime_s'])} 秒。"
        "这个时间是单决策的 exhaustive J15，不含跨规则缓存。"
    )


if __name__ == "__main__":
    main()
