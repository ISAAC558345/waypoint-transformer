"""Write the Stage D2 tables from the frozen D0 and D1 label files."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .analyze import Candidate, leave_one_out, material_regret_bits, state_analysis
from .protocol import (
    D0_LABELS,
    D0_SCENARIOS,
    D0_SCHEMA,
    D0_SELECTED,
    D0_TIMES,
    D1_LABELS,
    D1_SCHEMA,
    D1_SCENARIOS,
    D1_SELECTED,
    D1_SUMMARY,
    D1_TIME,
    DELTA_PAYLOAD_MAX,
    EPS_BITS,
    GATE_READY,
    HOLDOUT_SCENARIO,
    INSTRUCTION_COMMIT,
    LAMBDA_H,
    OUTPUT,
    PAYLOAD_SCALE,
    choose_gate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assignments(path: Path) -> dict[tuple[int, int, int], str]:
    found = {}
    for row in _load_rows(path):
        found[(int(row["scenario_id"]), int(row["time_step"]), int(row["candidate_id"]))] = str(row["assignment"])
    return found


def _verify(labels: list[dict], assignments: dict, schema: dict, *, source: str) -> None:
    if source == "d0":
        expected = {(int(scenario), int(time)) for scenario in D0_SCENARIOS for time in D0_TIMES}
        if [int(item) for item in schema["scenarios"]] != list(D0_SCENARIOS):
            raise RuntimeError("D0 schema scenarios do not match the pilot report")
        if [int(item) for item in schema["times"]] != list(D0_TIMES):
            raise RuntimeError("D0 schema times do not match the pilot report")
    else:
        expected = {(int(scenario), int(D1_TIME)) for scenario in D1_SCENARIOS}
        if [int(item) for item in schema["scenarios"]] != list(D1_SCENARIOS):
            raise RuntimeError("D1 schema scenarios do not match the pilot report")
        if int(schema["time_step"]) != int(D1_TIME):
            raise RuntimeError("D1 schema time does not match the pilot report")
    if schema.get("lambda_H") is not None:
        raise RuntimeError(f"{source} schema already selects lambda_H")
    grouped = defaultdict(list)
    for row in labels:
        scenario = int(row["scenario_id"])
        time_step = int(row["time_step"])
        key = (scenario, time_step)
        if key not in expected:
            raise RuntimeError(f"{source} labels include an unexpected state {key}")
        payload_bits = float(row["payload_bits"])
        payload_term = float(row["payload_term"])
        if abs(payload_bits - payload_term * float(PAYLOAD_SCALE)) > float(EPS_BITS):
            raise RuntimeError(f"{source} payload term is not payload bits / 1e6")
        urgent_bits = float(row["urgent_service_bits"])
        urgent_term = float(row["urgent_service_term"])
        if abs(urgent_bits - urgent_term * float(PAYLOAD_SCALE)) > float(EPS_BITS):
            raise RuntimeError(f"{source} urgent term is not urgent bits / 1e6")
        candidate_id = int(row["candidate_id"])
        if (scenario, time_step, candidate_id) not in assignments:
            raise RuntimeError(f"{source} candidate {candidate_id} has no saved assignment")
        grouped[key].append(candidate_id)
    if set(grouped) != expected:
        raise RuntimeError(f"{source} is missing a registered pilot state")
    for key, ids in grouped.items():
        if len(ids) != 128 or len(set(ids)) != 128:
            raise RuntimeError(f"{source} state {key} does not contain 128 unique candidates")


def _candidates(labels: list[dict], assignments: dict) -> dict[tuple[str, int, int], list[Candidate]]:
    grouped = defaultdict(list)
    for row in labels:
        scenario = int(row["scenario_id"])
        time_step = int(row["time_step"])
        source = "d1" if scenario in D1_SCENARIOS else "d0"
        candidate_id = int(row["candidate_id"])
        grouped[(source, scenario, time_step)].append(
            Candidate(
                candidate_id,
                float(row["payload_bits"]),
                float(row["urgent_service_bits"]),
                assignments[(scenario, time_step, candidate_id)],
            )
        )
    return grouped


def _reconcile_d1(records: list[dict], grouped: dict) -> dict:
    """Confirm the published D1 top-1 ids still match once 1-bit ties are applied."""
    published = json.loads(D1_SUMMARY.read_text(encoding="utf-8"))["rank_by_state"]
    if len(published) != len(records):
        return {"matched": False, "reason": "D1 summary state count differs from the label file"}
    max_gap = 0.0
    urgent_id_matches = 0
    payload_top_inside_tie = 0
    payload_top_id_differs = 0
    for row in published:
        scenario = int(row["scenario_id"])
        record = next(item for item in records if int(item["scenario_id"]) == scenario)
        if int(record["urgent_first_candidate_id"]) != int(row["urgent_top_candidate_id"]):
            return {"matched": False, "reason": f"urgent top for {scenario} does not match the D1 summary"}
        urgent_id_matches += 1
        gap = abs(float(record["raw_top1_payload_regret_bits"]) - float(row["top1_payload_regret_bits"]))
        max_gap = max(max_gap, gap)
        if gap > float(EPS_BITS):
            return {"matched": False, "reason": f"payload regret for {scenario} differs from the D1 summary by more than 1 bit"}
        candidates = grouped[("d1", scenario, int(D1_TIME))]
        payload_top = next(item for item in candidates if int(item.candidate_id) == int(row["payload_top_candidate_id"]))
        if abs(float(payload_top.payload_bits) - float(record["best_payload_bits"])) > float(EPS_BITS):
            return {"matched": False, "reason": f"D1 payload top for {scenario} is outside the 1-bit tie"}
        payload_top_inside_tie += 1
        if int(payload_top.candidate_id) != int(record["payload_first_candidate_id"]):
            payload_top_id_differs += 1
    return {
        "matched": True,
        "states": int(len(records)),
        "urgent_top_id_matches": int(urgent_id_matches),
        "max_abs_regret_gap_bits": float(max_gap),
        "published_payload_top_inside_one_bit": int(payload_top_inside_tie),
        "published_payload_top_id_differs_after_tiebreak": int(payload_top_id_differs),
    }


def _time_summary(records: list[dict], scope: str) -> dict:
    material = [float(item["material_top1_payload_regret_bits"]) for item in records]
    lambdas = [item["first_material_lambda"] for item in records if item["first_material_lambda"] is not None]
    return {
        "states": int(len(records)),
        "scope": scope,
        "mean_material_top1_payload_regret_bits": float(sum(material) / len(material)),
        "states_with_material_payload_loss": int(sum(value < 0.0 for value in material)),
        "states_urgent_opt_inside_payload_tie": int(sum(int(item["urgent_opt_inside_payload_tie"]) for item in records)),
        "mean_free_urgent_gain_bits": float(sum(float(item["free_urgent_gain_bits"]) for item in records) / len(records)),
        "mean_pareto_count": float(sum(int(item["pareto_count"]) for item in records) / len(records)),
        "mean_payload_tie_count": float(sum(int(item["payload_tie_count"]) for item in records) / len(records)),
        "mean_payload_pair_tie_fraction": float(sum(float(item["payload_pair_tie_fraction"]) for item in records) / len(records)),
        "states_with_material_lambda": int(len(lambdas)),
        "min_first_material_lambda": None if not lambdas else float(min(lambdas)),
        "max_first_material_lambda": None if not lambdas else float(max(lambdas)),
        "tiebreak_assignment_change_rate": float(sum(int(item["tiebreak_changes_assignment"]) for item in records) / len(records)),
        "urgent_rule_assignment_change_rate": float(sum(int(item["urgent_rule_changes_assignment"]) for item in records) / len(records)),
        "mean_lambda_assignment_changes": float(sum(int(item["lambda_assignment_changes"]) for item in records) / len(records)),
        "mean_positive_cross_count": float(sum(int(item["positive_cross_count"]) for item in records) / len(records)),
        "max_positive_cross_count": int(max(int(item["positive_cross_count"]) for item in records)),
    }


def execute() -> dict:
    d0_schema = json.loads(D0_SCHEMA.read_text(encoding="utf-8"))
    d1_schema = json.loads(D1_SCHEMA.read_text(encoding="utf-8"))
    d0_labels = _load_rows(D0_LABELS)
    d1_labels = _load_rows(D1_LABELS)
    assignments = _assignments(D0_SELECTED)
    assignments.update(_assignments(D1_SELECTED))
    reasons = []
    inputs_ok = True
    try:
        _verify(d0_labels, assignments, d0_schema, source="d0")
        _verify(d1_labels, assignments, d1_schema, source="d1")
    except RuntimeError as exc:
        inputs_ok = False
        reasons.append(str(exc))
    records = []
    pareto_rows = []
    lambda_rows = []
    constrained_steps = []
    grouped = {}
    if inputs_ok:
        grouped = _candidates(d0_labels + d1_labels, assignments)
        for (source, scenario, time_step), candidates in sorted(grouped.items()):
            analysis = state_analysis(candidates)
            by_id = {int(item.candidate_id): item for item in candidates}
            payload_choice = by_id[int(analysis["payload_first_candidate_id"])]
            payload_only = by_id[int(analysis["payload_id_only_candidate_id"])]
            urgent_choice = by_id[int(analysis["urgent_first_candidate_id"])]
            chain_assignments = [str(item["assignment"]) for item in analysis["segments"]]
            record = {
                "source": source,
                "scenario_id": int(scenario),
                "time_step": int(time_step),
                **{key: value for key, value in analysis.items() if key not in {"pareto", "segments"}},
                "payload_first_assignment": payload_choice.assignment,
                "payload_id_only_assignment": payload_only.assignment,
                "urgent_first_assignment": urgent_choice.assignment,
                "tiebreak_changes_assignment": int(payload_choice.assignment != payload_only.assignment),
                "urgent_rule_changes_assignment": int(urgent_choice.assignment != payload_choice.assignment),
                "lambda_assignment_changes": int(sum(left != right for left, right in zip(chain_assignments, chain_assignments[1:]))),
            }
            records.append(record)
            for candidate in analysis["pareto"]:
                pareto_rows.append(
                    {
                        "source": source,
                        "scenario_id": int(scenario),
                        "time_step": int(time_step),
                        "candidate_id": int(candidate.candidate_id),
                        "assignment": candidate.assignment,
                        "payload_bits": float(candidate.payload_bits),
                        "payload_term": float(candidate.payload_term),
                        "urgent_service_bits": float(candidate.urgent_bits),
                        "urgent_service_term": float(candidate.urgent_term),
                        "is_payload_first": int(candidate.candidate_id == analysis["payload_first_candidate_id"]),
                        "is_urgent_first": int(candidate.candidate_id == analysis["urgent_first_candidate_id"]),
                        "payload_regret_bits": float(candidate.payload_bits) - float(analysis["best_payload_bits"]),
                        "material_payload_regret_bits": material_regret_bits(candidate.payload_bits, analysis["best_payload_bits"]),
                        "urgent_gap_vs_best_bits": float(candidate.urgent_bits) - float(analysis["best_urgent_bits"]),
                    }
                )
            for segment in analysis["segments"]:
                lambda_rows.append({"source": source, "scenario_id": int(scenario), "time_step": int(time_step), **segment})
            for candidate in analysis["pareto"]:
                regret = material_regret_bits(candidate.payload_bits, analysis["best_payload_bits"])
                if regret < 0.0:
                    constrained_steps.append(
                        {
                            "source": source,
                            "scenario_id": int(scenario),
                            "time_step": int(time_step),
                            "candidate_id": int(candidate.candidate_id),
                            "payload_sacrifice_bits": float(-regret),
                            "urgent_gain_vs_payload_first_bits": float(candidate.urgent_bits) - float(payload_choice.urgent_bits),
                            "on_weighted_sum_path": int(candidate.candidate_id in {int(item["candidate_id"]) for item in analysis["segments"]}),
                        }
                    )
    frontier_ok = bool(records) and all(int(item["pareto_count"]) >= 1 and int(item["segment_count"]) >= 1 for item in records)
    rules_ok = bool(records) and all("payload_first_candidate_id" in item and "first_material_lambda" in item for item in records)
    by_time = {}
    for time_step in D0_TIMES:
        subset = [item for item in records if int(item["time_step"]) == int(time_step) and item["source"] == "d0"]
        if subset:
            by_time[str(time_step)] = _time_summary(subset, "D0 scenarios 322000-322003 only")
    d1_records = [item for item in records if item["source"] == "d1"]
    t30_records = [item for item in records if int(item["time_step"]) == int(D1_TIME)]
    if d1_records:
        by_time["d1_t30"] = _time_summary(d1_records, "D1 scenarios 322004-322015 at t=30")
    if t30_records:
        by_time["t30_all"] = _time_summary(t30_records, "D0 and D1 states at t=30; 322015 retained")
    reconciliation = _reconcile_d1(d1_records, grouped) if d1_records else {"matched": False, "reason": "no D1 records"}
    if not reconciliation.get("matched"):
        inputs_ok = False
        reasons.append(str(reconciliation.get("reason", "D1 summary does not match the recomputed labels")))
    loo = leave_one_out(d1_records, HOLDOUT_SCENARIO) if d1_records else None
    if loo is not None and int(loo["full_count"]) != 12:
        inputs_ok = False
        reasons.append("leave-one-out changed the D1 state count")
    gate = choose_gate(inputs_ok=inputs_ok and loo is not None, frontier_ok=frontier_ok, rules_ok=rules_ok)
    if gate != GATE_READY and not reasons:
        reasons.append("frontier or rule comparison is incomplete")
    return {
        "gate": gate,
        "reasons": reasons,
        "lambda_H": LAMBDA_H,
        "lambda_H_decided": False,
        "delta_payload_max": DELTA_PAYLOAD_MAX,
        "delta_payload_max_status": "needs_research_lead_confirmation",
        "full_dataset_supported": False,
        "trained_hgnn": False,
        "formal_system_modified": False,
        "rollouts_rerun": False,
        "eps_bits": float(EPS_BITS),
        "payload_scale": float(PAYLOAD_SCALE),
        "instruction_commit": INSTRUCTION_COMMIT,
        "d1_gate_unchanged": "T30_URGENCY_SIGNAL_SUPPORTED",
        "d0_gate_unchanged": "HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE",
        "holdout_scenario_retained": True,
        "coverage": {
            "states": int(len(records)),
            "candidates_per_state": 128,
            "full_pool_per_state": 13327,
            "missing_required_fields": 0,
            "sampled_frontier_is_not_the_full_pool": True,
        },
        "d1_reconciliation": reconciliation,
        "payload_constrained_steps": constrained_steps,
        "rules": {
            "payload_first": "maximum payload; urgent service only breaks ties inside 1 bit",
            "payload_constrained_urgency": "maximize urgent service inside an external payload-loss budget; no budget is selected",
            "weighted_sum": "report nonnegative critical intervals; no lambda_H is selected",
        },
        "by_time": by_time,
        "leave_one_state_out": loo,
        "states": [{key: value for key, value in item.items()} for item in records],
        "input_sha256": {
            "d0_labels": _sha256(D0_LABELS),
            "d0_selected": _sha256(D0_SELECTED),
            "d1_labels": _sha256(D1_LABELS),
            "d1_selected": _sha256(D1_SELECTED),
        },
        "pareto_rows": pareto_rows,
        "lambda_rows": lambda_rows,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv(row.get(key)) for key in fields})


def _csv(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.16g}"
    return value


def write_results(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pareto_fields = [
        "source",
        "scenario_id",
        "time_step",
        "candidate_id",
        "assignment",
        "payload_bits",
        "payload_term",
        "urgent_service_bits",
        "urgent_service_term",
        "is_payload_first",
        "is_urgent_first",
        "payload_regret_bits",
        "material_payload_regret_bits",
        "urgent_gap_vs_best_bits",
    ]
    lambda_fields = [
        "source",
        "scenario_id",
        "time_step",
        "segment_index",
        "lambda_low",
        "lambda_high",
        "lambda_high_unbounded",
        "candidate_id",
        "assignment",
        "payload_bits",
        "urgent_service_bits",
        "payload_term",
        "urgent_service_term",
        "payload_regret_bits",
        "urgent_gain_bits_vs_payload_first",
        "material_payload_loss",
    ]
    _write_csv(OUTPUT / "pareto_rows.csv", summary["pareto_rows"], pareto_fields)
    _write_csv(OUTPUT / "lambda_sensitivity.csv", summary["lambda_rows"], lambda_fields)
    public = {key: value for key, value in summary.items() if key not in {"pareto_rows", "lambda_rows"}}
    (OUTPUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "inputs": {
            "d0": "results/phase6i_urgency_highlevel_pilot labels for scenarios 322000-322003 at t=0,15,30",
            "d1": "results/phase6i_urgency_highlevel_t30_pilot labels for scenarios 322004-322015 at t=30",
            "sha256": summary["input_sha256"],
        },
        "candidate_scope": "128 sampled candidates per state, not the full 13327 pool",
        "eps_bits": float(EPS_BITS),
        "normalized_eps": float(EPS_BITS) / float(PAYLOAD_SCALE),
        "formulas": {
            "payload_term": "payload_bits / 1e6",
            "urgent_service_term": "urgent_service_bits / 1e6",
            "dominates": "payload and urgent are no worse than the other, and at least one is better by more than 1 bit",
            "lambda_cross": "(payload_i - payload_j) / (urgent_j - urgent_i) in the shared normalized units",
            "material_payload_regret_bits": "0 when the absolute payload gap is at most 1 bit, otherwise payload - best_payload",
            "utility": "payload_term + lambda_H * urgent_service_term",
        },
        "rules": {
            "payload_first": "maximum payload, urgent service breaks ties within 1 bit",
            "payload_constrained_urgency": "maximize urgent service subject to a payload-loss budget; the budget is not chosen",
            "weighted_sum": "critical intervals of nonnegative lambda_H from the convex sampled front; no single weight is selected",
        },
        "lambda_H": LAMBDA_H,
        "lambda_H_decided": False,
        "delta_payload_max": DELTA_PAYLOAD_MAX,
        "delta_payload_max_status": "needs_research_lead_confirmation",
        "held_out_splits_read": False,
        "rollouts_rerun": False,
        "instruction_commit": INSTRUCTION_COMMIT,
        "fields": {"pareto_rows.csv": pareto_fields, "lambda_sensitivity.csv": lambda_fields},
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def write_figures(summary: dict) -> None:
    """Optional plots. The tables remain the source of the reported numbers."""
    if summary.get("gate") != GATE_READY:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from matplotlib import font_manager

    for family in ("Microsoft YaHei", "SimHei"):
        if any(item.name == family for item in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [family]
            plt.rcParams["axes.unicode_minus"] = False
            break
    labels = _load_rows(D0_LABELS) + _load_rows(D1_LABELS)
    points = defaultdict(list)
    for row in labels:
        points[(int(row["scenario_id"]), int(row["time_step"]))].append(
            (float(row["payload_term"]), float(row["urgent_service_term"]), int(row["candidate_id"]))
        )
    material_states = [item for item in summary["states"] if float(item["material_top1_payload_regret_bits"]) < 0.0]
    figure, axes = plt.subplots(1, len(material_states), figsize=(4.4 * len(material_states), 4.4), squeeze=False)
    pareto_ids = defaultdict(set)
    for row in summary["pareto_rows"]:
        pareto_ids[(int(row["scenario_id"]), int(row["time_step"]))].add(int(row["candidate_id"]))
    for axis, state in zip(axes[0], material_states):
        key = (int(state["scenario_id"]), int(state["time_step"]))
        best_payload = float(state["best_payload_bits"])
        payload_first_urgent = float(state["payload_first_urgent_bits"])
        regret = [(item[0] * 1.0e6) - best_payload for item in points[key]]
        gain = [(item[1] * 1.0e6) - payload_first_urgent for item in points[key]]
        axis.scatter(regret, gain, s=12, c="#b0b7c3", label="抽样候选")
        front = [item for item in points[key] if item[2] in pareto_ids[key]]
        front_regret = [(item[0] * 1.0e6) - best_payload for item in front]
        front_gain = [(item[1] * 1.0e6) - payload_first_urgent for item in front]
        axis.scatter(front_regret, front_gain, s=36, c="#0f6c4c", label="抽样 Pareto", zorder=3)
        chosen = next(item for item in points[key] if item[2] == int(state["payload_first_candidate_id"]))
        urgent_choice = next(item for item in points[key] if item[2] == int(state["urgent_first_candidate_id"]))
        axis.scatter([(chosen[0] * 1.0e6) - best_payload], [(chosen[1] * 1.0e6) - payload_first_urgent], s=70, marker="s", c="#1d4e89", label="Payload-first", zorder=4)
        axis.scatter([(urgent_choice[0] * 1.0e6) - best_payload], [(urgent_choice[1] * 1.0e6) - payload_first_urgent], s=80, marker="D", c="#b45309", label="Urgent-first", zorder=4)
        low = min(front_regret) - 0.12 * abs(min(front_regret))
        high = max(front_gain)
        axis.set_xlim(low, abs(low) * 0.08)
        axis.set_ylim(-0.2 * high, high * 1.35 + 1.0)
        axis.axvline(0.0, color="#6b7280", lw=0.6)
        axis.set_title(f"{key[0]}  t={key[1]}")
        axis.set_xlabel("payload regret (bit)")
        axis.set_ylabel("urgent gain vs payload-first (bit)")
    axes[0][0].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(OUTPUT / "pareto_material_states.png", dpi=140)
    plt.close(figure)

    breakpoints = [item for item in summary["states"] if item["first_material_lambda"] is not None]
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    for index, state in enumerate(breakpoints):
        axis.plot([0.0, float(state["first_material_lambda"])], [index, index], color="#1d4e89", lw=2)
        axis.scatter([float(state["first_material_lambda"])], [index], color="#b45309", zorder=3)
        axis.text(float(state["first_material_lambda"]) + 0.15, index + 0.08, f"{float(state['first_material_lambda']):.3f}", fontsize=8)
    axis.set_yticks(range(len(breakpoints)))
    axis.set_yticklabels([f"{int(item['scenario_id'])} t={int(item['time_step'])}" for item in breakpoints])
    axis.set_xlabel("lambda_H")
    axis.set_title("出现实质 payload 损失的临界值")
    axis.set_xlim(left=0.0)
    figure.tight_layout()
    figure.savefig(OUTPUT / "lambda_breakpoints.png", dpi=140)
    plt.close(figure)


def main() -> dict:
    summary = execute()
    write_results(summary)
    write_figures(summary)
    print(summary["gate"], summary["by_time"].get("d1_t30"), summary["leave_one_state_out"], flush=True)
    return summary


if __name__ == "__main__":
    main()
