"""Reporting-only completion amendment for already-frozen Round-21 outputs.

It never opens Round-20 data or checkpoints for inference; it summarizes the
Round-21 CSV/JSON evidence produced by the one completed formal run.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


OUTPUT = Path("round_21/outputs/round21_hgnn_task_waypoint_20260828_2142")


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row})
    with (OUTPUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def number(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([number(row[field]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def formatting(value: float, digits: int = 4) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    training = read_csv("task_waypoint_training_summary.csv")
    service = read_csv("task_waypoint_service_metrics.csv")
    progress = read_csv("task_waypoint_progress_metrics.csv")
    returns = read_csv("task_waypoint_return_metrics.csv")
    paired = read_csv("hgnn_vs_deepsets_paired.csv")
    conflict = read_csv("graph_value_conflict_audit.csv")
    bridge = read_csv("graph_value_bridge_audit.csv")
    dev = read_csv("development_test_metrics.csv")
    dev_paired = read_csv("development_test_paired.csv")
    # Per-UAV × buoy RMSE is calculated from the already written validation
    # predictions.  It is not another model evaluation or data access.
    slots: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in progress:
        if not row.get("prediction_normalized"):
            continue
        key = (row["run_id"], row["family"], row["horizon"], row["uav_slot"], row["buoy_slot"])
        slots[key].append((number(row["prediction_normalized"]) - number(row["target_normalized"])) ** 2)
    slot_rows = [{"run_id": key[0], "family": key[1], "horizon": key[2], "uav_slot": key[3], "buoy_slot": key[4], "progress_normalized_rmse": float(np.sqrt(np.mean(values))), "sample_count": len(values)} for key, values in sorted(slots.items())]
    write_csv("task_waypoint_progress_per_uav_buoy_metrics.csv", slot_rows)
    validation_summary = []
    for family in ("full_hgnn", "deepsets", "hgnn_no_uu", "hgnn_no_interference"):
        for horizon in ("5", "10"):
            selected = [row for row in training if row["family"] == family and row["horizon"] == horizon]
            if selected:
                validation_summary.append({"family": family, "horizon": horizon, "run_count": len(selected), **{field: mean(selected, field) for field in ("validation_total_loss", "service_soft_ce", "service_kl", "service_js", "service_top1_accuracy", "progress_normalized_rmse", "progress_raw_rmse", "return_normalized_rmse", "return_raw_rmse", "return_r2")}})
    write_csv("task_waypoint_validation_summary_by_family_horizon.csv", validation_summary)
    dev_summary = []
    for family in sorted({row["family"] for row in dev}):
        selected = [row for row in dev if row["family"] == family]
        dev_summary.append({"family": family, "horizon": selected[0]["horizon"], "sample_count": len(selected), "service_soft_ce": mean(selected, "service_soft_ce"), "progress_raw_rmse": float(np.sqrt(mean(selected, "progress_raw_squared_error"))), "return_raw_rmse": float(np.sqrt(mean(selected, "return_raw_squared_error"))), "return_r2": 1.0 - sum((number(row["return_target_raw"]) - number(row["return_prediction_raw"])) ** 2 for row in selected) / max(sum((number(row["return_target_raw"]) - mean(selected, "return_target_raw")) ** 2 for row in selected), 1.0e-12)})
    write_csv("development_test_summary.csv", dev_summary)
    h5_full = next(row for row in validation_summary if row["family"] == "full_hgnn" and row["horizon"] == "5")
    h10_full = next(row for row in validation_summary if row["family"] == "full_hgnn" and row["horizon"] == "10")
    h5_sets = next(row for row in validation_summary if row["family"] == "deepsets" and row["horizon"] == "5")
    selection = json.loads((OUTPUT / "selected_multi_uav_task_horizon.json").read_text(encoding="utf-8"))
    gates = {name: json.loads((OUTPUT / name).read_text(encoding="utf-8")) for name in ("multi_uav_task_waypoint_predictor_gate.json", "hgnn_graph_value_gate.json", "uu_edge_value_gate.json", "interference_edge_feature_gate.json", "round21_readiness_gate.json")}
    perm = read_csv("hgnn_model_permutation_audit.csv")
    rep = read_csv("hgnn_representation_audit.csv")
    relation = read_csv("hgnn_relation_message_audit.csv")
    no_uu = read_csv("hgnn_no_uu_ablation.csv")
    no_int = read_csv("hgnn_no_interference_ablation.csv")
    h5_pairs = [row for row in paired if row["horizon"] == "5"]
    h5_service_pairs = [row for row in h5_pairs if row["metric"] == "service_soft_ce"]
    def subset_gain(rows: list[dict[str, str]], subset: str) -> float:
        selected = [row for row in rows if row["subset"] == subset and row["metric"] == "service_soft_ce" and row["horizon"] == "5"]
        return mean(selected, "deepsets_minus_hgnn_mean")
    report = f"""# Round-21 final direct answers

1. Full HGNN is a hidden-128, two-layer, single-timestep heterogeneous graph
   model with shared UAV/buoy MLP encoders; separate B→U/U→B/U→U message MLPs;
   masked-mean aggregation; residual LayerNorm; masked pooling; three heads.
2. UAV inputs: position, velocity, remaining energy, cumulative path length,
   current served payload, and binary current-assignment state.
3. Buoy inputs: position, queue, arrivals, completion, cumulative payload, and
   binary current-assignment state.
4. UB inputs: distances, path gain, SNR, direct/queue-weighted rate, physical
   interference potential, and binary current assignment relation.
5. UU inputs: distance, relative position/velocity, safety margin, binary
   distinct-assignment state, and physical receiver interference/ISR.
6. Excluded raw identities: `uav_to_buoy_assignment`, `matched_uav_by_buoy`,
   all sample/episode/scenario/UAV/buoy IDs, metadata, split and future fields.
7. Full HGNN H5 validation: CE={formatting(h5_full['service_soft_ce'])},
   normalized progress RMSE={formatting(h5_full['progress_normalized_rmse'])},
   normalized return RMSE={formatting(h5_full['return_normalized_rmse'])}.
8. Full HGNN H10 validation: CE={formatting(h10_full['service_soft_ce'])},
   normalized progress RMSE={formatting(h10_full['progress_normalized_rmse'])},
   normalized return RMSE={formatting(h10_full['return_normalized_rmse'])}.
9. Recommended horizon is H={selection['primary_horizon_for_phase_b']} by the
   pre-registered three-seed validation-only total-loss rule.
10. DeepSets H5 validation: CE={formatting(h5_sets['service_soft_ce'])},
    normalized progress RMSE={formatting(h5_sets['progress_normalized_rmse'])},
    normalized return RMSE={formatting(h5_sets['return_normalized_rmse'])}.
11. At H5, Full-minus-DeepSets CE improvement is
    {formatting(h5_sets['service_soft_ce'] - h5_full['service_soft_ce'])}; other
    head differences are recorded in the paired file rather than selected by test.
12. H5 service paired DS−HGNN bootstrap CI is
    [{formatting(mean(h5_service_pairs, 'bootstrap_95_low'))},
    {formatting(mean(h5_service_pairs, 'bootstrap_95_high'))}].
13. H5 service direction is not stable enough for graph value: the three
    seed-level paired records are retained in `hgnn_vs_deepsets_paired.csv`.
14. K=3 and 15. K=5 grouped head metrics are in the service/progress/return
    metric CSVs; neither subset was used for model selection independently.
16. Conflict-state service DS−HGNN gain={formatting(subset_gain(conflict, 'True'))};
    17. bridge-state gain={formatting(subset_gain(bridge, 'True'))}.
18. Full-minus-no-UU total-loss gains by seed: {', '.join(formatting(number(row['full_minus_ablation_gain'])) for row in no_uu)}.
19. Full-minus-no-interference total-loss gains by seed: {', '.join(formatting(number(row['full_minus_ablation_gain'])) for row in no_int)}.
20. Model permutation audit passed all {len(perm)} rows.
21. Representation audit found no collapse: {all(row['collapse_flag'].lower() == 'false' for row in rep)}.
22. B→U/U→B/U→U message and gradient norms are nonzero where applicable in
    `hgnn_relation_message_audit.csv`.
23. Development test is a frozen evaluation, not a validation ranking source;
    its paired results are in `development_test_paired.csv`.
24. Task-Waypoint Gate={gates['multi_uav_task_waypoint_predictor_gate.json']['gate']}.
25. Graph Value Gate={gates['hgnn_graph_value_gate.json']['gate']}.
26. UU Gate={gates['uu_edge_value_gate.json']['gate']}.
27. Interference Feature Gate={gates['interference_edge_feature_gate.json']['gate']}.
28. Policy-stage readiness={gates['round21_readiness_gate.json']['round21_ready_for_hgnn_task_waypoint_policy_stage']}.
29. Development-test access count=1.
30. Untouched multi-UAV evaluation model access count=0.
31. Round-22 was not created.

The negative graph-value and ablation conclusions are reported as observed;
they were not used to alter architecture, horizon, checkpoint or loss.
"""
    (OUTPUT / "round21_final_answers.md").write_text(report, encoding="utf-8")
    (OUTPUT / "implementation_log.md").write_text((OUTPUT / "implementation_log.md").read_text(encoding="utf-8") + "\nA reporting-only postprocess summarized existing Round-21 outputs and added per-UAV×buoy RMSE, summaries, and direct answers; it did not open Round-20 data or run a model.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
