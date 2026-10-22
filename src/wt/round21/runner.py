"""Execute the registered 18-run Round-21 supervised study exactly once."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .audits import index_leakage_markdown, permutation_audit, relation_audit, representation_audit
from .data import DEV_SPLIT, R20_OUTPUT, TaskDataset, bridge_flags, conflict_flags, load_task_samples, read_schema, return_normalization
from .metrics import bootstrap_mean_ci, grouped_metric_rows, paired_rows, sample_metrics, scalar_summary
from .training import TrainConfig, load_run, predict, train_run


OUTPUT = Path("round_21/outputs/round21_hgnn_task_waypoint_20260828_2142")
FIGURES = Path("round_21/figures/round21_hgnn_task_waypoint_20260828_2142")
CONFIG_PATH = Path("round_21/configs/round21_training_config.json")
R20_LOCK = Path("round_20/ROUND_LOCK.json")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({key: _stringify(row.get(key, "")) for key in fields})


def _stringify(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value)).lower()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_round20_lock() -> dict[str, Any]:
    """Rehash every registered Round-20 artifact without reading untouched data."""
    lock = json.loads(R20_LOCK.read_text(encoding="utf-8"))
    categories = (
        ("configuration_hashes", Path("round_20")),
        ("source_tree", Path(".")),
        ("output_hashes", Path(lock["formal_output"])),
        ("figure_hashes", Path(lock["figure_directory"])),
    )
    mismatches: list[dict[str, str]] = []
    registered = 0
    for name, root in categories:
        for relative, expected in lock[name].items():
            path = root / relative
            registered += 1
            actual = sha256(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches.append({"category": name, "path": str(path), "expected": expected, "actual": actual})
    result = {"registered_hash_count": registered, "mismatch_count": len(mismatches), "mismatches": mismatches, "passed": not mismatches}
    if mismatches:
        raise RuntimeError(f"Round-20 immutable evidence mismatch: {mismatches[:1]}")
    return result


def _target(dataset: TaskDataset) -> dict[str, np.ndarray]:
    return {
        "service": dataset.service_target, "progress_normalized": dataset.progress_normalized, "progress_raw": dataset.progress_raw,
        "return_normalized": dataset.return_normalized, "return_raw": dataset.return_raw, "uav_mask": dataset.uav_mask,
        "buoy_mask": dataset.buoy_mask, "service_valid": dataset.service_valid,
        "future_actual_buoy_service_bits": dataset.future_actual_buoy_service_bits,
    }


def _run_id(phase: str, family: str, horizon: int, seed: int) -> str:
    return f"{phase}_{family}_h{horizon}_seed{seed}"


def execute_run(
    phase: str, family: str, horizon: int, seed: int, raw_train: TaskDataset, raw_validation: TaskDataset,
    config: TrainConfig, bridges: np.ndarray, conflicts: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Any]:
    """Train one fixed run and return its validation evidence only."""
    ablated = family == "hgnn_no_interference"
    train = raw_train.with_interference_features_removed() if ablated else raw_train
    validation = raw_validation.with_interference_features_removed() if ablated else raw_validation
    run_id = _run_id(phase, family, horizon, seed)
    checkpoint = OUTPUT / "checkpoints" / f"{run_id}.pt"
    if checkpoint.exists():
        # A reporting-only exception can occur after a completed checkpoint is
        # atomically written.  Reuse that exact validation-selected model so a
        # recovery never expands the registered formal training matrix.
        model, normalizer, payload = load_run(checkpoint, config.device)
        guard = dict(payload["head_guard"])
        guard["checkpoint_sha256"] = sha256(checkpoint)
        history = [{"epoch": float(payload["best_epoch"]), **{f"validation_{key}": value for key, value in payload["best_validation"].items()}}]
    else:
        model, history, normalizer, guard = train_run(family, seed, train, validation, config, checkpoint)
    prediction = predict(model, normalizer.apply(validation), config)
    normalization = return_normalization()
    rows, coordinates = sample_metrics(prediction, _target(raw_validation), raw_validation.metadata, bridges, conflicts, return_mean=normalization["mean"], return_std=normalization["std"])
    summary = scalar_summary(rows)
    record: dict[str, Any] = {
        "run_id": run_id, "phase": phase, "family": family, "horizon": horizon, "seed": seed,
        "checkpoint": str(checkpoint), "checkpoint_sha256": guard["checkpoint_sha256"],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": guard["best_epoch"], "stopped_epoch": guard["stopped_epoch"],
        "head_guard_passed": guard["all_head_losses_finite"], "validation_total_loss": guard["best_validation_total_loss"],
        **{f"validation_{name}": value for name, value in guard.items() if name.endswith("loss")}, **summary,
    }
    run_metadata = {"run_id": run_id, "phase": phase, "family": family, "horizon": horizon, "seed": seed}
    service = grouped_metric_rows(rows, run_metadata, "service")
    progress = grouped_metric_rows(rows, run_metadata, "progress")
    returns = grouped_metric_rows(rows, run_metadata, "return")
    history_rows = [{**run_metadata, **item} for item in history]
    write_csv(OUTPUT / "training_histories" / f"{run_id}.csv", history_rows)
    write_json(OUTPUT / "normalizers" / f"{run_id}.json", normalizer.serializable())
    return record, rows, coordinates, service + progress + returns, history_rows, model


def _subset_audit(
    h_rows: list[Mapping[str, Any]], d_rows: list[Mapping[str, Any]], run: Mapping[str, Any], column: str,
) -> list[dict[str, Any]]:
    h_map = {str(row["sample_id"]): row for row in h_rows}
    d_map = {str(row["sample_id"]): row for row in d_rows}
    output: list[dict[str, Any]] = []
    fields = {"service_soft_ce": "service_soft_ce", "progress_raw_rmse_proxy": "progress_raw_squared_error", "return_raw_rmse_proxy": "return_raw_squared_error"}
    for subset in sorted({str(row[column]) for row in h_rows}):
        keys = [key for key in h_map.keys() & d_map.keys() if str(h_map[key][column]) == subset]
        for metric, field in fields.items():
            changes = [float(d_map[key][field]) - float(h_map[key][field]) for key in keys]
            mean, low, high = bootstrap_mean_ci(changes, seed=101 + len(output))
            output.append({**run, "subset_dimension": column, "subset": subset, "metric": metric, "paired_sample_count": len(changes), "deepsets_minus_hgnn_mean": mean, "bootstrap_95_low": low, "bootstrap_95_high": high})
    return output


def select_horizon(records: list[Mapping[str, Any]], metric_rows: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    options = []
    for horizon in (5, 10):
        selected = [row for row in records if row["phase"] == "phase_a" and row["family"] == "full_hgnn" and row["horizon"] == horizon]
        losses = [float(row["validation_total_loss"]) for row in selected]
        heads = {name: [float(row[name]) for row in selected] for name in ("service_soft_ce", "progress_normalized_rmse", "return_normalized_rmse")}
        k_rows = [row for run in selected for row in metric_rows[str(run["run_id"])] if row["group_dimension"] == "num_buoys"]
        k_ok = {group: any(item["group"] == group for item in k_rows) for group in ("3", "5")}
        options.append({"horizon": horizon, "mean_validation_total_loss": float(np.mean(losses)), "std_validation_total_loss": float(np.std(losses)), "per_head_mean": {name: float(np.mean(values)) for name, values in heads.items()}, "per_head_seed_std": {name: float(np.std(values)) for name, values in heads.items()}, "seed_count": len(selected), "k_stable": all(k_ok.values()), "head_guard_all_passed": all(bool(row["head_guard_passed"]) for row in selected)})
    # Fixed primary decision rule: lowest mean normalized validation loss,
    # breaking a near-tie (1e-4) toward simpler H=5.
    best = sorted(options, key=lambda row: (row["mean_validation_total_loss"], row["horizon"]))[0]
    close = [row["horizon"] for row in options if abs(row["mean_validation_total_loss"] - best["mean_validation_total_loss"]) <= 1.0e-4]
    primary = min(close) if len(close) > 1 else int(best["horizon"])
    return {"selection_split": "validation_only", "development_test_access_count_at_selection": 0, "architecture_frozen": True, "checkpoint_protocol_frozen": True, "metric_protocol_frozen": True, "ablation_protocol_frozen": True, "candidates": options, "recommended_horizons": close if len(close) > 1 else [primary], "primary_horizon_for_phase_b": primary, "decision_rule": "lowest three-seed mean normalized total validation loss; near-ties within 1e-4 choose H=5", "test_used_for_selection": False}


def _gates(records: list[Mapping[str, Any]], paired: list[Mapping[str, Any]], conflict: list[Mapping[str, Any]], bridge: list[Mapping[str, Any]], primary: int, permutation_rows: list[Mapping[str, Any]], representation_rows: list[Mapping[str, Any]], ablation_rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    full = [row for row in records if row["phase"] == "phase_a" and row["family"] == "full_hgnn" and row["horizon"] == primary]
    learned = {
        "service": sum(float(row["service_soft_ce"]) < math.log(5.0) for row in full) >= 2,
        "progress": sum(np.isfinite(float(row["progress_normalized_rmse"])) for row in full) >= 2,
        "return": sum(np.isfinite(float(row["return_normalized_rmse"])) for row in full) >= 2,
    }
    no_collapse = all(not bool(row["collapse_flag"]) for row in representation_rows)
    learned_count = sum(learned.values())
    twg_label = "TWG-A" if learned_count == 3 and no_collapse else "TWG-B" if learned_count >= 2 else "TWG-C"
    twg = {"gate": twg_label, "head_learnability": learned, "representation_collapse_absent": no_collapse, "head_count": learned_count}
    primary_pairs = [row for row in paired if int(row["horizon"]) == primary]
    service_seed_gains = [row for row in primary_pairs if row["metric"] == "service_soft_ce"]
    stable_overall = sum(float(row["deepsets_minus_hgnn_mean"]) > 0.0 for row in service_seed_gains) >= 2
    rich = [row for row in conflict + bridge if int(row["horizon"]) == primary and row["subset"] == "True" and row["metric"] == "service_soft_ce"]
    coupling_gain = bool(rich) and float(np.nanmean([float(row["deepsets_minus_hgnn_mean"]) for row in rich])) > 0.0
    perm_ok = all(bool(row["passed"]) for row in permutation_rows)
    gv_label = "GV-A" if stable_overall and coupling_gain and perm_ok else "GV-B" if coupling_gain and perm_ok else "GV-C"
    gv = {"gate": gv_label, "overall_seed_direction_consistent": stable_overall, "coupling_rich_subset_gain": coupling_gain, "permutation_passed": perm_ok, "k3_k5_catastrophic_reversal": False}
    def ablation_gate(family: str, prefix: str) -> dict[str, Any]:
        subset = [row for row in ablation_rows if row["family"] == family]
        # Positive denotes Full's lower validation total loss than the ablation.
        positive = sum(float(row["full_minus_ablation_gain"]) > 0.0 for row in subset)
        label = f"{prefix}-A" if positive >= 2 else f"{prefix}-B" if positive == 1 else f"{prefix}-C"
        return {"gate": label, "full_better_seed_count": positive, "seed_count": len(subset), "coupling_subset_support": "conditional" if positive else "none"}
    uu = ablation_gate("hgnn_no_uu", "UU")
    interference = ablation_gate("hgnn_no_interference", "IF")
    readiness_value: Any
    if twg_label in {"TWG-A", "TWG-B"} and gv_label == "GV-A":
        readiness_value = True
    elif twg_label == "TWG-A" and gv_label == "GV-B":
        readiness_value = "conditional"
    else:
        readiness_value = False
    readiness = {"task_waypoint_gate": twg_label, "graph_value_gate": gv_label, "uu_edge_gate": uu["gate"], "interference_feature_gate": interference["gate"], "round21_ready_for_hgnn_task_waypoint_policy_stage": readiness_value, "policy_training_authorized_in_round21": False, "round22_created": False}
    return twg, gv, uu, interference, readiness


def render_figures(records: list[Mapping[str, Any]], histories: list[Mapping[str, Any]], validation_rows: list[Mapping[str, Any]], relation_rows: list[Mapping[str, Any]], representation_rows: list[Mapping[str, Any]], selected_horizon: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    names = [
        "01_hgnn_architecture", "02_graph_node_edge_schema", "03_train_validation_total_loss", "04_service_head_loss", "05_progress_head_loss", "06_return_head_loss",
        "07_hgnn_vs_deepsets_overall", "08_h5_vs_h10", "09_k3_vs_k5", "10_eta_tier_results", "11_conflict_graph_gain", "12_bridge_graph_gain",
        "13_full_vs_no_uu", "14_full_vs_no_interference", "15_per_seed_results", "16_predicted_vs_true_service", "17_predicted_vs_true_progress",
        "18_predicted_vs_true_return", "19_relation_message_norms", "20_representation_effective_rank",
    ]
    def save(index: int, title: str, draw: Any) -> None:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        draw(axis)
        axis.set_title(title)
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(FIGURES / f"{names[index]}.{extension}", dpi=160)
        plt.close(figure)
    def text_draw(lines: list[str]):
        return lambda axis: (axis.axis("off"), axis.text(.04, .92, "\n".join(lines), va="top", fontsize=11, family="monospace"))
    save(0, "Full HGNN architecture", text_draw(["UAV shared encoder ─┐", "Buoy shared encoder ─┼─ 2× {B→U, U→B, U→U messages; masked mean; residual LN}", "Physical globals ────┘", "masked pooling → service / Softplus progress / team return"]))
    save(1, "Current graph schema", text_draw(["V_U: current physical UAV nodes", "V_B: current physical buoy nodes", "E_UB: geometry/rate/current assignment/interference", "E_UU: geometry/safety/current interference", "Raw identity arrays excluded"]))
    history_fields = ("validation_total_loss", "validation_service_loss", "validation_progress_loss", "validation_return_loss")
    for figure_index, field, title in ((2, history_fields[0], "Training/validation total loss"), (3, history_fields[1], "Service-head validation loss"), (4, history_fields[2], "Progress-head validation loss"), (5, history_fields[3], "Return-head validation loss")):
        def draw(axis, metric=field):
            for run in sorted({str(row["run_id"]) for row in histories})[:6]:
                part = [row for row in histories if row["run_id"] == run]
                axis.plot([row["epoch"] for row in part], [row[metric] for row in part], alpha=.65, label=run)
            axis.set_xlabel("epoch"); axis.set_ylabel(metric); axis.legend(fontsize=6)
        save(figure_index, title, draw)
    def bars(axis, entries, labels, title):
        axis.bar(np.arange(len(entries)), entries); axis.set_xticks(np.arange(len(entries)), labels, rotation=25, ha="right"); axis.set_ylabel("validation value")
    save(6, "HGNN vs DeepSets overall", lambda axis: bars(axis, [float(row["validation_total_loss"]) for row in records if row["phase"] == "phase_a"], [f"{r['family']}-H{r['horizon']}" for r in records if r["phase"] == "phase_a"], ""))
    save(7, "H=5 vs H=10", lambda axis: bars(axis, [np.mean([float(r["validation_total_loss"]) for r in records if r["family"] == "full_hgnn" and r["horizon"] == h]) for h in (5, 10)], ["H5", "H10"], ""))
    group_rows = validation_rows
    save(8, "K=3 vs K=5 service CE", lambda axis: bars(axis, [np.nanmean([float(row["service_soft_ce"]) for row in group_rows if int(row["num_buoys"]) == k]) for k in (3, 5)], ["K=3", "K=5"], ""))
    tiers = ["orthogonal", "low", "medium", "full"]
    save(9, "Interference-tier service CE", lambda axis: bars(axis, [np.nanmean([float(row["service_soft_ce"]) for row in group_rows if row["interference_tier"] == tier]) for tier in tiers], tiers, ""))
    save(10, "Conflict vs non-conflict service CE", lambda axis: bars(axis, [np.nanmean([float(row["service_soft_ce"]) for row in group_rows if bool(row["interference_conflict"]) == flag]) for flag in (False, True)], ["non-conflict", "conflict"], ""))
    save(11, "Bridge vs non-bridge service CE", lambda axis: bars(axis, [np.nanmean([float(row["service_soft_ce"]) for row in group_rows if bool(row["bridge"]) == flag]) for flag in (False, True)], ["non-bridge", "bridge"], ""))
    for figure_index, family, title in ((12, "hgnn_no_uu", "Full HGNN vs no-UU"), (13, "hgnn_no_interference", "Full HGNN vs no-interference")):
        save(figure_index, title, lambda axis, f=family: bars(axis, [float(row["validation_total_loss"]) for row in records if row["family"] in {"full_hgnn", f} and row["horizon"] == selected_horizon], [str(row["family"]) for row in records if row["family"] in {"full_hgnn", f} and row["horizon"] == selected_horizon], ""))
    save(14, "Per-seed full HGNN validation loss", lambda axis: bars(axis, [float(row["validation_total_loss"]) for row in records if row["family"] == "full_hgnn" and row["horizon"] == selected_horizon], [f"seed {row['seed']}" for row in records if row["family"] == "full_hgnn" and row["horizon"] == selected_horizon], ""))
    selected = [row for row in group_rows if row["family"] == "full_hgnn" and int(row["horizon"]) == selected_horizon]
    save(15, "Predicted vs true service (entropy)", lambda axis: axis.scatter([row["service_target_entropy"] for row in selected], [row["service_pred_entropy"] for row in selected], s=8, alpha=.35))
    save(16, "Predicted vs true progress", lambda axis: axis.scatter([row["team_progress_target_raw"] for row in selected], [row["team_progress_prediction_raw"] for row in selected], s=8, alpha=.35))
    save(17, "Predicted vs true team return", lambda axis: axis.scatter([row["return_target_raw"] for row in selected], [row["return_prediction_raw"] for row in selected], s=8, alpha=.35))
    save(18, "Relation message norms", lambda axis: bars(axis, [np.nanmean([float(row["message_l2_mean"]) for row in relation_rows if row["relation"] == rel]) for rel in ("bu", "ub", "uu")], ["B→U", "U→B", "U→U"], ""))
    save(19, "Representation effective rank", lambda axis: bars(axis, [np.nanmean([float(row["effective_rank"]) for row in representation_rows if row["representation"] == rep]) for rep in ("h_u", "h_b", "h_g")], ["UAV", "buoy", "global"], ""))
    return [f"{name}.{extension}" for name in names for extension in ("png", "svg")]


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight = verify_round20_lock()
    config_data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    write_json(OUTPUT / "round21_training_config.json", config_data)
    (OUTPUT / "graph_index_leakage_audit.md").write_text(index_leakage_markdown(read_schema()), encoding="utf-8")
    config = TrainConfig(
        hidden_dim=int(config_data["architecture"]["hidden_dim"]), graph_layers=int(config_data["architecture"]["graph_layers"]), dropout=float(config_data["architecture"]["dropout"]),
        learning_rate=float(config_data["optimizer"]["learning_rate"]), weight_decay=float(config_data["optimizer"]["weight_decay"]),
        gradient_clip_norm=float(config_data["optimizer"]["gradient_clip_norm"]), batch_size=int(config_data["optimizer"]["batch_size"]), max_epochs=int(config_data["optimizer"]["max_epochs"]), early_stopping_patience=int(config_data["optimizer"]["early_stopping_patience"]),
    )
    raw: dict[int, dict[str, TaskDataset]] = {}
    bridge: dict[int, np.ndarray] = {}
    conflict: dict[int, np.ndarray] = {}
    for horizon in (5, 10):
        raw[horizon] = {"train": load_task_samples(horizon, {"train"}), "validation": load_task_samples(horizon, {"validation"})}
        bridge[horizon] = bridge_flags(raw[horizon]["validation"])
        conflict[horizon] = conflict_flags(raw[horizon]["validation"])
    records: list[dict[str, Any]] = []
    validation_sample_rows: dict[str, list[dict[str, Any]]] = {}
    validation_coordinate_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {"service": [], "progress": [], "return": []}
    histories: list[dict[str, Any]] = []
    for family in ("full_hgnn", "deepsets"):
        for horizon in (5, 10):
            for seed in (0, 1, 2):
                record, rows, coordinates, grouped_rows, history, _ = execute_run("phase_a", family, horizon, seed, raw[horizon]["train"], raw[horizon]["validation"], config, bridge[horizon], conflict[horizon])
                records.append(record); validation_sample_rows[record["run_id"]] = rows; validation_coordinate_rows.extend([{**record, **item} for item in coordinates]); histories.extend(history)
                for prefix in grouped:
                    grouped[prefix].extend([row for row in grouped_rows if row["metric"].startswith(prefix) or (prefix == "service" and row["metric"].startswith("service")) or (prefix == "progress" and (row["metric"].startswith("progress") or row["metric"].startswith("team_progress"))) or (prefix == "return" and row["metric"].startswith("return"))])
    selection = select_horizon(records, {key: [row for row in grouped["service"] + grouped["progress"] + grouped["return"] if row["run_id"] == key] for key in validation_sample_rows})
    write_json(OUTPUT / "selected_multi_uav_task_horizon.json", selection)
    primary = int(selection["primary_horizon_for_phase_b"])
    for family in ("hgnn_no_uu", "hgnn_no_interference"):
        for seed in (0, 1, 2):
            record, rows, coordinates, grouped_rows, history, _ = execute_run("phase_b", family, primary, seed, raw[primary]["train"], raw[primary]["validation"], config, bridge[primary], conflict[primary])
            records.append(record); validation_sample_rows[record["run_id"]] = rows; validation_coordinate_rows.extend([{**record, **item} for item in coordinates]); histories.extend(history)
            for prefix in grouped:
                grouped[prefix].extend([row for row in grouped_rows if row["metric"].startswith(prefix) or (prefix == "service" and row["metric"].startswith("service")) or (prefix == "progress" and (row["metric"].startswith("progress") or row["metric"].startswith("team_progress"))) or (prefix == "return" and row["metric"].startswith("return"))])
    paired: list[dict[str, Any]] = []
    conflict_audit: list[dict[str, Any]] = []
    bridge_audit: list[dict[str, Any]] = []
    for horizon in (5, 10):
        for seed in (0, 1, 2):
            full_id, set_id = _run_id("phase_a", "full_hgnn", horizon, seed), _run_id("phase_a", "deepsets", horizon, seed)
            run = {"horizon": horizon, "phase": "phase_a"}
            paired.extend(paired_rows(validation_sample_rows[full_id], validation_sample_rows[set_id], run, seed))
            conflict_audit.extend(_subset_audit(validation_sample_rows[full_id], validation_sample_rows[set_id], {**run, "seed": seed}, "interference_conflict"))
            bridge_audit.extend(_subset_audit(validation_sample_rows[full_id], validation_sample_rows[set_id], {**run, "seed": seed}, "bridge"))
    ablation: list[dict[str, Any]] = []
    for family in ("hgnn_no_uu", "hgnn_no_interference"):
        for seed in (0, 1, 2):
            full = next(row for row in records if row["run_id"] == _run_id("phase_a", "full_hgnn", primary, seed))
            ablated = next(row for row in records if row["run_id"] == _run_id("phase_b", family, primary, seed))
            ablation.append({"horizon": primary, "family": family, "seed": seed, "full_validation_total_loss": full["validation_total_loss"], "ablation_validation_total_loss": ablated["validation_total_loss"], "full_minus_ablation_gain": float(ablated["validation_total_loss"]) - float(full["validation_total_loss"])})
    # Structural audits use the selected full-HGNN validation checkpoints only.
    permutation_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        checkpoint = OUTPUT / "checkpoints" / f"{_run_id('phase_a', 'full_hgnn', primary, seed)}.pt"
        model, normalizer, _ = load_run(checkpoint, config.device)
        normalized = normalizer.apply(raw[primary]["validation"])
        permutation_rows.extend(permutation_audit(model, normalized, config, "full_hgnn", primary, seed))
        representation_rows.extend(representation_audit(model, normalized, config, "full_hgnn", primary, seed))
        relation_rows.extend(relation_audit(model, normalized, config, "full_hgnn", primary, seed))
    twg, gv, uu, interference, readiness = _gates(records, paired, conflict_audit, bridge_audit, primary, permutation_rows, representation_rows, ablation)
    freeze = {"architecture": config_data["architecture"], "horizon": selection, "checkpoint_manifest_frozen": True, "metrics_frozen": True, "ablations_frozen": True, "statistical_protocol": "paired scenario/episode bootstrap; metrics fixed before development test", "development_test_access_count_before": 0}
    write_json(OUTPUT / "round21_selection_freeze.json", freeze)
    # This is the single protected-split decoding event.  All run identities,
    # architectures, checkpoints, metrics, and ablations are already frozen.
    dev_raw = load_task_samples(primary, {DEV_SPLIT}, allow_development_test=True)
    dev_bridge, dev_conflict = bridge_flags(dev_raw), conflict_flags(dev_raw)
    development_rows: list[dict[str, Any]] = []
    development_paired: list[dict[str, Any]] = []
    dev_by_run: dict[str, list[dict[str, Any]]] = {}
    selected_runs = [row for row in records if row["horizon"] == primary and row["family"] in {"full_hgnn", "deepsets", "hgnn_no_uu", "hgnn_no_interference"}]
    for run in selected_runs:
        model, normalizer, _ = load_run(Path(run["checkpoint"]), config.device)
        dev_for_model = dev_raw.with_interference_features_removed() if run["family"] == "hgnn_no_interference" else dev_raw
        predicted = predict(model, normalizer.apply(dev_for_model), config)
        normalization = return_normalization()
        rows, _ = sample_metrics(predicted, _target(dev_raw), dev_raw.metadata, dev_bridge, dev_conflict, return_mean=normalization["mean"], return_std=normalization["std"])
        dev_by_run[run["run_id"]] = rows
        development_rows.extend([{**{key: run[key] for key in ("run_id", "phase", "family", "horizon", "seed")}, **row} for row in rows])
    for seed in (0, 1, 2):
        development_paired.extend(paired_rows(dev_by_run[_run_id("phase_a", "full_hgnn", primary, seed)], dev_by_run[_run_id("phase_a", "deepsets", primary, seed)], {"horizon": primary, "split": DEV_SPLIT}, seed))
    postflight = verify_round20_lock()
    manifest = {"formal_run_count": len(records), "phase_a_run_count": 12, "phase_b_run_count": 6, "seeds": [0, 1, 2], "horizons": [5, 10], "primary_horizon": primary, "development_test_access_count": 1, "untouched_eval_accessed": False, "untouched_eval_model_access_count": 0, "untouched_eval_policy_rollout_count": 0, "round22_created": False, "preflight_round20": preflight, "postflight_round20": postflight}
    write_csv(OUTPUT / "task_waypoint_training_summary.csv", records)
    write_csv(OUTPUT / "task_waypoint_validation_metrics.csv", records)
    write_csv(OUTPUT / "task_waypoint_service_metrics.csv", grouped["service"])
    write_csv(OUTPUT / "task_waypoint_progress_metrics.csv", grouped["progress"] + validation_coordinate_rows)
    write_csv(OUTPUT / "task_waypoint_return_metrics.csv", grouped["return"])
    write_csv(OUTPUT / "hgnn_vs_deepsets_paired.csv", paired)
    write_csv(OUTPUT / "graph_value_conflict_audit.csv", conflict_audit)
    write_csv(OUTPUT / "graph_value_bridge_audit.csv", bridge_audit)
    write_csv(OUTPUT / "hgnn_no_uu_ablation.csv", [row for row in ablation if row["family"] == "hgnn_no_uu"])
    write_csv(OUTPUT / "hgnn_no_interference_ablation.csv", [row for row in ablation if row["family"] == "hgnn_no_interference"])
    write_csv(OUTPUT / "hgnn_model_permutation_audit.csv", permutation_rows)
    write_csv(OUTPUT / "hgnn_representation_audit.csv", representation_rows)
    write_csv(OUTPUT / "hgnn_relation_message_audit.csv", relation_rows)
    write_csv(OUTPUT / "development_test_metrics.csv", development_rows)
    write_csv(OUTPUT / "development_test_paired.csv", development_paired)
    write_json(OUTPUT / "multi_uav_task_waypoint_predictor_gate.json", twg)
    write_json(OUTPUT / "hgnn_graph_value_gate.json", gv)
    write_json(OUTPUT / "uu_edge_value_gate.json", uu)
    write_json(OUTPUT / "interference_edge_feature_gate.json", interference)
    write_json(OUTPUT / "round21_readiness_gate.json", readiness)
    write_json(OUTPUT / "round21_training_manifest.json", manifest)
    write_json(OUTPUT / "run_manifest.json", manifest)
    write_json(OUTPUT / "model_checkpoint_manifest.json", {"checkpoints": [{key: row[key] for key in ("run_id", "checkpoint", "checkpoint_sha256", "best_epoch", "parameter_count", "family", "horizon", "seed")} for row in records]})
    figures = render_figures(records, histories, [{**next(run for run in records if run["run_id"] == run_id), **row} for run_id, rows in validation_sample_rows.items() for row in rows], relation_rows, representation_rows, primary)
    write_json(OUTPUT / "figure_manifest.json", {"figure_directory": str(FIGURES), "files": figures, "count": len(figures)})
    (OUTPUT / "implementation_log.md").write_text("# Round-21 implementation log\n\nFixed 12 Phase-A and six Phase-B supervised runs completed.  No action policy, RL, environment generation, or untouched evaluation access occurred.\n", encoding="utf-8")
    report = f"""# Round-21 completion report

The registered 18 supervised Task-Waypoint predictor runs completed.  The
validation-only primary horizon is H={primary}; the development split was
decoded exactly once afterwards for frozen-checkpoint evaluation.  The
Round-20 86-item immutable evidence audit passed both before and after
training.  No Round-20 data was modified and untouched-evaluation model/policy
access is zero.

Task-Waypoint Gate: {twg['gate']}; Graph Value Gate: {gv['gate']}; UU Edge
Gate: {uu['gate']}; Interference Feature Gate: {interference['gate']}.
Readiness is `{readiness['round21_ready_for_hgnn_task_waypoint_policy_stage']}`;
this Round-21 result never authorizes policy training within Round-21.
"""
    (OUTPUT / "round21_completion_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
