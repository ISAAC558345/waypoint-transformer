"""Recompute design diagnostics from frozen Phase-2B checkpoints. No training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase2b.protocol import OLD_SEED1_CKPT, OLD_SEED2_CKPT, OUTPUT, PHASE1B_CKPT0, PHASE2B, ROOT, load_pools
from wt.lowlevel_v2.phase2b.run_phase2b import _csv, _dump, gate_decision, load_ckpt, plot_figures, write_report


def main() -> dict:
    _, design = load_pools()
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    l0w = ResidualWorker(high, actor, critic, service_lock=True, trust_region=False)
    tagged = {
        "L0": evaluate_design(l0w, design, zero_residual=True),
        "new0": evaluate_design(load_ckpt(OUTPUT / "ckpt_seed0_update100.pt", trust_region=True), design, zero_residual=False),
        "new1": evaluate_design(load_ckpt(OUTPUT / "ckpt_seed1_update100.pt", trust_region=True), design, zero_residual=False),
        "old0": evaluate_design(load_ckpt(PHASE1B_CKPT0, trust_region=False), design, zero_residual=False),
        "old1": evaluate_design(load_ckpt(OLD_SEED1_CKPT, trust_region=False), design, zero_residual=False),
        "old2": evaluate_design(load_ckpt(OLD_SEED2_CKPT, trust_region=False), design, zero_residual=False),
    }
    nav_rows, svc_rows, hl_rows, safe_rows, res_rows, feat_rows = [], [], [], [], [], []
    for name, rec in tagged.items():
        nav_rows.append({"tag": name, "mean_arrival_steps": rec["mean_arrival_steps"], "median_arrival_steps": rec["median_arrival_steps"], "p90_arrival_steps": rec["p90_arrival_steps"], "never_reach_count": rec["never_reach_count"], "never_reach_fraction": rec["never_reach_fraction"], "flight_distance": rec["flight_distance"], "time_in_range": rec["time_in_range"]})
        svc_rows.append({"tag": name, "time_in_range": rec["time_in_range"], "completed_blocks": rec["completed_blocks"], "unfinished_progress": rec["unfinished_progress"], "mean_active_sinr": rec["mean_active_sinr"], "mean_active_rate": rec["mean_active_rate"], "service_lock_violation_count": rec["service_lock_violation_count"]})
        hl_rows.append({"tag": name, "idle_fraction": rec["idle_fraction"], "assignment_change_count": rec["assignment_change_count"], "target_switch_count": rec["target_switch_count"], "high_level_decision_count": rec["high_level_decision_count"]})
        safe_rows.append({"tag": name, "collision_count": rec["collision_count"], "min_sep": rec["min_sep"], "min_sep_worst": rec["min_sep_worst"], "projector_failure_count": rec["projector_failure_count"], "service_lock_violation_count": rec["service_lock_violation_count"], "projector_intervention_rate": rec["projector_intervention_rate"], "PAIRWISE_COLLISION_among_intervened": rec["PAIRWISE_COLLISION_among_intervened"], "SERVICE_SAFE_ENVELOPE_among_intervened": rec["SERVICE_SAFE_ENVELOPE_among_intervened"], "MULTI_CONSTRAINT_among_intervened": rec["MULTI_CONSTRAINT_among_intervened"], "STEP_LIMIT_among_intervened": rec["STEP_LIMIT_among_intervened"], "OTHER_among_intervened": rec["OTHER_among_intervened"]})
        res_rows.append({"tag": name, "payload": rec["payload"], "mean_raw_residual": rec["mean_raw_residual"], "median_raw_residual": rec["median_raw_residual"], "p95_raw_residual": rec["p95_raw_residual"], "mean_used_residual": rec["mean_used_residual"], "median_used_residual": rec["median_used_residual"], "p95_used_residual": rec["p95_used_residual"], "cap_activation_rate": rec["cap_activation_rate"], "base_zero_freeze_rate": rec["base_zero_freeze_rate"], "mean_base_norm": rec["mean_base_norm"], "reverse_fraction": rec["reverse_fraction"], "mean_lateral": rec["mean_lateral"], "mean_forward": rec["mean_forward"]})
        feat_rows.append({"tag": name, "neighbor_shuffle": rec["neighbor_shuffle"], "interf_shuffle": rec["interf_shuffle"], "sinr_shuffle": rec["sinr_shuffle"]})
    gate = gate_decision(tagged)
    _csv(OUTPUT / "LLV2_PHASE2B_NAVIGATION.csv", nav_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_SERVICE.csv", svc_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_HIGHLEVEL_FEEDBACK.csv", hl_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_SAFETY.csv", safe_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_RESIDUAL_TRUST_REGION.csv", res_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_FEATURE_AUDIT.csv", feat_rows)
    _dump(OUTPUT / "LLV2_PHASE2B_GATE.json", gate)
    curve = list(csv.DictReader((OUTPUT / "LLV2_PHASE2B_EVAL_CURVE.csv").open(encoding="utf-8")))
    for row in curve:
        for k in row:
            if k != "model_seed" and k != "update":
                try:
                    row[k] = float(row[k])
                except Exception:
                    pass
            else:
                row[k] = int(float(row[k]))
    plot_figures(curve, tagged, OUTPUT / "figures")
    pre = json.loads((PHASE2B / "LLV2_PHASE2B_PREFREEZE.json").read_text(encoding="utf-8"))
    tests = json.loads((OUTPUT / "LLV2_PHASE2B_TESTS.json").read_text(encoding="utf-8"))
    report = write_report(pre, tests, tagged, gate, curve, [])
    _dump(OUTPUT / "HTW_LowLevelV2_Phase2B_ResidualTrustRegion_CN.md", report)
    for src in list(OUTPUT.glob("LLV2_PHASE2B_*")) + [OUTPUT / "HTW_LowLevelV2_Phase2B_ResidualTrustRegion_CN.md"]:
        if src.exists() and src.is_file():
            (PHASE2B / src.name).write_bytes(src.read_bytes())
            (ROOT / src.name).write_bytes(src.read_bytes())
    fig_dir = PHASE2B / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (OUTPUT / "figures").glob("*.png"):
        (fig_dir / png.name).write_bytes(png.read_bytes())
    return {
        "gate": gate["P2B"],
        "L0": tagged["L0"]["payload"],
        "new0": tagged["new0"]["payload"],
        "new1": tagged["new1"]["payload"],
        "old1": tagged["old1"]["payload"],
        "arr_L0": tagged["L0"]["mean_arrival_steps"],
        "arr_new1": tagged["new1"]["mean_arrival_steps"],
        "arr_old1": tagged["old1"]["mean_arrival_steps"],
        "tir_new1": tagged["new1"]["time_in_range"],
        "tir_L0": tagged["L0"]["time_in_range"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
