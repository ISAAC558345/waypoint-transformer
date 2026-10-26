"""Write the E1 prototype artifacts. No training and no formal edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import (
    BETA_TAG,
    GATE_BLOCKED,
    GATE_LEDGER_PASS,
    GATE_SCALE_READY,
    INSTRUCTION_COMMIT,
    LAMBDA_CANDIDATES,
    LAMBDA_REFERENCE,
    LAMBDA_SELECTED,
    OUTPUT,
    REPORT,
    ROOT,
    SCHEMA_VERSION,
    SMOKE_SCENARIO,
    SMOKE_STEPS,
    TAG_RATE,
    TAG_SEED,
    choose_gate,
)
from .report import render_report
from .reward_scale import load_saved_probe, scale_table
from .smoke import run_smoke

WATCHED = (
    "src/wt/benchmark_v2/bit_queue_v2.py",
    "src/wt/benchmark_v2/phase6i_urgency_stage_a/ledger.py",
    "src/wt/lowlevel_v2/observation.py",
    "src/wt/lowlevel_v2/critic_state.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py",
    "src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    "results/phase6i_urgency_label_rule_design/summary.json",
    "results/phase6i_urgency_lowlevel/smoke_summary.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    smoke = run_smoke(SMOKE_SCENARIO, SMOKE_STEPS)
    probe = load_saved_probe()
    scale = scale_table(smoke["rows"], probe)
    ledger_ok = bool(
        smoke["max_ledger_gap_bits"] <= 1.0
        and smoke["transition_gap"] == 0.0
        and smoke["repeat_gap"] == 0.0
        and smoke["tag_repeat_match"]
        and smoke["snap_source"] == "cumulative_snap_bits_delta"
    )
    tracking_ok = bool(
        all(abs(float(row["served_ordinary_bits"]) + float(row["served_tagged_bits"]) - float(row["served_bits"])) <= 1.0 for row in smoke["rows"])
        and all(abs(float(row["generated_bits"]) - float(row["admitted_bits"]) - float(row["dropped_bits"])) <= 1.0 for row in smoke["rows"])
    )
    gate = choose_gate(ledger_ok=ledger_ok, tracking_ok=tracking_ok, scale_reported=True)
    watched = {}
    for relative in WATCHED:
        path = ROOT / relative
        watched[relative] = _sha256(path) if path.is_file() else "missing"
    summary = {
        "gate": gate,
        "ledger_gate": GATE_LEDGER_PASS if ledger_ok and tracking_ok else GATE_BLOCKED,
        "scale_gate": GATE_SCALE_READY if gate == GATE_SCALE_READY else GATE_BLOCKED,
        "instruction_commit": INSTRUCTION_COMMIT,
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "tag_rate": TAG_RATE,
        "beta_tag": BETA_TAG,
        "tag_seed": TAG_SEED,
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "lambda_reference": LAMBDA_REFERENCE,
        "lambda_selected": LAMBDA_SELECTED,
        "trained": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "smoke": {key: value for key, value in smoke.items() if key != "rows"},
        "reward_scale": scale,
        "watched_sha256": watched,
    }
    schema = {
        "schema_version": SCHEMA_VERSION,
        "synthetic_priority_tag": "binary 0 ordinary, 1 synthetic urgent; not a maritime business label",
        "tag_rate": TAG_RATE,
        "beta_tag": BETA_TAG,
        "tag_seed_sequence": [TAG_SEED, "scenario_id", "admission_step", "buoy"],
        "reset_stock": "tag 0, age_origin reset_convention, admission_step 0",
        "urgency": "clip(clip(age/48,0,1)+beta_tag*tag,0,1), computed before service",
        "fifo_key": ["admission_step", "insert_index"],
        "tag_is_fifo_key": False,
        "formal_step_fields": ["served_bits", "generated_arrival_bits", "admitted_arrival_bits", "dropped_bits_step"],
        "snap_source": "cumulative_snap_bits delta because info does not contain float_snap_bits",
        "tolerance_bits": 1.0,
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "lambda_reference": LAMBDA_REFERENCE,
        "lambda_selected": None,
        "old_buoy_urgency_ratio_used": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "reward_scale.json").write_text(json.dumps(scale, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")
    print(gate, summary["ledger_gate"], smoke["max_ledger_gap_bits"], smoke["served_tagged_bits"], smoke["transition_gap"])


if __name__ == "__main__":
    main()
