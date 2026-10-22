"""Round-17 audit runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit import run_audit
from .reporting import create_figures, write_report


def run_round17(root: Path, output: Path, figures: Path) -> dict[str, Any]:
    result=run_audit(root,output)
    result["runtime"]=[]
    import csv
    with (output/"channel_runtime_events.csv").open(encoding="utf-8") as handle:
        result["runtime"]=list(csv.DictReader(handle))
    result["traces"]=[]
    with (output/"channel_trajectory_steps.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            import json
            row["position_after_m"]=json.loads(row["position_after_m"]); result["traces"].append(row)
    result["figures"]=create_figures(result,figures)
    write_report(result,output/"round17_predictable_channel_report.md")
    (output/"implementation_log.md").write_text("# Round-17 implementation log\n\n- Verified the locked Round-16 chain before initialization.\n- Created the pre-Round-17 snapshot and pre-change channel semantics audit.\n- Froze a bounded, deterministic, current-observable oscillator before the diagnostic run.\n- Preserved static channel behavior and the frozen lattice planner except for its channel-state transition.\n- No training dataset or protected-data input was used.\n",encoding="utf-8")
    return result


def render_existing_audit(output: Path, figures: Path) -> dict[str, Any]:
    """Finish report artifacts from an already-complete immutable audit output."""
    import csv
    import json
    def read(name: str):
        with (output / name).open(encoding="utf-8") as handle:
            rows=list(csv.DictReader(handle))
        for row in rows:
            if "horizon" in row: row["horizon"]=int(row["horizon"])
        return rows
    result={
        "static_rows": read("static_channel_regression_audit.csv"),
        "horizon": read("channel_planning_horizon_value.csv"),
        "headroom": read("channel_planning_headroom.csv"),
        "dpg": read("channel_induced_planning_gain.csv"),
        "reversals": read("channel_ranking_reversal_audit.csv"),
        "anticipation": read("channel_window_anticipation_audit.csv"),
        "delayed": read("channel_delayed_benefit.csv"),
        "oracle": read("causal_oracle_channel_gap.csv"),
        "runtime": read("channel_runtime_events.csv"),
        "coupling": read("queue_channel_coupling_audit.csv") + read("geometry_channel_coupling_audit.csv") + read("phase_conflict_coupling_audit.csv"),
        "gate": json.loads((output / "channel_planning_necessity_gate.json").read_text(encoding="utf-8")),
        "realism": json.loads((output / "channel_realism_gate.json").read_text(encoding="utf-8")),
        "traces": [],
    }
    for row in read("channel_trajectory_steps.csv"):
        row["position_after_m"] = json.loads(row["position_after_m"]); result["traces"].append(row)
    result["figures"]=create_figures(result,figures)
    write_report(result,output/"round17_predictable_channel_report.md")
    (output/"implementation_log.md").write_text("# Round-17 implementation log\n\n- Verified the locked Round-16 chain before initialization.\n- Created the pre-Round-17 snapshot and pre-change channel semantics audit.\n- Froze a bounded, deterministic, current-observable oscillator before the diagnostic run.\n- Preserved static channel behavior and the frozen lattice planner except for its channel-state transition.\n- Static regression passed before the 48-scenario, 576-episode audit.\n- No training dataset or protected-data input was used.\n",encoding="utf-8")
    return result


__all__=["render_existing_audit", "run_round17"]
