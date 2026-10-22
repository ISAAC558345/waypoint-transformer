"""Command-line entry point for the inference-only Round-14 audit."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .audit import run_mapping_audit
from .reporting import materialize_interface_artifacts, readiness_gate, render_figures


def run(root: Path, output: Path, figures: Path) -> dict:
    result = run_mapping_audit(root, output)
    postprocess(root, output, figures)
    return result


def postprocess(root: Path, output: Path, figures: Path) -> dict:
    """Complete text/figure materialization from an already-written core audit.

    This is intentionally idempotent only before the figure directory exists,
    which allows an interrupted display/client call to finish the immutable
    core CSV evidence rather than rerunning or replacing it.
    """
    root, output, figures = Path(root), Path(output), Path(figures)
    selected = json.loads((output / "selected_task_to_motion_mapping.json").read_text(encoding="utf-8"))
    gate = json.loads((output / "task_to_motion_mapping_gate.json").read_text(encoding="utf-8"))
    materialize_interface_artifacts(output, selected, gate)
    with (output / "mapping_quality_stratified_alignment.csv").open(encoding="utf-8", newline="") as handle:
        import csv
        alignment = list(csv.DictReader(handle))
    readiness = readiness_gate(gate, selected, alignment)
    (output / "round14_readiness_gate.json").write_text(json.dumps(readiness, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    rendered = render_figures(output, figures)
    manifest = json.loads((output / "round14_mapping_manifest.json").read_text(encoding="utf-8"))
    run_manifest = {
        **manifest,
        "output": str(output.relative_to(root)),
        "figures": str(figures.relative_to(root)),
        "figure_png_files": rendered,
        "readiness_gate": "round14_readiness_gate.json",
        "interface_artifacts": [
            "task_to_motion_mapping_spec.md",
            "actionable_motion_guidance_schema.json",
            "future_policy_mapping_interface.md",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"selected": selected, "gate": gate, "readiness": readiness, "figures": rendered}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round-14 Task-Waypoint-to-motion audit")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = args.output or root / "round_14/outputs" / f"round14_task_to_motion_mapping_{stamp}"
    figures = args.figures or root / "round_14/figures" / f"round14_task_to_motion_mapping_{stamp}"
    result = run(root, output, figures)
    print(json.dumps({"output": str(output), "figures": str(figures), "gate": result["gate"]["task_to_motion_mapping_gate"]}, indent=2))


if __name__ == "__main__":
    main()
