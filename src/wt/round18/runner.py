"""Round-18 formal-run entrypoint."""
from __future__ import annotations

from pathlib import Path

from .audit import run_audit
from .reporting import render_figures, write_report


FORMAL_STEM = "round18_multi_uav_audit_20260827_2055"


def run(root: Path) -> dict:
    output = root / "round_18" / "outputs" / FORMAL_STEM
    figures = root / "round_18" / "figures" / FORMAL_STEM
    result = run_audit(root, output)
    render_figures(figures, result)
    write_report(output / "round18_multi_uav_completion_report.md", result)
    return {"output": str(output), "figures": str(figures), **result}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(".")); args = parser.parse_args()
    completed = run(args.root.resolve()); print(completed["output"])
