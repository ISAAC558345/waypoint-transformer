"""Round-19 formal entrypoint."""
from __future__ import annotations

from pathlib import Path

from .audit import run_audit
from .reporting import render_figures, write_report


FORMAL_STEM = "round19_shared_spectrum_audit_20260828_1320"


def run(root: Path) -> dict:
    output = root / "round_19" / "outputs" / FORMAL_STEM
    figures = root / "round_19" / "figures" / FORMAL_STEM
    result = run_audit(root, output)
    render_figures(figures, result)
    write_report(output / "round19_shared_spectrum_completion_report.md", result)
    return {"output": str(output), "figures": str(figures), **result}


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path(".")); args=parser.parse_args()
    completed=run(args.root.resolve()); print(completed["output"])
