"""Command-line entry point for the self-contained Round-15 audit."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .audit import run_audit
from .reporting import create_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = (args.output or root / "round_15/outputs" / f"round15_planning_headroom_audit_{stamp}").resolve()
    figures = (args.figures or root / "round_15/figures" / output.name).resolve()
    result = run_audit(root, output)
    result["figures"] = create_figures(output, figures)
    (output / "round15_run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": str(figures), "gate": result["gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
