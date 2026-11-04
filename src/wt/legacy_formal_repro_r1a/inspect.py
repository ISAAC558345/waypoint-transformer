"""Print a saved R1 summary. This command writes no files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wt.legacy_formal_repro_r1a.read import build_view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a saved R1 manifest without rewriting it.")
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args(argv)
    view = build_view(args.artifact_root)
    summary = {
        "source_environment_steps": view["source_environment_steps"],
        "source_steps_are_a_new_run": False,
        "arms": [
            {
                "scenario_id": arm["scenario_id"],
                "policy": arm["policy"],
                "config_relative": arm["config_relative"],
                "payload_bits": arm["payload_bits"],
            }
            for arm in view["arms"]
        ],
        "pairs": view["pairs"],
        "missing_original_summaries": view["missing_original_summaries"],
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
