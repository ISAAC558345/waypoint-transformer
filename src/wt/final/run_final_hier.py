"""Demo / ordinary-scenario entry for frozen FINAL-HIER. Never runs untouched seeds."""
from __future__ import annotations

import argparse
import json

from . import _bootstrap  # noqa: F401
from .constants import DEFAULT_DEMO_SEED
from .pipeline import run_final_hier


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run frozen FINAL-HIER on an ordinary demo/design seed.")
    parser.add_argument("--seed", type=int, default=DEFAULT_DEMO_SEED, help="ordinary scenario seed (default 8). Untouched 106000-106063 are refused.")
    parser.add_argument("--no-gate", action="store_true", help="disable InRangeZeroMotionGate (debug only)")
    args = parser.parse_args(argv)
    result = run_final_hier(int(args.seed), gated=not bool(args.no_gate))
    printable = {
        "scenario": result["scenario"],
        "assignment": result["assignment"],
        "payload": result["payload"],
        "completed_blocks": result["completed_blocks"],
        "queue": result["queue"],
        "flight_distance": result["flight_distance"],
        "legality": result["legality"],
        "task_option": result["task_option"],
        "n_steps": result["n_steps"],
        "gated": result["gated"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "untouched_used": result["untouched_used"],
    }
    print(json.dumps(printable, indent=2))
    return result


if __name__ == "__main__":
    main()
