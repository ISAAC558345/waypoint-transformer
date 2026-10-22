"""Run one frozen closed-loop episode from reset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HOLDOUT_START = 322064
HOLDOUT_STOP = 322080


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen scheme-C planner with MAPPO-V2.")
    parser.add_argument("--scenario", type=int, default=322002)
    parser.add_argument("--stop-time", type=int, default=None)
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()
    scenario = int(args.scenario)
    if HOLDOUT_START <= scenario < HOLDOUT_STOP and not args.allow_holdout:
        raise SystemExit("scenario is in the reserved test split; pass --allow-holdout to read it")
    from policies import run_active_episode

    result = run_active_episode(scenario, stop_time=args.stop_time, allow_holdout=bool(args.allow_holdout))
    print(f"scenario={scenario} payload_bits={result['payload_bits']}")


if __name__ == "__main__":
    main()
