"""Formal single-scenario command. Importing this module does not load torch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCENARIO = 322000
STOP_TIME = 48
PROCESS_SEED = 1_000_000 + SCENARIO
REFERENCE_EPISODE = "results/legacy_formal_repro_r1/episodes/322000_scheme_c.json"
REFERENCE_CONFIG = "results/legacy_formal_repro_r1/configs/322000_scheme_c.json"
PROTECTED_OUTPUTS = {
    "results/legacy_formal_repro_r1",
    "results/legacy_formal_repro_r1a_view",
    "results/legacy_arrival_aware_ab_l2",
    "results/legacy_arrival_aware_ab_l3",
    "results/legacy_arrival_aware_ab_l4",
    "results/legacy_arrival_aware_ab_l5",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def preview(scenario: int, output: Path) -> dict:
    """Show the formal contract. This creates no directory and imports no model stack."""
    payload = {
        "mode": "config_only",
        "scenario_id": int(scenario),
        "allow_holdout": False,
        "high_level_policy": "scheme_c",
        "generator": "G1_RATE",
        "k": 6,
        "hold": 15,
        "historical_formal_k": 5,
        "historical_formal_k_role": "source label of the older formal candidate count; the active count is k=6",
        "stop_time": STOP_TIME,
        "process_seed": int(PROCESS_SEED if int(scenario) == SCENARIO else 1_000_000 + int(scenario)),
        "reset_seed": int(scenario),
        "decision_times": [0, 15, 30, 45],
        "training_updates": 0,
        "output": output.as_posix(),
        "writes_files": False,
        "loads_model": False,
        "reference_episode": REFERENCE_EPISODE,
        "reference_config": REFERENCE_CONFIG,
        "run_accepted": int(scenario) == SCENARIO,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _protected(output: Path) -> bool:
    posix = output.as_posix().replace("\\", "/").rstrip("/")
    return any(posix == item or posix.endswith("/" + item) for item in PROTECTED_OUTPUTS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview or run one formal Scheme C episode.")
    parser.add_argument("--scenario", type=int, default=SCENARIO)
    parser.add_argument("--output", type=Path, default=Path("results/legacy_system_l6"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    if not args.run:
        preview(int(args.scenario), args.output)
        return 0
    if int(args.scenario) != SCENARIO:
        print(json.dumps({"error": "this entry runs scenario 322000 only", "environment_steps": 0}), flush=True)
        return 2
    if _protected(output.resolve()):
        print(json.dumps({"error": "refusing to write into an existing result directory", "environment_steps": 0}), flush=True)
        return 2
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from wt.legacy_system.execute import run_once

    return run_once(_repo_root(), output.resolve(), int(args.scenario))


if __name__ == "__main__":
    sys.exit(main())
