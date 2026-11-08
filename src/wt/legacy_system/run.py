"""Formal single-scenario command. Importing this module does not load torch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wt.legacy_system.references import REGISTERED, ReferenceRefusal, require_reference, resolve


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
    spec = resolve(scenario)
    payload = {
        "mode": "config_only",
        "scenario_id": spec["scenario_id"],
        "allow_holdout": False,
        "high_level_policy": spec["policy"],
        "generator": "G1_RATE",
        "k": 6,
        "hold": 15,
        "historical_formal_k": 5,
        "historical_formal_k_role": "source label of the older formal candidate count; the active count is k=6",
        "stop_time": spec["stop_time"],
        "process_seed": spec["process_seed"],
        "reset_seed": spec["reset_seed"],
        "decision_times": [0, 15, 30, 45],
        "training_updates": 0,
        "output": output.as_posix(),
        "writes_files": False,
        "loads_model": False,
        "reference_episode": spec["reference_episode"],
        "reference_config": spec["reference_config"],
        "run_accepted": spec["run_accepted"],
        "registered_scenarios": list(REGISTERED),
        "coverage": "only registered R1 references can run; this is not a full-scenario regression",
    }
    if not spec["run_accepted"]:
        payload["reason"] = "scenario is not in the registered reference set"
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _protected(output: Path) -> bool:
    posix = output.as_posix().replace("\\", "/").rstrip("/")
    return any(posix == item or posix.endswith("/" + item) for item in PROTECTED_OUTPUTS)


def _refuse(message: str) -> int:
    print(json.dumps({"error": message, "environment_steps": 0, "exit_code": 2, "success": False}), flush=True)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview or run one formal Scheme C episode.")
    parser.add_argument("--scenario", type=int, default=REGISTERED[0])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    spec = resolve(int(args.scenario))
    output_arg = args.output if args.output is not None else Path(spec["default_output"])
    if not args.run:
        preview(int(args.scenario), output_arg)
        return 0
    if not spec["run_accepted"]:
        return _refuse("scenario is not in the registered reference set")
    try:
        require_reference(_repo_root(), int(args.scenario))
    except ReferenceRefusal as exc:
        return _refuse(str(exc))
    output = output_arg if output_arg.is_absolute() else Path.cwd() / output_arg
    if _protected(output.resolve()):
        return _refuse("refusing to write into an existing result directory")
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from wt.legacy_system.execute import run_once

    return run_once(_repo_root(), output.resolve(), int(args.scenario))


if __name__ == "__main__":
    sys.exit(main())
