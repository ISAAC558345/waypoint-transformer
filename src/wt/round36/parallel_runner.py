"""Engineering-only serial/parallel exact equivalence check."""
from __future__ import annotations

import json
from time import perf_counter

from .optimization import all_candidate_utilities, safe_worker_count
from .optimization_run import _state
from .harness import frozen_legacy_movement
from .infrastructure import configure_project_tempdir
from .run import _write_json
from .scenarios import DESIGN_FAMILY, build_clean_scenarios


def main() -> None:
    configure_project_tempdir("runtime")
    scenario = build_clean_scenarios(DESIGN_FAMILY)[0]
    state = _state(scenario, 0)
    movement = frozen_legacy_movement(state, scenario.config())
    started = perf_counter(); serial = all_candidate_utilities(state, movement, scenario.config(), scenario.eta, workers=1); serial_seconds = perf_counter() - started
    workers = safe_worker_count()
    started = perf_counter(); parallel = all_candidate_utilities(state, movement, scenario.config(), scenario.eta, workers=workers, executor="process"); parallel_seconds = perf_counter() - started
    same = serial == parallel
    result = {
        "schema_version": "round36_parallel_execution_equivalence_v1",
        "engineering_only": True, "scenario_id": scenario.scenario_id, "anchor": 0,
        "candidate_count": len(serial), "worker_count": workers,
        "serial_seconds": serial_seconds, "parallel_seconds": parallel_seconds, "executor": "process",
        "speedup": serial_seconds / parallel_seconds if parallel_seconds else 0.0,
        "all_candidate_utilities_equal": same,
        "gamma_h1_h5_h10_and_order_equal": same,
        "deterministic_tie_break_equal": same,
        "passed": same,
    }
    _write_json("parallel_execution_equivalence.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
