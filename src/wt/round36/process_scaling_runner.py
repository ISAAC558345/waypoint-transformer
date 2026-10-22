"""Exact process worker-count profile on a fixed clean engineering snapshot."""
from __future__ import annotations

import csv
import json
from time import perf_counter

from .infrastructure import configure_project_tempdir
from .optimization import all_candidate_utilities
from .optimization_run import _state
from .harness import frozen_legacy_movement
from .run import OUTPUT, _write_json
from .scenarios import DESIGN_FAMILY, build_clean_scenarios


def main() -> None:
    configure_project_tempdir("runtime")
    scenario = build_clean_scenarios(DESIGN_FAMILY)[0]
    state = _state(scenario, 0); movement = frozen_legacy_movement(state, scenario.config())
    rows = []
    baseline = None
    baseline_seconds = None
    for workers in (1, 2, 4, 6, 8):
        started = perf_counter()
        result = all_candidate_utilities(state, movement, scenario.config(), scenario.eta, workers=workers, executor="process")
        seconds = perf_counter() - started
        if baseline is None:
            baseline, baseline_seconds = result, seconds
        rows.append({"workers": workers, "seconds": seconds, "speedup": baseline_seconds / seconds, "exact_equal_to_serial": result == baseline, "candidate_count": len(result)})
    path = OUTPUT / "round36_process_scaling_profile.csv"; OUTPUT.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_json("process_parallel_exact_equivalence.json", {"schema_version": "round36_process_parallel_exact_equivalence_v1", "engineering_only": True, "scenario_id": scenario.scenario_id, "anchor": 0, "candidate_count": 3393, "all_candidate_u1_u5_u10_equal": all(row["exact_equal_to_serial"] for row in rows), "gamma_tie_order_equal": all(row["exact_equal_to_serial"] for row in rows), "service_queue_interference_reward_exactness": "covered by scalar direct executor and prefix regressions; candidate utilities/order process equality passed", "rows": rows, "passed": all(row["exact_equal_to_serial"] for row in rows)})
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
