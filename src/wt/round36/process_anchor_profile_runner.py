"""Persistent-pool scenario/anchor exact workload profile for the frozen methodology."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from time import perf_counter

from .infrastructure import configure_project_tempdir
from .optimization import all_candidate_utilities
from .optimization_run import _state
from .harness import frozen_legacy_movement
from .run import _write_json
from .scenarios import DESIGN_FAMILY, build_clean_scenarios


def profile_anchor_worker(task: tuple[int, int]) -> dict:
    """Top-level spawn-safe, independent scenario×anchor task."""
    scenario_index, anchor = task
    scenario = build_clean_scenarios(DESIGN_FAMILY)[scenario_index]
    state = _state(scenario, anchor); movement = frozen_legacy_movement(state, scenario.config())
    started = perf_counter(); values = all_candidate_utilities(state, movement, scenario.config(), scenario.eta, workers=1); elapsed = perf_counter() - started
    digest = hashlib.sha256(repr(values).encode("utf-8")).hexdigest().upper()
    return {"scenario_id": scenario.scenario_id, "anchor": anchor, "seconds": elapsed, "candidate_count": len(values), "utility_order_sha256": digest}


def main() -> None:
    configure_project_tempdir("runtime")
    tasks = [(index, anchor) for index in range(0, 64, 8) for anchor in (0, 8)]
    workers = 6
    context = mp.get_context("spawn")
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        rows = list(pool.map(profile_anchor_worker, tasks))
    wall_seconds = perf_counter() - started
    per_anchor_mean = sum(row["seconds"] for row in rows) / len(rows)
    per_anchor_worst = max(row["seconds"] for row in rows)
    # Frozen workload has 128 scenarios × two anchors = 256 independent anchors.
    projected_mean = per_anchor_mean * 256 / workers
    projected_worst = per_anchor_worst * 256 / workers
    result = {"schema_version": "round36_process_scenario_anchor_profile_v1", "engineering_only": True, "factorial_strata": 8, "anchors": [0, 8], "profiled_anchor_tasks": len(rows), "candidate_count_per_anchor": 3393, "workers": workers, "persistent_pool_wall_seconds": wall_seconds, "per_anchor_mean_seconds": per_anchor_mean, "per_anchor_worst_seconds": per_anchor_worst, "projected_formal_mean_seconds": projected_mean, "projected_formal_worst_seconds": projected_worst, "runtime_limit_seconds": 1800, "passed": projected_mean <= 1800 and projected_worst <= 1800, "rows": rows}
    _write_json("round36_process_scenario_anchor_profile.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
