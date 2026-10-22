"""Reproducible pre-outcome exact-runtime audit stages."""
from __future__ import annotations

import hashlib
import json
from time import perf_counter

import numpy as np

from wt.round35.level2_env import Level2SharedSpectrumMultiUAVEnv

from .harness import Round36AuditState, enumerate_legal_assignments, frozen_legacy_movement
from .optimization import equivalent_prefixes, prefix_trajectory, profile_candidate, separately_recomputed_prefixes
from .run import _write_json
from .scenarios import DESIGN_FAMILY, build_clean_scenarios


def _state(scenario, anchor: int) -> Round36AuditState:
    env = Level2SharedSpectrumMultiUAVEnv(scenario.config(), buoy_positions_m=scenario.buoy_positions_m(), interference_overlap_factor=scenario.eta)
    env.reset(seed=scenario.seed)
    base = env._shared._base
    state = Round36AuditState(base._uav_positions_m.copy(), base._remaining_energy.copy(), base._queue_bits.copy(), base._buoy_positions_m.copy(), 0, base._arrival_trace_bits[0].copy(), base._max_team_rate_bps)
    from .harness import audit_legacy_execute
    for _ in range(anchor):
        state = audit_legacy_execute(state, frozen_legacy_movement(state, scenario.config()), scenario.config(), interference_overlap_factor=scenario.eta).next_state
    return state


def run_prefix_and_profile() -> dict:
    scenarios = build_clean_scenarios(DESIGN_FAMILY)
    catalogue = enumerate_legal_assignments(4, 8)
    catalog_sha = hashlib.sha256(catalogue.tobytes()).hexdigest().upper()
    _write_json("legal_assignment_catalog.json", {"num_uavs": 4, "num_buoys": 8, "count": len(catalogue), "ordering": "frozen_round19_itertools_product_partial_injective", "sha256": catalog_sha, "assignments": catalogue})
    selected = np.linspace(0, len(catalogue) - 1, 12, dtype=np.int64)
    checks, profiles = [], []
    for scenario_index in range(0, 64, 8):
        scenario = scenarios[scenario_index]
        state = _state(scenario, 0)
        movement = frozen_legacy_movement(state, scenario.config())
        for index in selected:
            gamma = catalogue[index]
            old = separately_recomputed_prefixes(state, movement, gamma, scenario.config(), scenario.eta)
            new = prefix_trajectory(state, movement, gamma, scenario.config(), scenario.eta)
            checks.append(equivalent_prefixes(old, new))
        for gamma in catalogue[:8]:
            profiles.append(profile_candidate(state, movement, gamma, scenario.config(), scenario.eta))
    means = {key: float(np.mean([row[key] for row in profiles])) for key in ("first_step_direct_executor", "continuation_movement", "continuation_transition_and_matching", "prefix_utility_service")}
    total = sum(means.values())
    bottleneck = max(means, key=means.get)
    result = {"prefix_reuse_already_used_by_current_harness": True, "old_separate_transition_calls_per_candidate": 16, "new_prefix_transition_calls_per_candidate": 10, "candidate_cases": len(checks), "factorial_strata": 8, "passed": bool(all(checks)), "strict_atol": 1.0e-9}
    _write_json("prefix_utility_reuse_audit.json", result)
    _write_json("prefix_reuse_equivalence.json", {**result, "compared": ["utility", "state_prefix", "service_prefix", "queue_prefix", "interference_prefix", "reward_prefix"]})
    lines = ["# Round-36 runtime bottleneck audit", "", "Measured pre-outcome wall-clock decomposition over 8 factorial design strata and 8 legal assignments each.", "", "| Component | seconds/candidate | fraction | call count/candidate |", "|---|---:|---:|---:|"]
    calls = {"first_step_direct_executor": 1, "continuation_movement": 9, "continuation_transition_and_matching": 9, "prefix_utility_service": 9}
    for key, value in means.items(): lines.append(f"| {key} | {value:.8f} | {value / total:.3%} | {calls[key]} |")
    lines.extend(["", f"Largest measured component: `{bottleneck}`.", "Assignment catalogue enumeration is a one-time cached structural operation; scenario/anchor and JSON overhead are excluded from the inner candidate trajectory."])
    ( _write_json.__globals__["OUTPUT"] / "round36_runtime_bottleneck_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_prefix_and_profile(), sort_keys=True))
