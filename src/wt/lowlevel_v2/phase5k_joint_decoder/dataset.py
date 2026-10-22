"""Assignment pools labeled by Phase-5J J(A). Labels are not encoder inputs."""
from __future__ import annotations

import numpy as np
import torch

from wt.lowlevel_v2.phase5h_execution_audit.rollout import trace_h15
from wt.lowlevel_v2.phase5i_joint_utility.search import largest_complete_pool, top_options
from wt.lowlevel_v2.phase5j_joint_redesign.joint import joint_utility
from wt.lowlevel_v2.phase5j_joint_redesign.protocol import LAMBDA_CONFLICT
from wt.lowlevel_v2.phase5j_joint_redesign.run_phase5j import _sum_over, collect_marginals, replay_env
from wt.lowlevel_v2.phase5g_closed_loop.run_phase5g import _greedy_assignment

from .protocol import CACHE, CACHE_VERSION, OUTPUT, SCENARIOS, SEARCH_LIMIT, assert_phase5k_seed


def _clone_batch(batch: dict) -> dict[str, np.ndarray]:
    cloned = {}
    for key, value in batch.items():
        cloned[key] = np.asarray(value).copy()
    return cloned


def build_states() -> list[dict]:
    rows = []
    for seed in SCENARIOS:
        assert_phase5k_seed(int(seed))
        print(f"phase-5k marginals {seed}", flush=True)
        rows.extend(collect_marginals(int(seed)))
    grouped: dict[tuple[int, int], list[dict]] = {}
    for sample in rows:
        grouped.setdefault((int(sample["scenario_id"]), int(sample["time_step"])), []).append(sample)
    states = []
    for (scenario, moment), group in sorted(grouped.items()):
        assert_phase5k_seed(int(scenario))
        group = sorted(group, key=lambda sample: int(sample["uav_id"]))
        pools = {
            width: [top_options(sample["candidate_ids"], sample["utility_payload"], width) for sample in group]
            for width in (1, 2, 3)
        }
        width, assignments = largest_complete_pool(pools, SEARCH_LIMIT)
        if not assignments:
            raise RuntimeError(f"phase-5k found no exclusive assignment at {scenario} t={moment}")
        env, movement = replay_env(int(scenario), int(moment))
        labeled = []
        joints = []
        individuals = []
        flies = []
        for assignment in assignments:
            individual = _sum_over(group, assignment, "utility_payload")
            fly = float(trace_h15(env, assignment, None, movement, "fly")["payload"])
            labeled.append(np.asarray(assignment, dtype=np.int64))
            individuals.append(individual)
            flies.append(fly)
            joints.append(joint_utility(individual, fly, lam=LAMBDA_CONFLICT))
        local_utility = {int(sample["uav_id"]): sample["utility_payload"] for sample in group}
        states.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(moment),
                "pool_width": int(width),
                "urgency_batch": _clone_batch(group[0]["urgency_batch"]),
                "assignments": np.stack(labeled, axis=0),
                "joint": np.asarray(joints, dtype=np.float64),
                "individual": np.asarray(individuals, dtype=np.float64),
                "fly": np.asarray(flies, dtype=np.float64),
                "greedy": np.asarray(_greedy_assignment(group, local_utility), dtype=np.int64),
                "movement": np.asarray(movement, dtype=np.float64),
                "num_uavs": int(env.config.num_uavs),
                "num_buoys": int(env.config.num_buoys),
            }
        )
        print(f"phase-5k pool {scenario} t={moment} n={len(labeled)} width={width}", flush=True)
    return states


def load_states() -> list[dict]:
    if CACHE.is_file():
        payload = torch.load(CACHE, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) == int(CACHE_VERSION):
            return list(payload["states"])
    states = build_states()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"version": int(CACHE_VERSION), "states": states}, CACHE)
    return states


__all__ = ["build_states", "load_states"]
