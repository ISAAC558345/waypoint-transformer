"""Collect 64 pre-registered high-level decision states from LLV2_design."""
from __future__ import annotations

from wt.htw_phase100.high_level import decide
from wt.htw_phase945.candidate_pool import assignment_hash, generate_pool, is_legal
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase4b.snapshot import snapshot_env

from .protocol import HOLD_STEPS, N_STATES, POOL_SIZE, refuse_compat


def collect_hold_states(worker, design_seeds: list[int]) -> list[dict]:
    states = []
    for seed in design_seeds:
        refuse_compat(int(seed))
        info = worker.reset(int(seed))
        while True:
            if int(worker.pending) <= 0:
                env = worker.env
                comm = communication_features(env)
                assignment_ctx = np_assoc(worker)
                pool = generate_pool(env, seed_key=int(seed) * 1009 + int(env.t))
                if int(pool.shape[0]) != int(POOL_SIZE):
                    raise RuntimeError("candidate pool size mismatch")
                for row in pool:
                    if not is_legal(row, env.config.num_buoys):
                        raise RuntimeError("illegal candidate in frozen pool")
                policy = decide(worker.high, env, info, worker.last_arrival, worker.last_positions, worker.pending)
                states.append(
                    {
                        "state_id": len(states),
                        "scenario": int(seed),
                        "t": int(env.t),
                        "env": snapshot_env(env),
                        "last_arrival": np_copy(worker.last_arrival),
                        "last_positions": np_copy(worker.last_positions),
                        "last_movement": np_copy(worker.last_movement),
                        "hold_timing": {"remaining": int(worker.pending), "hold": int(HOLD_STEPS)},
                        "association_context": assignment_ctx,
                        "communication": {
                            "sinr": np_copy(comm["sinr"]),
                            "rate": np_copy(comm["rate"]),
                            "total_interference": np_copy(comm["total_interference"]),
                            "inr": np_copy(comm["interference_to_noise"]),
                        },
                        "queue": np_copy(env.queue),
                        "progress": np_copy(env.progress),
                        "candidates": np_copy(pool),
                        "candidate_hashes": [assignment_hash(row) for row in pool],
                        "policy_assignment": np_copy(policy),
                        "policy_hash": assignment_hash(policy),
                    }
                )
                if len(states) >= int(N_STATES):
                    return states
            info, rec = worker.step(info, deterministic=True, zero_residual=True)
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
    if len(states) != int(N_STATES):
        raise RuntimeError(f"expected {N_STATES} states, got {len(states)}")
    return states


def np_copy(value):
    import numpy as np

    return np.asarray(value).copy()


def np_assoc(worker):
    import numpy as np

    if worker.option is None:
        return np.full(int(worker.env.config.num_uavs), -1, dtype=np.int64)
    return np.asarray(worker.option.association, dtype=np.int64).copy()


__all__ = ["collect_hold_states"]
