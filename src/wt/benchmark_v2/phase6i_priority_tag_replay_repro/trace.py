"""Record one frozen L0 episode without changing the evaluator."""
from __future__ import annotations

import hashlib

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_reward.evaluate import _episode_row
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import eval_worker
from wt.lowlevel_v2.observation import communication_features

from .protocol import MODEL_SEED, SCENARIO_ID


def _sha256_arrays(parts: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        array = np.ascontiguousarray(part)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _observation_sha(obs: dict) -> str:
    keys = ("self", "task", "communication", "neighbor_set", "d_nearest")
    return _sha256_arrays([np.asarray(obs[key]) for key in keys])


def trace_episode(actor, *, instrumented: bool) -> dict:
    rollout = None
    original_follow = None
    if instrumented:
        import wt.benchmark_v2.phase6i_priority_tag_reward.rollout as rollout
        from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import follow_with_attribution

        original_follow = rollout.follow_step
        rollout.follow_step = follow_with_attribution
    try:
        worker = eval_worker(actor, int(SCENARIO_ID), lambda_u=0.0, use_ledger=True)
        captured: dict = {}
        original_project = worker.projector.project

        def project(env, desired, option):
            result = original_project(env, desired, option)
            captured["desired"] = np.asarray(desired, dtype=np.float64).copy()
            captured["projected"] = np.asarray(result.projected_actions, dtype=np.float64).copy()
            captured["intervention"] = float(result.intervention_norm)
            captured["constraint_status"] = str(result.constraint_status)
            return result

        worker.projector.project = project
        worker.reset()
        reset = {
            "t": int(worker.env.t),
            "payload": float(worker.env.payload_bits),
            "q_sha256": _sha256_arrays([np.asarray(worker.env.q_bits)]),
            "position_sha256": _sha256_arrays([np.asarray(worker.env.uav_positions)]),
            "q_bits": np.asarray(worker.env.q_bits, dtype=np.float64).copy(),
            "positions": np.asarray(worker.env.uav_positions, dtype=np.float64).copy(),
        }
        horizon = int(worker.env.config.episode_horizon)
        raw_rows = []
        steps = []
        for _ in range(horizon):
            if int(worker.env.t) >= horizon:
                break
            captured.clear()
            record = worker.step()
            raw_rows.append(record)
            comm = communication_features(worker.env)
            steps.append(
                {
                    "t": int(worker.env.t),
                    "obs_sha256": _observation_sha(record["obs"]),
                    "mu": np.asarray(record["pre_tanh"], dtype=np.float64).copy(),
                    "desired": np.asarray(captured["desired"], dtype=np.float64).copy(),
                    "projected": np.asarray(captured["projected"], dtype=np.float64).copy(),
                    "executed": np.asarray(record["action"], dtype=np.float64).copy(),
                    "position": np.asarray(worker.env.uav_positions, dtype=np.float64).copy(),
                    "association": np.asarray(worker.option.association, dtype=np.int64).copy(),
                    "q_bits": np.asarray(worker.env.q_bits, dtype=np.float64).copy(),
                    "served": float(record["served_bits"]),
                    "dropped": float(record["dropped_ordinary_bits"] + record["dropped_tagged_bits"]),
                    "snap": float(np.sum(worker.env.cumulative_snap_bits)),
                    "payload": float(record["payload"]),
                    "path_increment": float(record["path_step"]),
                    "sinr": float(record["sinr"]),
                    "intervention": float(captured["intervention"]),
                    "constraint_status": str(captured["constraint_status"]),
                    "interference": float(np.mean(comm["total_interference"])),
                    "ledger_gap": float(record["ledger_gap_bits"]),
                }
            )
        if len(steps) != horizon:
            raise RuntimeError(f"trace returned {len(steps)} steps")
        episode = _episode_row("L0", int(MODEL_SEED), int(SCENARIO_ID), raw_rows)
        return {"reset": reset, "steps": steps, "episode": episode}
    finally:
        if rollout is not None and original_follow is not None:
            rollout.follow_step = original_follow
