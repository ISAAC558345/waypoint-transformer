"""Hold-aligned train/val states with frozen candidate utilities."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase9.const import HOLD
from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment

from .candidate_pool import assignment_hash, generate_pool
from .protocol import POOL_SIZE
from .utility_rollout import shadow_h15


def collect_utility_split(seeds: tuple[int, ...], split: str) -> dict[str, Any]:
    keys = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")
    feats = {key: [] for key in keys}
    refs = []
    pools = []
    utilities = []
    sinr = []
    queue = []
    idle = []
    complete = []
    hashes = []
    meta = []
    uav_pos = []
    buoy_pos = []
    for seed in seeds:
        env = L3BEnv()
        info = env.reset(seed=int(seed))
        last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
        last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
        while not info.get("terminated", False):
            if int(env.t) % int(HOLD) == 0:
                batch = _batch_from_env(info, env, last_arrival, last_positions)
                ref = np.asarray(oracle_option_assignment(env), dtype=np.int64)
                pool = generate_pool(env, seed_key=int(seed) * 1009 + int(env.t))
                stats = [shadow_h15(env, row) for row in pool]
                for key in keys:
                    feats[key].append(batch[key][0])
                refs.append(np.pad(ref, (0, max(0, 6 - ref.shape[0])), constant_values=-1)[:6])
                pools.append(pool)
                utilities.append([row["payload_increment"] for row in stats])
                sinr.append([row["sinr_utility"] for row in stats])
                queue.append([row["queue_reduction"] for row in stats])
                idle.append([row["idle"] for row in stats])
                complete.append([row["completion"] for row in stats])
                hashes.append([assignment_hash(row) for row in pool])
                meta.append({"split": split, "scenario": int(seed), "t": int(env.t)})
                uav_pos.append(np.asarray(env.uav_positions, dtype=np.float64).copy())
                buoy_pos.append(np.asarray(env.buoy_positions, dtype=np.float64).copy())
            chosen = oracle_option_assignment(env)
            after = env.step(fly_toward(env, chosen), chosen)
            last_arrival = np.asarray(after["arrivals_blocks"], dtype=np.float32)
            last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
            info = after
    packed = {key: np.stack(value, axis=0) for key, value in feats.items()}
    packed["assignment"] = np.stack(refs, axis=0)
    packed["candidates"] = np.stack(pools, axis=0)
    packed["utility"] = np.asarray(utilities, dtype=np.float64)
    packed["sinr_utility"] = np.asarray(sinr, dtype=np.float64)
    packed["queue_reduction"] = np.asarray(queue, dtype=np.float64)
    packed["idle"] = np.asarray(idle, dtype=np.float64)
    packed["completion"] = np.asarray(complete, dtype=np.float64)
    packed["uav_positions"] = np.stack(uav_pos, axis=0)
    packed["buoy_positions"] = np.stack(buoy_pos, axis=0)
    packed["meta"] = meta
    packed["hashes"] = hashes
    packed["pool_size"] = int(POOL_SIZE)
    packed["split"] = split
    packed["seeds"] = list(seeds)
    return packed
