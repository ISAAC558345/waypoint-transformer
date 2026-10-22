"""Collect active agent-states from L0 frozen rollouts on train/design."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.lowlevel_v2.contracts import env_target
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase4b.protocol import refuse_phase4b
from wt.lowlevel_v2.phase4b.snapshot import snapshot_env


def collect_active_states(worker, seeds: list[int], *, pool: str, zero_residual: bool = True) -> list[dict]:
    rows = []
    for seed in seeds:
        refuse_phase4b(int(seed))
        info = worker.reset(int(seed))
        while True:
            worker._refresh_option(info)
            snap = snapshot_env(worker.env)
            option = worker.option.copy()
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
            comm_raw = communication_features(worker.env)
            team_interf = float(np.sum(comm_raw["total_interference"]))
            info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
            assoc = np.asarray(option.association, dtype=np.int64)
            obs = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in rec.obs.items()}
            for u in range(6):
                if float(rec.policy_active_mask[u]) < 0.5:
                    continue
                if bool(rec.service_lock[u]):
                    continue
                tgt = env_target(int(assoc[u]), int(worker.env.config.num_buoys))
                idle = int(assoc[u]) < 0
                invalid = (not idle) and tgt < 0
                empty = bool(tgt >= 0 and float(snap["queue"][tgt]) <= 1.0e-9)
                if idle or invalid or empty:
                    continue
                dist_tgt = float(np.linalg.norm(pos0[u] - worker.env.buoy_positions[tgt])) if tgt >= 0 else float("nan")
                d_uav = []
                for j in range(6):
                    if j == u:
                        continue
                    d_uav.append(float(np.linalg.norm(pos0[u] - pos0[j])))
                rows.append(
                    {
                        "pool": pool,
                        "scenario": int(seed),
                        "t": int(snap["t"]),
                        "uav": int(u),
                        "snap": snap,
                        "option": option,
                        "obs": obs,
                        "a_base": np.asarray(rec.a_base, dtype=np.float64).copy(),
                        "a_des": np.asarray(rec.a_des, dtype=np.float64).copy(),
                        "lock": np.asarray(rec.service_lock, dtype=bool).copy(),
                        "freeze": np.asarray(rec.base_zero_lock, dtype=bool).copy(),
                        "actor_comm": np.asarray(obs["communication"][u], dtype=np.float64).copy(),
                        "raw_sinr": float(comm_raw["sinr"][u]),
                        "raw_rate": float(comm_raw["rate"][u]),
                        "raw_interf": float(comm_raw["total_interference"][u]),
                        "raw_inr": float(comm_raw["interference_to_noise"][u]),
                        "team_interf": team_interf,
                        "d_nearest": float(np.min(d_uav) if d_uav else float("inf")),
                        "d_target": dist_tgt,
                    }
                )
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
    return rows


__all__ = ["collect_active_states"]
