"""In-range zero-motion gate. Reads pre-action state only; does not rematch."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv


class InRangeZeroMotionGate:
    """Replace movement with (0,0) iff active, valid, nonempty-queue, in-range."""

    def flags(self, env: L3BEnv, option: TaskOption) -> np.ndarray:
        if not np.array_equal(option.association, option.targets):
            raise RuntimeError("gate refused rematching: association != targets")
        m_count = int(env.config.num_uavs)
        k_count = int(env.config.num_buoys)
        targets = np.asarray(option.targets, dtype=np.int64)
        gated = np.zeros(m_count, dtype=bool)
        for uav in range(m_count):
            tgt = int(targets[uav]) if uav < targets.shape[0] else -1
            if tgt < 0 or tgt >= k_count:
                continue
            if float(env.queue[tgt]) <= 1.0e-9:
                continue
            if not env._in_range(uav, tgt):
                continue
            gated[uav] = True
        return gated

    def apply(self, env: L3BEnv, option: TaskOption, movement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        move = np.asarray(movement, dtype=np.float64).reshape(int(env.config.num_uavs), 2).copy()
        gated = self.flags(env, option)
        move[gated] = 0.0
        return move, gated


__all__ = ["InRangeZeroMotionGate"]
