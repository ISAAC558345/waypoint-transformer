"""Rule executor adapter. Reuses L3-B fly-toward. Never rematches."""
from __future__ import annotations

import numpy as np

from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward

from .contract import TaskOption


class RuleExecutorAdapter:
    """Movement-only adapter around existing fly-toward."""

    name = "rule_fly_toward"

    def act(self, env: L3BEnv, option: TaskOption) -> np.ndarray:
        association = np.asarray(option.association, dtype=np.int64)
        if association.shape != (int(env.config.num_uavs),):
            raise ValueError("association rank does not match L3-B num_uavs")
        if not np.array_equal(association, np.asarray(option.targets, dtype=np.int64)):
            raise RuntimeError("adapter refused rematching: association != targets")
        return fly_toward(env, association)


__all__ = ["RuleExecutorAdapter"]
