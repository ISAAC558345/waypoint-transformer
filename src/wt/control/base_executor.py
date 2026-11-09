"""L0 base executor: frozen fly-toward + optional InRangeZeroMotionGate."""
from __future__ import annotations

import numpy as np

from wt.planning.task_option.contract import TaskOption
from wt.planning.task_option.rule_executor import RuleExecutorAdapter
from wt.legacy.htw_historical103.gate import InRangeZeroMotionGate
from wt.envs.queue.env import L3BEnv

from wt.planning.contracts import refuse_rematch


class BaseExecutor:
    """Movement-only. Never rematches Task Option."""

    def __init__(self, *, gated: bool = True) -> None:
        self.gated = bool(gated)
        self._rule = RuleExecutorAdapter()
        self._gate = InRangeZeroMotionGate()

    def act(self, env: L3BEnv, option: TaskOption) -> tuple[np.ndarray, np.ndarray]:
        refuse_rematch(option)
        raw = np.asarray(self._rule.act(env, option), dtype=np.float64)
        if self.gated:
            return self._gate.apply(env, option, raw)
        flags = np.zeros(int(env.config.num_uavs), dtype=bool)
        return raw.copy(), flags


__all__ = ["BaseExecutor"]