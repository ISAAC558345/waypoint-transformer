"""Frozen-assignment Residual worker. Does not train. Does not rematch during H=15."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.high_level import make_option
from wt.lowlevel_v2.phase1b.controller import ResidualWorker
from wt.lowlevel_v2.phase2b.train import TrustWorker

from .protocol import HOLD_STEPS, refuse_compat


class CompatWorker(TrustWorker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frozen_assignment = None

    def reset(self, seed: int):
        refuse_compat(int(seed))
        self.frozen_assignment = None
        return ResidualWorker.reset(self, int(seed))

    def bind_assignment(self, assignment: np.ndarray, info: dict) -> None:
        chosen = np.asarray(assignment, dtype=np.int64).copy()
        self.frozen_assignment = chosen
        self.pending = int(HOLD_STEPS)
        self.option = make_option(self.env, info, chosen, self.pending)

    def _refresh_option(self, info: dict) -> None:
        if self.frozen_assignment is None:
            super()._refresh_option(info)
            return
        if self.pending <= 0:
            self.pending = int(HOLD_STEPS)
            self.option = make_option(self.env, info, self.frozen_assignment, self.pending)
        assert self.option is not None
        self.option.remaining_hold = int(self.pending)
        self.option.association = np.asarray(self.frozen_assignment, dtype=np.int64).copy()
        self.option.targets = np.asarray(self.frozen_assignment, dtype=np.int64).copy()


__all__ = ["CompatWorker"]
