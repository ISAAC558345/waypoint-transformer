"""First-hold override worker. Does not train. Later holds stay official greedy."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.high_level import make_option
from wt.lowlevel_v2.integration_phase1.protocol import HOLD_STEPS
from wt.lowlevel_v2.integration_phase1.worker import CompatWorker


class FirstHoldOverrideWorker(CompatWorker):
    def __init__(self, *args, first_assignment=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._first_override = None if first_assignment is None else np.asarray(first_assignment, dtype=np.int64).copy()
        self._override_used = False

    def set_first_assignment(self, assignment: np.ndarray | None) -> None:
        self._first_override = None if assignment is None else np.asarray(assignment, dtype=np.int64).copy()
        self._override_used = False

    def reset(self, seed: int):
        self._override_used = False
        return super().reset(int(seed))

    def _refresh_option(self, info: dict) -> None:
        if (
            self._first_override is not None
            and not self._override_used
            and self.frozen_assignment is None
            and int(self.pending) <= 0
        ):
            chosen = np.asarray(self._first_override, dtype=np.int64).copy()
            self._override_used = True
            self.pending = int(HOLD_STEPS)
            self.option = make_option(self.env, info, chosen, self.pending)
            self.option.remaining_hold = int(self.pending)
            return
        super()._refresh_option(info)


__all__ = ["FirstHoldOverrideWorker"]
