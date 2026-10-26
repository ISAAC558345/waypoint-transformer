"""Synthetic binary tags. One arrival event draws one tag."""
from __future__ import annotations

import numpy as np

from .protocol import TAG_RATE, TAG_SEED


def draw_tag(scenario_id: int, admission_step: int, buoy: int, tag_rate: float = TAG_RATE) -> int:
    """Return 0 or 1. The draw ignores served bits and future arrivals."""
    if not 0.0 <= float(tag_rate) <= 1.0:
        raise RuntimeError("tag rate is outside [0, 1]")
    sequence = np.random.SeedSequence([int(TAG_SEED), int(scenario_id), int(admission_step), int(buoy)])
    sample = float(np.random.default_rng(sequence).random())
    return int(sample < float(tag_rate))
