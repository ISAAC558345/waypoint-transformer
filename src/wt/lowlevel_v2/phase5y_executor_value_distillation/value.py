"""Fresh non-additive value head. The frozen encoder is not part of this module."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from wt.lowlevel_v2.phase5u_representation_repair.scorer import RelationalSetScorer
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import FINAL_REL_DIM

from .protocol import HUBER_DELTA


class ExecutorConditionedJointValue(RelationalSetScorer):
    """Assignment-level scalar. Parameters start from the module's random initialization."""

    def __init__(self) -> None:
        super().__init__(rel_dim=FINAL_REL_DIM)


def state_huber(prediction: Tensor, target_blocks: Tensor) -> Tensor:
    """Mean Huber inside one state. States are averaged by the training loop."""
    if prediction.shape != target_blocks.shape:
        raise ValueError("prediction and J15 blocks must describe the same assignments")
    return F.huber_loss(prediction, target_blocks, reduction="mean", delta=float(HUBER_DELTA))


__all__ = ["ExecutorConditionedJointValue", "state_huber"]
