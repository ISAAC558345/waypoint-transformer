"""Attach frozen TW_old (service, progress, return) without retraining the predictor."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wt.round20.dataset import build_graph_state
from wt.round22.data import FrozenDeepSetsPredictor
from wt.round30.features import MAX_UAVS, UAV_DIM

OLD_TW_EXTRA = 11
OLD_CONDITIONED_UAV_DIM = UAV_DIM + OLD_TW_EXTRA

_PREDICTOR: FrozenDeepSetsPredictor | None = None


def frozen_old_predictor() -> FrozenDeepSetsPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = FrozenDeepSetsPredictor()
    return _PREDICTOR


def predict_tw_old(info: Mapping[str, Any], config: Any) -> np.ndarray:
    graph = build_graph_state(info, config)
    flat = {key: np.asarray(graph[key])[None, ...] for key in ("uav_nodes", "buoy_nodes", "uav_mask", "buoy_mask", "global_features")}
    return frozen_old_predictor().phi(flat)[0]


def attach_tw_old(features: Mapping[str, np.ndarray], phi: np.ndarray) -> dict[str, np.ndarray]:
    extra = np.asarray(phi, dtype=np.float32)
    if extra.shape != (MAX_UAVS, OLD_TW_EXTRA):
        raise ValueError("TW_old phi must have shape [M, 11]")
    uav = np.asarray(features["uav"], dtype=np.float32)
    if uav.ndim != 2 or uav.shape[-1] != UAV_DIM:
        raise ValueError("TW_old adapter expects unbatched Round-30 UAV features")
    extra = extra * np.asarray(features["uav_mask"], dtype=np.float32)[:, None]
    output = {key: np.asarray(value).copy() for key, value in features.items()}
    output["uav"] = np.concatenate((uav, extra), axis=-1)
    return output


__all__ = ["OLD_CONDITIONED_UAV_DIM", "OLD_TW_EXTRA", "attach_tw_old", "frozen_old_predictor", "predict_tw_old"]
