"""Static HGNN feature inventory. Does not edit the encoder."""
from __future__ import annotations

import numpy as np

from wt.htw_phase9.const import BUOY_DIM, EDGE_UB_DIM, EDGE_UU_DIM, UAV_DIM
from wt.htw_phase9.features import feature_dict_l3b
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.phase5_hgnn_urgency.encoder import Phase5HGNNEncoder


def hgnn_feature_table() -> list[dict[str, str]]:
    encoder = Phase5HGNNEncoder(buoy_dim=BUOY_DIM)
    env = L3BEnv()
    info = env.reset(seed=310100)
    feats = feature_dict_l3b(info, env, last_arrival=np.zeros(env.config.num_buoys, dtype=np.float32), last_positions=env.uav_positions.astype(np.float32))
    energy_is_zero = bool(np.all(feats["uav"][:, 7] == 0.0))
    has_uu = hasattr(encoder, "uu_msg") and hasattr(encoder, "uu_enc")
    has_ub = hasattr(encoder, "ub_msg") and hasattr(encoder, "edge_enc")
    has_bb = hasattr(encoder, "bb_msg")
    if has_bb:
        raise RuntimeError("diagnosis found an unexpected buoy-buoy module")
    if int(EDGE_UU_DIM) != 2 or int(EDGE_UB_DIM) != 3 or int(UAV_DIM) != 8 or int(BUOY_DIM) != 7:
        raise RuntimeError("phase-9 feature widths changed under the diagnosis")
    return [
        {
            "feature": "UAV-UAV edge",
            "current": "yes" if has_uu else "no",
            "required": "yes",
            "note": "Distance and current interference power ratio. No buoy-buoy edge.",
        },
        {
            "feature": "UAV-Buoy edge",
            "current": "yes" if has_ub else "no",
            "required": "yes",
            "note": "Distance, hypothetical rate, and ETA.",
        },
        {
            "feature": "interference feature",
            "current": "yes, UAV-UAV channel",
            "required": "yes",
            "note": "Taken from active links. It is zero when no UAV is in service.",
        },
        {
            "feature": "channel quality",
            "current": "yes, rate_if_assigned on UAV-Buoy",
            "required": "yes",
            "note": "Current hypothetical rate, not a future channel.",
        },
        {
            "feature": "energy cost",
            "current": "slot exists but is filled with 0" if energy_is_zero else "nonzero energy channel",
            "required": "not available in L3-B",
            "note": "The environment does not expose energy. The diagnosis does not invent one.",
        },
        {
            "feature": "queue state",
            "current": "yes, buoy queue/capacity and assigned-queue on the UAV",
            "required": "yes",
            "note": "Current queue only.",
        },
        {
            "feature": "urgency",
            "current": "queue ratio appended only on the Phase-5 full checkpoint; measured assignment did not change when it was zeroed",
            "required": "a shared score that can change the assignment",
            "note": "Plain HGNN checkpoint has no urgency channel.",
        },
    ]


__all__ = ["hgnn_feature_table"]
