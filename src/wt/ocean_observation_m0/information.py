"""Shore fusion, command versions, and the four required timelines."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from wt.ocean_observation_m0.codec import decode_product, encode_product
from wt.ocean_observation_m0.parameters import (
    CURRENT_FROM_EAST_RAD,
    CURRENT_M_S,
    EXAMPLE_SEED,
    HS_FIELD_AMP_M,
    HS_NOISE_STD_M,
    HS_PRIOR_MEAN_M,
    HS_PRIOR_VAR_M2,
    PRODUCT_VALIDITY_S,
    SAMPLE_INTERVAL_S,
    STOKES_M_S,
    WARMUP_START_S,
    WINDAGE_FRACTION,
)


def hs_true_m(x_m: float, y_m: float) -> float:
    return HS_PRIOR_MEAN_M + HS_FIELD_AMP_M * math.sin(2.0 * math.pi * x_m / 4000.0) * math.cos(
        2.0 * math.pi * y_m / 4000.0
    )


def drift_velocity_m_s(wind_m_s: Tuple[float, float]) -> Tuple[float, float]:
    current_x = CURRENT_M_S * math.cos(CURRENT_FROM_EAST_RAD)
    current_y = CURRENT_M_S * math.sin(CURRENT_FROM_EAST_RAD)
    speed = math.hypot(wind_m_s[0], wind_m_s[1])
    if speed == 0.0:
        return (current_x, current_y)
    ux = wind_m_s[0] / speed
    uy = wind_m_s[1] / speed
    stokes_x = STOKES_M_S * ux
    stokes_y = STOKES_M_S * uy
    windage_x = WINDAGE_FRACTION * speed * ux
    windage_y = WINDAGE_FRACTION * speed * uy
    return (current_x + stokes_x + windage_x, current_y + stokes_y + windage_y)


def advance(point: Tuple[float, float], wind_m_s: Tuple[float, float], dt_s: float) -> Tuple[float, float]:
    vx, vy = drift_velocity_m_s(wind_m_s)
    return (point[0] + vx * dt_s, point[1] + vy * dt_s)


class ShoreBelief:
    """The map changes only for a complete, checksum-valid, new, unexpired product."""

    def __init__(self) -> None:
        self.mean_m = HS_PRIOR_MEAN_M
        self.variance_m2 = HS_PRIOR_VAR_M2
        self.fused = {}
        self.events: List[dict] = []
        self.information_updates = 0

    def receive(self, blob: bytes, receive_time_s: float, complete: bool = True) -> dict:
        if not complete:
            event = {"status": "partial_no_fusion", "receive_time_s": receive_time_s, "map_updated": False}
            self.events.append(event)
            return event
        decoded = decode_product(blob)
        if not decoded.get("ok"):
            event = {
                "status": decoded.get("reason", "rejected"),
                "receive_time_s": receive_time_s,
                "map_updated": False,
            }
            self.events.append(event)
            return event
        window_end = float(decoded["window_end_s"])
        key = (int(decoded["buoy_id"]), int(decoded["seq"]))
        age = receive_time_s - window_end
        if age > PRODUCT_VALIDITY_S:
            event = {
                "status": "stale_recorded_not_fused",
                "receive_time_s": receive_time_s,
                "window_end_s": window_end,
                "age_s": age,
                "map_updated": False,
                "sampling_time_rewritten": False,
            }
            self.events.append(event)
            return event
        if key in self.fused:
            event = {
                "status": "duplicate_no_fusion",
                "receive_time_s": receive_time_s,
                "key": list(key),
                "map_updated": False,
            }
            self.events.append(event)
            return event
        meas = float(decoded["hs_m"])
        meas_var = float(decoded["measurement_variance_m2"])
        if meas_var <= 0.0:
            event = {"status": "bad_variance", "map_updated": False}
            self.events.append(event)
            return event
        prior_var = self.variance_m2
        prior_mean = self.mean_m
        post_var = 1.0 / (1.0 / prior_var + 1.0 / meas_var)
        post_mean = post_var * (prior_mean / prior_var + meas / meas_var)
        self.mean_m = post_mean
        self.variance_m2 = post_var
        self.information_updates += 1
        self.fused[key] = {
            "hs_m": meas,
            "measurement_variance_m2": meas_var,
            "window_start_s": decoded["window_start_s"],
            "window_end_s": window_end,
            "receive_time_s": receive_time_s,
            "mean_x_m": decoded["mean_x_m"],
            "mean_y_m": decoded["mean_y_m"],
        }
        event = {
            "status": "fused",
            "receive_time_s": receive_time_s,
            "window_end_s": window_end,
            "sampling_time_rewritten": False,
            "map_updated": True,
            "mean_before_m": prior_mean,
            "mean_after_m": post_mean,
            "variance_before_m2": prior_var,
            "variance_after_m2": post_var,
        }
        self.events.append(event)
        return event


def command_outcome(delivered_to_uav: bool, ack_received: bool, now_s: float, valid_until_s: float) -> dict:
    if not delivered_to_uav:
        uav = "keeps_last_activated_task"
        shore = "pending"
        assumed_unexecuted = False
    elif now_s > valid_until_s:
        uav = "not_activated_expired"
        shore = "pending" if not ack_received else "confirmed_but_uav_rejected_expiry"
        assumed_unexecuted = False
    else:
        uav = "activated"
        if ack_received:
            shore = "confirmed"
        else:
            shore = "pending_not_assumed_unexecuted"
        assumed_unexecuted = False
    return {
        "uav": uav,
        "shore": shore,
        "assumed_unexecuted": assumed_unexecuted,
        "shore_knows_execution": shore == "confirmed",
    }


def worked_product(position_at_t0: Tuple[float, float], wind_m_s: Tuple[float, float], buoy_id: int = 1) -> dict:
    """Window mean along the drift track. The shore does not see it until receive()."""
    vx, vy = drift_velocity_m_s(wind_m_s)
    times = []
    t = WARMUP_START_S
    while t <= 1e-9:
        times.append(t)
        t += SAMPLE_INTERVAL_S
    rng = random.Random(EXAMPLE_SEED)
    samples = []
    xs = []
    ys = []
    for sample_t in times:
        x = position_at_t0[0] + vx * sample_t
        y = position_at_t0[1] + vy * sample_t
        noisy = hs_true_m(x, y) + rng.gauss(0.0, HS_NOISE_STD_M)
        samples.append(noisy)
        xs.append(x)
        ys.append(y)
    n = len(samples)
    mean = sum(samples) / n
    variance = (HS_NOISE_STD_M ** 2) / n
    blob = encode_product(
        buoy_id,
        1,
        times[0],
        times[-1],
        sum(xs) / n,
        sum(ys) / n,
        mean,
        variance,
        0,
        n,
    )
    belief = ShoreBelief()
    before = {"mean_m": belief.mean_m, "variance_m2": belief.variance_m2}
    fused = belief.receive(blob, receive_time_s=30.0)
    return {
        "n_samples": n,
        "window_start_s": times[0],
        "window_end_s": times[-1],
        "mean_hs_m": mean,
        "measurement_variance_m2": variance,
        "mean_x_m": sum(xs) / n,
        "mean_y_m": sum(ys) / n,
        "framed_bytes": len(blob),
        "belief_before": before,
        "fusion": fused,
        "belief_after": {"mean_m": belief.mean_m, "variance_m2": belief.variance_m2},
        "readable_before_receive": False,
        "seed": EXAMPLE_SEED,
    }


def four_timelines(product_blob: bytes) -> Dict[str, List[dict]]:
    normal = ShoreBelief()
    normal_events = [
        {"t_s": 0.0, "actor": "浮标", "fact": "窗口结束，产品完成编码", "map_updated": False},
        {"t_s": 30.0, "actor": "无人机", "fact": "完整包进入机载缓存，岸站仍是先验", "map_updated": False},
    ]
    delivered = normal.receive(product_blob, 40.0)
    normal_events.append({"t_s": 40.0, "actor": "岸站", "fact": "完整包校验通过并融合", "map_updated": delivered["map_updated"]})
    duplicate = normal.receive(product_blob, 50.0)
    normal_events.append(
        {"t_s": 50.0, "actor": "岸站", "fact": "同一产品再次到达，不重复融合", "map_updated": duplicate["map_updated"]}
    )
    if delivered["status"] != "fused" or duplicate["status"] != "duplicate_no_fusion":
        raise RuntimeError("timeline statuses drifted")

    outage = [
        {"t_s": 0.0, "actor": "浮标", "fact": "产品已编码", "map_updated": False},
        {"t_s": 20.0, "actor": "无人机", "fact": "包留在机载缓存", "map_updated": False},
        {"t_s": 20.0, "actor": "岸站", "fact": "回传窗口关闭，地图保持先验", "map_updated": False},
        {"t_s": 560.0, "actor": "无人机", "fact": "仍执行最后一份已激活任务：返航并交付", "map_updated": False},
        {"t_s": 560.0, "actor": "岸站", "fact": "断链期间无人机状态变旧，海况尚未融合", "map_updated": False},
    ]

    lost = command_outcome(False, False, 0.0, 1800.0)
    lost_events = [
        {"t_s": 0.0, "actor": "岸站", "fact": "命令版本已发出", "shore": "pending", "map_updated": False},
        {"t_s": 5.0, "actor": "无人机", "fact": "没有收到命令，保持上一份已激活任务", "assumed_unexecuted": lost["assumed_unexecuted"]},
        {"t_s": 5.0, "actor": "岸站", "fact": "没有确认；状态保持待确认，不能假定任务没有执行", "shore": lost["shore"]},
    ]

    unacked = command_outcome(True, False, 0.0, 1800.0)
    unacked_events = [
        {"t_s": 0.0, "actor": "无人机", "fact": "命令已送达并激活", "assumed_unexecuted": False},
        {"t_s": 1.0, "actor": "岸站", "fact": "确认丢失，状态保持待确认，不把任务当成未执行", "shore_knows_execution": False},
    ]
    return {
        "normal_then_duplicate": normal_events,
        "backhaul_outage": outage,
        "lost_command": lost_events,
        "unacked_command": unacked_events,
    }
