"""Match the loaded actor's current outputs to a small turn and the old speed."""
from __future__ import annotations

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment
from wt.benchmark_v2.phase6il_direct_action.action import residual_command
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL

from .action import adaptive_command, heading_delta_deg
from .protocol import CALIBRATION_SCENARIOS, INIT_TURN_TARGET_DEG, MAX_TURN_DEG
from .worker import apply_command


def _median_abs_turn(pre0: np.ndarray, center: float, temperature: float, max_turn: float) -> float:
    turn = np.tanh((pre0 - float(center)) / float(temperature)) * float(max_turn)
    return float(np.median(np.abs(turn)))


def _temperature(pre0: np.ndarray, center: float) -> float:
    if _median_abs_turn(pre0, center, 1.0, MAX_TURN_DEG) <= float(INIT_TURN_TARGET_DEG):
        return 1.0
    low, high = 1.0, 30.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if _median_abs_turn(pre0, center, mid, MAX_TURN_DEG) > float(INIT_TURN_TARGET_DEG):
            low = mid
        else:
            high = mid
    return float(high)


def _speed_bias(pre1: np.ndarray, target_speed: float) -> float:
    target = float(np.clip(target_speed, 0.02, 0.98))
    unit = float(np.clip(2.0 * target - 1.0, -0.999, 0.999))
    return float(np.arctanh(unit) - np.median(pre1))


def collect_loaded_outputs(actor) -> dict[str, np.ndarray]:
    """Roll the frozen residual policy and record the pre-activation it already emits."""
    actor.eval()
    pre_rows = []
    speed_rows = []
    heading_rows = []
    for scenario in CALIBRATION_SCENARIOS:
        env = L3BBitQueueEnvV2()
        env.reset(seed=int(scenario))
        rng = np.random.default_rng(7000 + int(scenario))
        option = None
        pending = 0
        last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
        while int(env.t) < int(env.config.episode_horizon):
            if pending <= 0:
                option = make_option(env, sample_assignment(env, rng))
                pending = 15
            association = np.asarray(option.association, dtype=np.int64).copy()
            obs = build_actor_observation(env, option, last)
            with torch.no_grad():
                mu, _log_std = actor(
                    torch.as_tensor(obs["self"], dtype=torch.float32),
                    torch.as_tensor(obs["task"], dtype=torch.float32),
                    torch.as_tensor(obs["communication"], dtype=torch.float32),
                    torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                    torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
                )
            pre = mu.detach().cpu().numpy()
            base = fly_toward(env, association)
            command, _delta = residual_command(env, association, pre)
            executed, _report = apply_command(env, option, command)
            active = np.linalg.norm(base, axis=-1) > 1.0e-8
            if np.any(active):
                pre_rows.append(pre[active])
                speed_rows.append(np.linalg.norm(command[active], axis=-1))
                heading_rows.append(heading_delta_deg(command, base)[active])
            env.step(executed, association)
            pending -= 1
            option.remaining_hold = int(pending)
            last = executed.copy()
    pre_all = np.concatenate(pre_rows, axis=0)
    return {
        "pre": pre_all,
        "old_speed": np.concatenate(speed_rows, axis=0),
        "old_heading_deg": np.concatenate(heading_rows, axis=0),
        "alpha_residual": np.asarray([ALPHA_RESIDUAL], dtype=np.float64),
    }


def build_calibration(actor) -> dict[str, float]:
    sample = collect_loaded_outputs(actor)
    pre = sample["pre"]
    center = float(np.median(pre[:, 0]))
    temperature = _temperature(pre[:, 0], center)
    target_speed = float(np.median(sample["old_speed"]))
    bias = _speed_bias(pre[:, 1], target_speed)
    mapped = adaptive_command(
        np.tile(np.array([[1.0, 0.0]]), (len(pre), 1)),
        pre,
        steer_center=center,
        steer_temperature=temperature,
        speed_bias=bias,
    )
    turn = np.tanh((pre[:, 0] - center) / temperature) * float(MAX_TURN_DEG)
    speed = np.linalg.norm(mapped, axis=-1)
    return {
        "steer_center": center,
        "steer_temperature": float(temperature),
        "speed_bias": float(bias),
        "max_turn_deg": float(MAX_TURN_DEG),
        "max_speed_fraction": 1.0,
        "official_config": False,
        "samples": int(len(pre)),
        "old_pre0_median": center,
        "old_pre0_std": float(np.std(pre[:, 0])),
        "old_pre1_median": float(np.median(pre[:, 1])),
        "old_pre1_std": float(np.std(pre[:, 1])),
        "old_speed_median": target_speed,
        "old_heading_abs_median": float(np.median(np.abs(sample["old_heading_deg"]))),
        "init_turn_abs_median": float(np.median(np.abs(turn))),
        "init_speed_median": float(np.median(speed)),
        "init_turn_abs_p90": float(np.percentile(np.abs(turn), 90)),
    }


__all__ = ["build_calibration", "collect_loaded_outputs"]
