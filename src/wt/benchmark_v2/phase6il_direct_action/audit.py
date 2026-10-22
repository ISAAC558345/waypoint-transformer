"""Measure the frozen residual reachable set and the checkpoint's restore path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6g_lowlevel_mappo.train import probe_initialization
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_actor, load_primary
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.observation import build_actor_observation
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL, LOG_STD_INIT

from .action import residual_command, residual_limits
from .protocol import AUDIT_OUTPUT, SOURCE_ACTOR, assert_phase_output
from .train import source_sha256
from .worker import execute_command


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1.0e-8 or right_norm < 1.0e-8:
        return float("nan")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def rollout_frozen(scenarios: tuple[int, ...] = (321000, 321001)) -> dict:
    actor = load_primary()
    rows = []
    for scenario in scenarios:
        env = L3BBitQueueEnvV2()
        env.reset(seed=int(scenario))
        rng = np.random.default_rng(900 + int(scenario))
        option = None
        pending = 0
        last = np.zeros((6, 2), dtype=np.float64)
        for _step in range(int(env.config.episode_horizon)):
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
            command, delta = residual_command(env, association, pre)
            executed, safety = execute_command(env, option, command)
            for uav, buoy in enumerate(association):
                if int(buoy) < 0:
                    continue
                dist = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
                rows.append(
                    {
                        "distance_m": dist,
                        "in_radius": int(dist <= float(env.config.service_radius_m)),
                        "pre_abs": float(np.mean(np.abs(pre[uav]))),
                        "delta_norm": float(np.linalg.norm(delta[uav])),
                        "command_speed": float(np.linalg.norm(command[uav])),
                        "heading_deg": _angle_deg(command[uav], base[uav]),
                        "executed_heading_deg": _angle_deg(executed[uav], base[uav]),
                        "intervention_norm": float(safety["intervention_norm"]),
                    }
                )
            env.step(executed, association)
            pending -= 1
            option.remaining_hold = int(pending)
            last = executed.copy()
    table = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}
    inside = table["in_radius"] > 0.5
    outside = ~inside

    def pack(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {}
        return {
            "count": int(np.sum(mask)),
            "pre_abs": float(np.nanmean(table["pre_abs"][mask])),
            "delta_norm": float(np.nanmean(table["delta_norm"][mask])),
            "command_speed": float(np.nanmean(table["command_speed"][mask])),
            "heading_deg": float(np.nanmean(table["heading_deg"][mask])),
            "heading_deg_max": float(np.nanmax(table["heading_deg"][mask])),
            "executed_heading_deg": float(np.nanmean(table["executed_heading_deg"][mask])),
            "executed_heading_deg_max": float(np.nanmax(table["executed_heading_deg"][mask])),
            "intervention_norm": float(np.nanmean(table["intervention_norm"][mask])),
        }

    return {
        "scenarios": [int(item) for item in scenarios],
        "samples": int(len(rows)),
        "inside_radius": pack(inside),
        "outside_radius": pack(outside),
        "all": pack(np.ones(len(rows), dtype=bool)),
    }


def restore_audit() -> dict:
    primary = load_primary()
    restored = load_actor(0)
    same = all(torch.equal(restored.state_dict()[key], primary.state_dict()[key]) for key in restored.state_dict())
    payload = torch.load(SOURCE_ACTOR, map_location="cpu", weights_only=False)
    return {
        "inference_restores_actor_strict": bool(same),
        "checkpoint_contains_critic": bool(isinstance(payload, dict) and "critic" in payload),
        "inference_uses_critic": False,
        "inference_uses_optimizer": False,
        "phase6g_train_resumes_phase6g_checkpoint": False,
        "phase6g_train_actor_init": probe_initialization(),
        "phase6g_observation_normalization": False,
        "log_std_init": float(LOG_STD_INIT),
        "alpha_residual": float(ALPHA_RESIDUAL),
        "source_sha256": source_sha256(),
        "source_actor": str(SOURCE_ACTOR),
    }


def write_audit() -> dict:
    payload = {"limits": residual_limits(), "frozen_rollout": rollout_frozen(), "restore": restore_audit()}
    destination = assert_phase_output(AUDIT_OUTPUT / "residual_action_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = ["restore_audit", "rollout_frozen", "write_audit"]
