"""Independent ordinary scenario registry. Freeze before any model evaluation."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from wt.l3b.config import L3BConfig
from wt.l3b.env import L3BEnv

from .protocol import (
    ENV_FILE,
    HOLD,
    N_SCENARIOS,
    OUTPUT,
    SCENARIO_BASE,
    all_blocked_scenario_seeds,
    dump_frozen,
    sha256_bytes,
)


def select_independent_seeds(n: int = N_SCENARIOS, base: int = SCENARIO_BASE) -> list[int]:
    blocked = all_blocked_scenario_seeds()
    seeds: list[int] = []
    cursor = int(base)
    while len(seeds) < int(n):
        if int(cursor) not in blocked:
            seeds.append(int(cursor))
        cursor += 1
    overlap = set(seeds) & blocked
    if overlap:
        raise RuntimeError(f"independent seeds overlap history: {sorted(overlap)[:20]}")
    return seeds


def build_scenario_manifest(*, consumed: bool = False) -> dict[str, Any]:
    seeds = select_independent_seeds()
    cfg = L3BConfig()
    env = L3BEnv()
    info = env.reset(seed=int(seeds[0]))
    return {
        "label": "Phase-10.4 independent ordinary validation, not final untouched test",
        "consumed": bool(consumed),
        "n_scenarios": len(seeds),
        "scenario_seeds": seeds,
        "generation_rule": f"first {len(seeds)} unused integers starting at {SCENARIO_BASE}, excluding historical blocked/manifest seeds",
        "blocked_protocol_count": len(all_blocked_scenario_seeds()),
        "env_sha256": sha256_bytes(ENV_FILE.read_bytes()),
        "config_sha256": sha256_bytes(ENV_FILE.parent.joinpath("config.py").read_bytes()),
        "l3b_config": asdict(cfg) | {"step_length_m": cfg.step_length_m},
        "hold": HOLD,
        "horizon_T": int(cfg.episode_horizon),
        "num_uavs_M": int(cfg.num_uavs),
        "num_buoys_K": int(cfg.num_buoys),
        "service_radius_m": float(cfg.service_radius_m),
        "arrival": {
            "queue_capacity_blocks": int(cfg.queue_capacity_blocks),
            "num_hot": int(cfg.num_hot),
            "reset_seed_drives_arrivals": True,
            "probe_reset_keys": sorted(info.keys()),
        },
        "sinr": {
            "bandwidth_hz": float(cfg.bandwidth_hz),
            "tx_power": float(cfg.tx_power),
            "noise_power": float(cfg.noise_power),
            "path_loss_exp": float(cfg.path_loss_exp),
            "kappa_same_cluster": float(cfg.kappa_same_cluster),
            "kappa_cross_cluster": float(cfg.kappa_cross_cluster),
        },
        "not_confirmation": True,
        "not_protected": True,
        "not_untouched": True,
        "not_design_8_15": True,
        "not_train_0_7": True,
    }


def freeze_scenario_manifest() -> dict[str, Any]:
    path = OUTPUT / "phase104_independent_scenario_manifest.json"
    if path.is_file():
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        if payload.get("scenario_seeds"):
            return payload
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = build_scenario_manifest(consumed=False)
    dump_frozen(path, payload)
    return payload


__all__ = ["build_scenario_manifest", "freeze_scenario_manifest", "select_independent_seeds"]
