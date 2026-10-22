"""Train low-level PPO seeds 1-4 with the frozen Phase-10.1 recipe. Seed 0 is copied."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from wt.round30.ppo import PPOAgent, PPOConfig, load_agent, seed_everything
from wt.round30.runner import collect

from wt.htw_phase101.features import build_schema
from wt.htw_phase101.high_level import high_grad_count, load_frozen_high, parameter_sha, refuse_m5
from wt.htw_phase101.models import L3BGoalActorCritic
from wt.htw_phase101.protocol import CHECKPOINT_EVERY, DESIGN_TRAIN_SEEDS, HIGH_CKPT
from wt.htw_phase101.vector import L3BLowLevelVector

from .protocol import OUTPUT, PPO, SEED0_FINAL, UPDATES, VECTOR_SEED_PILOT, sha256_bytes


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def load_low(path: Path, schema: dict[str, Any], config: PPOConfig | None = None) -> PPOAgent:
    refuse_m5(path)
    payload = torch.load(path, map_location="cpu")
    extra = payload.get("extra", {})
    if int(payload.get("model_shape", {}).get("max_uavs", -1)) == 5 or extra.get("max_uavs") == 5:
        raise RuntimeError("checkpoint declares max_uavs=5")
    model = L3BGoalActorCritic(schema)
    return load_agent(payload, config or PPO, model)


def _save_ckpt(agent: PPOAgent, path: Path, seed: int, extra: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    refuse_m5(path)
    torch.save(agent.checkpoint(seed=int(seed), steps=int(extra.get("environment_steps", 0)), extra=dict(extra)), path)
    return sha256_bytes(path.read_bytes())


def train_one_seed(low_seed: int, high, high_file_sha: str, high_param_sha: str, schema: dict[str, Any]) -> dict[str, Any]:
    ckpt_dir = OUTPUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / f"seed{int(low_seed)}_final.pt"
    hist_path = OUTPUT / f"phase104_training_history_seed{int(low_seed)}.csv"
    if int(low_seed) == 0:
        if not SEED0_FINAL.is_file():
            raise FileNotFoundError(str(SEED0_FINAL))
        shutil.copy2(SEED0_FINAL, final_path)
        return {
            "low_seed": 0,
            "retrained": False,
            "copied_from": str(SEED0_FINAL),
            "final_path": str(final_path),
            "sha256": sha256_bytes(final_path.read_bytes()),
            "source_sha256": sha256_bytes(SEED0_FINAL.read_bytes()),
            "updates": UPDATES,
            "environment_steps": UPDATES * PPO.rollout_length * PPO.num_envs,
            "history": [],
            "runtime": [],
            "checkpoints": [{"tag": "final", "update": UPDATES, "sha256": sha256_bytes(final_path.read_bytes()), "path": str(final_path)}],
        }
    if final_path.is_file() and hist_path.is_file() and sum(1 for _ in hist_path.open(encoding="utf-8")) > UPDATES:
        return {
            "low_seed": int(low_seed),
            "retrained": False,
            "resumed": True,
            "final_path": str(final_path),
            "sha256": sha256_bytes(final_path.read_bytes()),
            "updates": UPDATES,
            "environment_steps": UPDATES * PPO.rollout_length * PPO.num_envs,
            "history": list(csv.DictReader(hist_path.open(encoding="utf-8"))),
            "runtime": [],
            "checkpoints": [{"tag": "final", "update": UPDATES, "sha256": sha256_bytes(final_path.read_bytes()), "path": str(final_path)}],
        }
    seed_everything(int(low_seed))
    agent = PPOAgent(L3BGoalActorCritic(schema), PPO)
    low_ids = {id(p) for p in agent.model.parameters()}
    high_ids = {id(p) for p in high.parameters()}
    opt_ids = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    if opt_ids & high_ids or not opt_ids <= low_ids:
        raise RuntimeError("low-level optimizer would touch high-level parameters")
    vector = L3BLowLevelVector(DESIGN_TRAIN_SEEDS, PPO.num_envs, VECTOR_SEED_PILOT, high_model=high, schema=schema)
    history: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    ckpts: list[dict[str, Any]] = []
    init_path = ckpt_dir / f"seed{int(low_seed)}_init.pt"
    ckpts.append({"tag": "init", "update": 0, "sha256": _save_ckpt(agent, init_path, low_seed, {"environment_steps": 0, "max_uavs": 6, "low_seed": int(low_seed)}), "path": str(init_path)})
    t_train0 = perf_counter()
    for update in range(1, int(UPDATES) + 1):
        t0 = perf_counter()
        vector.reward_parts = {"r_env": [], "r_aux": [], "r_train": []}
        storage, events, health = collect(agent, vector)
        if storage["pre_tanh"].shape[-2:] != (int(schema["max_uavs"]), 2):
            raise RuntimeError("sampled action is not (6,2)")
        stats = agent.update(storage)
        elapsed = perf_counter() - t0
        if sha256_bytes(HIGH_CKPT.read_bytes()) != high_file_sha or parameter_sha(high) != high_param_sha or high_grad_count(high) != 0:
            raise RuntimeError("high-level changed during low-level training")
        row = {
            "low_seed": int(low_seed),
            "label": f"seed{int(low_seed)}",
            "update": update,
            "environment_steps": vector.environment_steps,
            "seconds": elapsed,
            "completed_episodes": len(events),
            "mean_completed_payload_bits": float(np.mean([item["payload_bits"] for item in events])) if events else float("nan"),
            "mean_completed_env_return": float(np.mean([item["return"] for item in events])) if events else float("nan"),
            **vector.parts_mean(),
            **health,
            **{key: float(value) for key, value in stats.items()},
        }
        if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise RuntimeError(f"seed {low_seed} produced NaN/Inf at update {update}")
        history.append(row)
        runtime.append({"low_seed": int(low_seed), "update": update, "seconds": elapsed, "environment_steps": vector.environment_steps})
        if update % int(CHECKPOINT_EVERY) == 0 or update == int(UPDATES):
            path = ckpt_dir / f"seed{int(low_seed)}_update{update:03d}.pt"
            ckpts.append({"tag": f"update_{update}", "update": update, "sha256": _save_ckpt(agent, path, low_seed, {"environment_steps": vector.environment_steps, "max_uavs": 6, "low_seed": int(low_seed)}), "path": str(path)})
        print(f"seed{low_seed} update {update}/{UPDATES} steps={vector.environment_steps} sec={elapsed:.3f} payload={row['mean_completed_payload_bits']}", flush=True)
    _save_ckpt(agent, final_path, low_seed, {"environment_steps": vector.environment_steps, "max_uavs": 6, "low_seed": int(low_seed), "primary": True})
    ckpts.append({"tag": "final", "update": int(UPDATES), "sha256": sha256_bytes(final_path.read_bytes()), "path": str(final_path)})
    _write_csv(hist_path, history)
    return {
        "low_seed": int(low_seed),
        "retrained": True,
        "final_path": str(final_path),
        "sha256": sha256_bytes(final_path.read_bytes()),
        "updates": UPDATES,
        "environment_steps": vector.environment_steps,
        "wall_clock_s": perf_counter() - t_train0,
        "history": history,
        "runtime": runtime,
        "checkpoints": ckpts,
        "vector_seed": VECTOR_SEED_PILOT,
        "train_scenarios": list(DESIGN_TRAIN_SEEDS),
    }


def attach_seed0_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hist = Path(__file__).resolve().parents[1] / "htw_phase101" / "outputs" / "pilot" / "phase101_training_history.csv"
    if not hist.is_file():
        return rows
    copied = []
    for row in csv.DictReader(hist.open(encoding="utf-8")):
        item = dict(row)
        item["low_seed"] = 0
        item["copied_from_phase101"] = True
        copied.append(item)
    return copied + rows


__all__ = ["attach_seed0_history", "load_low", "train_one_seed"]
