"""Frozen current-state baselines and the selected locked State-only BC policy."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.round19.planner import independent_decision, joint_plan
from wt.round20.dataset import build_graph_state
from wt.round22.closed_loop import _model_inputs
from wt.round22.models import STATE_ONLY_TRANSFORMER, rollout_action
from wt.round22.training import load_checkpoint


BC_CHECKPOINT = Path("round_22/outputs/round22_deepsets_task_waypoint_policy_20260829_1153/formal/state_only_transformer/seed_0/best.pt")
BC_CHECKPOINT_SHA256 = "7BAA38E91B9CC212EF8EC614565357C1881E2BD8198749EF251EE611887BB8DF"


class FrozenBaseline:
    policy_id = ""
    matching_mode = "legacy"

    def reset(self) -> None: pass
    def action(self, env: Any, info: Mapping[str, Any]) -> np.ndarray: raise NotImplementedError


class RandomValid(FrozenBaseline):
    policy_id = "random_valid"

    def __init__(self, seed: int) -> None: self.rng = np.random.default_rng(seed)

    def action(self, env: Any, info: Mapping[str, Any]) -> np.ndarray:
        action = self.rng.uniform(-1.0, 1.0, size=env.action_shape)
        return action / np.maximum(np.linalg.norm(action, axis=1, keepdims=True), 1.0)


class IndependentMyopic(FrozenBaseline):
    policy_id = "independent_myopic"

    def action(self, env: Any, info: Mapping[str, Any]) -> np.ndarray:
        return independent_decision(env, info).action


class JointPlanner(FrozenBaseline):
    matching_mode = "joint_aware"

    def __init__(self, horizon: int) -> None:
        self.horizon, self.policy_id = int(horizon), f"joint_h{int(horizon)}"

    def action(self, env: Any, info: Mapping[str, Any]) -> np.ndarray:
        return joint_plan(env, info, horizon=self.horizon).action


class FrozenStateOnlyBC(FrozenBaseline):
    policy_id = "state_only_bc_transformer"

    def __init__(self) -> None:
        if not BC_CHECKPOINT.is_file() or hashlib.sha256(BC_CHECKPOINT.read_bytes()).hexdigest().upper() != BC_CHECKPOINT_SHA256:
            raise RuntimeError("Round-30 requires the locked selected Round-22 state-only BC checkpoint")
        family, model, normalizer, payload = load_checkpoint(BC_CHECKPOINT, "cpu")
        if family != STATE_ONLY_TRANSFORMER or not payload.get("deployable") or payload.get("historical_action_input"):
            raise RuntimeError("selected Round-22 checkpoint is not the frozen state-only BC policy")
        self.model, self.normalizer, self.history = model.eval(), normalizer, []

    def reset(self) -> None: self.history = []

    def action(self, env: Any, info: Mapping[str, Any]) -> np.ndarray:
        graph = build_graph_state(info, env.config)
        self.history.append((graph, np.zeros((2, 11), dtype=np.float32)))
        inputs = _model_inputs(self.history, self.normalizer, "cpu")
        inputs["phi"] = None
        with torch.no_grad():
            return rollout_action(STATE_ONLY_TRANSFORMER, self.model, **inputs).cpu().numpy()[0]


def baseline_factories(scenario_seed: int) -> list[FrozenBaseline]:
    return [RandomValid(scenario_seed * 101 + 17), IndependentMyopic(), JointPlanner(1), JointPlanner(5), JointPlanner(10), FrozenStateOnlyBC()]


__all__ = ["BC_CHECKPOINT", "BC_CHECKPOINT_SHA256", "FrozenBaseline", "FrozenStateOnlyBC", "IndependentMyopic", "JointPlanner", "RandomValid", "baseline_factories"]
