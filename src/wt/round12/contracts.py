"""Non-negotiable Round-12 policy-conditioning contracts.

The selected Round-11 task-waypoint predictor is an immutable inference-only
dependency.  Keeping the checks here (rather than in a training loop) makes
it difficult to accidentally turn it into a trainable policy submodule.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from wt.round11.task_waypoint import TaskWaypointConfig, TaskWaypointMLP


PREDICTOR_PATH = Path("round_11/checkpoints/H10/seed_2/best.pt")
PREDICTOR_SHA256 = "A3E8B6C361CCA997A05766B847521EBB68E68985CAC273667DE95181ACCCEF65"
H_TASK = 10
CONTEXT_LENGTH = 10
ACTION_DIM = 2
ALLOWED_SPLITS = frozenset(("train", "validation"))
FORBIDDEN_SPLITS = frozenset(("test", "old_development_test", "holdout", "untouched_holdout"))


def sha256(path: Path) -> str:
    """Return the uppercase SHA-256 digest without loading a checkpoint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def assert_selected_predictor(root: Path) -> Path:
    """Validate the only Round-11 checkpoint allowed in Round-12."""
    path = Path(root) / PREDICTOR_PATH
    if not path.is_file():
        raise FileNotFoundError(f"selected Round-11 predictor is missing: {path}")
    actual = sha256(path)
    if actual != PREDICTOR_SHA256:
        raise RuntimeError(
            "selected Round-11 predictor hash changed: "
            f"expected {PREDICTOR_SHA256}, found {actual}"
        )
    return path


def _checkpoint_config(payload: dict, state_dim: int, max_buoys: int) -> TaskWaypointConfig:
    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise ValueError("Round-11 checkpoint does not contain a configuration")
    expected_state = int(raw.get("state_dim", state_dim))
    expected_buoys = int(raw.get("max_buoys", max_buoys))
    if (expected_state, expected_buoys) != (int(state_dim), int(max_buoys)):
        raise ValueError(
            "Round-12 predictor dimensions must match the selected checkpoint: "
            f"checkpoint={(expected_state, expected_buoys)}, "
            f"requested={(state_dim, max_buoys)}"
        )
    return TaskWaypointConfig(
        state_dim=expected_state,
        max_buoys=expected_buoys,
        progress_scale=float(raw["progress_scale"]),
        return_mean=float(raw["return_mean"]),
        return_std=float(raw["return_std"]),
        gamma=float(raw.get("gamma", 0.99)),
    )


def assert_predictor_frozen(model: TaskWaypointMLP) -> None:
    """Raise unless a predictor is in its required inference-only state."""
    if model.training:
        raise RuntimeError("Round-12 predictor must remain in eval mode")
    for parameter in model.parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError("Round-12 predictor must have no trainable parameters or gradients")


def load_frozen_predictor(root: Path, state_dim: int, max_buoys: int) -> TaskWaypointMLP:
    """Load the hash-pinned predictor as a detached, inference-only module."""
    path = assert_selected_predictor(root)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("selected Round-11 predictor checkpoint is malformed")
    model = TaskWaypointMLP(_checkpoint_config(payload, state_dim, max_buoys))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    assert_predictor_frozen(model)
    return model


def predicted_phi(
    model: TaskWaypointMLP, state: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Produce deployable Phi=[service, gross progress, H-step return].

    True task-waypoint labels are intentionally absent from this API.  The
    explicit mask also guarantees exact zeroes in padded service/progress
    slots even if a caller supplies a non-canonical predictor implementation.
    """
    assert_predictor_frozen(model)
    if state.ndim != 2 or mask.ndim != 2:
        raise ValueError("predictor inputs must be [B,state_dim] and [B,max_buoys]")
    if state.shape[0] != mask.shape[0]:
        raise ValueError("predictor state and mask batch sizes disagree")
    with torch.no_grad():
        output = model(state, mask)
        service = output["service_distribution"] * mask.to(state.dtype)
        progress = output["gross_service_progress"] * mask.to(state.dtype)
        result = torch.cat((service, progress, output["H_step_return"].unsqueeze(-1)), dim=-1)
    if result.requires_grad:
        raise RuntimeError("frozen predictor output unexpectedly requires gradients")
    return result


__all__ = [
    "ACTION_DIM",
    "ALLOWED_SPLITS",
    "CONTEXT_LENGTH",
    "FORBIDDEN_SPLITS",
    "H_TASK",
    "PREDICTOR_PATH",
    "PREDICTOR_SHA256",
    "assert_predictor_frozen",
    "assert_selected_predictor",
    "load_frozen_predictor",
    "predicted_phi",
    "sha256",
]
