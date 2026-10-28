"""Build and apply an explicit runtime manifest.

The global seed value 2 is the leftover from the last training call before the
saved evaluation. It is not a declared evaluation-protocol seed.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from .protocol import (
    ASSIGNMENT_RNG_SEED,
    CONTEXT_LABEL,
    ENV_RESET_SEED,
    GLOBAL_SEED_ROLE,
    HASHED_RELATIVE,
    HISTORICAL_INHERITED_GLOBAL_SEED,
    REPORTED_MODEL_SEED,
    ROOT,
    SCENARIO_ID,
    TORCH_THREADS,
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(*, versions: dict, commit: str, code_hashes: dict) -> dict:
    missing = [relative for relative in HASHED_RELATIVE if relative not in code_hashes]
    if missing:
        raise RuntimeError("manifest is missing code hashes")
    return {
        "context_label": CONTEXT_LABEL,
        "historical_note": (
            "torch threads=1 and deterministic_algorithms=True reproduced the saved episode in E7 "
            "only together with seed_everything(2). Seed 2 is the last train_variant seed before evaluate_all, "
            "not an evaluation-protocol seed. A canonical explicit evaluation seed was not found."
        ),
        "reported_model_seed": int(REPORTED_MODEL_SEED),
        "scenario_id": int(SCENARIO_ID),
        "env_reset_seed": int(ENV_RESET_SEED),
        "assignment_rng_seed": int(ASSIGNMENT_RNG_SEED),
        "torch_num_threads": int(TORCH_THREADS),
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "global_rng": {
            "applied_seed": int(HISTORICAL_INHERITED_GLOBAL_SEED),
            "role": GLOBAL_SEED_ROLE,
            "canonical_explicit_seed": None,
            "canonical_explicit_seed_status": "unconfirmed",
            "python": True,
            "numpy_global": True,
            "torch_cpu": True,
            "torch_cuda": False,
        },
        "actor_eval_mode": True,
        "deterministic_actions": True,
        "observation_normalization": "none_in_formal_builder",
        "versions": versions,
        "git_commit": commit,
        "code_hashes": code_hashes,
    }


def manifest_is_explicit(manifest: dict) -> bool:
    rng = manifest.get("global_rng") or {}
    return bool(
        manifest.get("context_label") == CONTEXT_LABEL
        and int(manifest.get("reported_model_seed")) == int(REPORTED_MODEL_SEED)
        and int(manifest.get("env_reset_seed")) == int(ENV_RESET_SEED)
        and int(manifest.get("assignment_rng_seed")) == int(ASSIGNMENT_RNG_SEED)
        and int(manifest.get("scenario_id")) == int(SCENARIO_ID)
        and int(manifest.get("torch_num_threads")) == int(TORCH_THREADS)
        and manifest.get("deterministic_algorithms") is True
        and rng.get("role") == GLOBAL_SEED_ROLE
        and rng.get("canonical_explicit_seed") is None
        and rng.get("role") != "eval_protocol_seed"
        and int(rng.get("applied_seed")) == int(HISTORICAL_INHERITED_GLOBAL_SEED)
        and manifest.get("actor_eval_mode") is True
        and manifest.get("deterministic_actions") is True
    )


def apply_manifest(manifest: dict) -> dict:
    if not manifest_is_explicit(manifest):
        raise RuntimeError("manifest mixes the historical inherited seed with a canonical eval seed")
    seed = int(manifest["global_rng"]["applied_seed"])
    torch.set_num_threads(int(manifest["torch_num_threads"]))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=bool(manifest["deterministic_warn_only"]))
    cudnn_available = bool(getattr(torch.backends, "cudnn", None) is not None and torch.backends.cudnn.is_available())
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_initial_seed": int(torch.initial_seed()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_applies": bool(torch.cuda.is_available()),
        "cudnn_available": cudnn_available,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "interop_threads_recorded_not_set": int(torch.get_num_interop_threads()),
    }


def code_hashes() -> dict[str, str]:
    return {relative: file_sha256(ROOT / relative) for relative in HASHED_RELATIVE}
