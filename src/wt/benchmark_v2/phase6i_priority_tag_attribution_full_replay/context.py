"""Apply the E8 process context without relabeling it as an evaluation seed.

Seed 2 is reapplied before every episode. It remains the inherited post-training
process state, not a canonical evaluation-protocol seed. Environment reset and
assignment RNGs stay on the original per-scenario values.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from .protocol import (
    CONTEXT_LABEL,
    EVAL_ASSIGNMENT_SEED,
    GLOBAL_SEED_ROLE,
    HISTORICAL_INHERITED_GLOBAL_SEED,
    TORCH_THREADS,
)


def assignment_seed(scenario_id: int) -> int:
    return int(EVAL_ASSIGNMENT_SEED) + int(scenario_id)


def apply_historical_context() -> dict:
    seed = int(HISTORICAL_INHERITED_GLOBAL_SEED)
    torch.set_num_threads(int(TORCH_THREADS))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "context_label": CONTEXT_LABEL,
        "torch_num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_initial_seed": int(torch.initial_seed()),
        "process_rng_seed": seed,
        "process_rng_role": GLOBAL_SEED_ROLE,
        "canonical_explicit_seed": None,
        "actor_eval_required": True,
    }


def context_is_historical(applied: dict) -> bool:
    return bool(
        applied.get("context_label") == CONTEXT_LABEL
        and int(applied.get("torch_num_threads")) == int(TORCH_THREADS)
        and applied.get("deterministic_algorithms") is True
        and int(applied.get("torch_initial_seed")) == int(HISTORICAL_INHERITED_GLOBAL_SEED)
        and applied.get("process_rng_role") == GLOBAL_SEED_ROLE
        and applied.get("process_rng_role") != "eval_protocol_seed"
        and applied.get("canonical_explicit_seed") is None
        and int(applied.get("process_rng_seed")) == int(HISTORICAL_INHERITED_GLOBAL_SEED)
    )
