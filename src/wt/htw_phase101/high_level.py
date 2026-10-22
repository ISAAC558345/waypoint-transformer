"""Frozen DeepSets-util high-level wrapper. No optimizer, no M=5 load."""
from __future__ import annotations

from pathlib import Path

import torch

from wt.htw_phase100.high_level import decide, load_deepsets_util, make_option
from wt.htw_phase941.protocol import sha256_bytes

from .protocol import FORBIDDEN_M5_CKPT, HIGH_CKPT, HIGH_MODEL_SEED


def refuse_m5(path: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == Path(FORBIDDEN_M5_CKPT).resolve() or "low_level_c05_d005" in resolved.name:
        raise RuntimeError(f"refused to load Phase-6.2 M=5 checkpoint: {resolved}")


def load_frozen_high(path: Path | None = None):
    ckpt = Path(HIGH_CKPT if path is None else path)
    refuse_m5(ckpt)
    if ckpt.resolve() == Path(FORBIDDEN_M5_CKPT).resolve():
        raise RuntimeError("high-level loader received the M=5 low-level path")
    model = load_deepsets_util(HIGH_MODEL_SEED, ckpt)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, sha256_bytes(ckpt.read_bytes())


def parameter_sha(model) -> str:
    blobs = []
    for key, value in model.state_dict().items():
        blobs.append(key.encode("utf-8"))
        blobs.append(bytes(value.detach().cpu().contiguous().numpy()))
    return sha256_bytes(b"".join(blobs))


def high_grad_count(model) -> int:
    return int(sum(int(p.requires_grad) for p in model.parameters()))


__all__ = ["high_grad_count", "load_frozen_high", "parameter_sha", "refuse_m5"]
