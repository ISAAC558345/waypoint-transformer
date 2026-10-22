"""One optimizer step on the Phase-6I revision. The original checkpoints stay put."""
from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset, one_state_batch
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import (
    DECODER_CKPT,
    ENCODER_CKPT,
    TRAIN_POOL,
    VALIDATION_POOL,
)
from wt.htw_phase9.const import BUOY_DIM
from wt.lowlevel_v2.phase5k_joint_decoder.decoders import JointAssignmentDecoder
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN, append_queue_edge, batch_tensors
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import LEARNING_RATE, WEIGHT_DECAY


REPORT = Path(__file__).resolve().parents[4] / "docs" / "archive" / "phase_reports" / "HTW_Phase6I_JointUtilityHGNN_SmokeTest_CN.md"


class _MemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
        ("quota_nonpaged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    )


def _process_rss_bytes() -> int:
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.working_set_size)


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _run_forward(model, decoder, sample: dict, device: torch.device) -> dict:
    conditioned = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
    labels = torch.as_tensor(sample["j15_bits"], dtype=torch.float32, device=device)
    started = time.perf_counter()
    embeddings = model.condition(tensors)
    scores = decoder.score_assignments(embeddings[0], embeddings[1], assignments)
    loss = joint_rank_loss(scores, labels)
    elapsed = time.perf_counter() - started
    return {
        "embeddings": embeddings,
        "scores": scores,
        "loss": loss,
        "seconds": elapsed,
        "input_shapes": {key: list(value.shape) for key, value in tensors.items()},
    }


def _manifest_candidate_total(split: str, pool: tuple[int, ...]) -> int:
    total = 0
    root = Path(__file__).resolve().parents[1] / "phase6ib_highlevel_dataset" / "revision_hgnn_input" / "dataset" / split
    for scenario in pool:
        folder = root / f"scenario_{int(scenario)}"
        for time_step in (0, 15, 30):
            payload = json.loads((folder / f"t{time_step:03d}_manifest.json").read_text(encoding="utf-8"))
            total += int(payload["candidate_count"])
    return total


def run_smoke() -> dict:
    if not ENCODER_CKPT.is_file() or not DECODER_CKPT.is_file():
        raise FileNotFoundError("phase-6ic checkpoint is missing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    encoder_payload = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(encoder_payload["state_dict"])
    model.to(device)
    decoder = JointAssignmentDecoder("matching")
    decoder_payload = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    decoder.load_state_dict(decoder_payload["state_dict"])
    decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(DecisionDataset("train"), batch_size=1, shuffle=False, num_workers=0, collate_fn=one_state_batch)
    validation_loader = DataLoader(DecisionDataset("validation"), batch_size=1, shuffle=False, num_workers=0, collate_fn=one_state_batch)
    train_sample = next(iter(train_loader))
    validation_sample = next(iter(validation_loader))
    if int(train_sample["scenario_id"]) not in TRAIN_POOL or int(validation_sample["scenario_id"]) not in VALIDATION_POOL:
        raise RuntimeError("smoke loader crossed the scenario split")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    watched = model.encoder.uav_enc[0].weight.detach().clone()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    graph_finite = all(np.isfinite(np.asarray(value, dtype=np.float32)).all() for value in train_sample["graph"].values())
    label_finite = bool(np.isfinite(train_sample["j15_bits"]).all() and np.isfinite(validation_sample["j15_bits"]).all())
    trained = _run_forward(model, decoder, train_sample, device)
    loss = trained["loss"]
    if not graph_finite or not label_finite or not _finite(loss) or not _finite(trained["scores"]) or not _finite(trained["embeddings"][0]) or not _finite(trained["embeddings"][1]):
        raise RuntimeError("non-finite forward")
    backward_started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - backward_started
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads or any(not _finite(grad) for grad in grads):
        raise RuntimeError("non-finite gradient")
    grad_norm = float(clip_grad_norm_(model.parameters(), 1.0))
    if not np.isfinite(grad_norm):
        raise RuntimeError("non-finite gradient")
    step_started = time.perf_counter()
    optimizer.step()
    step_seconds = time.perf_counter() - step_started
    updated = model.encoder.uav_enc[0].weight.detach()
    parameter_delta = float(torch.max(torch.abs(updated - watched)).item())
    model.eval()
    with torch.no_grad():
        validated = _run_forward(model, decoder, validation_sample, device)
    if not _finite(validated["loss"]) or not _finite(validated["scores"]):
        raise RuntimeError("non-finite validation forward")
    train_candidates = _manifest_candidate_total("train", TRAIN_POOL)
    validation_candidates = _manifest_candidate_total("validation", VALIDATION_POOL)
    train_states = len(TRAIN_POOL) * 3
    validation_states = len(VALIDATION_POOL) * 3
    batch_seconds = float(trained["seconds"])
    return {
        "device": device.type,
        "train_scenario": int(train_sample["scenario_id"]),
        "train_time": int(train_sample["time_step"]),
        "validation_scenario": int(validation_sample["scenario_id"]),
        "validation_time": int(validation_sample["time_step"]),
        "train_candidates": int(train_sample["candidate_count"]),
        "validation_candidates": int(validation_sample["candidate_count"]),
        "input_shapes": trained["input_shapes"],
        "embedding_shapes": [list(trained["embeddings"][0].shape), list(trained["embeddings"][1].shape)],
        "score_shape": list(trained["scores"].shape),
        "train_loss": float(loss.detach().cpu()),
        "validation_loss": float(validated["loss"].detach().cpu()),
        "grad_norm": grad_norm,
        "grad_tensors": len(grads),
        "parameter_delta": parameter_delta,
        "forward_loss_seconds": batch_seconds,
        "backward_seconds": float(backward_seconds),
        "optimizer_step_seconds": float(step_seconds),
        "batch_seconds": float(batch_seconds + backward_seconds + step_seconds),
        "validation_forward_seconds": float(validated["seconds"]),
        "rss_bytes": _process_rss_bytes(),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "train_states": train_states,
        "validation_states": validation_states,
        "train_candidate_total": train_candidates,
        "validation_candidate_total": validation_candidates,
        "epoch_seconds_estimate": float(batch_seconds + backward_seconds + step_seconds) * train_states,
        "decoder_requires_grad": any(parameter.requires_grad for parameter in decoder.parameters()),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), sort_keys=True), flush=True)
