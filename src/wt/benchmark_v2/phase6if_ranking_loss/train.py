"""Short Phase-6I-F runs. Epoch 6 and the frozen encoder are read-only."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset, one_state_batch
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import GRAPH_KEYS, RESERVED_TEST
from wt.benchmark_v2.phase6id_joint_training.metrics import mean_metrics, state_metrics
from wt.benchmark_v2.phase6id_joint_training.protocol import (
    ENCODER_CKPT,
    GRAD_CLIP,
    LEARNING_RATE,
    OUTPUT as PHASE6ID_OUTPUT,
    WEIGHT_DECAY,
)
from wt.benchmark_v2.phase6id_joint_training.train import _build, _finite_tensor, _sha256
from wt.benchmark_v2.phase6if_ranking_loss.losses import LOSSES
from wt.lowlevel_v2.phase5l_joint_encoder.model import append_queue_edge, batch_tensors


ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT = ROOT / "results/phase6if_ranking_loss" / "runs"
EPOCH6 = PHASE6ID_OUTPUT / "epoch_006.pt"
SEED = 0


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
    return int(counters.peak_working_set_size)


def _scores(model, decoder, sample: dict, device: torch.device) -> torch.Tensor:
    if tuple(sample["graph"]) != GRAPH_KEYS or "j15_bits" in sample["graph"] or "utility_rank" in sample["graph"]:
        raise RuntimeError("graph batch contains a label")
    conditioned = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
    embeddings = model.condition(tensors)
    return decoder.score_assignments(embeddings[0], embeddings[1], assignments)


def _evaluate(model, decoder, device: torch.device, objective) -> tuple[dict, list[dict]]:
    rows = []
    losses = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(DataLoader(DecisionDataset("validation"), batch_size=1, shuffle=False, num_workers=0, collate_fn=one_state_batch), start=1):
            if int(sample["scenario_id"]) in RESERVED_TEST:
                raise RuntimeError("validation loader opened a reserved scenario")
            scores = _scores(model, decoder, sample, device)
            if not _finite_tensor(scores):
                raise RuntimeError("non-finite validation score")
            labels = torch.as_tensor(sample["j15_bits"], dtype=scores.dtype)
            loss = objective(scores, labels)
            if not _finite_tensor(loss):
                raise RuntimeError("non-finite validation loss")
            losses.append(float(loss.detach().cpu()))
            measured = state_metrics(scores.detach().cpu().numpy(), sample["j15_bits"], sample["utility_rank"])
            measured["scenario_id"] = float(sample["scenario_id"])
            measured["time_step"] = float(sample["time_step"])
            rows.append(measured)
            if index == 1 or index % 16 == 0 or index == 48:
                print(f"eval {index}/48 {int(sample['scenario_id'])} t={int(sample['time_step'])}", flush=True)
    summary = mean_metrics(rows)
    summary["loss"] = float(np.mean(losses))
    return summary, rows


def _selection(summary: dict) -> tuple[float, float, float, float, float]:
    return (
        float(summary["payload"]),
        float(summary["top1"]),
        -float(summary["normalized_regret"]),
        float(summary["top3"]),
        float(summary["top5"]),
    )


def _paired(rows: list[dict], reference_path: Path) -> dict[str, float]:
    reference = {(int(row["scenario_id"]), int(row["time_step"])): row for row in json.loads(reference_path.read_text(encoding="utf-8"))}
    delta = []
    for row in rows:
        key = (int(row["scenario_id"]), int(row["time_step"]))
        delta.append(float(row["payload"]) - float(reference[key]["payload"]))
    array = np.asarray(delta, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive": float(np.sum(array > 1.0)),
        "negative": float(np.sum(array < -1.0)),
    }


def run_scheme(name: str, epochs: int) -> dict:
    if name not in LOSSES:
        raise RuntimeError(f"unknown scheme {name}")
    if not EPOCH6.is_file():
        raise FileNotFoundError(str(EPOCH6))
    output = RUN_ROOT / name
    if EPOCH6.resolve().is_relative_to(output.resolve()) or ENCODER_CKPT.resolve().is_relative_to(output.resolve()):
        raise RuntimeError("refusing to write over a baseline checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    model, decoder, optimizer = _build(device)
    payload = torch.load(EPOCH6, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != 6:
        raise RuntimeError("initialization is not epoch 6")
    before_epoch6 = _sha256(EPOCH6)
    before_encoder = _sha256(ENCODER_CKPT)
    model.load_state_dict(payload["model"])
    model.to(device)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    objective = LOSSES[name]
    history = []
    completed = 0
    for epoch in range(1, int(epochs) + 1):
        existing = output / f"epoch_{epoch:03d}.pt"
        if existing.is_file():
            saved = torch.load(existing, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            history = list(saved["history"])
            completed = epoch
            continue
        if completed != epoch - 1:
            raise RuntimeError("missing earlier epoch checkpoint")
        model.train()
        losses = []
        grad_norms = []
        peak_rss = _process_rss_bytes()
        started = time.perf_counter()
        loader = DataLoader(DecisionDataset("train"), batch_size=1, shuffle=False, num_workers=0, collate_fn=one_state_batch)
        for index, sample in enumerate(loader, start=1):
            if int(sample["scenario_id"]) in RESERVED_TEST:
                raise RuntimeError("training loader opened a reserved scenario")
            optimizer.zero_grad(set_to_none=True)
            scores = _scores(model, decoder, sample, device)
            labels = torch.as_tensor(sample["j15_bits"], dtype=scores.dtype)
            loss = objective(scores, labels)
            if not _finite_tensor(loss) or not _finite_tensor(scores):
                raise RuntimeError(f"non-finite scheme {name} at epoch {epoch}")
            loss.backward()
            grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not grads or any(not _finite_tensor(grad) for grad in grads):
                raise RuntimeError(f"non-finite gradient in scheme {name}")
            grad_norm = float(clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(grad_norm)
            peak_rss = max(peak_rss, _process_rss_bytes())
            del scores, loss
            if index == 1 or index % 12 == 0 or index == len(loader):
                print(f"scheme {name} epoch {epoch} {index}/{len(loader)} loss={losses[-1]:.4f} grad={grad_norm:.4f}", flush=True)
        train_seconds = time.perf_counter() - started
        valid_started = time.perf_counter()
        summary, rows = _evaluate(model, decoder, device, objective)
        valid_seconds = time.perf_counter() - valid_started
        peak_rss = max(peak_rss, _process_rss_bytes())
        summary.update(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": float(summary["loss"]),
                "grad_norm_mean": float(np.mean(grad_norms)),
                "grad_norm_max": float(np.max(grad_norms)),
                "train_seconds": train_seconds,
                "validation_seconds": valid_seconds,
                "peak_rss_bytes": int(peak_rss),
                "selection": list(_selection(summary)),
            }
        )
        history.append(summary)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "scheme": name, "init": "epoch6"}, output / f"epoch_{epoch:03d}.pt")
        (output / f"epoch_{epoch:03d}_states.json").write_text(json.dumps(rows), encoding="utf-8")
        print(
            f"scheme {name} epoch {epoch} payload={summary['payload']:.3f} top1={summary['top1']:.4f} regret={summary['normalized_regret']:.4f} seconds={train_seconds + valid_seconds:.1f}",
            flush=True,
        )
        completed = epoch
    best_index = max(range(len(history)), key=lambda index: tuple(history[index]["selection"]))
    best = history[best_index]
    result = {
        "scheme": name,
        "best_epoch": int(best["epoch"]),
        "history": history,
        "paired_vs_frozen": _paired(json.loads((output / f"epoch_{int(best['epoch']):03d}_states.json").read_text(encoding="utf-8")), PHASE6ID_OUTPUT / "baseline_states.json"),
        "paired_vs_epoch6": _paired(json.loads((output / f"epoch_{int(best['epoch']):03d}_states.json").read_text(encoding="utf-8")), PHASE6ID_OUTPUT / "best_states.json"),
        "epoch6_unchanged": _sha256(EPOCH6) == before_epoch6,
        "encoder_unchanged": _sha256(ENCODER_CKPT) == before_encoder,
    }
    result["frozen_sha256"] = _sha256(ENCODER_CKPT)
    result["epoch6_sha256"] = _sha256(EPOCH6)
    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", required=True, choices=tuple(LOSSES))
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps({"scheme": args.scheme, "best_epoch": run_scheme(args.scheme, int(args.epochs))["best_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
