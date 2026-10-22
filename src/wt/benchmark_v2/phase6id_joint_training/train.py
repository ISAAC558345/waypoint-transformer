"""Controlled Phase-6I-D training. The Phase-5L checkpoint file is read-only."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset, one_state_batch
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import GRAPH_KEYS
from wt.benchmark_v2.phase6id_joint_training.metrics import mean_metrics, state_metrics
from wt.benchmark_v2.phase6id_joint_training.protocol import (
    COLLAPSE_STOP,
    DECODER_CKPT,
    ENCODER_CKPT,
    GRAD_CLIP,
    LEARNING_RATE,
    MAX_EPOCHS,
    OUTPUT,
    PATIENCE,
    RESERVED_TEST,
    TRAIN_POOL,
    VALIDATION_POOL,
    WEIGHT_DECAY,
    assert_output_path,
    clear_improvement,
    selection_key,
)
from wt.htw_phase9.const import BUOY_DIM
from wt.lowlevel_v2.phase5k_joint_decoder.decoders import JointAssignmentDecoder
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN, append_queue_edge, batch_tensors


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


def _rss_bytes() -> int:
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.working_set_size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    target = assert_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _save_checkpoint(path: Path, payload: dict) -> None:
    target = assert_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def confirm_contract() -> dict:
    """Stop before the first optimizer step when the smoke contract is not intact."""
    source = inspect.getsource(joint_rank_loss)
    if "joints[:, None] > joints[None, :]" not in source:
        raise RuntimeError("pairwise loss no longer uses strict J15 order")
    if float(LEARNING_RATE) != 3.0e-4 or float(WEIGHT_DECAY) != 1.0e-4 or float(GRAD_CLIP) != 1.0:
        raise RuntimeError("optimizer settings drifted from the smoke test")
    if not ENCODER_CKPT.is_file() or not DECODER_CKPT.is_file():
        raise RuntimeError("initialization checkpoint is missing")
    assert_output_path(OUTPUT)
    train_index = DecisionDataset("train").index
    valid_index = DecisionDataset("validation").index
    if any(int(scenario) in RESERVED_TEST for scenario, _time_step in train_index + valid_index):
        raise RuntimeError("dataset index contains a reserved test scenario")
    if [int(scenario) for scenario, _time_step in train_index] != [int(scenario) for scenario in TRAIN_POOL for _time in range(3)]:
        raise RuntimeError("training scenarios are not 322000-322047")
    if [int(scenario) for scenario, _time_step in valid_index] != [int(scenario) for scenario in VALIDATION_POOL for _time in range(3)]:
        raise RuntimeError("validation scenarios are not 322048-322063")
    sample = next(iter(DataLoader(DecisionDataset("train"), batch_size=1, shuffle=False, num_workers=0, collate_fn=one_state_batch)))
    if tuple(sample["graph"]) != GRAPH_KEYS or "j15_bits" in sample["graph"] or "utility_rank" in sample["graph"]:
        raise RuntimeError("graph batch contains a label field")
    count = int(sample["assignments"].shape[0])
    if count != int(sample["j15_bits"].shape[0]) or count != int(sample["utility_rank"].shape[0]) or count < 1000:
        raise RuntimeError("assignment rows are not aligned with J15 labels")
    model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    encoder_payload = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(encoder_payload["state_dict"])
    decoder = JointAssignmentDecoder("matching")
    decoder_payload = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    decoder.load_state_dict(decoder_payload["state_dict"])
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    decoder_ids = {id(parameter) for parameter in decoder.parameters()}
    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise RuntimeError("matching decoder was unfrozen")
    if any(id(parameter) in decoder_ids for parameter in optimizer.param_groups[0]["params"]):
        raise RuntimeError("optimizer includes the matching decoder")
    conditioned = append_queue_edge(sample["graph"])
    if int(conditioned["edge"].shape[-1]) != 4 or int(sample["graph"]["uav"].shape[-1]) != 8 or int(sample["graph"]["buoy"].shape[-1]) != 8:
        raise RuntimeError("checkpoint input widths are not 8/8/4/2")
    return {
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "grad_clip": float(GRAD_CLIP),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "decoder_parameters": int(sum(parameter.numel() for parameter in decoder.parameters())),
        "decoder_requires_grad": False,
        "encoder_sha256": _sha256(ENCODER_CKPT),
        "decoder_sha256": _sha256(DECODER_CKPT),
        "loss": "mean(-logsigmoid(s_i-s_j)) for j15_i > j15_j",
        "init_checkpoint": str(ENCODER_CKPT),
        "decoder_checkpoint": str(DECODER_CKPT),
    }


def _build(device: torch.device):
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
    return model, decoder, optimizer


def _forward(model, decoder, sample: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    conditioned = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
    labels = torch.as_tensor(sample["j15_bits"], dtype=torch.float32, device=device)
    embeddings = model.condition(tensors)
    scores = decoder.score_assignments(embeddings[0], embeddings[1], assignments)
    loss = joint_rank_loss(scores, labels)
    return scores, loss


def _evaluate(model, decoder, device: torch.device, split: str) -> tuple[dict, list[dict]]:
    loader = DataLoader(DecisionDataset(split), batch_size=1, shuffle=False, num_workers=0, collate_fn=one_state_batch)
    rows = []
    losses = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(loader, start=1):
            scores, loss = _forward(model, decoder, sample, device)
            if not _finite_tensor(scores) or not _finite_tensor(loss):
                raise RuntimeError(f"non-finite validation at {sample['scenario_id']} t={sample['time_step']}")
            measured = state_metrics(scores.detach().cpu().numpy(), sample["j15_bits"], sample["utility_rank"])
            measured["scenario_id"] = float(sample["scenario_id"])
            measured["time_step"] = float(sample["time_step"])
            measured["loss"] = float(loss.detach().cpu())
            rows.append(measured)
            losses.append(measured["loss"])
            if index == 1 or index % 8 == 0 or index == len(loader):
                print(f"eval {split} {index}/{len(loader)} {int(sample['scenario_id'])} t={int(sample['time_step'])} loss={measured['loss']:.4f}", flush=True)
            del scores, loss
    summary = mean_metrics(rows)
    summary["loss"] = float(np.mean(np.asarray(losses, dtype=np.float64)))
    return summary, rows


def _train_epoch(model, decoder, optimizer, device: torch.device, epoch: int) -> dict:
    loader = DataLoader(DecisionDataset("train"), batch_size=1, shuffle=False, num_workers=0, collate_fn=one_state_batch)
    losses = []
    grad_norms = []
    model.train()
    for index, sample in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        scores, loss = _forward(model, decoder, sample, device)
        if not _finite_tensor(scores) or not _finite_tensor(loss):
            raise RuntimeError(f"non-finite train forward at epoch {epoch} {sample['scenario_id']} t={sample['time_step']}")
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not grads or any(not _finite_tensor(grad) for grad in grads):
            raise RuntimeError(f"non-finite gradient at epoch {epoch}")
        grad_norm = float(clip_grad_norm_(model.parameters(), GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at epoch {epoch}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(grad_norm)
        if index == 1 or index % 12 == 0 or index == len(loader):
            print(f"train epoch {epoch} {index}/{len(loader)} {int(sample['scenario_id'])} t={int(sample['time_step'])} loss={losses[-1]:.4f} grad={grad_norm:.4f}", flush=True)
        del scores, loss
    return {
        "loss": float(np.mean(np.asarray(losses, dtype=np.float64))),
        "grad_norm_mean": float(np.mean(np.asarray(grad_norms, dtype=np.float64))),
        "grad_norm_max": float(np.max(np.asarray(grad_norms, dtype=np.float64))),
        "steps": int(len(losses)),
    }


def _load_resume(model, optimizer, device: torch.device) -> tuple[list[dict], int, dict | None]:
    history = []
    best = None
    completed = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        path = OUTPUT / f"epoch_{epoch:03d}.pt"
        if not path.is_file():
            break
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != epoch:
            raise RuntimeError(f"checkpoint epoch mismatch in {path.name}")
        history = list(payload["history"])
        best = payload.get("best")
        completed = epoch
    if completed:
        payload = torch.load(OUTPUT / f"epoch_{completed:03d}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        model.to(device)
    return history, completed, best


def _fmt(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def _gate(status: str, comparison: dict | None) -> str:
    if status == "blocked" or not comparison:
        return "JOINT_UTILITY_TRAINING_BLOCKED"
    if not comparison.get("encoder_unchanged") or not comparison.get("decoder_unchanged"):
        return "JOINT_UTILITY_TRAINING_BLOCKED"
    if comparison.get("clear_improvement"):
        return "JOINT_UTILITY_TRAINING_COMPLETE"
    return "JOINT_UTILITY_TRAINING_INCONCLUSIVE"


def _paired(trained_rows: list[dict], frozen_rows: list[dict]) -> dict[str, float]:
    frozen = {(int(row["scenario_id"]), int(row["time_step"])): row for row in frozen_rows}
    deltas = []
    for row in trained_rows:
        key = (int(row["scenario_id"]), int(row["time_step"]))
        if key not in frozen:
            raise RuntimeError("trained and frozen validation states differ")
        deltas.append(float(row["payload"]) - float(frozen[key]["payload"]))
    array = np.asarray(deltas, dtype=np.float64)
    return {"mean_payload_delta": float(np.mean(array)), "positive_states": float(np.sum(array > 0.0)), "n_states": float(array.size)}


def run_training(epochs: int) -> dict:
    limit = min(int(epochs), int(MAX_EPOCHS))
    if limit < 1:
        raise RuntimeError("epoch limit must be positive")
    contract = confirm_contract()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, decoder, optimizer = _build(device)
    history, completed, best = _load_resume(model, optimizer, device)
    baseline_path = OUTPUT / "baseline_validation.json"
    if baseline_path.is_file():
        baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))["summary"]
        baseline_rows = json.loads((OUTPUT / "baseline_states.json").read_text(encoding="utf-8"))
    else:
        started = time.perf_counter()
        baseline_summary, baseline_rows = _evaluate(model, decoder, device, "validation")
        baseline_summary["seconds"] = time.perf_counter() - started
        _write_json(baseline_path, {"summary": baseline_summary, "encoder_sha256": contract["encoder_sha256"]})
        _write_json(OUTPUT / "baseline_states.json", baseline_rows)
        print(f"frozen validation loss={_fmt(baseline_summary['loss'])} spearman={_fmt(baseline_summary['spearman'])} seconds={baseline_summary['seconds']:.1f}", flush=True)
    status = "running"
    failure = ""
    while completed < limit and status == "running":
        epoch = completed + 1
        started = time.perf_counter()
        try:
            trained = _train_epoch(model, decoder, optimizer, device, epoch)
            train_seconds = time.perf_counter() - started
            valid_started = time.perf_counter()
            summary, rows = _evaluate(model, decoder, device, "validation")
            valid_seconds = time.perf_counter() - valid_started
        except (RuntimeError, MemoryError, ValueError) as exc:
            status = "blocked"
            failure = str(exc)
            _write_json(OUTPUT / "failure.json", {"epoch": epoch, "failure": failure, "history": history})
            print(f"blocked epoch {epoch}: {failure}", flush=True)
            break
        summary["validation_loss"] = float(summary["loss"])
        summary["train_loss"] = float(trained["loss"])
        summary["grad_norm_mean"] = float(trained["grad_norm_mean"])
        summary["grad_norm_max"] = float(trained["grad_norm_max"])
        summary["steps"] = int(trained["steps"])
        summary["loss"] = float(summary["validation_loss"])
        summary["epoch"] = epoch
        summary["train_seconds"] = train_seconds
        summary["validation_seconds"] = valid_seconds
        summary["epoch_seconds"] = train_seconds + valid_seconds
        summary["rss_bytes"] = _rss_bytes()
        summary["selection_key"] = list(selection_key(summary))
        history.append(summary)
        if best is None or selection_key(summary) > tuple(best["selection_key"]):
            best = {"epoch": epoch, "selection_key": list(selection_key(summary)), "summary": summary}
            stall = 0
        else:
            stall = int(history[-2]["stall"]) + 1 if len(history) > 1 else 1
        summary["stall"] = stall
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best": best,
            "contract": contract,
            "decoder_frozen": True,
        }
        _save_checkpoint(OUTPUT / f"epoch_{epoch:03d}.pt", payload)
        _write_json(OUTPUT / "history.json", {"history": history, "best_epoch": int(best["epoch"]), "status": "running"})
        _write_json(OUTPUT / f"epoch_{epoch:03d}_validation.json", {"summary": summary, "rows": rows})
        print(
            f"epoch {epoch} train_loss={_fmt(summary['train_loss'])} val_loss={_fmt(summary['validation_loss'])} spearman={_fmt(summary['spearman'])} top1={_fmt(summary['top1'])} seconds={summary['epoch_seconds']:.1f}",
            flush=True,
        )
        completed = epoch
        if float(summary["collapsed"]) > float(COLLAPSE_STOP):
            status = "blocked"
            failure = f"validation score collapse fraction {summary['collapsed']:.3f}"
            break
        if stall >= int(PATIENCE):
            status = "patience"
            break
    if status == "running":
        status = "complete"
    if best is None:
        result = {"status": status, "failure": failure, "gate": _gate(status, None), "history": history, "contract": contract, "device": device.type}
        _write_json(OUTPUT / "result.json", result)
        return result
    best_payload = torch.load(OUTPUT / f"epoch_{int(best['epoch']):03d}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model"])
    model.to(device)
    best_summary, best_rows = _evaluate(model, decoder, device, "validation")
    paired = _paired(best_rows, baseline_rows)
    oracle_payload = float(np.mean([float(row["oracle_payload"]) for row in best_rows]))
    comparison = {
        "frozen": baseline_summary,
        "best": best_summary,
        "best_epoch": int(best["epoch"]),
        "paired": paired,
        "oracle_payload": oracle_payload,
        "oracle_normalized_regret": 0.0,
        "oracle_top1": 1.0,
        "oracle_top3": 1.0,
        "oracle_top5": 1.0,
        "clear_improvement": clear_improvement(
            {**best_summary, "payload": best_summary["payload"]},
            {**baseline_summary, "payload": baseline_summary["payload"]},
        ),
        "encoder_sha256_after": _sha256(ENCODER_CKPT),
        "decoder_sha256_after": _sha256(DECODER_CKPT),
        "encoder_unchanged": _sha256(ENCODER_CKPT) == contract["encoder_sha256"],
        "decoder_unchanged": _sha256(DECODER_CKPT) == contract["decoder_sha256"],
    }
    _save_checkpoint(OUTPUT / "best.pt", {"epoch": int(best["epoch"]), "model": model.state_dict(), "contract": contract, "decoder_frozen": True})
    _write_json(OUTPUT / "best_states.json", best_rows)
    if failure:
        status = "blocked"
    result = {
        "status": status,
        "failure": failure,
        "gate": _gate(status, comparison),
        "device": device.type,
        "history": history,
        "comparison": comparison,
        "contract": contract,
    }
    _write_json(OUTPUT / "result.json", result)
    _write_json(OUTPUT / "history.json", {"history": history, "best_epoch": int(best["epoch"]), "status": status, "failure": failure})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    print(json.dumps(_json_ready({"status": run_training(int(args.epochs))["status"]}), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
