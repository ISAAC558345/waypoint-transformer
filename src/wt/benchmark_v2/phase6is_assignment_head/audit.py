"""Train only a new assignment-conditioned scoring head.

The Epoch 6 encoder and the matching decoder stay frozen. Reserved scenarios
are not opened. Existing checkpoint files are hashed before and after.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import _paths, load_decision
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import PRIMARY_TIMES, assert_open
from wt.benchmark_v2.phase6id_joint_training.metrics import state_metrics
from wt.benchmark_v2.phase6is_assignment_head.features import assignment_raw_features
from wt.benchmark_v2.phase6is_assignment_head.head import AssignmentConditionedHead
from wt.benchmark_v2.phase6is_assignment_head.loss import pairwise_rank_loss
from wt.benchmark_v2.phase6is_assignment_head.protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DECODER_CKPT,
    ENCODER_CKPT,
    EPOCH6_CKPT,
    GRAD_CLIP,
    LEARNING_RATE,
    MATERIAL_BITS,
    MAX_EPOCHS,
    MIN_POSITIVE_SCENARIOS,
    OUTPUT,
    PUBLISHED_TOP1_COUNTS,
    RAW_NAMES,
    SCHEME_C_CKPT,
    SEEDS,
    SWAP_PROBE,
    TRAIN_POOL,
    VALIDATION_POOL,
    WEIGHT_DECAY,
    assert_output_path,
    assign_gate,
    seed_supported,
)
from wt.htw_phase9.const import BUOY_DIM
from wt.lowlevel_v2.phase5k_joint_decoder.decoders import JointAssignmentDecoder
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN, append_queue_edge, batch_tensors

PERMUTATION_SCORE_ATOL = 1.0e-3
PERMUTATION_FEATURE_ATOL = 1.0e-6
TOKEN_CHANGE_ATOL = 1.0e-4


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


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
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_hashes() -> dict[str, str]:
    return {
        "frozen_encoder": _sha256(ENCODER_CKPT),
        "decoder": _sha256(DECODER_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "scheme_c": _sha256(SCHEME_C_CKPT),
    }


def _write_json(path: Path, payload: dict) -> None:
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _split_of(scenario: int) -> str:
    if int(scenario) in TRAIN_POOL:
        return "train"
    if int(scenario) in VALIDATION_POOL:
        return "validation"
    raise RuntimeError(f"phase-6is refused scenario {scenario}")


def _load_bundle(scenario: int, time_step: int) -> tuple[dict, dict[str, np.ndarray]]:
    assert_open(int(scenario), _split_of(scenario))
    sample = load_decision(int(scenario), int(time_step))
    state_path, _candidates, _manifest = _paths(int(scenario), int(time_step))
    with np.load(state_path) as state_file:
        arrays = {key: np.asarray(state_file[key]) for key in state_file.files}
    if "service_progress" not in arrays:
        raise RuntimeError(f"scenario {scenario} t={time_step} has no service_progress")
    if int(sample["assignments"].shape[0]) != int(sample["j15_bits"].shape[0]):
        raise RuntimeError("assignment rows and J15 rows differ")
    return sample, arrays


def _load_hgnn(path: Path, device: torch.device) -> JointUtilityHGNN:
    model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_decoder(device: torch.device) -> JointAssignmentDecoder:
    decoder = JointAssignmentDecoder("matching")
    payload = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    decoder.load_state_dict(payload["state_dict"])
    decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def _encode(model: JointUtilityHGNN, sample: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conditioned = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    with torch.no_grad():
        h_u, h_b = model.condition(tensors)
        edge_h = model.encoder.edge_enc(tensors["edge"])
    return h_u[0].detach().cpu(), h_b[0].detach().cpu(), edge_h[0].detach().cpu()


def _state_record(scenario: int, time_step: int, model: JointUtilityHGNN, device: torch.device, config) -> dict:
    sample, arrays = _load_bundle(scenario, time_step)
    h_u, h_b, edge_h = _encode(model, sample, device)
    raw = torch.as_tensor(assignment_raw_features(arrays, sample["assignments"], config), dtype=torch.float32)
    return {
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "sample": sample,
        "arrays": arrays,
        "h_u": h_u,
        "h_b": h_b,
        "edge_h": edge_h,
        "raw": raw,
        "assignments": torch.as_tensor(sample["assignments"], dtype=torch.long),
        "labels": torch.as_tensor(sample["j15_bits"], dtype=torch.float32),
        "ranks": np.asarray(sample["utility_rank"], dtype=np.int64),
    }


def _score_head(head: AssignmentConditionedHead, record: dict, device: torch.device) -> torch.Tensor:
    return head(
        record["h_u"].to(device),
        record["h_b"].to(device),
        record["edge_h"].to(device),
        record["raw"].to(device),
        record["assignments"].to(device),
    )


def _ranking_row(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray, scenario: int, time_step: int) -> dict:
    measured = state_metrics(scores, labels, ranks)
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    place = np.empty(values.shape[0], dtype=np.int64)
    place[order] = np.arange(1, values.shape[0] + 1)
    optimal = np.asarray(ranks) == 1
    measured["group_rank"] = float(np.min(place[optimal]))
    measured["exact_rank"] = float(place[int(np.argmax(np.asarray(labels, dtype=np.float64)))])
    measured["scenario_id"] = int(scenario)
    measured["time_step"] = int(time_step)
    measured["top1_top2_gap"] = float(values[order[0]] - values[order[1]])
    return measured


def _permute_graph(graph: dict, perm: np.ndarray) -> dict:
    updated = {key: np.array(value, copy=True) for key, value in graph.items()}
    updated["uav"] = graph["uav"][:, perm]
    updated["uav_mask"] = graph["uav_mask"][:, perm]
    updated["edge"] = graph["edge"][:, perm]
    updated["edge_uu"] = graph["edge_uu"][:, perm][:, :, perm]
    return updated


def _run_checks(model: JointUtilityHGNN, device: torch.device, config) -> dict:
    torch.manual_seed(0)
    head = AssignmentConditionedHead().to(device)
    train_record = _state_record(322000, 15, model, device, config)
    probe_scenario, probe_time, swap = SWAP_PROBE
    probe = _state_record(probe_scenario, probe_time, model, device, config)
    head.train()
    scores = _score_head(head, train_record, device)
    loss = pairwise_rank_loss(scores, train_record["labels"].to(device))
    loss.backward()
    head_grads = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    encoder_open = any(bool(parameter.requires_grad) for parameter in model.parameters())
    grad_finite = bool(head_grads) and all(bool(torch.isfinite(grad).all()) for grad in head_grads)
    grad_nonzero = bool(head_grads) and float(max(float(grad.abs().max()) for grad in head_grads)) > 0.0
    score_finite = bool(torch.isfinite(scores).all()) and bool(torch.isfinite(loss).all())
    score_std = float(scores.detach().std())
    head.eval()
    with torch.no_grad():
        first = _score_head(head, probe, device)
        second = _score_head(head, probe, device)
        repeat_gap = float((first - second).abs().max())
        other = AssignmentConditionedHead().to(device).eval()
        other_scores = _score_head(other, probe, device)
        seed_gap = float((first - other_scores).abs().mean())
        base = probe["assignments"].numpy()
        swapped = base.copy()
        left, right = int(swap[0]), int(swap[1])
        swapped[:, [left, right]] = swapped[:, [right, left]]
        swapped_record = dict(probe)
        swapped_record["assignments"] = torch.as_tensor(swapped, dtype=torch.long)
        swapped_record["raw"] = torch.as_tensor(assignment_raw_features(probe["arrays"], swapped, config), dtype=torch.float32)
        base_tokens = head.tokens(
            probe["h_u"].to(device), probe["h_b"].to(device), probe["edge_h"].to(device), probe["raw"].to(device), probe["assignments"].to(device)
        )
        swap_tokens = head.tokens(
            swapped_record["h_u"].to(device),
            swapped_record["h_b"].to(device),
            swapped_record["edge_h"].to(device),
            swapped_record["raw"].to(device),
            swapped_record["assignments"].to(device),
        )
        changed = float((base_tokens[:, [left, right]] - swap_tokens[:, [left, right]]).abs().mean())
        unchanged_ids = [index for index in range(int(base.shape[1])) if index not in {left, right}]
        unchanged = float((base_tokens[:, unchanged_ids, : head.width * 2] - swap_tokens[:, unchanged_ids, : head.width * 2]).abs().max())
        edge_changed = float((base_tokens[:, [left, right], head.width * 2 : head.width * 3] - swap_tokens[:, [left, right], head.width * 2 : head.width * 3]).abs().mean())
        perm = np.asarray([2, 0, 1, 4, 3, 5], dtype=np.int64)
        permuted = dict(probe)
        permuted["sample"] = dict(probe["sample"])
        permuted["sample"]["graph"] = _permute_graph(probe["sample"]["graph"], perm)
        encoded = _encode(model, permuted["sample"], device)
        permuted["h_u"], permuted["h_b"], permuted["edge_h"] = encoded
        permuted["assignments"] = probe["assignments"][:, perm]
        permuted["raw"] = probe["raw"][:, perm]
        permuted_scores = _score_head(head, permuted, device)
        permutation_gap = float((first - permuted_scores).abs().max())
        feature_perm = assignment_raw_features(_permute_arrays(probe["arrays"], perm), probe["assignments"].numpy()[:, perm], config)
        feature_gap = float(np.max(np.abs(feature_perm - probe["raw"].numpy()[:, perm])))
    passed = bool(
        score_finite
        and grad_finite
        and grad_nonzero
        and not encoder_open
        and score_std > 1.0e-6
        and repeat_gap <= 1.0e-6
        and seed_gap > 1.0e-6
        and changed > TOKEN_CHANGE_ATOL
        and edge_changed > TOKEN_CHANGE_ATOL
        and unchanged <= TOKEN_CHANGE_ATOL
        and permutation_gap <= PERMUTATION_SCORE_ATOL
        and feature_gap <= PERMUTATION_FEATURE_ATOL
    )
    return {
        "passed": passed,
        "score_finite": score_finite,
        "loss": float(loss.detach()),
        "score_std": score_std,
        "grad_finite": grad_finite,
        "grad_nonzero": grad_nonzero,
        "encoder_trainable": encoder_open,
        "repeat_gap": repeat_gap,
        "init_seed_mean_abs_gap": seed_gap,
        "swap_token_mean_abs": changed,
        "swap_edge_mean_abs": edge_changed,
        "unchanged_uav_embedding_max_abs": unchanged,
        "permutation_score_gap": permutation_gap,
        "permutation_feature_gap": feature_gap,
        "probe_assignment_rows": int(base.shape[0]),
        "raw_dim": int(probe["raw"].shape[-1]),
        "raw_names": list(RAW_NAMES),
        "h_u_shape": list(probe["h_u"].shape),
        "h_b_shape": list(probe["h_b"].shape),
        "edge_h_shape": list(probe["edge_h"].shape),
        "graph_shapes": {key: list(np.asarray(value).shape) for key, value in probe["sample"]["graph"].items()},
    }


def _permute_arrays(arrays: dict[str, np.ndarray], perm: np.ndarray) -> dict[str, np.ndarray]:
    updated = {key: np.array(value, copy=True) for key, value in arrays.items()}
    for key in (
        "uav_position_m",
        "uav_step_displacement_m",
        "association",
        "service_progress",
        "uav_buoy_distance_m",
        "uav_buoy_gain",
        "uav_buoy_in_range",
        "uav_buoy_hypothetical_rate",
    ):
        updated[key] = np.asarray(arrays[key])[perm]
    distance = np.asarray(arrays["uav_uav_distance_m"])
    updated["uav_uav_distance_m"] = distance[perm][:, perm]
    return updated


def _score_matching(model: JointUtilityHGNN, decoder: JointAssignmentDecoder, sample: dict, device: torch.device) -> np.ndarray:
    conditioned = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
    with torch.no_grad():
        h_u, h_b = model.condition(tensors)
        scores = decoder.score_assignments(h_u, h_b, assignments)
    return scores.detach().cpu().numpy()


def _evaluate_matching(name: str, model: JointUtilityHGNN, decoder: JointAssignmentDecoder, device: torch.device) -> dict:
    rows = []
    started = time.perf_counter()
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            sample, _arrays = _load_bundle(scenario, time_step)
            one = time.perf_counter()
            scores = _score_matching(model, decoder, sample, device)
            elapsed = time.perf_counter() - one
            row = _ranking_row(scores, sample["j15_bits"], sample["utility_rank"], scenario, time_step)
            row["seconds"] = float(elapsed)
            rows.append(row)
            print(f"baseline {name} {scenario} t={time_step} top1={row['top1']:.0f} seconds={elapsed:.3f}", flush=True)
    hits = int(sum(int(row["top1"]) for row in rows))
    return {
        "name": name,
        "rows": rows,
        "top1_hits": hits,
        "seconds_total": float(time.perf_counter() - started),
        "seconds_median": float(np.median([row["seconds"] for row in rows])),
        "matches_published": hits == int(PUBLISHED_TOP1_COUNTS[name]),
    }


def _build_cache(model: JointUtilityHGNN, device: torch.device, config, split: str) -> list[dict]:
    pool = TRAIN_POOL if split == "train" else VALIDATION_POOL
    cache = []
    for scenario in pool:
        for time_step in PRIMARY_TIMES:
            record = _state_record(scenario, time_step, model, device, config)
            record.pop("sample")
            record.pop("arrays")
            cache.append(record)
            print(f"cache {split} {scenario} t={time_step}", flush=True)
    return cache


def _epoch_rows(head: AssignmentConditionedHead, cache: list[dict], device: torch.device) -> list[dict]:
    head.eval()
    rows = []
    with torch.no_grad():
        for record in cache:
            scores = _score_head(head, record, device).detach().cpu().numpy()
            rows.append(_ranking_row(scores, record["labels"].numpy(), record["ranks"], record["scenario_id"], record["time_step"]))
    return rows


def _summarize(rows: list[dict]) -> dict:
    payloads = np.asarray([float(row["payload"]) for row in rows], dtype=np.float64)
    return {
        "n_states": int(len(rows)),
        "top1": float(np.mean([row["top1"] for row in rows])),
        "top3": float(np.mean([row["top3"] for row in rows])),
        "top5": float(np.mean([row["top5"] for row in rows])),
        "normalized_regret": float(np.mean([row["normalized_regret"] for row in rows])),
        "payload_mean": float(np.mean(payloads)),
        "payload_median": float(np.median(payloads)),
        "group_rank_median": float(np.median([row["group_rank"] for row in rows])),
        "exact_rank_median": float(np.median([row["exact_rank"] for row in rows])),
        "score_gap_median": float(np.median([row["top1_top2_gap"] for row in rows])),
    }


def _slice_summary(rows: list[dict], time_step: int) -> dict:
    chosen = [row for row in rows if int(row["time_step"]) == int(time_step)]
    packed = _summarize(chosen)
    packed["time_step"] = int(time_step)
    return packed


def _paired_matrix(rows: list[dict], baseline_rows: list[dict]) -> np.ndarray:
    baseline = {(int(row["scenario_id"]), int(row["time_step"])): row for row in baseline_rows}
    scenarios = list(VALIDATION_POOL)
    times = list(PRIMARY_TIMES)
    matrix = np.zeros((len(scenarios), len(times)), dtype=np.float64)
    for row_index, scenario in enumerate(scenarios):
        for column, time_step in enumerate(times):
            key = (int(scenario), int(time_step))
            current = next(row for row in rows if int(row["scenario_id"]) == key[0] and int(row["time_step"]) == key[1])
            matrix[row_index, column] = float(current["payload"]) - float(baseline[key]["payload"])
    return matrix


def _interval(matrix: np.ndarray) -> dict:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, matrix.shape[0], size=(BOOTSTRAP_DRAWS, matrix.shape[0]))
    means = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    medians = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index, draw in enumerate(draws):
        sample = matrix[draw].reshape(-1)
        means[index] = float(np.mean(sample))
        medians[index] = float(np.median(sample))
    totals = matrix.sum(axis=1)
    return {
        "mean": float(np.mean(matrix)),
        "mean_low": float(np.percentile(means, 2.5)),
        "mean_high": float(np.percentile(means, 97.5)),
        "median": float(np.median(matrix)),
        "positive_scenarios": int(np.sum(totals > MATERIAL_BITS)),
        "negative_scenarios": int(np.sum(totals < -MATERIAL_BITS)),
    }


def _decision_flags(rows: list[dict], epoch6_rows: list[dict], scheme_rows: list[dict]) -> dict:
    summary = _summarize(rows)
    t15 = _slice_summary(rows, 15)
    epoch6 = _summarize(epoch6_rows)
    scheme = _summarize(scheme_rows)
    epoch6_t15 = _slice_summary(epoch6_rows, 15)
    scheme_t15 = _slice_summary(scheme_rows, 15)
    versus_epoch6 = _interval(_paired_matrix(rows, epoch6_rows))
    versus_scheme = _interval(_paired_matrix(rows, scheme_rows))
    t15_epoch6 = _interval(_paired_matrix(rows, epoch6_rows)[:, 1:2])
    t15_scheme = _interval(_paired_matrix(rows, scheme_rows)[:, 1:2])
    flags = {
        "summary": summary,
        "t15": t15,
        "versus_epoch6": versus_epoch6,
        "versus_scheme_c": versus_scheme,
        "t15_versus_epoch6": t15_epoch6,
        "t15_versus_scheme_c": t15_scheme,
        "top1_above_epoch6": bool(summary["top1"] > epoch6["top1"]),
        "top1_above_scheme_c": bool(summary["top1"] > scheme["top1"]),
        "t15_top1_above_epoch6": bool(t15["top1"] > epoch6_t15["top1"]),
        "t15_top1_above_scheme_c": bool(t15["top1"] > scheme_t15["top1"]),
        "mean_ci_above_epoch6": bool(versus_epoch6["mean_low"] > 0.0),
        "mean_ci_above_scheme_c": bool(versus_scheme["mean_low"] > 0.0),
        "median_above_epoch6": bool(versus_epoch6["median"] > MATERIAL_BITS),
        "median_above_scheme_c": bool(versus_scheme["median"] > MATERIAL_BITS),
        "t15_mean_ci_above_epoch6": bool(t15_epoch6["mean_low"] > 0.0),
        "t15_mean_ci_above_scheme_c": bool(t15_scheme["mean_low"] > 0.0),
        "scenarios_consistent_epoch6": bool(
            versus_epoch6["positive_scenarios"] >= MIN_POSITIVE_SCENARIOS
            and versus_epoch6["positive_scenarios"] > versus_epoch6["negative_scenarios"]
        ),
        "scenarios_consistent_scheme_c": bool(
            versus_scheme["positive_scenarios"] >= MIN_POSITIVE_SCENARIOS
            and versus_scheme["positive_scenarios"] > versus_scheme["negative_scenarios"]
        ),
    }
    flags["supported"] = seed_supported(flags)
    return flags


def _train_seed(seed: int, train_cache: list[dict], valid_cache: list[dict], baselines: dict, device: torch.device) -> dict:
    torch.manual_seed(int(seed))
    head = AssignmentConditionedHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    best = None
    folder = assert_output_path(OUTPUT / f"seed_{seed}")
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        losses = []
        grad_norms = []
        started = time.perf_counter()
        for record in train_cache:
            optimizer.zero_grad(set_to_none=True)
            scores = _score_head(head, record, device)
            loss = pairwise_rank_loss(scores, record["labels"].to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at seed {seed} epoch {epoch}")
            loss.backward()
            grad_norm = float(clip_grad_norm_(head.parameters(), GRAD_CLIP))
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(grad_norm)
        rows = _epoch_rows(head, valid_cache, device)
        summary = _summarize(rows)
        summary["epoch"] = epoch
        summary["train_loss"] = float(np.mean(losses))
        summary["grad_norm_mean"] = float(np.mean(grad_norms))
        summary["grad_norm_max"] = float(np.max(grad_norms))
        summary["seconds"] = float(time.perf_counter() - started)
        summary["rss_bytes"] = _rss_bytes()
        history.append(summary)
        key = (summary["payload_mean"], summary["top1"], -summary["normalized_regret"], -summary["train_loss"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch, "rows": rows}
            torch.save({"epoch": epoch, "seed": int(seed), "head": head.state_dict(), "encoder": str(EPOCH6_CKPT)}, folder / "head.pt")
        print(
            f"seed {seed} epoch {epoch} loss={summary['train_loss']:.4f} top1={summary['top1']:.4f} payload={summary['payload_mean']:.1f} seconds={summary['seconds']:.1f}",
            flush=True,
        )
    flags = _decision_flags(best["rows"], baselines["epoch6"]["rows"], baselines["scheme_c"]["rows"])
    flags["seed"] = int(seed)
    flags["selected_epoch"] = int(best["epoch"])
    flags["history"] = history
    flags["rows"] = best["rows"]
    flags["chosen_disagreements"] = int(
        sum(
            int(row["chosen"]) != int(other["chosen"])
            for row, other in zip(best["rows"], baselines["epoch6"]["rows"])
        )
    )
    return flags


def _cold_timing(head: AssignmentConditionedHead, model: JointUtilityHGNN, device: torch.device, config) -> dict:
    head.eval()
    elapsed = []
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            sample, arrays = _load_bundle(scenario, time_step)
            started = time.perf_counter()
            with torch.no_grad():
                h_u, h_b, edge_h = _encode(model, sample, device)
                raw = torch.as_tensor(assignment_raw_features(arrays, sample["assignments"], config), dtype=torch.float32)
                head(h_u.to(device), h_b.to(device), edge_h.to(device), raw.to(device), torch.as_tensor(sample["assignments"], dtype=torch.long, device=device))
            elapsed.append(time.perf_counter() - started)
    return {"seconds_median": float(np.median(elapsed)), "seconds_max": float(np.max(elapsed)), "seconds_total": float(np.sum(elapsed)), "rss_bytes": _rss_bytes()}


def _public_rows(rows: list[dict]) -> list[dict]:
    kept = ("scenario_id", "time_step", "top1", "top3", "top5", "payload", "normalized_regret", "group_rank", "exact_rank", "chosen", "top1_top2_gap")
    return [{key: row[key] for key in kept} for row in rows]


def run_audit() -> dict:
    device = torch.device("cpu")
    config = L3BBitQueueEnvV2().config
    before = _checkpoint_hashes()
    model = _load_hgnn(EPOCH6_CKPT, device)
    checks = _run_checks(model, device, config)
    print(f"checks passed={checks['passed']}", flush=True)
    blocked = not bool(checks["passed"])
    baselines = None
    seeds = []
    timing = None
    if not blocked:
        decoder = _load_decoder(device)
        epoch6 = _evaluate_matching("epoch6", model, decoder, device)
        scheme = _evaluate_matching("scheme_c", _load_hgnn(SCHEME_C_CKPT, device), decoder, device)
        baselines = {"epoch6": epoch6, "scheme_c": scheme}
        blocked = not (bool(epoch6["matches_published"]) and bool(scheme["matches_published"]))
        print(f"published top1 epoch6={epoch6['top1_hits']} scheme_c={scheme['top1_hits']} blocked={blocked}", flush=True)
    if not blocked:
        train_cache = _build_cache(model, device, config, "train")
        valid_cache = _build_cache(model, device, config, "validation")
        for seed in SEEDS:
            seeds.append(_train_seed(int(seed), train_cache, valid_cache, baselines, device))
        selected = AssignmentConditionedHead().to(device)
        payload = torch.load(OUTPUT / "seed_0" / "head.pt", map_location="cpu", weights_only=False)
        selected.load_state_dict(payload["head"])
        timing = _cold_timing(selected, model, device, config)
    after = _checkpoint_hashes()
    if before != after:
        blocked = True
    gate = assign_gate(blocked, bool(checks["passed"]) and before == after, seeds)
    trainable = int(sum(parameter.numel() for parameter in AssignmentConditionedHead().parameters()))
    frozen = int(sum(parameter.numel() for parameter in model.parameters()))
    result = {
        "gate": gate,
        "device": device.type,
        "checks": checks,
        "checkpoint_hashes_before": before,
        "checkpoint_hashes_after": after,
        "hashes_unchanged": before == after,
        "encoder_parameters_frozen": frozen,
        "head_parameters": trainable,
        "raw_names": list(RAW_NAMES),
        "excluded_inputs": ["j15_bits", "utility_rank", "future_sinr", "mappo_rollout", "kinematic_bits", "straight_line_bits"],
        "hold_constant": int(config.option_hold),
        "loss": "mean(-logsigmoid(s_i-s_j)) for j15_i > j15_j",
        "epochs": int(MAX_EPOCHS),
        "seeds": [int(seed) for seed in SEEDS],
        "learning_rate": float(LEARNING_RATE),
        "baselines": None
        if baselines is None
        else {
            "epoch6": {key: value for key, value in baselines["epoch6"].items() if key != "rows"} | {"summary": _summarize(baselines["epoch6"]["rows"]), "t15": _slice_summary(baselines["epoch6"]["rows"], 15)},
            "scheme_c": {key: value for key, value in baselines["scheme_c"].items() if key != "rows"} | {"summary": _summarize(baselines["scheme_c"]["rows"]), "t15": _slice_summary(baselines["scheme_c"]["rows"], 15)},
        },
        "seed_reports": [
            {
                key: value
                for key, value in report.items()
                if key != "rows"
            }
            | {"rows": _public_rows(report["rows"])}
            for report in seeds
        ],
        "inference": timing,
        "reserved_scenarios_read": 0,
    }
    _write_json(OUTPUT / "summary.json", result)
    print(json.dumps({"gate": gate, "checks": checks["passed"], "head_parameters": trainable}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
