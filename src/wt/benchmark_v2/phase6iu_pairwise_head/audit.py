"""Train only the pairwise interaction head on frozen Epoch 6 unary scores.

No planner, no MAPPO, no encoder update, and no reserved-test reads.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import _paths
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import PRIMARY_TIMES, assert_open
from wt.benchmark_v2.phase6im_local_swap.audit import single_swaps
from wt.benchmark_v2.phase6is_assignment_head.audit import (
    _checkpoint_hashes,
    _load_bundle,
    _load_decoder,
    _load_hgnn,
    _ranking_row,
    _rss_bytes,
    _score_head,
    _score_matching,
    _state_record,
)
from wt.benchmark_v2.phase6is_assignment_head.head import AssignmentConditionedHead
from wt.benchmark_v2.phase6is_assignment_head.loss import pairwise_rank_loss
from wt.benchmark_v2.phase6it_interaction.audit import decompositions
from wt.benchmark_v2.phase6iu_pairwise_head.head import PairwiseInteractionHead
from wt.benchmark_v2.phase6iu_pairwise_head.protocol import (
    ACTIVE_FRACTION,
    BIT_TIE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONCENTRATION,
    EPOCH6_CKPT,
    GRAD_CLIP,
    HEAD_OUTPUT,
    LEARNING_RATE,
    MATERIAL_BITS,
    MAX_EPOCHS,
    MIN_POSITIVE_SCENARIOS,
    OUTPUT,
    PUBLISHED_TOP1_COUNTS,
    REVISION_DATASET,
    SCHEME_C_CKPT,
    SCORE_DUST,
    SEEDS,
    SIGN_AGREEMENT,
    TRAIN_POOL,
    VALIDATION_POOL,
    WEIGHT_DECAY,
    assert_output_path,
    assign_gate,
    seed_supported,
)
from wt.lowlevel_v2.phase5l_joint_encoder.model import append_queue_edge, batch_tensors

SAVED_SCHEME = Path(__file__).resolve().parents[4] / "phase6im_local_swap" / "states.json"


def verify_split() -> dict:
    """Scenario folders stay inside one split. Reserved files are not opened."""
    found = {}
    for split, pool in (("train", TRAIN_POOL), ("validation", VALIDATION_POOL)):
        folder = REVISION_DATASET / split
        names = sorted(int(path.name.split("_")[1]) for path in folder.iterdir() if path.is_dir() and path.name.startswith("scenario_"))
        if tuple(names) != tuple(pool):
            raise RuntimeError(f"phase-6iu {split} folders do not match the scenario protocol")
        for scenario in pool:
            assert_open(int(scenario), split)
            for time_step in PRIMARY_TIMES:
                state_path, candidate_path, manifest_path = _paths(int(scenario), int(time_step))
                if not state_path.is_file() or not candidate_path.is_file() or not manifest_path.is_file():
                    raise RuntimeError(f"phase-6iu missing decision files for {scenario} t={time_step}")
        found[split] = int(len(names))
    if set(TRAIN_POOL) & set(VALIDATION_POOL):
        raise RuntimeError("phase-6iu train and validation scenarios overlap")
    return found


def _frozen_cache(model, decoder, device: torch.device, split: str) -> list[dict]:
    pool = TRAIN_POOL if split == "train" else VALIDATION_POOL
    cache = []
    for scenario in pool:
        for time_step in PRIMARY_TIMES:
            sample, _arrays = _load_bundle(int(scenario), int(time_step))
            if int(sample["assignments"].shape[0]) != 13327:
                raise RuntimeError(f"scenario {scenario} t={time_step} does not have 13327 candidates")
            graph = append_queue_edge(sample["graph"])
            tensors = {key: value.to(device) for key, value in batch_tensors(graph).items()}
            assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
            with torch.no_grad():
                h_u, h_b = model.condition(tensors)
                unary = decoder.score_assignments(h_u, h_b, assignments).detach().cpu()
                edge_h = model.encoder.edge_enc(tensors["edge"])[0].detach().cpu()
            cache.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(time_step),
                    "edge_h": edge_h,
                    "assignments": assignments.cpu(),
                    "unary": unary,
                    "labels": torch.as_tensor(sample["j15_bits"], dtype=torch.float64),
                    "ranks": np.asarray(sample["utility_rank"], dtype=np.int64),
                }
            )
            print(f"cache {split} {scenario} t={time_step}", flush=True)
    return cache


def _ranking(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray, scenario: int, time_step: int) -> dict:
    row = _ranking_row(scores, labels, ranks, scenario, time_step)
    optimal = np.asarray(ranks) == 1
    row["high_gain_margin"] = float(np.max(scores[optimal]) - np.max(scores[~optimal]))
    return row


def _score_interaction(head: PairwiseInteractionHead, record: dict, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return head(record["edge_h"].to(device), record["assignments"].to(device), record["unary"].to(device)).detach().cpu().numpy()


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
        "high_gain_margin_median": float(np.median([row["high_gain_margin"] for row in rows])),
        "collapsed_states": int(sum(float(row["collapsed"]) > 0.0 for row in rows)),
    }


def _slice_summary(rows: list[dict], time_step: int) -> dict:
    packed = _summarize([row for row in rows if int(row["time_step"]) == int(time_step)])
    packed["time_step"] = int(time_step)
    return packed


def _paired_matrix(rows: list[dict], baseline_rows: list[dict]) -> np.ndarray:
    baseline = {(int(row["scenario_id"]), int(row["time_step"])): row for row in baseline_rows}
    matrix = np.zeros((len(VALIDATION_POOL), len(PRIMARY_TIMES)), dtype=np.float64)
    for row_index, scenario in enumerate(VALIDATION_POOL):
        for column, time_step in enumerate(PRIMARY_TIMES):
            current = next(row for row in rows if int(row["scenario_id"]) == int(scenario) and int(row["time_step"]) == int(time_step))
            matrix[row_index, column] = float(current["payload"]) - float(baseline[(int(scenario), int(time_step))]["payload"])
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
        "positive_scenarios": int(np.sum(totals > BIT_TIE)),
        "negative_scenarios": int(np.sum(totals < -BIT_TIE)),
    }


def _quad_accuracy(scores: list[float], labels: list[float]) -> float | None:
    correct = 0
    total = 0
    for left in range(4):
        for right in range(4):
            if float(labels[left]) <= float(labels[right]) + BIT_TIE:
                continue
            total += 1
            correct += int(float(scores[left]) > float(scores[right]))
    if total == 0:
        return None
    return float(correct) / float(total)


def _interaction(record: dict, scores: np.ndarray, scheme_scores: np.ndarray, epoch6_scores: np.ndarray) -> dict:
    assignments = np.asarray(record["assignments"], dtype=np.int64)
    labels = np.asarray(record["labels"], dtype=np.float64)
    table = {tuple(int(item) for item in row): index for index, row in enumerate(assignments)}
    base_index = int(np.argsort(-np.asarray(scheme_scores, dtype=np.float64), kind="stable")[0])
    base = assignments[base_index]
    missing = 0
    j15_values = []
    score_values = []
    epoch6_values = []
    scheme_values = []
    quad_new = []
    quad_scheme = []
    quad_epoch6 = []
    for item in decompositions(base):
        if item["kind"] != "disjoint":
            continue
        indexes = [table.get(tuple(int(value) for value in item[key])) for key in ("joint", "first", "second")]
        if any(index is None for index in indexes):
            missing += 1
            continue
        joint, first, second = (int(index) for index in indexes)
        j15_interaction = float(labels[joint] - labels[first] - labels[second] + labels[base_index])
        new_interaction = float(scores[joint] - scores[first] - scores[second] + scores[base_index])
        old_interaction = float(epoch6_scores[joint] - epoch6_scores[first] - epoch6_scores[second] + epoch6_scores[base_index])
        scheme_interaction = float(scheme_scores[joint] - scheme_scores[first] - scheme_scores[second] + scheme_scores[base_index])
        j15_values.append(j15_interaction)
        score_values.append(new_interaction)
        epoch6_values.append(old_interaction)
        scheme_values.append(scheme_interaction)
        if abs(j15_interaction) <= MATERIAL_BITS:
            continue
        label_rows = [float(labels[base_index]), float(labels[first]), float(labels[second]), float(labels[joint])]
        quad_new.append(_quad_accuracy([float(scores[base_index]), float(scores[first]), float(scores[second]), float(scores[joint])], label_rows))
        quad_scheme.append(_quad_accuracy([float(scheme_scores[base_index]), float(scheme_scores[first]), float(scheme_scores[second]), float(scheme_scores[joint])], label_rows))
        quad_epoch6.append(_quad_accuracy([float(epoch6_scores[base_index]), float(epoch6_scores[first]), float(epoch6_scores[second]), float(epoch6_scores[joint])], label_rows))
    single_labels = []
    single_scores = []
    for _left, _right, row in single_swaps(base):
        index = table.get(tuple(int(item) for item in row))
        if index is None:
            missing += 1
            continue
        single_labels.append(float(labels[index] - labels[base_index]))
        single_scores.append(float(scores[index] - scores[base_index]))
    return {
        "base": [int(item) for item in base],
        "missing": int(missing),
        "j15_interaction": j15_values,
        "score_interaction": score_values,
        "epoch6_interaction": epoch6_values,
        "scheme_interaction": scheme_values,
        "quad_new": quad_new,
        "quad_scheme": quad_scheme,
        "quad_epoch6": quad_epoch6,
        "single_labels": single_labels,
        "single_scores": single_scores,
    }


def _pool_interaction(parts: list[dict]) -> dict:
    j15 = np.asarray([value for part in parts for value in part["j15_interaction"]], dtype=np.float64)
    scores = np.asarray([value for part in parts for value in part["score_interaction"]], dtype=np.float64)
    epoch6 = np.asarray([value for part in parts for value in part["epoch6_interaction"]], dtype=np.float64)
    scheme = np.asarray([value for part in parts for value in part["scheme_interaction"]], dtype=np.float64)
    material = np.abs(j15) > MATERIAL_BITS
    active = material & (np.abs(scores) > SCORE_DUST)
    sign = float(np.mean(np.sign(j15[active]) == np.sign(scores[active]))) if int(np.sum(active)) else 0.0
    quads = [value for part in parts for value in part["quad_new"] if value is not None]
    scheme_quads = [value for part in parts for value in part["quad_scheme"] if value is not None]
    epoch6_quads = [value for part in parts for value in part["quad_epoch6"] if value is not None]
    single_labels = np.asarray([value for part in parts for value in part["single_labels"]], dtype=np.float64)
    single_scores = np.asarray([value for part in parts for value in part["single_scores"]], dtype=np.float64)
    single_material = np.abs(single_labels) > MATERIAL_BITS
    single_active = single_material & (np.abs(single_scores) > SCORE_DUST)
    return {
        "states": int(len(parts)),
        "missing": int(sum(part["missing"] for part in parts)),
        "disjoint": int(j15.size),
        "material": int(np.sum(material)),
        "active_fraction": float(np.mean(np.abs(scores[material]) > SCORE_DUST)) if int(np.sum(material)) else 0.0,
        "sign_agreement": sign,
        "active_count": int(np.sum(active)),
        "j15_interaction_abs_median": float(np.median(np.abs(j15))) if j15.size else None,
        "score_interaction_abs_median": float(np.median(np.abs(scores[material]))) if int(np.sum(material)) else None,
        "epoch6_interaction_abs_max": float(np.max(np.abs(epoch6))) if epoch6.size else None,
        "scheme_interaction_abs_max": float(np.max(np.abs(scheme))) if scheme.size else None,
        "quadruple_accuracy": float(np.mean(quads)) if quads else None,
        "scheme_quadruple_accuracy": float(np.mean(scheme_quads)) if scheme_quads else None,
        "epoch6_quadruple_accuracy": float(np.mean(epoch6_quads)) if epoch6_quads else None,
        "single_count": int(single_labels.size),
        "single_material": int(np.sum(single_material)),
        "single_sign_agreement": float(np.mean(np.sign(single_labels[single_active]) == np.sign(single_scores[single_active]))) if int(np.sum(single_active)) else 0.0,
        "single_score_abs_median": float(np.median(np.abs(single_scores[single_material]))) if int(np.sum(single_material)) else None,
    }


def _saved_scheme_matches(parts: list[dict]) -> int:
    saved = {int(row["scenario_id"]): [int(item) for item in row["assignments"]["scheme_c"]] for row in json.loads(SAVED_SCHEME.read_text(encoding="utf-8"))}
    return int(sum(part["base"] == saved[int(part["scenario_id"])] for part in parts if int(part["scenario_id"]) in saved))


def _decision_flags(rows: list[dict], scheme_rows: list[dict], interaction: dict) -> dict:
    summary = _summarize(rows)
    scheme = _summarize(scheme_rows)
    versus = _interval(_paired_matrix(rows, scheme_rows))
    deltas = _paired_matrix(rows, scheme_rows).reshape(-1)
    positive = deltas[deltas > BIT_TIE]
    concentrated = positive.size == 0 or float(np.max(positive)) >= CONCENTRATION * float(np.sum(positive))
    flags = {
        "summary": summary,
        "t0": _slice_summary(rows, 0),
        "t15": _slice_summary(rows, 15),
        "t30": _slice_summary(rows, 30),
        "versus_scheme_c": versus,
        "top1_above_scheme_c": bool(summary["top1"] > scheme["top1"]),
        "regret_not_worse": bool(summary["normalized_regret"] <= scheme["normalized_regret"]),
        "mean_ci_above_scheme_c": bool(versus["mean_low"] > 0.0),
        "scenarios_consistent": bool(
            versus["positive_scenarios"] >= MIN_POSITIVE_SCENARIOS and versus["positive_scenarios"] > versus["negative_scenarios"]
        ),
        "gain_not_concentrated": bool(not concentrated),
        "interaction_active": bool(float(interaction["active_fraction"]) >= ACTIVE_FRACTION),
        "interaction_sign": bool(float(interaction["sign_agreement"]) >= SIGN_AGREEMENT and int(interaction["active_count"]) > 0),
        "quadruple_improves": bool(
            interaction["quadruple_accuracy"] is not None
            and interaction["scheme_quadruple_accuracy"] is not None
            and float(interaction["quadruple_accuracy"]) > float(interaction["scheme_quadruple_accuracy"])
        ),
        "interaction": interaction,
    }
    flags["supported"] = seed_supported(flags)
    return flags


def _public_rows(rows: list[dict]) -> list[dict]:
    kept = ("scenario_id", "time_step", "top1", "top3", "top5", "payload", "normalized_regret", "group_rank", "exact_rank", "chosen", "top1_top2_gap", "high_gain_margin", "collapsed")
    return [{key: row[key] for key in kept} for row in rows]


def _run_checks(model, decoder, device: torch.device) -> dict:
    torch.manual_seed(0)
    head = PairwiseInteractionHead().to(device)
    sample, _arrays = _load_bundle(322000, 15)
    graph = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(graph).items()}
    assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
    with torch.no_grad():
        h_u, h_b = model.condition(tensors)
        unary = decoder.score_assignments(h_u, h_b, assignments)
        edge_h = model.encoder.edge_enc(tensors["edge"])[0]
    head.train()
    order = np.argsort(np.asarray(sample["j15_bits"], dtype=np.float64), kind="stable")
    probe_index = torch.as_tensor(order[np.linspace(0, order.size - 1, 32).astype(np.int64)], dtype=torch.long, device=device)
    scores = head(edge_h, assignments[probe_index], unary[probe_index])
    labels = torch.as_tensor(sample["j15_bits"], dtype=torch.float64, device=device)[probe_index]
    loss = pairwise_rank_loss(scores, labels)
    loss.backward()
    grad_norm = float(torch.linalg.vector_norm(torch.cat([parameter.grad.detach().reshape(-1) for parameter in head.parameters()])))
    finite = bool(torch.isfinite(scores).all().item()) and bool(torch.isfinite(loss).item()) and bool(np.isfinite(grad_norm))
    nonzero = grad_norm > 0.0
    frozen = all(not parameter.requires_grad for parameter in model.parameters()) and all(not parameter.requires_grad for parameter in decoder.parameters())
    with torch.no_grad():
        repeated = head(edge_h, assignments[:1].repeat(2, 1), unary[:1].repeat(2))
        repeat_gap = float((repeated[0] - repeated[1]).abs())
        perm = torch.tensor([1, 0, 2, 3, 4, 5], device=device)
        probe = assignments[:1]
        probe_unary = unary[:1]
        permuted_edges = edge_h[perm]
        permuted_assignment = probe[:, perm]
        logits = decoder.logits(h_u, h_b)
        chosen = permuted_assignment[0].clamp(min=0)
        valid = (permuted_assignment[0] >= 0).to(logits.dtype)
        permuted_unary = (logits[perm, chosen] * valid).sum()
        original = head(edge_h, probe, probe_unary)
        swapped = head(permuted_edges, permuted_assignment, permuted_unary.reshape(1))
        permutation_gap = float((original - swapped).abs())
        pair_only = head.pair_score(edge_h, probe)
        pair_swapped = head.pair_score(permuted_edges, permuted_assignment)
        pair_gap = float((pair_only - pair_swapped).abs())
    passed = bool(finite and nonzero and frozen and repeat_gap < 1.0e-6 and permutation_gap < 1.0e-4 and pair_gap < 1.0e-4)
    return {
        "passed": passed,
        "finite": finite,
        "grad_norm": grad_norm,
        "frozen": frozen,
        "repeat_gap": repeat_gap,
        "permutation_gap": permutation_gap,
        "pair_gap": pair_gap,
        "lambda_pair": float(head.lambda_pair),
    }


def _matching_rows(name: str, model, decoder, device: torch.device) -> dict:
    rows = []
    t15_scores = {}
    elapsed = []
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            sample, _arrays = _load_bundle(int(scenario), int(time_step))
            started = time.perf_counter()
            scores = _score_matching(model, decoder, sample, device)
            elapsed.append(time.perf_counter() - started)
            rows.append(_ranking(scores, sample["j15_bits"], sample["utility_rank"], int(scenario), int(time_step)))
            if int(time_step) == 15:
                t15_scores[int(scenario)] = scores
            print(f"baseline {name} {scenario} t={time_step} top1={rows[-1]['top1']:.0f}", flush=True)
    hits = int(sum(int(row["top1"]) for row in rows))
    return {
        "name": name,
        "rows": rows,
        "t15_scores": t15_scores,
        "top1_hits": hits,
        "seconds_median": float(np.median(elapsed)),
        "matches_published": hits == int(PUBLISHED_TOP1_COUNTS[name]),
    }


def _conditioned_rows(seed: int, device: torch.device, config) -> dict:
    head = AssignmentConditionedHead().to(device)
    payload = torch.load(HEAD_OUTPUT / f"seed_{seed}" / "head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(payload["head"])
    head.eval()
    model = _load_hgnn(EPOCH6_CKPT, device)
    rows = []
    t15_scores = {}
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            record = _state_record(int(scenario), int(time_step), model, device, config)
            scores = _score_head(head, record, device).detach().cpu().numpy()
            rows.append(_ranking(scores, record["labels"].numpy(), record["ranks"], int(scenario), int(time_step)))
            if int(time_step) == 15:
                t15_scores[int(scenario)] = scores
            print(f"baseline conditioned_seed{seed} {scenario} t={time_step} top1={rows[-1]['top1']:.0f}", flush=True)
    return {"name": f"conditioned_seed{seed}", "rows": rows, "t15_scores": t15_scores, "selected_epoch": int(payload["epoch"])}


def _train_seed(seed: int, train_cache: list[dict], valid_cache: list[dict], device: torch.device) -> tuple[PairwiseInteractionHead, list[dict]]:
    torch.manual_seed(int(seed))
    head = PairwiseInteractionHead().to(device)
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
            scores = head(record["edge_h"].to(device), record["assignments"].to(device), record["unary"].to(device))
            loss = pairwise_rank_loss(scores, record["labels"].to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at seed {seed} epoch {epoch}")
            loss.backward()
            grad_norm = float(clip_grad_norm_(head.parameters(), GRAD_CLIP))
            if not np.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient at seed {seed} epoch {epoch}")
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(grad_norm)
        rows = []
        head.eval()
        for record in valid_cache:
            values = _score_interaction(head, record, device)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"non-finite validation score at seed {seed} epoch {epoch}")
            rows.append(_ranking(values, record["labels"].numpy(), record["ranks"], record["scenario_id"], record["time_step"]))
        summary = _summarize(rows)
        summary.update(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "grad_norm_mean": float(np.mean(grad_norms)),
                "grad_norm_max": float(np.max(grad_norms)),
                "seconds": float(time.perf_counter() - started),
                "rss_bytes": _rss_bytes(),
            }
        )
        history.append(summary)
        key = (summary["payload_mean"], summary["top1"], -summary["normalized_regret"], -summary["train_loss"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch}
            torch.save(
                {
                    "epoch": epoch,
                    "seed": int(seed),
                    "head": head.state_dict(),
                    "lambda_pair": float(head.lambda_pair),
                    "encoder": str(EPOCH6_CKPT),
                    "unary": "epoch6_matching_decoder",
                },
                folder / "head.pt",
            )
        print(
            f"seed {seed} epoch {epoch} loss={summary['train_loss']:.4f} top1={summary['top1']:.4f} regret={summary['normalized_regret']:.4f} grad={summary['grad_norm_max']:.3f} seconds={summary['seconds']:.1f}",
            flush=True,
        )
    chosen = PairwiseInteractionHead().to(device)
    payload = torch.load(folder / "head.pt", map_location="cpu", weights_only=False)
    chosen.load_state_dict(payload["head"])
    chosen.eval()
    return chosen, history


def _cold_timing(head: PairwiseInteractionHead, model, decoder, device: torch.device) -> dict:
    head.eval()
    full = []
    head_only = []
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            sample, _arrays = _load_bundle(int(scenario), int(time_step))
            started = time.perf_counter()
            graph = append_queue_edge(sample["graph"])
            tensors = {key: value.to(device) for key, value in batch_tensors(graph).items()}
            assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
            with torch.no_grad():
                h_u, h_b = model.condition(tensors)
                unary = decoder.score_assignments(h_u, h_b, assignments)
                edge_h = model.encoder.edge_enc(tensors["edge"])[0]
                head(edge_h, assignments, unary)
            full.append(time.perf_counter() - started)
            started = time.perf_counter()
            with torch.no_grad():
                head(edge_h, assignments, unary)
            head_only.append(time.perf_counter() - started)
    return {
        "full_seconds_median": float(np.median(full)),
        "full_seconds_max": float(np.max(full)),
        "head_seconds_median": float(np.median(head_only)),
        "head_seconds_max": float(np.max(head_only)),
        "rss_bytes": _rss_bytes(),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_trained_head(seed: int, device: torch.device) -> tuple[PairwiseInteractionHead, list[dict], int]:
    folder = OUTPUT / f"seed_{seed}"
    payload = torch.load(folder / "head.pt", map_location="cpu", weights_only=False)
    head = PairwiseInteractionHead().to(device)
    head.load_state_dict(payload["head"])
    head.eval()
    history_path = folder / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    return head, history, int(payload["epoch"])


def _pack_baseline(result: dict) -> dict:
    rows = result["rows"]
    return {
        "name": result["name"],
        "top1_hits": int(sum(int(row["top1"]) for row in rows)),
        "seconds_median": result.get("seconds_median"),
        "selected_epoch": result.get("selected_epoch"),
        "matches_published": result.get("matches_published"),
        "summary": _summarize(rows),
        "t0": _slice_summary(rows, 0),
        "t15": _slice_summary(rows, 15),
        "t30": _slice_summary(rows, 30),
        "rows": _public_rows(rows),
    }


def run_audit() -> dict:
    device = torch.device("cpu")
    config = L3BBitQueueEnvV2().config
    before = _checkpoint_hashes()
    split = verify_split()
    model = _load_hgnn(EPOCH6_CKPT, device)
    decoder = _load_decoder(device)
    checks = _run_checks(model, decoder, device)
    print(f"checks passed={checks['passed']} grad={checks['grad_norm']:.4f}", flush=True)
    blocked = not bool(checks["passed"])
    baselines = {}
    seeds = []
    timing = None
    if not blocked:
        epoch6 = _matching_rows("epoch6", model, decoder, device)
        scheme_model = _load_hgnn(SCHEME_C_CKPT, device)
        scheme = _matching_rows("scheme_c", scheme_model, decoder, device)
        conditioned = {seed: _conditioned_rows(seed, device, config) for seed in (0, 1)}
        baselines = {"epoch6": epoch6, "scheme_c": scheme, "conditioned": conditioned}
        blocked = not (bool(epoch6["matches_published"]) and bool(scheme["matches_published"]))
        print(f"published top1 epoch6={epoch6['top1_hits']} scheme_c={scheme['top1_hits']} blocked={int(blocked)}", flush=True)
    resumed = False
    if not blocked:
        resumed = all((OUTPUT / f"seed_{seed}" / "head.pt").is_file() for seed in SEEDS)
        valid_cache = _frozen_cache(model, decoder, device, "validation")
        if not resumed:
            train_cache = _frozen_cache(model, decoder, device, "train")
        for seed in SEEDS:
            if resumed:
                head, history, _epoch = _load_trained_head(int(seed), device)
                print(f"resume seed {seed} epoch {_epoch}", flush=True)
            else:
                head, history = _train_seed(int(seed), train_cache, valid_cache, device)
            rows = []
            parts = []
            t15_records = [record for record in valid_cache if int(record["time_step"]) == 15]
            for record in valid_cache:
                values = _score_interaction(head, record, device)
                rows.append(_ranking(values, record["labels"].numpy(), record["ranks"], record["scenario_id"], record["time_step"]))
            for record in t15_records:
                values = _score_interaction(head, record, device)
                part = _interaction(record, values, scheme["t15_scores"][int(record["scenario_id"])], record["unary"].numpy())
                part["scenario_id"] = int(record["scenario_id"])
                parts.append(part)
            if _saved_scheme_matches(parts) != 16 or any(part["missing"] for part in parts):
                blocked = True
            interaction = _pool_interaction(parts)
            flags = _decision_flags(rows, scheme["rows"], interaction)
            flags["seed"] = int(seed)
            flags["selected_epoch"] = int(torch.load(OUTPUT / f"seed_{seed}" / "head.pt", map_location="cpu", weights_only=False)["epoch"])
            flags["history"] = history
            flags["rows"] = rows
            seeds.append(flags)
            print(
                f"seed {seed} selected_epoch={flags['selected_epoch']} supported={flags['supported']} active={interaction['active_fraction']:.3f} quad={interaction['quadruple_accuracy']}",
                flush=True,
            )
        timing = _cold_timing(head, model, decoder, device)
    after = _checkpoint_hashes()
    if before != after:
        blocked = True
    gate = assign_gate(blocked, bool(checks["passed"]) and before == after, seeds)
    result = {
        "gate": gate,
        "device": device.type,
        "checks": checks,
        "split": split,
        "checkpoint_hashes_before": before,
        "checkpoint_hashes_after": after,
        "hashes_unchanged": before == after,
        "blocked": int(blocked),
        "lambda_pair": 1.0,
        "lambda_reason": "fixed at 1 because the last linear layer learns the pair scale; validation and reserved test are not used to choose it",
        "unary": "epoch6 conditioned pair logits, frozen matching decoder",
        "edge_state": "epoch6 edge_enc output, shape [6, 12, 128], selected by the assignment",
        "hidden": 32,
        "head_parameters": int(sum(parameter.numel() for parameter in PairwiseInteractionHead().parameters())),
        "encoder_parameters_frozen": int(sum(parameter.numel() for parameter in model.parameters())),
        "decoder_parameters_frozen": int(sum(parameter.numel() for parameter in decoder.parameters())),
        "epochs": int(MAX_EPOCHS),
        "seeds": [int(seed) for seed in SEEDS],
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "loss": "mean(-logsigmoid(s_i-s_j)) for j15_i > j15_j",
        "baselines": {name: _pack_baseline(baselines[name]) for name in ("epoch6", "scheme_c")} if baselines else None,
        "conditioned": {str(seed): _pack_baseline(baselines["conditioned"][seed]) for seed in (0, 1)} if baselines else None,
        "seed_reports": [{key: value for key, value in report.items() if key != "rows"} | {"rows": _public_rows(report["rows"])} for report in seeds],
        "inference": timing,
        "resumed_from_checkpoints": int(resumed),
        "reserved_scenarios_read": 0,
    }
    path = assert_output_path(OUTPUT / "summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate": gate, "checks": checks["passed"], "head_parameters": result["head_parameters"]}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
