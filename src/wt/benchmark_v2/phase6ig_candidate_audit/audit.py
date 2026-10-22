"""Time-stratified audit of the scheme-C epoch-1 checkpoint.

Scores are read-only. J15 labels are used after scoring and never reorder candidates.
Reserved scenarios are refused. Nothing here updates a checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6id_joint_training.metrics import state_metrics
from wt.benchmark_v2.phase6id_joint_training.protocol import ENCODER_CKPT, OUTPUT as PHASE6ID_OUTPUT
from wt.benchmark_v2.phase6id_joint_training.train import _build, _finite_tensor, _sha256
from wt.benchmark_v2.phase6if_ranking_loss.train import EPOCH6, _scores


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6ig_candidate_audit"
SCHEME_C = ROOT / "results/phase6if_ranking_loss" / "runs" / "c" / "epoch_001.pt"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
MATERIAL_BITS = 1.0
NEAR_GAPS = (1.0, 1.0e4, 1.0e6)
RANK_BINS = (1, 3, 5, 10, 20, 50, 100, 500, 2000, 13327)


def decision_view(scores: np.ndarray, labels: np.ndarray, utility_rank: np.ndarray) -> dict:
    """Offline view of one scored state. The score order is not recomputed from J15."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    groups = np.asarray(utility_rank, dtype=np.int64).reshape(-1)
    if values.shape != truth.shape or values.shape != groups.shape:
        raise ValueError("scores, labels, and utility_rank are not aligned")
    order = np.argsort(-values, kind="stable")
    rank_of = np.empty(values.size, dtype=np.int64)
    rank_of[order] = np.arange(1, values.size + 1)
    reverse = np.lexsort((-np.arange(values.size), -values))
    metric = state_metrics(values, truth, groups)
    exact = truth == float(np.max(truth))
    best = groups == 1
    top = order[: min(5, order.size)]
    outside = np.ones(values.size, dtype=bool)
    outside[top] = False
    near = {}
    for gap in NEAR_GAPS:
        mask = truth >= float(np.max(truth) - gap)
        subset = np.flatnonzero(mask)
        subset_order = subset[np.argsort(-values[subset], kind="stable")]
        margin = float("nan")
        label_gap = float("nan")
        if subset_order.size >= 2:
            margin = float(values[subset_order[0]] - values[subset_order[1]])
            label_gap = float(truth[subset_order[0]] - truth[subset_order[1]])
        near[f"{gap:g}"] = {
            "count": int(subset.size),
            "best_predicted_rank": int(np.min(rank_of[mask])),
            "chosen_inside": int(bool(mask[int(order[0])])),
            "score_margin": margin,
            "label_gap_of_margin": label_gap,
            "tiny_score_reorders_material_gap": int(bool(np.isfinite(margin) and margin < 1.0e-4 and label_gap > 1.0e4)),
        }
    return {
        "metric": metric,
        "chosen": int(order[0]),
        "score_margin": float(values[order[0]] - values[order[1]]),
        "score_tie_count": int(np.sum(np.equal(values, values[order[0]]))),
        "tie_break_changes_top1": int(bool(best[int(order[0])]) != bool(best[int(reverse[0])])),
        "exact_max_predicted_rank": int(np.min(rank_of[exact])),
        "best_group_predicted_rank": int(np.min(rank_of[best])),
        "exact_max_in_top3": int(np.min(rank_of[exact]) <= 3),
        "exact_max_in_top5": int(np.min(rank_of[exact]) <= 5),
        "best_group_in_top3_not_top1": int(np.min(rank_of[best]) <= 3 and not bool(best[int(order[0])])),
        "best_group_in_top5_not_top1": int(np.min(rank_of[best]) <= 5 and not bool(best[int(order[0])])),
        "top5_payload_min": float(np.min(truth[top])),
        "top5_payload_median": float(np.median(truth[top])),
        "top5_payload_max": float(np.max(truth[top])),
        "outside_top5_payload_max": float(np.max(truth[outside])) if np.any(outside) else float("nan"),
        "best_group_size": int(np.sum(best)),
        "exact_max_count": int(np.sum(exact)),
        "regret_bits": float(metric["regret"]),
        "top1_matches_one_bit_regret": int(bool(metric["top1"]) == bool(float(metric["regret"]) <= float(EPS_BITS))),
        "near": near,
    }


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _median(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def cluster_means(groups: list[np.ndarray], draws: int, seed: int) -> np.ndarray:
    """Resample whole scenarios. Each group is one scenario's paired deltas."""
    if not groups or any(np.asarray(group).size == 0 for group in groups):
        raise ValueError("bootstrap groups must be non-empty scenarios")
    packed = [np.asarray(group, dtype=np.float64).reshape(-1) for group in groups]
    width = {int(group.size) for group in packed}
    if len(width) != 1:
        raise ValueError("scenarios do not share the same decision times")
    generator = np.random.default_rng(int(seed))
    picks = generator.integers(0, len(packed), size=(int(draws), len(packed)))
    stacked = np.stack(packed, axis=0)
    return np.mean(stacked[picks], axis=(1, 2))


def interval(samples: np.ndarray) -> list[float]:
    low, high = np.quantile(np.asarray(samples, dtype=np.float64), [0.025, 0.975])
    return [float(low), float(high)]


def _sign_counts(delta: np.ndarray) -> dict[str, float]:
    return {
        "n": float(delta.size),
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "positive_strict": float(np.mean(delta > 0.0)),
        "negative_strict": float(np.mean(delta < 0.0)),
        "positive_material": float(np.mean(delta > MATERIAL_BITS)),
        "negative_material": float(np.mean(delta < -MATERIAL_BITS)),
        "within_one_bit": float(np.mean(np.abs(delta) <= MATERIAL_BITS)),
    }


def _paired_block(groups: list[np.ndarray], draws: int, seed: int) -> dict:
    flat = np.concatenate(groups)
    scenario_sums = np.asarray([float(np.sum(group)) for group in groups], dtype=np.float64)
    samples = cluster_means(groups, draws, seed)
    leave = []
    for index in range(len(groups)):
        held = np.concatenate([group for position, group in enumerate(groups) if position != index])
        leave.append(float(np.mean(held)))
    return {
        **_sign_counts(flat),
        "n_scenarios": float(len(groups)),
        "scenarios_positive": float(np.sum(scenario_sums > MATERIAL_BITS)),
        "scenarios_negative": float(np.sum(scenario_sums < -MATERIAL_BITS)),
        "mean_ci95": interval(samples),
        "median_ci95": _median_samples(groups, draws, seed),
        "leave_one_scenario_mean_min": float(np.min(leave)),
        "leave_one_scenario_mean_max": float(np.max(leave)),
    }


def _median_samples(groups: list[np.ndarray], draws: int, seed: int) -> list[float]:
    stacked = np.stack([np.asarray(group, dtype=np.float64).reshape(-1) for group in groups], axis=0)
    generator = np.random.default_rng(int(seed))
    picks = generator.integers(0, stacked.shape[0], size=(int(draws), stacked.shape[0]))
    drawn = stacked[picks].reshape(int(draws), -1)
    return interval(np.median(drawn, axis=1))


def assign_gate(versus_epoch6: dict, versus_frozen: dict, by_time_epoch6: list[dict]) -> str:
    """Pre-registered support rule. A positive mean alone is not enough."""
    times_not_negative = all(float(item["mean_ci95"][1]) >= 0.0 for item in by_time_epoch6)
    supported = bool(
        float(versus_epoch6["mean_ci95"][0]) > 0.0
        and float(versus_frozen["mean_ci95"][0]) > 0.0
        and float(versus_epoch6["median"]) > MATERIAL_BITS
        and float(versus_epoch6["scenarios_positive"]) >= float(versus_epoch6["n_scenarios"]) / 2.0
        and times_not_negative
    )
    if supported:
        return "RANKING_CANDIDATE_GAIN_SUPPORTED"
    return "RANKING_CANDIDATE_GAIN_INCONCLUSIVE"


def _slice_metrics(rows: list[dict], model: str) -> dict:
    payloads = [float(row[model]["payload"]) for row in rows]
    regrets = [float(row[model]["normalized_regret"]) for row in rows]
    oracle_ranks = [float(row[model]["exact_max_predicted_rank"]) for row in rows]
    group_ranks = [float(row[model]["best_group_predicted_rank"]) for row in rows]
    return {
        "n": float(len(rows)),
        "top1": _mean([row[model]["top1"] for row in rows]),
        "top3": _mean([row[model]["top3"] for row in rows]),
        "top5": _mean([row[model]["top5"] for row in rows]),
        "spearman": _mean([row[model]["spearman"] for row in rows]),
        "kendall": _mean([row[model]["kendall"] for row in rows]),
        "normalized_regret": _mean(regrets),
        "payload_mean": _mean(payloads),
        "payload_median": _median(payloads),
        "exact_max_rank_median": _median(oracle_ranks),
        "exact_max_rank_min": float(np.min(oracle_ranks)),
        "exact_max_rank_max": float(np.max(oracle_ranks)),
        "best_group_rank_median": _median(group_ranks),
        "exact_max_top3": _mean([row[model]["exact_max_in_top3"] for row in rows]),
        "exact_max_top5": _mean([row[model]["exact_max_in_top5"] for row in rows]),
        "crowded_out_of_top1_inside_top5": _mean([row[model]["best_group_in_top5_not_top1"] for row in rows]),
    }


def _rank_hist(rows: list[dict], model: str) -> dict[str, int]:
    ranks = np.asarray([int(row[model]["exact_max_predicted_rank"]) for row in rows], dtype=np.int64)
    edges = [0, *RANK_BINS]
    hist = {}
    for left, right in zip(edges[:-1], edges[1:]):
        hist[f"{left + 1}-{right}"] = int(np.sum((ranks > left) & (ranks <= right)))
    return hist


def _groups_for(rows: list[dict], reference: str) -> list[np.ndarray]:
    by_scenario: dict[int, list[float]] = {}
    for row in rows:
        by_scenario.setdefault(int(row["scenario_id"]), []).append(float(row["scheme_c"]["payload"]) - float(row[reference]["payload"]))
    ordered = []
    for scenario in sorted(by_scenario):
        values = by_scenario[scenario]
        if len(values) != 3:
            raise RuntimeError(f"scenario {scenario} does not have three decision times")
        ordered.append(np.asarray(values, dtype=np.float64))
    return ordered


def _time_blocks(rows: list[dict], reference: str, time_step: int, draws: int, seed: int) -> dict:
    groups = []
    for group in _groups_for(rows, reference):
        # Groups follow t=0, t=15, t=30 because the dataset index is scenario-major.
        groups.append(group[ {0: 0, 15: 1, 30: 2}[int(time_step)] :][:1])
    flat = np.concatenate(groups)
    return {**_sign_counts(flat), "mean_ci95": interval(cluster_means(groups, draws, seed)), "n_scenarios": float(len(groups))}


def _model_record(view: dict) -> dict:
    metric = view["metric"]
    record = {
        "payload": float(metric["payload"]),
        "normalized_regret": float(metric["normalized_regret"]),
        "spearman": float(metric["spearman"]),
        "kendall": float(metric["kendall"]),
        "top1": float(metric["top1"]),
        "top3": float(metric["top3"]),
        "top5": float(metric["top5"]),
        "chosen": int(view["chosen"]),
        "score_margin": float(view["score_margin"]),
        "score_tie_count": int(view["score_tie_count"]),
        "tie_break_changes_top1": int(view["tie_break_changes_top1"]),
        "exact_max_predicted_rank": int(view["exact_max_predicted_rank"]),
        "best_group_predicted_rank": int(view["best_group_predicted_rank"]),
        "exact_max_in_top3": int(view["exact_max_in_top3"]),
        "exact_max_in_top5": int(view["exact_max_in_top5"]),
        "best_group_in_top3_not_top1": int(view["best_group_in_top3_not_top1"]),
        "best_group_in_top5_not_top1": int(view["best_group_in_top5_not_top1"]),
        "top5_payload_min": float(view["top5_payload_min"]),
        "top5_payload_median": float(view["top5_payload_median"]),
        "top5_payload_max": float(view["top5_payload_max"]),
        "outside_top5_payload_max": float(view["outside_top5_payload_max"]),
        "best_group_size": int(view["best_group_size"]),
        "exact_max_count": int(view["exact_max_count"]),
        "regret_bits": float(view["regret_bits"]),
        "top1_matches_one_bit_regret": int(view["top1_matches_one_bit_regret"]),
        "near": view["near"],
    }
    return record


def _reproduce(saved: list[dict], fresh: list[dict], name: str) -> None:
    if len(saved) != len(fresh):
        raise RuntimeError(f"{name} row count drifted")
    saved_by = {(int(row["scenario_id"]), int(row["time_step"])): row for row in saved}
    for row in fresh:
        key = (int(row["scenario_id"]), int(row["time_step"]))
        left = saved_by[key]
        if int(left["chosen"]) != int(row["chosen"]):
            raise RuntimeError(f"{name} chosen candidate drifted at {key[0]} t={key[1]}")
        if abs(float(left["payload"]) - float(row["payload"])) > 1.0e-4:
            raise RuntimeError(f"{name} payload drifted at {key[0]} t={key[1]}")
        if int(left["top1"]) != int(row["top1"]):
            raise RuntimeError(f"{name} top1 drifted at {key[0]} t={key[1]}")


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario, _time in DecisionDataset("validation").index):
        raise RuntimeError("validation index contains a reserved scenario")
    if not SCHEME_C.is_file() or not EPOCH6.is_file():
        raise FileNotFoundError("scheme C or epoch 6 checkpoint is missing")
    before = {path.name: _sha256(path) for path in (ENCODER_CKPT, EPOCH6, SCHEME_C)}
    device = torch.device("cpu")
    frozen, decoder, _optimizer = _build(device)
    epoch6_payload = torch.load(EPOCH6, map_location="cpu", weights_only=False)
    scheme_payload = torch.load(SCHEME_C, map_location="cpu", weights_only=False)
    if int(epoch6_payload["epoch"]) != 6 or int(scheme_payload["epoch"]) != 1 or scheme_payload.get("scheme") != "c":
        raise RuntimeError("checkpoint headers do not match the three comparison models")
    epoch6, _decoder_epoch6, _optimizer_epoch6 = _build(device)
    scheme, _decoder_scheme, _optimizer_scheme = _build(device)
    epoch6.load_state_dict(epoch6_payload["model"])
    scheme.load_state_dict(scheme_payload["model"])
    for model in (frozen, epoch6, scheme, decoder):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    rows = []
    with torch.no_grad():
        for sample in DecisionDataset("validation"):
            if int(sample["scenario_id"]) in RESERVED_TEST:
                raise RuntimeError("validation loader opened a reserved scenario")
            labels = np.asarray(sample["j15_bits"], dtype=np.float64)
            ranks = np.asarray(sample["utility_rank"], dtype=np.int64)
            scored = {}
            for name, model in (("frozen", frozen), ("epoch6", epoch6), ("scheme_c", scheme)):
                tensor = _scores(model, decoder, sample, device)
                if not _finite_tensor(tensor):
                    raise RuntimeError(f"non-finite score from {name}")
                scored[name] = decision_view(tensor.detach().cpu().numpy(), labels, ranks)
                del tensor
            row = {
                "scenario_id": int(sample["scenario_id"]),
                "time_step": int(sample["time_step"]),
                "candidate_count": int(labels.shape[0]),
                "oracle_payload": float(np.max(labels)),
            }
            for name, view in scored.items():
                row[name] = _model_record(view)
            row["delta_vs_epoch6"] = float(row["scheme_c"]["payload"] - row["epoch6"]["payload"])
            row["delta_vs_frozen"] = float(row["scheme_c"]["payload"] - row["frozen"]["payload"])
            rows.append(row)
            print(
                f"audit {row['scenario_id']} t={row['time_step']} d6={row['delta_vs_epoch6']:.3f} rank={row['scheme_c']['exact_max_predicted_rank']}",
                flush=True,
            )
    _reproduce(
        json.loads((PHASE6ID_OUTPUT / "baseline_states.json").read_text(encoding="utf-8")),
        [{"scenario_id": row["scenario_id"], "time_step": row["time_step"], "chosen": row["frozen"]["chosen"], "payload": row["frozen"]["payload"], "top1": row["frozen"]["top1"]} for row in rows],
        "frozen",
    )
    _reproduce(
        json.loads((PHASE6ID_OUTPUT / "best_states.json").read_text(encoding="utf-8")),
        [{"scenario_id": row["scenario_id"], "time_step": row["time_step"], "chosen": row["epoch6"]["chosen"], "payload": row["epoch6"]["payload"], "top1": row["epoch6"]["top1"]} for row in rows],
        "epoch6",
    )
    _reproduce(
        json.loads((SCHEME_C.parent / "epoch_001_states.json").read_text(encoding="utf-8")),
        [{"scenario_id": row["scenario_id"], "time_step": row["time_step"], "chosen": row["scheme_c"]["chosen"], "payload": row["scheme_c"]["payload"], "top1": row["scheme_c"]["top1"]} for row in rows],
        "scheme_c",
    )
    by_time = {}
    for time_step in (0, 15, 30):
        chosen = [row for row in rows if int(row["time_step"]) == time_step]
        by_time[str(time_step)] = {
            "models": {name: _slice_metrics(chosen, name) for name in ("frozen", "epoch6", "scheme_c")},
            "scheme_c_exact_rank_hist": _rank_hist(chosen, "scheme_c"),
            "versus_epoch6": _time_blocks(rows, "epoch6", time_step, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
            "versus_frozen": _time_blocks(rows, "frozen", time_step, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        }
    versus_epoch6 = _paired_block(_groups_for(rows, "epoch6"), BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    versus_frozen = _paired_block(_groups_for(rows, "frozen"), BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    gate = assign_gate(versus_epoch6, versus_frozen, [by_time[str(time_step)]["versus_epoch6"] for time_step in (0, 15, 30)])
    after = {path.name: _sha256(path) for path in (ENCODER_CKPT, EPOCH6, SCHEME_C)}
    if after != before:
        raise RuntimeError("a comparison checkpoint changed during the audit")
    t15 = [row for row in rows if int(row["time_step"]) == 15]
    summary = {
        "gate": gate,
        "n_states": len(rows),
        "n_scenarios": 16,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "versus_epoch6": versus_epoch6,
        "versus_frozen": versus_frozen,
        "by_time": by_time,
        "t15_tie_break_changes": int(sum(row["scheme_c"]["tie_break_changes_top1"] for row in t15)),
        "t15_crowded_out_top5": int(sum(row["scheme_c"]["best_group_in_top5_not_top1"] for row in t15)),
        "t15_exact_outside_top5": int(sum(not row["scheme_c"]["exact_max_in_top5"] for row in t15)),
        "checkpoints_unchanged": True,
        "rule_mismatches": int(sum(row[name]["top1_matches_one_bit_regret"] == 0 for row in rows for name in ("frozen", "epoch6", "scheme_c"))),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "states.json").write_text(json.dumps(rows), encoding="utf-8")
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps({"gate": run_audit()["gate"]}), flush=True)
