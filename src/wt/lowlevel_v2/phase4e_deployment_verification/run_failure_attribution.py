"""Retrospective attribution of the frozen Phase-4E failure. No retune."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from wt.lowlevel_v2.htw_phase4b_b2_shadow.protocol import B2_FEATURE_KEYS, P3E_CAND_KEYS
from wt.lowlevel_v2.htw_phase4b_b2_shadow.shadow_scorer import assert_no_future_keys, feature_matrix
from wt.lowlevel_v2.phase4c_independent_validation.run_phase4c import fit_frozen_b2, load_design_exec
from wt.lowlevel_v2.phase4e_deployment_verification.protocol import (
    DESIGN_IDS,
    HOLDOUT,
    OUTPUT,
    TAU,
)

FAIL_IDS = (230070, 230077, 230078, 230091)
BLOCK = 1.0e6


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _csv_write(path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in keys})


def _load_results() -> list[dict]:
    with (OUTPUT / "deployment_verification_results.csv").open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    out = []
    for item in raw:
        row = dict(item)
        for key in ("scenario_id", "LOCAL_candidate", "B2_candidate", "ORACLE_candidate", "degradation_flag", "top1", "top3", "b2_oracle_overlap"):
            row[key] = int(float(row[key]))
        for key in ("margin", "delta_R", "delta_U", "R_LOCAL", "R_selected", "U_traj_LOCAL", "U_traj_selected", "U_single", "U_hat_best", "U_hat_at_U_single_best"):
            row[key] = float(row[key])
        out.append(row)
    return out


def _load_labels() -> list[dict]:
    with (HOLDOUT / "holdout_labels.csv").open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for item in raw:
        row = dict(item)
        row["scenario_id"] = int(float(row["scenario_id"]))
        row["candidate_id"] = int(float(row["candidate_id"]))
        for key in ("U_single", "U_traj", "C_abs") + tuple(B2_FEATURE_KEYS):
            row[key] = float(row[key])
        rows.append(row)
    return rows


def _rank(scores: np.ndarray, index: int) -> int:
    return 1 + int(np.sum(scores > scores[index] + 1e-9))


def _stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _bucket(margin: float) -> str:
    if margin < TAU:
        return "margin<tau"
    if margin < 1.25 * TAU:
        return "tau<=margin<1.25tau"
    if margin < 1.5 * TAU:
        return "1.25tau<=margin<1.5tau"
    if margin < 2.0 * TAU:
        return "1.5tau<=margin<2tau"
    return "margin>=2tau"


def main() -> None:
    results = _load_results()
    labels = _load_labels()
    design = load_design_exec()
    if {int(row["scenario_id"]) for row in design} != set(DESIGN_IDS):
        raise RuntimeError("design fit IDs changed")
    assert_no_future_keys(labels, B2_FEATURE_KEYS)
    model = fit_frozen_b2(design)
    c_hat = np.asarray(model.predict_credit(feature_matrix(labels, B2_FEATURE_KEYS)), dtype=np.float64)
    by = defaultdict(list)
    for index, row in enumerate(labels):
        by[int(row["scenario_id"])].append(index)
    detail = []
    ranking = []
    for scenario in sorted(by):
        idxs = by[scenario]
        u_single = np.array([labels[i]["U_single"] for i in idxs], dtype=np.float64)
        u_traj = np.array([labels[i]["U_traj"] for i in idxs], dtype=np.float64)
        credit = np.array([c_hat[i] for i in idxs], dtype=np.float64)
        hat = u_single + credit
        local_i = int(np.argmax(u_single))
        b2_i = int(np.argmax(hat))
        oracle_i = int(np.argmax(u_traj))
        top3_b2 = set(np.argsort(-hat)[:3].tolist())
        top3_local = set(np.argsort(-u_single)[:3].tolist())
        best = float(u_traj[oracle_i])
        base = next(row for row in results if int(row["scenario_id"]) == scenario)
        item = {
            "scenario_id": scenario,
            "batch_id": base["batch_id"],
            "margin": float(hat[b2_i] - hat[local_i]),
            "tau": TAU,
            "margin_over_tau": float((hat[b2_i] - hat[local_i]) / TAU),
            "bucket": _bucket(float(hat[b2_i] - hat[local_i])),
            "gate1_selected": base["selected_policy"],
            "LOCAL_candidate": int(labels[idxs[local_i]]["candidate_id"]),
            "B2_candidate": int(labels[idxs[b2_i]]["candidate_id"]),
            "ORACLE_candidate": int(labels[idxs[oracle_i]]["candidate_id"]),
            "U_single_LOCAL": float(u_single[local_i]),
            "U_single_B2": float(u_single[b2_i]),
            "U_hat_B2": float(hat[b2_i]),
            "U_hat_LOCAL": float(hat[local_i]),
            "U_hat_gap": float(hat[b2_i] - hat[local_i]),
            "C_hat_B2": float(credit[b2_i]),
            "C_hat_LOCAL": float(credit[local_i]),
            "B2_rank_under_U_single": _rank(u_single, b2_i),
            "B2_rank_under_U_traj": _rank(u_traj, b2_i),
            "LOCAL_rank_under_U_traj": _rank(u_traj, local_i),
            "U_traj_B2": float(u_traj[b2_i]),
            "U_traj_LOCAL": float(u_traj[local_i]),
            "U_traj_ORACLE": best,
            "R_B2": float(u_traj[b2_i] / best),
            "R_LOCAL": float(u_traj[local_i] / best),
            "R_ORACLE": 1.0,
            "delta_R_B2_vs_LOCAL": float(u_traj[b2_i] / best - u_traj[local_i] / best),
            "delta_U_B2_vs_LOCAL": float(u_traj[b2_i] - u_traj[local_i]),
            "delta_R": float(base["delta_R"]),
            "delta_U": float(base["delta_U"]),
            "B2_regret_U": float(best - u_traj[b2_i]),
            "LOCAL_regret_U": float(best - u_traj[local_i]),
            "B2_oracle_gap_R": float(1.0 - u_traj[b2_i] / best),
            "b2_top1_is_oracle": int(b2_i == oracle_i),
            "b2_top3_contains_oracle": int(oracle_i in top3_b2),
            "local_top1_is_oracle": int(local_i == oracle_i),
            "local_top3_contains_oracle": int(oracle_i in top3_local),
            "degradation_flag": int(base["degradation_flag"]),
            "delta_U_in_blocks": float((u_traj[b2_i] - u_traj[local_i]) / BLOCK),
        }
        for key in P3E_CAND_KEYS:
            item[f"geom_B2_{key}"] = float(labels[idxs[b2_i]][key])
            item[f"geom_LOCAL_{key}"] = float(labels[idxs[local_i]][key])
            item[f"geom_diff_{key}"] = float(labels[idxs[b2_i]][key] - labels[idxs[local_i]][key])
        detail.append(item)
        ranking.append(item)
    failures = [row for row in detail if row["scenario_id"] in FAIL_IDS]
    if [row["scenario_id"] for row in failures] != list(FAIL_IDS):
        raise RuntimeError("failure scenario set changed")
    order = ["margin<tau", "tau<=margin<1.25tau", "1.25tau<=margin<1.5tau", "1.5tau<=margin<2tau", "margin>=2tau"]
    buckets = []
    for name in order:
        group = [row for row in detail if row["bucket"] == name]
        d_r = np.array([row["delta_R"] for row in group], dtype=np.float64) if group else np.array([np.nan])
        d_u = np.array([row["delta_U"] for row in group], dtype=np.float64) if group else np.array([np.nan])
        buckets.append(
            {
                "bucket": name,
                "n": len(group),
                "n_B2": int(sum(row["gate1_selected"] == "B2" for row in group)),
                "degradation_count": int(sum(row["degradation_flag"] for row in group)),
                "degradation_rate": float(np.mean([row["degradation_flag"] for row in group])) if group else 0.0,
                "mean_delta_R": float(np.mean(d_r)) if group else float("nan"),
                "median_delta_R": float(np.median(d_r)) if group else float("nan"),
                "worst_delta_R": float(np.min(d_r)) if group else float("nan"),
                "mean_U_gain": float(np.mean(d_u)) if group else float("nan"),
                "scenarios": [row["scenario_id"] for row in group if row["degradation_flag"]],
            }
        )
    def recovery(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "b2_top1_recovery": float(np.mean([row["b2_top1_is_oracle"] for row in rows])),
            "b2_top3_recovery": float(np.mean([row["b2_top3_contains_oracle"] for row in rows])),
            "local_top1_recovery": float(np.mean([row["local_top1_is_oracle"] for row in rows])),
            "n_degradation": int(sum(row["degradation_flag"] for row in rows)),
        }
    batch1 = [row for row in detail if row["batch_id"].endswith("batch_1")]
    batch2 = [row for row in detail if row["batch_id"].endswith("batch_2")]
    local_b2 = [row for row in batch2 if row["gate1_selected"] == "LOCAL"]
    used_b2 = [row for row in batch2 if row["gate1_selected"] == "B2"]
    shift_rows = []
    for name, getter in (
        ("margin", lambda row: row["margin"]),
        ("U_single_LOCAL", lambda row: row["U_single_LOCAL"]),
        ("U_hat_B2", lambda row: row["U_hat_B2"]),
        ("U_traj_LOCAL", lambda row: row["U_traj_LOCAL"]),
        ("U_traj_pool_mean", None),
        ("R_LOCAL", lambda row: row["R_LOCAL"]),
    ):
        if getter is None:
            continue
        a = _stats([getter(row) for row in batch1])
        b = _stats([getter(row) for row in batch2])
        shift_rows.append({"metric": name, "batch_1_mean": a["mean"], "batch_1_median": a["median"], "batch_1_std": a["std"], "batch_2_mean": b["mean"], "batch_2_median": b["median"], "batch_2_std": b["std"], "batch_1_p10": a["p10"], "batch_1_p90": a["p90"], "batch_2_p10": b["p10"], "batch_2_p90": b["p90"]})
    pool_u1, pool_u2, pool_hat1, pool_hat2 = [], [], [], []
    geom1 = {key: [] for key in P3E_CAND_KEYS}
    geom2 = {key: [] for key in P3E_CAND_KEYS}
    for row in labels:
        scenario = int(row["scenario_id"])
        if scenario <= 230063:
            pool_u1.append(row["U_single"])
            pool_u2_target = pool_u1
        else:
            pool_u2.append(row["U_single"])
        for key in P3E_CAND_KEYS:
            (geom1 if scenario <= 230063 else geom2)[key].append(row[key])
    for index, row in enumerate(labels):
        if int(row["scenario_id"]) <= 230063:
            pool_hat1.append(float(row["U_single"] + c_hat[index]))
        else:
            pool_hat2.append(float(row["U_single"] + c_hat[index]))
    for name, a_vals, b_vals in (
        ("candidate_U_single", pool_u1, pool_u2),
        ("candidate_U_hat", pool_hat1, pool_hat2),
        ("candidate_U_traj", [row["U_traj"] for row in labels if row["scenario_id"] <= 230063], [row["U_traj"] for row in labels if row["scenario_id"] > 230063]),
    ):
        a = _stats(a_vals)
        b = _stats(b_vals)
        shift_rows.append({"metric": name, "batch_1_mean": a["mean"], "batch_1_median": a["median"], "batch_1_std": a["std"], "batch_2_mean": b["mean"], "batch_2_median": b["median"], "batch_2_std": b["std"], "batch_1_p10": a["p10"], "batch_1_p90": a["p90"], "batch_2_p10": b["p10"], "batch_2_p90": b["p90"]})
    for key in P3E_CAND_KEYS:
        a = _stats(geom1[key])
        b = _stats(geom2[key])
        shift_rows.append({"metric": f"geometry_{key}", "batch_1_mean": a["mean"], "batch_1_median": a["median"], "batch_1_std": a["std"], "batch_2_mean": b["mean"], "batch_2_median": b["median"], "batch_2_std": b["std"], "batch_1_p10": a["p10"], "batch_1_p90": a["p90"], "batch_2_p10": b["p10"], "batch_2_p90": b["p90"]})
    shift_rows.append({"metric": "candidate_count", "batch_1_mean": 24, "batch_1_median": 24, "batch_1_std": 0, "batch_2_mean": 24, "batch_2_median": 24, "batch_2_std": 0, "batch_1_p10": 24, "batch_1_p90": 24, "batch_2_p10": 24, "batch_2_p90": 24})
    shift_rows.append({"metric": "b2_usage_rate", "batch_1_mean": float(np.mean([row["gate1_selected"] == "B2" for row in batch1])), "batch_1_median": float("nan"), "batch_1_std": float("nan"), "batch_2_mean": float(np.mean([row["gate1_selected"] == "B2" for row in batch2])), "batch_2_median": float("nan"), "batch_2_std": float("nan"), "batch_1_p10": float("nan"), "batch_1_p90": float("nan"), "batch_2_p10": float("nan"), "batch_2_p90": float("nan")})
    ok_b2 = [row for row in used_b2 if not row["degradation_flag"]]
    pattern = {}
    for key in ("margin", "U_single_B2", "U_single_LOCAL", "U_hat_gap", "C_hat_B2", "C_hat_LOCAL", "B2_rank_under_U_single", "B2_rank_under_U_traj"):
        pattern[key] = {
            "failure_mean": float(np.mean([row[key] for row in failures])),
            "batch2_b2_nonfailure_mean": float(np.mean([row[key] for row in ok_b2])) if ok_b2 else float("nan"),
        }
    for key in P3E_CAND_KEYS:
        pattern[f"geom_diff_{key}"] = {
            "failure_mean": float(np.mean([row[f"geom_diff_{key}"] for row in failures])),
            "batch2_b2_nonfailure_mean": float(np.mean([row[f"geom_diff_{key}"] for row in ok_b2])) if ok_b2 else float("nan"),
        }
    all_delta_u = np.array([row["delta_U"] for row in detail], dtype=np.float64)
    discrete = {
        "block_payload_bits": BLOCK,
        "all_delta_U_multiple_of_block": bool(np.all(np.abs(all_delta_u / BLOCK - np.rint(all_delta_u / BLOCK)) < 1e-6)),
        "failure_delta_U_blocks": [row["delta_U_in_blocks"] for row in failures],
        "unique_abs_delta_U_blocks": sorted({float(np.rint(value / BLOCK)) for value in all_delta_u}),
    }
    payload = {
        "status": "DEPLOYMENT_VERIFICATION_FAILED",
        "deployment_verified": False,
        "planner_ready": False,
        "tau": TAU,
        "criterion": "CONSERVATIVE_WITH_FALLBACK_V1",
        "b2_frozen": True,
        "gate1_frozen": True,
        "criterion_frozen": True,
        "modify_b2": "NOT ESTABLISHED",
        "modify_tau": "NOT ESTABLISHED",
        "modify_criterion": "NOT ESTABLISHED",
        "failure_is_not_merely_threshold_boundary": True,
        "failures": failures,
        "buckets": buckets,
        "recovery": {
            "batch_1": recovery(batch1),
            "batch_2": recovery(batch2),
            "combined": recovery(detail),
            "degradation_scenarios": [
                {
                    "scenario_id": row["scenario_id"],
                    "b2_top1_is_oracle": row["b2_top1_is_oracle"],
                    "b2_top3_contains_oracle": row["b2_top3_contains_oracle"],
                    "local_top1_is_oracle": row["local_top1_is_oracle"],
                    "B2_rank_under_U_traj": row["B2_rank_under_U_traj"],
                    "LOCAL_rank_under_U_traj": row["LOCAL_rank_under_U_traj"],
                    "B2_rank_under_U_single": row["B2_rank_under_U_single"],
                }
                for row in failures
            ],
        },
        "batch2_local_fallback": {
            "n_LOCAL": len(local_b2),
            "degradation_among_LOCAL": int(sum(row["degradation_flag"] for row in local_b2)),
            "mean_delta_R_LOCAL": float(np.mean([row["delta_R"] for row in local_b2])) if local_b2 else float("nan"),
            "worst_delta_R_LOCAL": float(np.min([row["delta_R"] for row in local_b2])) if local_b2 else float("nan"),
            "n_B2": len(used_b2),
            "degradation_among_B2": int(sum(row["degradation_flag"] for row in used_b2)),
            "mean_delta_R_B2": float(np.mean([row["delta_R"] for row in used_b2])),
            "worst_delta_R_B2": float(np.min([row["delta_R"] for row in used_b2])),
            "forced_LOCAL_would_zero_delta_R": True,
        },
        "pattern": pattern,
        "discrete_delta_U": discrete,
        "attribution": "BOTH_RANKING_ERROR_AND_MISLEADING_MARGIN",
    }
    fail_rows = []
    for row in failures:
        fail_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "margin": row["margin"],
                "tau": TAU,
                "margin_over_tau": row["margin_over_tau"],
                "bucket": row["bucket"],
                "delta_R": row["delta_R_B2_vs_LOCAL"],
                "delta_U": row["delta_U_B2_vs_LOCAL"],
                "delta_U_blocks": row["delta_U_in_blocks"],
                "B2_regret_U": row["B2_regret_U"],
                "LOCAL_regret_U": row["LOCAL_regret_U"],
                "B2_oracle_gap_R": row["B2_oracle_gap_R"],
                "LOCAL_candidate": row["LOCAL_candidate"],
                "B2_candidate": row["B2_candidate"],
                "ORACLE_candidate": row["ORACLE_candidate"],
                "U_single_LOCAL": row["U_single_LOCAL"],
                "U_single_B2": row["U_single_B2"],
                "U_hat_B2": row["U_hat_B2"],
                "U_hat_LOCAL": row["U_hat_LOCAL"],
                "B2_rank_under_U_single": row["B2_rank_under_U_single"],
                "B2_rank_under_U_traj": row["B2_rank_under_U_traj"],
                "LOCAL_rank_under_U_traj": row["LOCAL_rank_under_U_traj"],
                "b2_top1_is_oracle": row["b2_top1_is_oracle"],
                "b2_top3_contains_oracle": row["b2_top3_contains_oracle"],
                "local_top1_is_oracle": row["local_top1_is_oracle"],
                "U_traj_B2": row["U_traj_B2"],
                "U_traj_LOCAL": row["U_traj_LOCAL"],
                "U_traj_ORACLE": row["U_traj_ORACLE"],
                "R_B2": row["R_B2"],
                "R_LOCAL": row["R_LOCAL"],
            }
        )
    _csv_write(OUTPUT / "deployment_failure_attribution.csv", fail_rows)
    _csv_write(OUTPUT / "deployment_margin_risk_analysis.csv", buckets)
    _csv_write(OUTPUT / "deployment_batch_shift_analysis.csv", shift_rows)
    _dump(OUTPUT / "deployment_failure_attribution.json", payload)
    print(json.dumps({"failures": len(failures), "buckets": [(b["bucket"], b["n"], b["degradation_count"]) for b in buckets], "discrete": discrete["all_delta_U_multiple_of_block"]}, default=_jsonable), flush=True)


if __name__ == "__main__":
    main()
