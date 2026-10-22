"""Phase-4B B2 shadow integration audit. Official planner default is unchanged."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold

from wt.htw_phase100.high_level import decide as official_decide
from wt.lowlevel_v2.htw_phase4a_hindsight_credit.protocol import SOURCE as P4A_SOURCE
from wt.lowlevel_v2.htw_phase4b_b2_shadow.protocol import (
    B2_FEATURE_KEYS,
    CONS_ATOL_R,
    ET,
    FIG,
    LOCKED_LSO_CSV,
    LOCKED_R_B2_FOLD,
    LOCKED_R_B2_STATE,
    LOCKED_R_LOCAL_FOLD,
    LOCKED_R_LOCAL_STATE,
    N_PERM,
    N_SPLITS,
    OUTPUT,
    P3E_CAND_KEYS,
    PERM_DROP_EXISTING,
    PERM_SEED,
    PLANNER_PATHS,
    PRIMARY,
    REPORT_NAME,
    ROOT,
)
from wt.lowlevel_v2.htw_phase4b_b2_shadow.shadow_scorer import (
    B2ShadowScorer,
    assert_no_future_keys,
    extract_candidate_geometry,
    feature_matrix,
)
from wt.lowlevel_v2.htw_phase4b_b2_validation.run_phase4b_b2 import B2_KEYS as LOCKED_B2_KEYS
from wt.lowlevel_v2.htw_phase4b_credit_learner.protocol import LOCKED_FV_U, LOCKED_TRAJ_U
from wt.lowlevel_v2.htw_phase4b_credit_learner.run_phase4b import (
    attach_observables,
    ext_sum,
    load_phase4a,
    parse_gamma,
)
from wt.lowlevel_v2.integration_phase2.protocol import file_sha
from wt.lowlevel_v2.phase3e_future_value_audit.run_phase3e import reset_env


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _fmt(x: float) -> str:
    if not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.3f}"


def _sci(x: float) -> str:
    return f"{float(x):.3e}"


def _close(a: float, b: float, atol: float = CONS_ATOL_R) -> bool:
    return abs(float(a) - float(b)) <= atol


def grouped(rows: list[dict]) -> dict[int, list[int]]:
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[int(row["scenario_id"])].append(i)
    return dict(sorted(by.items()))


def load_locked_lso() -> list[dict]:
    if not LOCKED_LSO_CSV.is_file():
        raise FileNotFoundError(f"missing locked LSO ranking: {LOCKED_LSO_CSV}")
    with LOCKED_LSO_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def planner_integrity() -> dict:
    src = PLANNER_PATHS[0].read_text(encoding="utf-8")
    leaked = [tok for tok in ("U_hat", "C_hat", "B2Shadow", "ExtraTrees") if tok in src]
    return {
        "official_decide_patched": False,
        "official_decide_name": official_decide.__name__,
        "official_decide_module": official_decide.__module__,
        "high_level_has_b2_tokens": leaked,
        "default_still_decode_assignment": "decode_assignment" in src and "def decide(" in src,
        "sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in PLANNER_PATHS if p.is_file()},
        "note": "shadow scorer is a side channel; high_level.decide was not imported for selection",
    }


def live_geometry_check(rows: list[dict]) -> dict:
    env_cache = {}
    abs_err = []
    n_ok = 0
    for row in rows:
        sc = int(row["scenario_id"])
        if sc not in env_cache:
            env_cache[sc] = reset_env(sc)
        live = extract_candidate_geometry(env_cache[sc], parse_gamma(row["gamma"]))
        err = [abs(live[k] - float(row[k])) for k in P3E_CAND_KEYS]
        abs_err.extend(err)
        n_ok += int(max(err) <= 1e-9)
    arr = np.asarray(abs_err, dtype=np.float64)
    return {
        "n_rows": len(rows),
        "n_exact": int(n_ok),
        "max_abs_err": float(np.max(arr)) if arr.size else float("nan"),
        "mean_abs_err": float(np.mean(arr)) if arr.size else float("nan"),
        "keys": list(P3E_CAND_KEYS),
        "matches_attach_observables": bool(arr.size and float(np.max(arr)) <= 1e-9),
    }


def feature_diagnostics(rows: list[dict], live: dict) -> dict:
    by = grouped(rows)
    orders = {sc: [int(rows[i]["candidate_id"]) for i in idxs] for sc, idxs in by.items()}
    expected = list(range(24))
    return {
        "B2_keys": list(B2_FEATURE_KEYS),
        "matches_locked_B2_KEYS": list(B2_FEATURE_KEYS) == list(LOCKED_B2_KEYS),
        "normalization": "none",
        "n_scenarios": len(by),
        "pool_sizes": {str(sc): len(idxs) for sc, idxs in by.items()},
        "candidate_id_contiguous_0_23": all(orders[sc] == expected for sc in orders),
        "scenario_ids": sorted(by),
        "live_geometry": live,
        "phase4a_source": {k: str(v) for k, v in P4A_SOURCE.items()},
        "forbidden_keys_in_model": [],
    }


def fit_lso(rows: list[dict]):
    assert_no_future_keys(rows, B2_FEATURE_KEYS)
    x = feature_matrix(rows, B2_FEATURE_KEYS)
    y = np.array([r["C_abs"] for r in rows], dtype=np.float64)
    groups = np.array([int(r["scenario_id"]) for r in rows], dtype=np.int64)
    oof_c = np.full(len(rows), np.nan)
    oof_u = np.full(len(rows), np.nan)
    models = []
    folds = []
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(set(groups.tolist()))))
    for fold, (tr, te) in enumerate(gkf.split(x, groups=groups)):
        scorer = B2ShadowScorer()
        scorer.fit(x[tr], y[tr])
        c_hat, u_hat = scorer.score_matrix(x[te])
        oof_c[te] = c_hat
        oof_u[te] = u_hat
        models.append((scorer, np.asarray(te), sorted({int(groups[i]) for i in te})))
        folds.append(
            {
                "fold": fold,
                "train_scenarios": sorted({int(groups[i]) for i in tr}),
                "test_scenarios": sorted({int(groups[i]) for i in te}),
                "n_train": int(tr.size),
                "n_test": int(te.size),
            }
        )
    return oof_c, oof_u, folds, models, x


def state_pick(rows: list[dict], idxs: list[int], score) -> dict:
    sc = np.asarray(score, dtype=np.float64)
    ut = np.array([rows[i]["U_traj"] for i in idxs], dtype=np.float64)
    ih = int(np.argmax(sc))
    it = int(np.argmax(ut))
    best = float(np.max(ut))
    gi = idxs[ih]
    return {
        "idx": ih,
        "gidx": gi,
        "oracle_idx": it,
        "candidate_id": int(rows[gi]["candidate_id"]),
        "assignment_hash": rows[gi]["assignment_hash"],
        "U_single": float(rows[gi]["U_single"]),
        "U_traj": float(ut[ih]),
        "R": float(ut[ih] / best) if abs(best) > 0 else float("nan"),
        "top1": float(ih == it),
        "top3": float(it in set(np.argsort(-sc)[:3].tolist())),
        "best": best,
        "oracle_cand": int(rows[idxs[it]]["candidate_id"]),
    }


def mechanism_label(d_u: float, d_c: float) -> str:
    if d_u <= 0.0 and d_c > 0.0:
        return "residual-driven"
    if d_u > 0.0 and d_c > 0.0:
        return "local+residual"
    if d_u > 0.0 and d_c <= 0.0:
        return "local-driven"
    return "conflicting"


def build_decisions(rows: list[dict], oof_c: np.ndarray, oof_u: np.ndarray) -> list[dict]:
    out = []
    for sc, idxs in grouped(rows).items():
        us = [rows[i]["U_single"] for i in idxs]
        ut = [rows[i]["U_traj"] for i in idxs]
        loc = state_pick(rows, idxs, us)
        b2 = state_pick(rows, idxs, [oof_u[i] for i in idxs])
        ora = state_pick(rows, idxs, ut)
        loc_c = float(oof_c[loc["gidx"]])
        b2_c = float(oof_c[b2["gidx"]])
        loc_h = float(oof_u[loc["gidx"]])
        b2_h = float(oof_u[b2["gidx"]])
        d_u = float(b2["U_single"] - loc["U_single"])
        d_c = float(b2_c - loc_c)
        d_h = float(b2_h - loc_h)
        changed = int(b2["candidate_id"] != loc["candidate_id"])
        mech = mechanism_label(d_u, d_c) if changed else "unchanged"
        geom = {f"B2_{k}": float(rows[b2["gidx"]][k]) for k in P3E_CAND_KEYS}
        geom_l = {f"LOCAL_{k}": float(rows[loc["gidx"]][k]) for k in P3E_CAND_KEYS}
        out.append(
            {
                "scenario_id": sc,
                "LOCAL_candidate": loc["candidate_id"],
                "B2_candidate": b2["candidate_id"],
                "ORACLE_candidate": ora["candidate_id"],
                "hash_LOCAL": loc["assignment_hash"],
                "hash_B2": b2["assignment_hash"],
                "hash_ORACLE": ora["assignment_hash"],
                "U_single_LOCAL": loc["U_single"],
                "U_single_B2": b2["U_single"],
                "U_traj_LOCAL": loc["U_traj"],
                "U_traj_B2": b2["U_traj"],
                "U_traj_ORACLE": ora["U_traj"],
                "C_hat_LOCAL": loc_c,
                "C_hat_B2": b2_c,
                "U_hat_LOCAL": loc_h,
                "U_hat_B2": b2_h,
                "R_LOCAL": loc["R"],
                "R_B2": b2["R"],
                "R_ORACLE": 1.0,
                "delta_R": float(b2["R"] - loc["R"]),
                "B2_changed": changed,
                "B2_matches_oracle": int(b2["candidate_id"] == ora["candidate_id"]),
                "LOCAL_matches_oracle": int(loc["candidate_id"] == ora["candidate_id"]),
                "top1": b2["top1"],
                "top3": b2["top3"],
                "dU_single": d_u,
                "dC": d_c,
                "dU_hat": d_h,
                "mechanism": mech,
                "sign_conflict": int(d_u * d_c < 0.0) if changed else 0,
                **geom,
                **geom_l,
            }
        )
    return out


def fold_metrics(rows: list[dict], oof_c: np.ndarray, oof_u: np.ndarray, folds: list[dict]) -> list[dict]:
    by = grouped(rows)
    out = []
    for rec in folds:
        r_b2, r_loc, top1, top3, u_b2 = [], [], [], [], []
        for sc in rec["test_scenarios"]:
            idxs = by[sc]
            loc = state_pick(rows, idxs, [rows[i]["U_single"] for i in idxs])
            b2 = state_pick(rows, idxs, [oof_u[i] for i in idxs])
            r_b2.append(b2["R"])
            r_loc.append(loc["R"])
            top1.append(b2["top1"])
            top3.append(b2["top3"])
            u_b2.append(b2["U_traj"])
        out.append(
            {
                "fold": rec["fold"],
                "test_scenarios": rec["test_scenarios"],
                "R_pred_B2": float(np.nanmean(r_b2)),
                "R_pred_LOCAL": float(np.nanmean(r_loc)),
                "top1": float(np.mean(top1)),
                "top3": float(np.mean(top3)),
                "mean_U_traj_B2": float(np.mean(u_b2)),
            }
        )
    return out


def compare_locked(decisions: list[dict], locked: list[dict]) -> dict:
    lock = {int(r["scenario_id"]): r for r in locked}
    n_match = 0
    mismatches = []
    for row in decisions:
        sc = int(row["scenario_id"])
        ref = lock[sc]
        ok = (
            int(row["LOCAL_candidate"]) == int(float(ref["cand_LOCAL"]))
            and int(row["B2_candidate"]) == int(float(ref["cand_B2"]))
            and int(row["ORACLE_candidate"]) == int(float(ref["cand_ORACLE"]))
        )
        n_match += int(ok)
        if not ok:
            mismatches.append(
                {
                    "scenario_id": sc,
                    "shadow": [row["LOCAL_candidate"], row["B2_candidate"], row["ORACLE_candidate"]],
                    "locked": [int(float(ref["cand_LOCAL"])), int(float(ref["cand_B2"])), int(float(ref["cand_ORACLE"]))],
                }
            )
    return {
        "n_states": len(decisions),
        "n_decision_match": n_match,
        "all_match": n_match == len(decisions),
        "mismatches": mismatches,
    }


def permute_geometry(rows: list[dict], models, x: np.ndarray) -> dict:
    rng = np.random.default_rng(PERM_SEED)
    cols = [B2_FEATURE_KEYS.index(k) for k in P3E_CAND_KEYS]
    r_real, r_perm = [], []
    for _ in range(N_PERM):
        fold_r, fold_p = [], []
        for scorer, te, _test_sc in models:
            x_te = x[te].copy()
            x_perm = x_te.copy()
            local = defaultdict(list)
            for j, i in enumerate(te):
                local[int(rows[i]["scenario_id"])].append(j)
            for idxs in local.values():
                order = rng.permutation(len(idxs))
                sl = np.asarray(idxs, dtype=np.int64)
                block = x_perm[sl][:, cols]
                tmp = x_perm[sl]
                tmp[:, cols] = block[order]
                x_perm[sl] = tmp
            _c, u_hat = scorer.score_matrix(x_te)
            _cp, u_hat_p = scorer.score_matrix(x_perm)
            for idxs in local.values():
                glob = [int(te[j]) for j in idxs]
                ut = [rows[i]["U_traj"] for i in glob]
                fold_r.append(state_pick(rows, glob, [u_hat[k] for k in idxs])["R"])
                fold_p.append(state_pick(rows, glob, [u_hat_p[k] for k in idxs])["R"])
        r_real.append(float(np.nanmean(fold_r)))
        r_perm.append(float(np.nanmean(fold_p)))
    return {"n_perm": N_PERM, "R_pred_real_repeat": ext_sum(r_real), "R_pred_perm": ext_sum(r_perm)}


def worse_geometry(decisions: list[dict]) -> dict:
    worse = [r for r in decisions if r["delta_R"] < -1e-12]
    better = [r for r in decisions if r["delta_R"] > 1e-12]
    keys = list(P3E_CAND_KEYS)

    def _delta_mean(group: list[dict]) -> dict:
        if not group:
            return {}
        out = {}
        for k in keys:
            out[k] = float(np.mean([r[f"B2_{k}"] - r[f"LOCAL_{k}"] for r in group]))
        return out

    hashes = Counter(r["hash_B2"] for r in worse)
    return {
        "n_worse": len(worse),
        "worse_scenarios": [r["scenario_id"] for r in worse],
        "worse_B2_hashes": [r["hash_B2"] for r in worse],
        "worse_B2_candidates": [r["B2_candidate"] for r in worse],
        "shared_worse_hash": None if len(set(hashes)) != 1 else next(iter(hashes)),
        "mean_geom_delta_worse": _delta_mean(worse),
        "mean_geom_delta_better": _delta_mean(better),
        "note": "descriptive only; no new features and no retune",
    }


def summarize_mech(decisions: list[dict]) -> dict:
    changed = [r for r in decisions if r["B2_changed"]]
    counts = Counter(r["mechanism"] for r in changed)
    n = max(len(changed), 1)
    return {
        "n_changed": len(changed),
        "counts": {
            "residual-driven": int(counts.get("residual-driven", 0)),
            "local+residual": int(counts.get("local+residual", 0)),
            "local-driven": int(counts.get("local-driven", 0)),
            "conflicting": int(counts.get("conflicting", 0)),
        },
        "fractions": {
            "residual-driven": counts.get("residual-driven", 0) / n,
            "local+residual": counts.get("local+residual", 0) / n,
            "local-driven": counts.get("local-driven", 0) / n,
            "conflicting": counts.get("conflicting", 0) / n,
        },
        "n_sign_conflict": int(sum(r["sign_conflict"] for r in changed)),
        "mean_dU_single_changed": float(np.mean([r["dU_single"] for r in changed])) if changed else float("nan"),
        "mean_dC_changed": float(np.mean([r["dC"] for r in changed])) if changed else float("nan"),
        "mean_dU_hat_changed": float(np.mean([r["dU_hat"] for r in changed])) if changed else float("nan"),
    }


def gates(decisions: list[dict], folds: list[dict], perm: dict, cons: dict, feat: dict) -> dict:
    r_b2 = float(np.mean([r["R_B2"] for r in decisions]))
    r_loc = float(np.mean([r["R_LOCAL"] for r in decisions]))
    r_b2_fold = float(np.mean([f["R_pred_B2"] for f in folds]))
    r_loc_fold = float(np.mean([f["R_pred_LOCAL"] for f in folds]))
    dR = np.array([r["delta_R"] for r in decisions], dtype=np.float64)
    n_pos = int(np.sum(dR > 1e-12))
    n_neg = int(np.sum(dR < -1e-12))
    n_zero = int(len(dR) - n_pos - n_neg)
    perm_drop = float(perm["R_pred_real_repeat"]["mean"] - perm["R_pred_perm"]["mean"])
    gate1 = bool(r_b2 > r_loc)
    gate2 = bool(perm_drop > PERM_DROP_EXISTING)
    cons_ok = bool(
        cons["all_match"]
        and feat["matches_locked_B2_KEYS"]
        and feat["live_geometry"]["matches_attach_observables"]
        and _close(r_b2, LOCKED_R_B2_STATE)
        and _close(r_loc, LOCKED_R_LOCAL_STATE)
        and _close(r_loc_fold, LOCKED_R_LOCAL_FOLD)
    )
    if not cons_ok:
        overall = "FAIL"
        text = "shadow scorer 未能复现冻结 LSO 决策或 R_pred。未改模型；见 join / feature / ordering 诊断。"
    elif gate1 and gate2:
        overall = "PARTIAL"
        text = "shadow scorer 复现了 LSO 的 U_hat first-hold 选择，Gate-1/2 通过。仍缺少真正 held-out independent scenario，Gate-3 未定义 deployment threshold，因此不能正式替换 planner。"
    elif gate1:
        overall = "PARTIAL"
        text = "shadow 能复现选择且均值高于 LOCAL，但 permutation 证据不足。不能正式替换 planner。"
    else:
        overall = "FAIL"
        text = "shadow 路径未显示相对 LOCAL 的增益。不能正式替换 planner。"
    return {
        "gate": overall,
        "text": text,
        "planner_ready": False,
        "independent_available": False,
        "Gate-1_B2_gt_LOCAL": gate1,
        "Gate-2_perm_below_real": gate2,
        "Gate-3": "未定义 deployment threshold",
        "consistency_ok": cons_ok,
        "mean_R_B2": r_b2,
        "median_R_B2": float(np.median([r["R_B2"] for r in decisions])),
        "std_R_B2": float(np.std([r["R_B2"] for r in decisions], ddof=1)),
        "mean_R_LOCAL": r_loc,
        "median_R_LOCAL": float(np.median([r["R_LOCAL"] for r in decisions])),
        "std_R_LOCAL": float(np.std([r["R_LOCAL"] for r in decisions], ddof=1)),
        "fold_R_B2": r_b2_fold,
        "fold_R_LOCAL": r_loc_fold,
        "fold_R_B2_std": float(np.std([f["R_pred_B2"] for f in folds], ddof=1)),
        "n_improve": n_pos,
        "n_degraded": n_neg,
        "n_unchanged": n_zero,
        "top1": float(np.mean([r["top1"] for r in decisions])),
        "top3": float(np.mean([r["top3"] for r in decisions])),
        "oracle_overlap": int(sum(r["B2_matches_oracle"] for r in decisions)),
        "local_oracle_overlap": int(sum(r["LOCAL_matches_oracle"] for r in decisions)),
        "n_changed": int(sum(r["B2_changed"] for r in decisions)),
        "payload_gain_vs_LOCAL": float(np.mean([r["U_traj_B2"] - r["U_traj_LOCAL"] for r in decisions])),
        "residual_gap_vs_ORACLE": float(np.mean([r["U_traj_ORACLE"] - r["U_traj_B2"] for r in decisions])),
        "mean_U_B2": float(np.mean([r["U_traj_B2"] for r in decisions])),
        "mean_U_LOCAL": float(np.mean([r["U_traj_LOCAL"] for r in decisions])),
        "mean_U_ORACLE": float(np.mean([r["U_traj_ORACLE"] for r in decisions])),
        "perm_drop": perm_drop,
        "perm_R": float(perm["R_pred_perm"]["mean"]),
        "locked_R_B2": LOCKED_R_B2_STATE,
        "locked_R_LOCAL_state": LOCKED_R_LOCAL_STATE,
        "locked_R_LOCAL_fold": LOCKED_R_LOCAL_FOLD,
    }


def figures(decisions: list[dict], mech: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = ["residual-driven", "local+residual", "local-driven", "conflicting"]
    vals = [mech["counts"][k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(labels, vals, color=["#54A24B", "#4C78A8", "#F58518", "#E45756"])
    ax.set_ylabel("changed states")
    ax.set_title("B2 shadow decision mechanism")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "decision_mechanism.png", dpi=140)
    plt.close(fig)

    ch = [r for r in decisions if r["B2_changed"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    colors = {"residual-driven": "#54A24B", "local+residual": "#4C78A8", "local-driven": "#F58518", "conflicting": "#E45756"}
    for name in labels:
        xs = [r["dU_single"] for r in ch if r["mechanism"] == name]
        ys = [r["dC"] for r in ch if r["mechanism"] == name]
        ax.scatter(xs, ys, label=name, color=colors[name], s=40)
    ax.axhline(0.0, color="#333", lw=0.8)
    ax.axvline(0.0, color="#333", lw=0.8)
    ax.set_xlabel("dU_single = U_single(B2)-U_single(LOCAL)")
    ax.set_ylabel("dC = C_hat(B2)-C_hat(LOCAL)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "decision_mechanism_scatter.png", dpi=140)
    plt.close(fig)


def write_report(
    integ: dict,
    feat: dict,
    cons: dict,
    decisions: list[dict],
    folds: list[dict],
    perm: dict,
    mech: dict,
    worse: dict,
    gate: dict,
) -> str:
    lines = [
        "# HTW Phase-4B B2 Shadow Integration Audit",
        "",
        "本阶段只验证：把冻结 B2 的 `U_hat = U_single + C_hat` 当作 first-hold shadow score 时，能否复现已锁定的 leave-scenario-out 结果。",
        "未训练 Low-Level，未改 reward / utility / decoder / pool / checkpoint，未改正式 planner 默认行为，未重跑 Phase-3 counterfactual，未调 ExtraTrees，未进入 Phase-4C。",
        "",
        f"**门控：{gate['gate']}** — {gate['text']}",
        "",
        f"**planner_ready = {gate['planner_ready']}**。即使 Gate-1 / Gate-2 通过，也不能正式替换 planner：当前仍缺少真正 held-out independent scenario。",
        "",
        "## 固定方法",
        "",
        "`C_hat = ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)`，`U_hat = U_single + C_hat`。",
        f"ExtraTrees 固定：{ET}。`GroupKFold({N_SPLITS})`，group=`scenario_id`，同一 state 的 24 candidates 不拆分。",
        "LOCAL = argmax U_single；B2-SHADOW = argmax U_hat；ORACLE = argmax U_traj（仅评价）。",
        "U_single 复用冻结 Phase-4A，不重新调用 Low-Level / shadow_h15。U_traj / F / remaining_arrivals 不进入输入。",
        "",
        "## 正式 planner 未被替换",
        "",
        f"- official decide module: `{integ['official_decide_module']}.{integ['official_decide_name']}`",
        f"- 默认仍走 `decode_assignment`: {integ['default_still_decode_assignment']}",
        f"- high_level.py 中的 B2 token: {integ['high_level_has_b2_tokens'] or 'none'}",
        f"- sha256: {integ['sha256']}",
        "",
        "## A. Shadow scorer consistency",
        "",
        f"- 特征键与冻结 LSO B2_KEYS 一致: {feat['matches_locked_B2_KEYS']}",
        f"- 无 normalization: {feat['normalization']}",
        f"- candidate_id 连续 0-23: {feat['candidate_id_contiguous_0_23']}",
        f"- live Phase-3E geometry 与 attach_observables 一致: {feat['live_geometry']['matches_attach_observables']} (max abs err={feat['live_geometry']['max_abs_err']})",
        f"- 与冻结 ranking CSV 决策一致: {cons['n_decision_match']}/{cons['n_states']}",
        f"- shadow mean R_B2={_fmt(gate['mean_R_B2'])} vs locked 0.916 ({_fmt(gate['locked_R_B2'])})",
        f"- shadow mean R_LOCAL(state)={_fmt(gate['mean_R_LOCAL'])} vs locked 0.758 ({_fmt(gate['locked_R_LOCAL_state'])})",
        f"- shadow fold-mean R_LOCAL={_fmt(gate['fold_R_LOCAL'])} vs locked 0.754 ({_fmt(gate['locked_R_LOCAL_fold'])})",
        f"- shadow fold-mean R_B2={_fmt(gate['fold_R_B2'])} ± {_fmt(gate['fold_R_B2_std'])}",
        f"- consistency_ok={gate['consistency_ok']}",
        "",
        "上一阶段报告中的 LOCAL≈0.754 是 **fold mean**；state-mean LOCAL 是 0.758。两者都复现则不诊断为 join 错误。",
        "",
        "| fold | test scenarios | R_B2 | R_LOCAL | top-1 | top-3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for f in folds:
        lines.append(
            f"| {f['fold']} | {f['test_scenarios']} | {_fmt(f['R_pred_B2'])} | {_fmt(f['R_pred_LOCAL'])} | {_fmt(f['top1'])} | {_fmt(f['top3'])} |"
        )
    if cons["mismatches"]:
        lines.extend(["", "决策不一致（应为空）：", str(cons["mismatches"])])
    lines.extend(
        [
            "",
            "## B. Shadow first-hold decision audit",
            "",
            "| scenario | LOCAL | B2 | ORACLE | U_single_L | U_single_B2 | U_traj_L | U_traj_B2 | U_traj_O | R_L | R_B2 | dR | changed | vsO |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in decisions:
        lines.append(
            f"| {r['scenario_id']} | {r['LOCAL_candidate']} | {r['B2_candidate']} | {r['ORACLE_candidate']} | {_sci(r['U_single_LOCAL'])} | {_sci(r['U_single_B2'])} | {_sci(r['U_traj_LOCAL'])} | {_sci(r['U_traj_B2'])} | {_sci(r['U_traj_ORACLE'])} | {_fmt(r['R_LOCAL'])} | {_fmt(r['R_B2'])} | {_fmt(r['delta_R'])} | {r['B2_changed']} | {r['B2_matches_oracle']} |"
        )
    lines.extend(
        [
            "",
            f"- B2 改变 first-hold: {gate['n_changed']}/16",
            f"- 改变后改善 / 恶化 / 持平: {gate['n_improve']} / {gate['n_degraded']} / {gate['n_unchanged']}",
            f"- B2-ORACLE overlap: {gate['oracle_overlap']}/16（LOCAL-ORACLE={gate['local_oracle_overlap']}/16）",
            f"- 恶化 scenario: {worse['worse_scenarios']}",
            f"- 恶化侧 B2 hash: {worse['worse_B2_hashes']}",
            f"- 恶化侧是否共享同一 assignment hash: {worse['shared_worse_hash']}",
            "",
            "恶化 scenario 的 B2-LOCAL geometry 差（Phase-3E 已有特征，不新增）：",
            "",
            f"`{worse['mean_geom_delta_worse']}`",
            "",
            "改善 scenario 的对应差：",
            "",
            f"`{worse['mean_geom_delta_better']}`",
            "",
            "## Decision mechanism（只统计 B2 改变的 state）",
            "",
            "| class | n | fraction |",
            "|---|---:|---:|",
        ]
    )
    for name in ("residual-driven", "local+residual", "local-driven", "conflicting"):
        lines.append(f"| {name} | {mech['counts'][name]} | {_fmt(mech['fractions'][name])} |")
    lines.extend(
        [
            "",
            f"- 改变 state 上 mean dU_single={_sci(mech['mean_dU_single_changed'])}，mean dC={_sci(mech['mean_dC_changed'])}，mean dU_hat={_sci(mech['mean_dU_hat_changed'])}",
            f"- sign(dU_single)*sign(dC)<0 的改变数: {mech['n_sign_conflict']}/{mech['n_changed']}",
            "- residual-driven 表示选了更低或持平的 U_single，但 C_hat 更高：这才是 residual correction，而不是重发现 LOCAL。",
            "",
            "## C. Conservative deployment gate",
            "",
            f"- Gate-1 B2 mean R > LOCAL mean R: **{gate['Gate-1_B2_gt_LOCAL']}** ({_fmt(gate['mean_R_B2'])} > {_fmt(gate['mean_R_LOCAL'])})",
            f"- Gate-2 permutation 明显低于真实 B2: **{gate['Gate-2_perm_below_real']}** (real={_fmt(perm['R_pred_real_repeat']['mean'])}, perm={_fmt(perm['R_pred_perm']['mean'])} ± {_fmt(perm['R_pred_perm']['std'])}, drop={_fmt(gate['perm_drop'])}；沿用既有审计 drop>{PERM_DROP_EXISTING})",
            f"- Gate-3 恶化数阈值: **{gate['Gate-3']}**。工程中没有预先定义 deployment threshold，不自行发明。当前恶化数={gate['n_degraded']}。",
            "",
            f"- mean/median/std R_B2 = {_fmt(gate['mean_R_B2'])} / {_fmt(gate['median_R_B2'])} / {_fmt(gate['std_R_B2'])}",
            f"- mean/median/std R_LOCAL = {_fmt(gate['mean_R_LOCAL'])} / {_fmt(gate['median_R_LOCAL'])} / {_fmt(gate['std_R_LOCAL'])}",
            f"- improved / degraded / unchanged = {gate['n_improve']} / {gate['n_degraded']} / {gate['n_unchanged']}",
            f"- top-1 / top-3 = {_fmt(gate['top1'])} / {_fmt(gate['top3'])}",
            f"- B2-ORACLE overlap = {gate['oracle_overlap']}/16",
            f"- average payload gain vs LOCAL = {_sci(gate['payload_gain_vs_LOCAL'])}",
            f"- residual gap vs ORACLE = {_sci(gate['residual_gap_vs_ORACLE'])}",
            "",
            "## 对照",
            "",
            "| method | mean R | mean U_traj |",
            "|---|---:|---:|",
            f"| LOCAL | {_fmt(gate['mean_R_LOCAL'])} | {_sci(gate['mean_U_LOCAL'])} |",
            f"| Phase-3E locked FV | (cite) | {_sci(LOCKED_FV_U)} |",
            f"| B2-SHADOW | {_fmt(gate['mean_R_B2'])} | {_sci(gate['mean_U_B2'])} |",
            f"| TRAJ-ORACLE | 1.000 | {_sci(LOCKED_TRAJ_U)} |",
            "",
            "## PASS / PARTIAL / FAIL",
            "",
            f"**{gate['gate']}**。{gate['text']}",
            "",
            "本阶段停止。不修改 B2，不修改 planner，不进入 Phase-4C。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    raw = [r for r in attach_observables(load_phase4a()) if r["family"] == PRIMARY]
    integ = planner_integrity()
    live = live_geometry_check(raw)
    feat = feature_diagnostics(raw, live)
    oof_c, oof_u, folds, models, x = fit_lso(raw)
    fold_tab = fold_metrics(raw, oof_c, oof_u, folds)
    decisions = build_decisions(raw, oof_c, oof_u)
    locked = load_locked_lso()
    cons = compare_locked(decisions, locked)
    perm = permute_geometry(raw, models, x)
    mech = summarize_mech(decisions)
    worse = worse_geometry(decisions)
    gate = gates(decisions, fold_tab, perm, cons, feat)
    figures(decisions, mech)
    drop_geom = {f"B2_{k}" for k in P3E_CAND_KEYS} | {f"LOCAL_{k}" for k in P3E_CAND_KEYS}
    slim = [{k: v for k, v in r.items() if k not in drop_geom} for r in decisions]
    _csv_write(OUTPUT / "b2_shadow_decisions.csv", slim)
    _dump(
        OUTPUT / "b2_shadow_summary.json",
        {
            "integrity": integ,
            "features": feat,
            "consistency": cons,
            "folds": folds,
            "fold_metrics": fold_tab,
            "perm": perm,
            "mechanism": mech,
            "worse_geometry": worse,
            "gate": gate,
            "et": ET,
            "B2_keys": list(B2_FEATURE_KEYS),
        },
    )
    body = write_report(integ, feat, cons, decisions, fold_tab, perm, mech, worse, gate)
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate["gate"],
                "consistency_ok": gate["consistency_ok"],
                "R_B2": gate["mean_R_B2"],
                "R_LOCAL": gate["mean_R_LOCAL"],
                "R_LOCAL_fold": gate["fold_R_LOCAL"],
                "Gate-1": gate["Gate-1_B2_gt_LOCAL"],
                "Gate-2": gate["Gate-2_perm_below_real"],
                "Gate-3": gate["Gate-3"],
                "planner_ready": False,
                "n_changed": gate["n_changed"],
                "mechanism": mech["counts"],
            },
            indent=2,
        ),
        flush=True,
    )
    return gate


if __name__ == "__main__":
    run()
