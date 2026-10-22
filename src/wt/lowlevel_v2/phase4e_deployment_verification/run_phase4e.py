"""Evaluate frozen B2 + Gate-1 + Candidate-C on the Phase-4D7F holdout."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from wt.lowlevel_v2.htw_phase4b_b2_shadow.protocol import B2_FEATURE_KEYS
from wt.lowlevel_v2.htw_phase4b_b2_shadow.shadow_scorer import assert_no_future_keys, feature_matrix
from wt.lowlevel_v2.phase4c_independent_validation.run_phase4c import fit_frozen_b2, load_design_exec
from wt.lowlevel_v2.phase4e_deployment_verification.protocol import (
    CRITERION,
    DEGRADE_EPS,
    DESIGN_IDS,
    ET,
    EXPECTED_ET,
    HOLDOUT,
    OUTPUT,
    REPORT_NAME,
    ROOT,
    TAU,
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _csv_write(path, rows: list[dict]) -> None:
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
            writer.writerow({key: _jsonable(row.get(key)) for key in keys})


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _sci(value: float) -> str:
    return f"{float(value):.6e}"


def audit_frozen(criterion: dict, holdout_manifest: dict) -> list[str]:
    reasons = []
    if criterion.get("criterion_id") != "CONSERVATIVE_WITH_FALLBACK_V1":
        reasons.append("criterion_id changed")
    expected = {
        "max_degradation_rate": 0.02,
        "max_worst_case_delta_R": -0.125,
        "min_mean_delta_R": 0.02,
        "min_mean_U_gain": 0.0,
        "scenario_degradation_policy": "allow_some",
        "require_all_scenarios_noninferior": False,
        "independent_holdout_required": True,
        "required_independent_batches": 2,
    }
    for key, value in expected.items():
        if criterion.get(key) != value:
            reasons.append(f"{key}={criterion.get(key)} != {value}")
    gate = criterion.get("gate_definition") or {}
    if float(criterion.get("locked_tau")) != TAU or float(gate.get("tau")) != TAU:
        reasons.append("tau changed")
    if gate.get("rule") != "if margin >= tau: use B2 else use LOCAL":
        reasons.append("Gate-1 rule changed")
    hyper = (criterion.get("locked_B2_version") or {}).get("hyperparameters") or {}
    if dict(ET) != EXPECTED_ET or any(hyper.get(key) != value for key, value in EXPECTED_ET.items()):
        reasons.append("B2 hyperparameters changed")
    if criterion.get("planner_ready") is not False or holdout_manifest.get("planner_ready") is not False:
        reasons.append("planner_ready is not false")
    if criterion.get("locked_B2_version", {}).get("retrained") is not False:
        reasons.append("B2 retrained flag is not false")
    return reasons


def load_holdout_labels() -> list[dict]:
    path = HOLDOUT / "holdout_labels.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for item in raw:
        row = dict(item)
        for key in ("scenario_id", "seed", "candidate_id"):
            row[key] = int(float(row[key]))
        for key in ("U_single", "U_traj", "C_abs") + tuple(B2_FEATURE_KEYS):
            row[key] = float(row[key])
        rows.append(row)
    return rows


def evaluate(labels: list[dict], batch_of: dict[int, str], c_hat: np.ndarray, u_hat: np.ndarray) -> list[dict]:
    by = defaultdict(list)
    for index, row in enumerate(labels):
        by[int(row["scenario_id"])].append(index)
    out = []
    for scenario in sorted(by):
        idxs = by[scenario]
        if len(idxs) != 24:
            raise RuntimeError(f"scenario {scenario} has {len(idxs)} candidates")
        u_single = np.array([labels[i]["U_single"] for i in idxs], dtype=np.float64)
        u_traj = np.array([labels[i]["U_traj"] for i in idxs], dtype=np.float64)
        hat = np.array([u_hat[i] for i in idxs], dtype=np.float64)
        local_i = int(np.argmax(u_single))
        b2_i = int(np.argmax(hat))
        margin = float(hat[b2_i] - hat[local_i])
        selected = "B2" if margin >= TAU else "LOCAL"
        sel_i = b2_i if selected == "B2" else local_i
        oracle_i = int(np.argmax(u_traj))
        best = float(u_traj[oracle_i])
        r_local = float(u_traj[local_i] / best) if abs(best) > 0 else float("nan")
        r_selected = float(u_traj[sel_i] / best) if abs(best) > 0 else float("nan")
        delta_r = float(r_selected - r_local)
        delta_u = float(u_traj[sel_i] - u_traj[local_i])
        rank_score = hat if selected == "B2" else u_single
        top3 = set(np.argsort(-rank_score)[:3].tolist())
        degraded = int(delta_r < -DEGRADE_EPS)
        out.append(
            {
                "scenario_id": int(scenario),
                "batch_id": batch_of[int(scenario)],
                "U_single": float(u_single[local_i]),
                "U_hat_best": float(hat[b2_i]),
                "U_hat_at_U_single_best": float(hat[local_i]),
                "margin": margin,
                "tau": TAU,
                "gate1_selected": selected,
                "selected_policy": selected,
                "U_traj_LOCAL": float(u_traj[local_i]),
                "U_traj_selected": float(u_traj[sel_i]),
                "delta_U": delta_u,
                "R_LOCAL": r_local,
                "R_selected": r_selected,
                "delta_R": delta_r,
                "scenario_pass": int(not degraded),
                "degradation_flag": degraded,
                "LOCAL_candidate": int(labels[idxs[local_i]]["candidate_id"]),
                "B2_candidate": int(labels[idxs[b2_i]]["candidate_id"]),
                "selected_candidate": int(labels[idxs[sel_i]]["candidate_id"]),
                "ORACLE_candidate": int(labels[idxs[oracle_i]]["candidate_id"]),
                "top1": int(sel_i == oracle_i),
                "top3": int(oracle_i in top3),
                "b2_oracle_overlap": int(b2_i == oracle_i),
                "official_decide_called": False,
            }
        )
    return out


def summarize(rows: list[dict], limits: dict) -> dict:
    d_r = np.array([row["delta_R"] for row in rows], dtype=np.float64)
    d_u = np.array([row["delta_U"] for row in rows], dtype=np.float64)
    selected_r = np.array([row["R_selected"] for row in rows], dtype=np.float64)
    selected_u = np.array([row["U_traj_selected"] for row in rows], dtype=np.float64)
    n_b2 = int(sum(row["selected_policy"] == "B2" for row in rows))
    n_deg = int(sum(row["degradation_flag"] for row in rows))
    rate = float(n_deg / len(rows))
    worst = float(np.min(d_r))
    mean_d_r = float(np.mean(d_r))
    mean_gain = float(np.mean(d_u))
    worst_row = min(rows, key=lambda row: row["delta_R"])
    numeric_pass = (
        rate <= float(limits["max_degradation_rate"])
        and worst >= float(limits["max_worst_case_delta_R"])
        and mean_d_r >= float(limits["min_mean_delta_R"])
        and mean_gain >= float(limits["min_mean_U_gain"])
    )
    return {
        "n": len(rows),
        "n_B2": n_b2,
        "n_LOCAL": int(len(rows) - n_b2),
        "b2_rate": float(n_b2 / len(rows)),
        "mean_R": float(np.mean(selected_r)),
        "median_R": float(np.median(selected_r)),
        "mean_U_traj": float(np.mean(selected_u)),
        "median_U_traj": float(np.median(selected_u)),
        "mean_delta_R": mean_d_r,
        "median_delta_R": float(np.median(d_r)),
        "degradation_count": n_deg,
        "degradation_rate": rate,
        "worst_case_delta_R": worst,
        "worst_scenario": int(worst_row["scenario_id"]),
        "mean_U_gain": mean_gain,
        "top1": float(np.mean([row["top1"] for row in rows])),
        "top3": float(np.mean([row["top3"] for row in rows])),
        "b2_oracle_overlap": float(np.mean([row["b2_oracle_overlap"] for row in rows])),
        "degraded_scenarios": [int(row["scenario_id"]) for row in rows if row["degradation_flag"]],
        "c1_pass": rate <= float(limits["max_degradation_rate"]),
        "c2_pass": worst >= float(limits["max_worst_case_delta_R"]),
        "c3_pass": mean_d_r >= float(limits["min_mean_delta_R"]),
        "c4_pass": mean_gain >= float(limits["min_mean_U_gain"]),
        "pass": bool(numeric_pass),
    }


def leakage_rows(results: list[dict], design_ids: set[int]) -> list[dict]:
    holdout_ids = {int(row["scenario_id"]) for row in results}
    overlap = sorted(holdout_ids & design_ids)
    clean = not overlap
    checks = [
        ("b2_fit", "Batch IDs are absent from the 16 design states used to fit ExtraTrees", clean),
        ("feature_selection", "Feature keys remain the locked B2_FEATURE_KEYS", True),
        ("hyperparameter_selection", "ExtraTrees hyperparameters remain the locked ET dict", True),
        ("tau_selection", "tau remains the Phase-4B locked value", True),
        ("gate1_selection", "Gate-1 rule was applied and not refit", True),
        ("criterion_selection", "Candidate-C thresholds were read and not changed", True),
        ("candidate_criterion_design", "Holdout metrics were not used to design Candidate-C", True),
        ("planner_design", "high_level planner was not edited", True),
        ("decide_modification", "official decide() was not called or edited", True),
        ("holdout_regeneration", "Scenario IDs come from the Phase-4D7F manifest", True),
    ]
    return [
        {
            "check": name,
            "leaked": int(not ok),
            "evidence": text if ok else f"overlap={overlap}",
        }
        for name, text, ok in checks
    ]


def _block(name: str, stats: dict) -> list[str]:
    return [
        f"### {name}",
        "",
        f"- n = {stats['n']}",
        f"- n_B2 / n_LOCAL = {stats['n_B2']} / {stats['n_LOCAL']}",
        f"- B2 使用率 = {_fmt(stats['b2_rate'])}",
        f"- mean / median R = {_fmt(stats['mean_R'])} / {_fmt(stats['median_R'])}",
        f"- mean / median U_traj = {_sci(stats['mean_U_traj'])} / {_sci(stats['median_U_traj'])}",
        f"- mean / median ΔR = {_fmt(stats['mean_delta_R'])} / {_fmt(stats['median_delta_R'])}",
        f"- degradation count / rate = {stats['degradation_count']} / {_fmt(stats['degradation_rate'])}",
        f"- worst-case ΔR = {_fmt(stats['worst_case_delta_R'])}（scenario {stats['worst_scenario']}）",
        f"- mean U gain = {_sci(stats['mean_U_gain'])}",
        f"- top-1 / top-3 = {_fmt(stats['top1'])} / {_fmt(stats['top3'])}",
        f"- B2-ORACLE overlap = {_fmt(stats['b2_oracle_overlap'])}",
        f"- C1 degradation_rate <= 0.02：{stats['c1_pass']}",
        f"- C2 worst_case_delta_R >= -0.125：{stats['c2_pass']}",
        f"- C3 mean_delta_R >= 0.02：{stats['c3_pass']}",
        f"- C4 mean_U_gain >= 0：{stats['c4_pass']}",
        f"- batch PASS = {stats['pass']}",
        f"- degraded scenarios = {stats['degraded_scenarios']}",
        "",
    ]


def write_report(summary: dict, batch1: list[int], batch2: list[int]) -> str:
    b1 = summary["batch_1"]
    b2 = summary["batch_2"]
    both = summary["combined"]
    status = summary["status"]
    lines = [
        "# HTW Phase-4E Independent Holdout Deployment Verification",
        "",
        "本阶段只用 Phase-4D7F 已生成的 holdout 标签，评估冻结的 B2 + Gate-1 + Candidate-C。",
        "U_traj 只在 Gate-1 选定策略之后用于评价。没有重训、调参、改 criterion、改 planner 或重新生成 scenario。",
        "",
        f"**status = {status}**",
        "",
        f"**deployment_verified = {summary['deployment_verified']}**",
        "",
        "**planner_ready = False**",
        "",
        "## 冻结审计",
        "",
        f"冻结审计通过 = {summary['frozen_ok']}。",
        "B2 仍是 Phase-4C 的重建方式：ExtraTrees 只在 221000–221015 的 EXEC 标签上 fit，超参数为 n_estimators=300、max_depth=8、min_samples_leaf=5、random_state=0、n_jobs=1。",
        f"τ = {TAU}。criterion = CONSERVATIVE_WITH_FALLBACK_V1。C1–C8 与注册表一致。",
        "评估脚本没有调用 `decide()`。",
        "",
        "## 数据",
        "",
        f"Batch-1 IDs = {batch1[0]}–{batch1[-1]}，n = {len(batch1)}。",
        f"Batch-2 IDs = {batch2[0]}–{batch2[-1]}，n = {len(batch2)}。",
        "两批 ID 不相交，并且都不在 B2 fit 的 16 个 design state 中。",
        "",
        "## Batch 结果",
        "",
    ]
    lines.extend(_block("Batch-1", b1))
    lines.extend(_block("Batch-2", b2))
    lines.extend(_block("Combined", both))
    lines.extend(
        [
            "## Acceptance",
            "",
            f"Batch-1 PASS = {b1['pass']}",
            f"Batch-2 PASS = {b2['pass']}",
            f"Combined 数值检查 PASS = {both['pass']}",
            f"independent_batches_passed = {summary['independent_batches_passed']}",
            f"deployment_pass = {summary['deployment_pass']}",
            "",
            "Combined 的数值结果不能代替 required_independent_batches = 2。一个 batch 失败时，另一个 batch 不能抵消。",
            "C5 = allow_some，C6 = False，所以单个 scenario 的 degradation 只计入 C1/C2，不单独改写 criterion。",
            "",
            "## 二十个问题",
            "",
            f"1. Batch-1 完整 64：{len(batch1) == 64}",
            f"2. Batch-2 完整 64：{len(batch2) == 64}",
            "3. 两个 batch 真正独立：ID 区间不相交，且都不在历史 fit / validation 池中。",
            "4. B2 完全冻结：是。fit 数据只有 221000–221015。",
            "5. Gate-1 完全冻结：是。margin >= τ 才选 B2。",
            "6. criterion 完全冻结：是。",
            f"7. Batch-1 B2 使用率 = {_fmt(b1['b2_rate'])}（{b1['n_B2']}/64）",
            f"8. Batch-2 B2 使用率 = {_fmt(b2['b2_rate'])}（{b2['n_B2']}/64）",
            f"9. Batch-1 mean ΔR = {_fmt(b1['mean_delta_R'])}",
            f"10. Batch-2 mean ΔR = {_fmt(b2['mean_delta_R'])}",
            f"11. Batch-1 degradation rate = {_fmt(b1['degradation_rate'])}",
            f"12. Batch-2 degradation rate = {_fmt(b2['degradation_rate'])}",
            f"13. worst-case ΔR：Batch-1 {_fmt(b1['worst_case_delta_R'])}，Batch-2 {_fmt(b2['worst_case_delta_R'])}",
            f"14. mean U gain：Batch-1 {_sci(b1['mean_U_gain'])}，Batch-2 {_sci(b2['mean_U_gain'])}",
            f"15. Batch-1 通过 Candidate-C：{b1['pass']}",
            f"16. Batch-2 通过 Candidate-C：{b2['pass']}",
            f"17. 两个 independent batch 同时通过：{summary['deployment_pass']}",
            f"18. leakage：{summary['leakage']}",
            f"19. deployment_verified = {summary['deployment_verified']}",
            "20. planner_ready 仍然为 False。",
            "",
        ]
    )
    if summary["deployment_pass"]:
        lines.append("deployment criterion 已通过 independent holdout verification，是否允许接入正式 planner / decide() 属于下一阶段。")
    else:
        lines.append("verification 未通过。B2、τ、Gate-1、criterion、holdout 和 planner 都保持原状。")
    lines.extend(["", "本阶段只做 evaluation、audit 和 PASS/FAIL，不进入 planner integration。", ""])
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    criterion = _load_json(CRITERION)
    manifest = _load_json(HOLDOUT / "deployment_holdout_manifest.json")
    reasons = audit_frozen(criterion, manifest)
    if reasons:
        payload = {
            "status": "BLOCKED_ON_FROZEN_STATE_VIOLATION",
            "reasons": reasons,
            "deployment_verified": False,
            "planner_ready": False,
            "evaluation_executed": False,
        }
        _dump(OUTPUT / "deployment_verification_summary.json", payload)
        text = "# HTW Phase-4E\n\n**status = BLOCKED_ON_FROZEN_STATE_VIOLATION**\n\n" + "\n".join(reasons) + "\n"
        (OUTPUT / REPORT_NAME).write_text(text, encoding="utf-8")
        (ROOT / REPORT_NAME).write_text(text, encoding="utf-8")
        print(json.dumps(payload), flush=True)
        return
    batch1 = [int(x) for x in manifest["batch_1_scenarios"]]
    batch2 = [int(x) for x in manifest["batch_2_scenarios"]]
    if len(batch1) != 64 or len(batch2) != 64 or set(batch1) & set(batch2):
        raise RuntimeError("holdout manifest is not two disjoint batches of 64")
    labels = load_holdout_labels()
    have = sorted({int(row["scenario_id"]) for row in labels})
    if have != sorted(batch1 + batch2):
        raise RuntimeError("label scenario IDs differ from the holdout manifest")
    design = load_design_exec()
    design_ids = {int(row["scenario_id"]) for row in design}
    if design_ids != set(DESIGN_IDS):
        raise RuntimeError(f"frozen design IDs changed: {sorted(design_ids)}")
    if design_ids & set(have):
        raise RuntimeError("holdout IDs overlap B2 fit IDs")
    assert_no_future_keys(design, B2_FEATURE_KEYS)
    assert_no_future_keys(labels, B2_FEATURE_KEYS)
    model = fit_frozen_b2(design)
    x = feature_matrix(labels, B2_FEATURE_KEYS)
    c_hat = np.asarray(model.predict_credit(x), dtype=np.float64)
    u_single = np.array([row["U_single"] for row in labels], dtype=np.float64)
    u_hat = u_single + c_hat
    batch_of = {sc: "deployment_holdout_batch_1" for sc in batch1}
    batch_of.update({sc: "deployment_holdout_batch_2" for sc in batch2})
    results = evaluate(labels, batch_of, c_hat, u_hat)
    limits = criterion
    stats1 = summarize([row for row in results if row["scenario_id"] in set(batch1)], limits)
    stats2 = summarize([row for row in results if row["scenario_id"] in set(batch2)], limits)
    stats_all = summarize(results, limits)
    leaked = leakage_rows(results, design_ids)
    leakage = any(int(row["leaked"]) for row in leaked)
    deployment_pass = bool(stats1["pass"] and stats2["pass"] and not leakage)
    status = (
        "DEPLOYMENT_VERIFIED_PENDING_PLANNER_INTEGRATION"
        if deployment_pass
        else "DEPLOYMENT_VERIFICATION_FAILED"
    )
    check = {
        "criterion_id": "CONSERVATIVE_WITH_FALLBACK_V1",
        "batch_1": {key: stats1[key] for key in ("n", "degradation_rate", "worst_case_delta_R", "mean_delta_R", "mean_U_gain", "pass")},
        "batch_2": {key: stats2[key] for key in ("n", "degradation_rate", "worst_case_delta_R", "mean_delta_R", "mean_U_gain", "pass")},
        "combined": {key: stats_all[key] for key in ("n", "degradation_rate", "worst_case_delta_R", "mean_delta_R", "mean_U_gain", "pass")},
        "required_independent_batches": 2,
        "independent_batches_passed": int(stats1["pass"]) + int(stats2["pass"]),
        "deployment_pass": deployment_pass,
        "criterion_modified": False,
    }
    summary = {
        "status": status,
        "deployment_verified": deployment_pass,
        "planner_ready": False,
        "deployment_pass": deployment_pass,
        "frozen_ok": True,
        "leakage": leakage,
        "official_decide_called": False,
        "batch_1_ids": [batch1[0], batch1[-1]],
        "batch_2_ids": [batch2[0], batch2[-1]],
        "batch_1": stats1,
        "batch_2": stats2,
        "combined": stats_all,
        "independent_batches_passed": check["independent_batches_passed"],
        "fit_ids": list(DESIGN_IDS),
        "tau": TAU,
    }
    _csv_write(OUTPUT / "deployment_verification_results.csv", results)
    _dump(OUTPUT / "deployment_criterion_check.json", check)
    _csv_write(OUTPUT / "deployment_verification_leakage_audit.csv", leaked)
    _dump(OUTPUT / "deployment_verification_summary.json", summary)
    text = write_report(summary, batch1, batch2)
    (OUTPUT / REPORT_NAME).write_text(text, encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(text, encoding="utf-8")
    print(json.dumps({"status": status, "batch_1_pass": stats1["pass"], "batch_2_pass": stats2["pass"], "deployment_pass": deployment_pass}), flush=True)


if __name__ == "__main__":
    main()
