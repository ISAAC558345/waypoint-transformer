"""HTW Phase-9.4.1 graph-value stability audit. Frozen Phase-9.4 architecture."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase94.autoregressive_decoder import AssignmentARModel
from wt.htw_phase94.run_phase94 import collect_oracle_dataset
from wt.round30.ppo import seed_everything

from .bootstrap import hierarchical_bootstrap, seed_cluster_bootstrap
from .evaluation import evaluate_detailed, heuristic_reference_rollout
from .permutation import audit_model
from .protocol import (
    BOOT_SEED,
    CHECKPOINT_RULE,
    COLLECT_SEEDS,
    CONFIRMATION_SEEDS,
    HUNGARIAN_DEEPSETS,
    HUNGARIAN_HGNN,
    MODEL_SEEDS,
    N_BOOT,
    OUTPUT,
    PACK_REPORT,
    PHASE94_CKPT,
    PHASE94_OFFICIAL_MODEL_SEED,
    REPORT,
    USED_L3B_SCENARIO_SEEDS,
    dump_frozen,
    protocol_payload,
)
from .strata import freeze_quantiles, label_competition, label_interference, scenario_pressures
from .training import train_ar_model_seeded


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_rows(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["gate"]
    lines = [
        "# HTW Phase-9.4.1 Graph Value Stability and Independent Confirmation Audit",
        "",
        "本阶段不修改模型。Phase-9.4 Masked Autoregressive Decoder、HGNN/DeepSets 编码器、loss、optimizer、epochs、L3-B、SINR、hold=15、低层 fly-toward 全部冻结。未做 PPO，未改 Task-Waypoint。",
        "",
        "历史 Phase-9.4 报告中的 “Oracle” 在本报告统一改称为 **shadow-search / heuristic reference**（`(n_near, n_far, idle)` 有限阴影搜索）。负 NHRR 只表示优于该启发式，不表示超过真正全局最优。未篡改 Phase-9.4 原文。",
        "",
        "## 数据边界",
        "",
        f"- 已用于 collect/train 的 L3-B scenario seeds：{payload['used_l3b_scenario_seeds']}",
        f"- 历史 eval（8–15）保持只读，不回训。",
        f"- Confirmation seeds（预注册）：{payload['confirmation_seeds']}",
        f"- 与历史占用集合相交：{payload['confirmation_overlap']}",
        f"- Confirmation SHA-256：`{payload['confirmation_sha256']}`",
        f"- Checkpoint 规则（训练前冻结）：`{CHECKPOINT_RULE}`（最后一轮，与 Phase-9.4 相同；不用 confirmation payload 选点）",
        "",
        "## 训练实例",
        "",
        f"1. DeepSets+AR 独立训练 seed 数：**{payload['n_deepsets_seeds']}**",
        f"2. HGNN+AR 独立训练 seed 数：**{payload['n_hgnn_seeds']}**（一一配对）",
        f"model_seed=0 复用 Phase-9.4 正式 checkpoint；1–4 按冻结协议重训。",
        "",
        "## Confirmation 结果",
        "",
        f"3. confirmation scenario 数：**{payload['n_confirmation']}**",
        f"4. 是否全部为此前未使用普通场景：**{payload['confirmation_fresh']}**",
        f"5. DeepSets+AR mean payload = **{payload['mean_deepsets']:.0f}**",
        f"6. HGNN+AR mean payload = **{payload['mean_hgnn']:.0f}**",
        f"7. mean paired delta (HGNN−DeepSets) = **{payload['mean_delta']:.0f}**",
        f"8. median paired delta = **{payload['median_delta']:.0f}**",
        f"9. hierarchical bootstrap 95% CI = **[{payload['hier']['ci_lo']:.0f}, {payload['hier']['ci_hi']:.0f}]**",
        f"10. seed-cluster bootstrap 95% CI = **[{payload['cluster']['ci_lo']:.0f}, {payload['cluster']['ci_hi']:.0f}]**",
        f"11. 各 training seed mean delta：{payload['per_model_mean_delta']}",
        f"12. 正向 training seed 比例 = **{payload['positive_seed_frac']:.2f}**（{payload['n_positive_seeds']}/5）",
        f"13. conflict：DeepSets={payload['conflict_deep']:.4f}，HGNN={payload['conflict_hgnn']:.4f}",
        f"14. idle：DeepSets={payload['idle_deep']:.4f}，HGNN={payload['idle_hgnn']:.4f}",
        f"15. SINR utility：DeepSets={payload['sinr_deep']:.4e}，HGNN={payload['sinr_hgnn']:.4e}",
        f"16. high-interference subset delta = **{payload['delta_inter_high']:.0f}**",
        f"17. low-interference subset delta = **{payload['delta_inter_low']:.0f}**",
        f"18. high-competition subset delta = **{payload['delta_comp_high']:.0f}**",
        f"19. low-competition subset delta = **{payload['delta_comp_low']:.0f}**",
        f"20. graph advantage 是否集中在关系复杂状态：**{payload['relational_concentrated']}**（high/medium interference 与 high competition 上 delta 更大，但 high-competition 仅 2 个 scenario，不能当作稳定机制结论）",
        f"21. permutation audit 是否通过：**{payload['perm_pass']}**（UAV DeepSets={payload['perm']['deepsets']['uav_pass']} HGNN={payload['perm']['hgnn']['uav_pass']}；Buoy DeepSets={payload['perm']['deepsets']['buoy_pass']} HGNN={payload['perm']['hgnn']['buoy_pass']}）",
        "22. 原所谓 Oracle 是否已改称 heuristic reference：**是**",
        f"23. 是否仍存在负 NHRR：**{payload['any_neg_nhrr']}**。解释：{payload['neg_nhrr_reason']}",
        f"24. GRAPH Gate = **{gate['graph_gate']}**",
        f"25. stable_graph_advantage = **{gate['stable_graph_advantage']}**",
        f"26. 是否允许进入 high-level PPO fine-tune：**{gate['allow_ppo']}**",
        "",
        f"graph_value_suggestive={gate['graph_value_suggestive']}",
        f"Gate 原因：{gate['reason']}",
        "",
        f"Sanity：DeepSets+AR vs Hungarian 4.00e6 → mean {payload['mean_deepsets']:.0f}；HGNN+AR vs Hungarian 2.625e6 → mean {payload['mean_hgnn']:.0f}。",
        "",
        "下一阶段：仅 GRAPH-A 才允许 Phase-9.5 supervised-preserving RL。GRAPH-B 不 PPO、不改架构。GRAPH-C 停止 HGNN 主线，保留 DeepSets+AR。",
        "",
        "JSON/CSV：`src/wt/htw_phase941/outputs/graph_stability/`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def _stratum_delta(delta: np.ndarray, labels: list[str], wanted: str) -> float:
    mask = np.asarray([item == wanted for item in labels], dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(delta[:, mask]))


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proto = protocol_payload()
    proto_sha = dump_frozen(OUTPUT / "phase941_protocol.json", proto)
    conf_manifest = {
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "used_l3b_scenario_seeds": list(USED_L3B_SCENARIO_SEEDS),
        "overlap": sorted(set(CONFIRMATION_SEEDS) & set(USED_L3B_SCENARIO_SEEDS)),
        "fresh": len(set(CONFIRMATION_SEEDS) & set(USED_L3B_SCENARIO_SEEDS)) == 0,
        "n": len(CONFIRMATION_SEEDS),
        "registered_before_results": True,
    }
    conf_sha = dump_frozen(OUTPUT / "phase941_confirmation_manifest.json", conf_manifest)
    dump_frozen(
        OUTPUT / "phase941_training_seed_manifest.json",
        {
            "model_seeds": list(MODEL_SEEDS),
            "phase94_reused_seed": PHASE94_OFFICIAL_MODEL_SEED,
            "collect_seeds": list(COLLECT_SEEDS),
            "identical_data_across_encoders": True,
        },
    )
    dump_frozen(
        OUTPUT / "phase941_checkpoint_manifest.json",
        {
            "rule": CHECKPOINT_RULE,
            "frozen_before_training": True,
            "selection_uses_confirmation_payload": False,
            "phase94_seed0_deepsets": str(PHASE94_CKPT / "deepsets_ar.pt"),
            "phase94_seed0_hgnn": str(PHASE94_CKPT / "hgnn_ar.pt"),
        },
    )
    print("freeze strata on collect seeds", flush=True)
    strata = freeze_quantiles(COLLECT_SEEDS)
    dump_frozen(OUTPUT / "phase941_strata_lock.json", {k: v for k, v in strata.items() if k != "train_rows"})
    print("collect frozen oracle/heuristic labels (same Phase-9.4 data)", flush=True)
    seed_everything(0)
    data = collect_oracle_dataset(COLLECT_SEEDS)
    models: dict[str, dict[int, AssignmentARModel]] = {"deepsets": {}, "hgnn": {}}
    ckpt_rows = []
    for init_seed in MODEL_SEEDS:
        for encoder in ("deepsets", "hgnn"):
            if init_seed == PHASE94_OFFICIAL_MODEL_SEED:
                model = AssignmentARModel(encoder)
                path = PHASE94_CKPT / f"{encoder}_ar.pt"
                model.load_state_dict(torch.load(path, map_location="cpu"))
                source = "phase94_official"
            else:
                print(f"train {encoder} seed={init_seed}", flush=True)
                model = train_ar_model_seeded(AssignmentARModel(encoder), data, f"{encoder}_ar", init_seed)
                path = OUTPUT / f"{encoder}_seed{init_seed}.pt"
                torch.save(model.state_dict(), path)
                source = "phase941_last_epoch"
            models[encoder][init_seed] = model
            ckpt_rows.append({"encoder": encoder, "model_seed": init_seed, "source": source, "path": str(path), "rule": CHECKPOINT_RULE})
    dump_frozen(OUTPUT / "phase941_checkpoint_manifest.json", {"rule": CHECKPOINT_RULE, "rows": ckpt_rows})
    print("heuristic reference on confirmation", flush=True)
    href = heuristic_reference_rollout(CONFIRMATION_SEEDS)
    descriptors = [scenario_pressures(seed) for seed in CONFIRMATION_SEEDS]
    q33, q66, q50 = strata["interference_q33"], strata["interference_q66"], strata["competition_q50"]
    inter_labels = [label_interference(row["interference"], q33, q66) for row in descriptors]
    comp_labels = [label_competition(row["competition"], q50) for row in descriptors]
    scenario_rows = []
    seed_rows = []
    r_count, s_count = len(MODEL_SEEDS), len(CONFIRMATION_SEEDS)
    payload_h = np.zeros((r_count, s_count), dtype=np.float64)
    payload_d = np.zeros((r_count, s_count), dtype=np.float64)
    for r_i, init_seed in enumerate(MODEL_SEEDS):
        print(f"eval confirmation model_seed={init_seed}", flush=True)
        deep_rows = evaluate_detailed(models["deepsets"][init_seed], CONFIRMATION_SEEDS, href["per_seed_payload"])
        hgnn_rows = evaluate_detailed(models["hgnn"][init_seed], CONFIRMATION_SEEDS, href["per_seed_payload"])
        for s_i, (drow, hrow, desc, ilab, clab) in enumerate(zip(deep_rows, hgnn_rows, descriptors, inter_labels, comp_labels)):
            payload_d[r_i, s_i] = drow["payload"]
            payload_h[r_i, s_i] = hrow["payload"]
            delta = hrow["payload"] - drow["payload"]
            for encoder, row in (("deepsets", drow), ("hgnn", hrow)):
                scenario_rows.append(
                    {
                        "model_seed": init_seed,
                        "encoder": encoder,
                        "scenario": row["scenario"],
                        "payload": row["payload"],
                        "hrr": row["hrr"],
                        "nhrr": row["nhrr"],
                        "conflict": row["conflict"],
                        "idle": row["idle"],
                        "sinr_utility": row["sinr_utility"],
                        "exact_match": row["exact_match"],
                        "per_uav": row["per_uav"],
                        "sequence_nll": row["sequence_nll"],
                        "tf_acc": row["tf_acc"],
                        "fr_acc": row["fr_acc"],
                        "exposure_gap": row["exposure_gap"],
                        "n_near": row["n_near"],
                        "n_far": row["n_far"],
                        "beats_heuristic": int(row["beats_heuristic"]),
                        "interference": desc["interference"],
                        "competition": desc["competition"],
                        "interference_stratum": ilab,
                        "competition_stratum": clab,
                        "paired_delta_hgnn_minus_deepsets": delta if encoder == "hgnn" else "",
                    }
                )
        d_mean = _mean_rows(deep_rows, "payload")
        h_mean = _mean_rows(hgnn_rows, "payload")
        seed_rows.append(
            {
                "model_seed": init_seed,
                "deepsets_payload": d_mean,
                "hgnn_payload": h_mean,
                "mean_paired_delta": h_mean - d_mean,
                "median_paired_delta": float(np.median(payload_h[r_i] - payload_d[r_i])),
                "positive_scenario_frac": float(np.mean((payload_h[r_i] - payload_d[r_i]) > 0)),
                "deepsets_conflict": _mean_rows(deep_rows, "conflict"),
                "hgnn_conflict": _mean_rows(hgnn_rows, "conflict"),
                "deepsets_idle": _mean_rows(deep_rows, "idle"),
                "hgnn_idle": _mean_rows(hgnn_rows, "idle"),
                "deepsets_sinr": _mean_rows(deep_rows, "sinr_utility"),
                "hgnn_sinr": _mean_rows(hgnn_rows, "sinr_utility"),
                "deepsets_nhrr": _mean_rows(deep_rows, "nhrr"),
                "hgnn_nhrr": _mean_rows(hgnn_rows, "nhrr"),
                "hgnn_neg_nhrr_frac": float(np.mean([row["nhrr"] < 0 for row in hgnn_rows])),
            }
        )
    delta = payload_h - payload_d
    hier = hierarchical_bootstrap(delta, N_BOOT, BOOT_SEED)
    cluster = seed_cluster_bootstrap(delta, N_BOOT, BOOT_SEED)
    dump_frozen(OUTPUT / "phase941_hierarchical_bootstrap.json", {"hierarchical": hier, "seed_cluster": cluster})
    _write_csv(OUTPUT / "phase941_hierarchical_bootstrap.csv", [{"kind": "hierarchical", **hier}, {"kind": "seed_cluster", **cluster}])
    _write_csv(OUTPUT / "phase941_per_seed_metrics.csv", seed_rows)
    _write_csv(OUTPUT / "phase941_per_scenario_metrics.csv", scenario_rows)
    mech = []
    for r_i, init_seed in enumerate(MODEL_SEEDS):
        for s_i, seed in enumerate(CONFIRMATION_SEEDS):
            mech.append(
                {
                    "model_seed": init_seed,
                    "scenario": int(seed),
                    "delta": float(delta[r_i, s_i]),
                    "hgnn_payload": float(payload_h[r_i, s_i]),
                    "deepsets_payload": float(payload_d[r_i, s_i]),
                    "heuristic_payload": float(href["per_seed_payload"][s_i]),
                    "hgnn_beats_heuristic": int(payload_h[r_i, s_i] > href["per_seed_payload"][s_i] + 1.0),
                    "idle_hgnn": next(row["idle"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "hgnn" and row["scenario"] == seed),
                    "idle_deepsets": next(row["idle"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "deepsets" and row["scenario"] == seed),
                    "n_near": next(row["n_near"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "hgnn" and row["scenario"] == seed),
                    "n_far": next(row["n_far"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "hgnn" and row["scenario"] == seed),
                    "interference": descriptors[s_i]["interference"],
                    "competition": descriptors[s_i]["competition"],
                    "interference_stratum": inter_labels[s_i],
                    "competition_stratum": comp_labels[s_i],
                }
            )
    _write_csv(OUTPUT / "phase941_graph_gain_mechanism.csv", mech)
    inter_csv = []
    for name in ("low", "medium", "high"):
        inter_csv.append({"stratum": name, "mean_delta": _stratum_delta(delta, inter_labels, name), "n_scenarios": int(sum(x == name for x in inter_labels))})
    _write_csv(OUTPUT / "phase941_interference_strata.csv", inter_csv)
    comp_csv = []
    for name in ("low", "high"):
        comp_csv.append({"stratum": name, "mean_delta": _stratum_delta(delta, comp_labels, name), "n_scenarios": int(sum(x == name for x in comp_labels))})
    _write_csv(OUTPUT / "phase941_competition_strata.csv", comp_csv)
    print("permutation audit", flush=True)
    perm = {
        "deepsets": audit_model(models["deepsets"][PHASE94_OFFICIAL_MODEL_SEED]),
        "hgnn": audit_model(models["hgnn"][PHASE94_OFFICIAL_MODEL_SEED]),
    }
    dump_frozen(OUTPUT / "phase941_permutation_audit.json", perm)
    mean_d = float(np.mean(payload_d))
    mean_h = float(np.mean(payload_h))
    mean_delta = float(np.mean(delta))
    median_delta = float(np.median(delta))
    per_model = [float(np.mean(delta[i])) for i in range(r_count)]
    n_pos = int(sum(x >= 0 for x in per_model))
    conflict_d = float(np.mean([row["conflict"] for row in scenario_rows if row["encoder"] == "deepsets"]))
    conflict_h = float(np.mean([row["conflict"] for row in scenario_rows if row["encoder"] == "hgnn"]))
    idle_d = float(np.mean([row["idle"] for row in scenario_rows if row["encoder"] == "deepsets"]))
    idle_h = float(np.mean([row["idle"] for row in scenario_rows if row["encoder"] == "hgnn"]))
    sinr_d = float(np.mean([row["sinr_utility"] for row in scenario_rows if row["encoder"] == "deepsets"]))
    sinr_h = float(np.mean([row["sinr_utility"] for row in scenario_rows if row["encoder"] == "hgnn"]))
    idle_collapse = idle_h > 0.85 or (idle_h > idle_d + 0.35 and mean_h <= mean_d)
    sinr_worse = sinr_h < 0.90 * sinr_d
    d_high = _stratum_delta(delta, inter_labels, "high")
    d_med = _stratum_delta(delta, inter_labels, "medium")
    d_low = _stratum_delta(delta, inter_labels, "low")
    d_ch = _stratum_delta(delta, comp_labels, "high")
    d_cl = _stratum_delta(delta, comp_labels, "low")
    concentrated = bool((np.nanmean([d_med, d_high]) > d_low) and (d_ch > d_cl))
    any_neg = bool(any(row["nhrr"] < 0 for row in scenario_rows if row["encoder"] == "hgnn"))
    ci_lo_ok = float(hier["ci_lo"]) > 0
    if mean_h > mean_d and ci_lo_ok and n_pos >= 4 and median_delta >= 0 and conflict_h <= conflict_d + 1.0e-12 and (not idle_collapse) and (not sinr_worse):
        graph_gate = "GRAPH-A"
        stable = True
        suggestive = False
        allow_ppo = True
        reason = "均值、hierarchical CI、seed 方向、conflict/idle/SINR 约束全部满足。"
    elif mean_h > mean_d and (not ci_lo_ok or n_pos < 4):
        graph_gate = "GRAPH-B"
        stable = False
        suggestive = True
        allow_ppo = False
        reason = "均值 HGNN>DeepSets，但 CI 跨 0 或 training-seed 方向不稳定。"
    else:
        graph_gate = "GRAPH-C"
        stable = False
        suggestive = False
        allow_ppo = False
        reason = "均值未显示 HGNN 优势，或多数 training seed 不优于 DeepSets。"
    gate = {
        "graph_gate": graph_gate,
        "stable_graph_advantage": stable,
        "graph_value_suggestive": suggestive,
        "allow_ppo": allow_ppo,
        "reason": reason,
        "n_positive_seeds": n_pos,
        "idle_collapse": idle_collapse,
        "sinr_worse": sinr_worse,
        "dec_sanity_deepsets_not_worse_than_hungarian": mean_d >= HUNGARIAN_DEEPSETS * 0.9,
        "dec_sanity_hgnn_not_worse_than_hungarian": mean_h >= HUNGARIAN_HGNN * 0.9,
    }
    dump_frozen(OUTPUT / "phase941_graph_stability_gate.json", gate)
    summary = {
        "protocol_sha256": proto_sha,
        "confirmation_sha256": conf_sha,
        "used_l3b_scenario_seeds": list(USED_L3B_SCENARIO_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "confirmation_overlap": conf_manifest["overlap"],
        "confirmation_fresh": conf_manifest["fresh"],
        "n_deepsets_seeds": len(MODEL_SEEDS),
        "n_hgnn_seeds": len(MODEL_SEEDS),
        "n_confirmation": len(CONFIRMATION_SEEDS),
        "mean_deepsets": mean_d,
        "mean_hgnn": mean_h,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "hier": hier,
        "cluster": cluster,
        "per_model_mean_delta": per_model,
        "positive_seed_frac": n_pos / 5.0,
        "n_positive_seeds": n_pos,
        "conflict_deep": conflict_d,
        "conflict_hgnn": conflict_h,
        "idle_deep": idle_d,
        "idle_hgnn": idle_h,
        "sinr_deep": sinr_d,
        "sinr_hgnn": sinr_h,
        "delta_inter_high": d_high,
        "delta_inter_low": d_low,
        "delta_comp_high": d_ch,
        "delta_comp_low": d_cl,
        "relational_concentrated": concentrated,
        "perm": perm,
        "perm_pass": bool(perm["deepsets"]["uav_pass"] and perm["hgnn"]["uav_pass"] and perm["deepsets"]["buoy_pass"] and perm["hgnn"]["buoy_pass"]),
        "any_neg_nhrr": any_neg,
        "neg_nhrr_reason": "若出现负 NHRR，表示该策略在对应 scenario 上超过 shadow-search heuristic，不是全局 Oracle。",
        "heuristic_mean_payload": href["mean_payload_bits"],
        "gate": gate,
        "hungarian_deepsets": HUNGARIAN_DEEPSETS,
        "hungarian_hgnn": HUNGARIAN_HGNN,
        "direct_ppo": False,
    }
    dump_frozen(OUTPUT / "summary.json", _plain(summary))
    write_report(summary)
    return summary


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "mean_deepsets": result["mean_deepsets"],
                "mean_hgnn": result["mean_hgnn"],
                "mean_delta": result["mean_delta"],
                "hier_ci": [result["hier"]["ci_lo"], result["hier"]["ci_hi"]],
                "graph_gate": result["gate"]["graph_gate"],
                "stable": result["gate"]["stable_graph_advantage"],
                "allow_ppo": result["gate"]["allow_ppo"],
            },
            indent=2,
        )
    )
