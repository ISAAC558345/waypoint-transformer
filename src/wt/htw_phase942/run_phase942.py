"""Phase-9.4.2: frozen-model conditional graph + decoder-order audit. No training."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase94.autoregressive_decoder import AssignmentARModel
from wt.htw_phase941.bootstrap import hierarchical_bootstrap
from wt.htw_phase941.evaluation import evaluate_detailed, heuristic_reference_rollout
from wt.htw_phase941.protocol import MODEL_SEEDS

from .decode_order import assignment_entropy, evaluate_payload_order, freeze_orders
from .pressures import freeze_competition_tertiles, label_three, regime_of, scenario_pressures
from .protocol import (
    BLOCKED_SCENARIO_SEEDS,
    BOOT_SEED,
    COLLECT_SEEDS,
    N_BOOT,
    N_PER_REGIME,
    ORDER_SEED,
    OUTPUT,
    PACK_REPORT,
    PHASE941_CKPT,
    PHASE941_LOCK,
    REPORT,
    SCAN_LIMIT,
    SCAN_START,
    dump_frozen,
)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_models() -> dict[str, dict[int, AssignmentARModel]]:
    manifest = _load_json(PHASE941_CKPT)
    models: dict[str, dict[int, AssignmentARModel]] = {"deepsets": {}, "hgnn": {}}
    for row in manifest["rows"]:
        model = AssignmentARModel(row["encoder"])
        model.load_state_dict(torch.load(row["path"], map_location="cpu"))
        model.eval()
        models[row["encoder"]][int(row["model_seed"])] = model
    return models


def freeze_thresholds() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _load_json(PHASE941_LOCK)
    inter = {
        "metric": "episode-mean interference_uav_uav under exclusive matching (Phase-9.4.1 I(s))",
        "source": "reused phase941_strata_lock.json; COLLECT_SEEDS only",
        "interference_q33": float(lock["interference_q33"]),
        "interference_q66": float(lock["interference_q66"]),
        "low": "I(s) <= q33",
        "high": "I(s) > q66",
        "medium": "q33 < I(s) <= q66",
    }
    comp_extra = freeze_competition_tertiles(COLLECT_SEEDS)
    comp = {
        **comp_extra,
        "low": "C(s) <= q33",
        "high": "C(s) > q67",
        "medium": "q33 < C(s) <= q67",
        "phase941_competition_q50_reference": float(lock["competition_q50"]),
    }
    dump_frozen(OUTPUT / "phase942_interference_thresholds.json", inter)
    dump_frozen(OUTPUT / "phase942_competition_thresholds.json", comp)
    return inter, comp


def classify(press: dict[str, float], inter: Mapping[str, Any], comp: Mapping[str, Any]) -> tuple[str, str, str | None]:
    i_lab = label_three(float(press["interference"]), float(inter["interference_q33"]), float(inter["interference_q66"]))
    c_lab = label_three(float(press["competition"]), float(comp["competition_q33"]), float(comp["competition_q67"]))
    return i_lab, c_lab, regime_of(i_lab, c_lab)


def select_scenarios(inter: Mapping[str, Any], comp: Mapping[str, Any]) -> dict[str, Any]:
    blocked = set(BLOCKED_SCENARIO_SEEDS)
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in ("R1", "R2", "R3", "R4")}
    scanned = 0
    seed = int(SCAN_START)
    while any(len(buckets[key]) < N_PER_REGIME for key in buckets) and seed < int(SCAN_LIMIT):
        if seed in blocked:
            seed += 1
            continue
        scanned += 1
        press = scenario_pressures(seed)
        i_lab, c_lab, regime = classify(press, inter, comp)
        if regime is not None and len(buckets[regime]) < N_PER_REGIME:
            buckets[regime].append(
                {
                    "scenario": int(seed),
                    "regime": regime,
                    "interference": press["interference"],
                    "competition": press["competition"],
                    "i_label": i_lab,
                    "c_label": c_lab,
                }
            )
        seed += 1
    selected = []
    for key in ("R1", "R2", "R3", "R4"):
        selected.extend(buckets[key])
    return {
        "scanned_candidates": scanned,
        "scan_end_exclusive": seed,
        "blocked": list(BLOCKED_SCENARIO_SEEDS),
        "counts": {key: len(buckets[key]) for key in buckets},
        "rows": selected,
        "seeds": [int(row["scenario"]) for row in selected],
    }


def interaction_bootstrap(high: np.ndarray, low: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n_r = high.shape[0]
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        rows = rng.integers(0, n_r, size=n_r)
        acc_h, acc_l = [], []
        for row in rows:
            acc_h.append(high[int(row), rng.integers(0, high.shape[1], size=high.shape[1])])
            acc_l.append(low[int(row), rng.integers(0, low.shape[1], size=low.shape[1])])
        stats[index] = float(np.mean(np.concatenate(acc_h)) - np.mean(np.concatenate(acc_l)))
    lo, hi = np.percentile(stats, (2.5, 97.5))
    point = float(np.mean(high) - np.mean(low))
    return {"mean": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": int(n_boot)}


def regime_stats(delta: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    if delta.size == 0 or delta.shape[1] == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "positive_scenario_frac": float("nan"), "positive_seed_frac": float("nan"), "n_pos_seeds": 0}
    hier = hierarchical_bootstrap(delta, n_boot, seed)
    seed_means = delta.mean(axis=1)
    return {
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "ci_lo": hier["ci_lo"],
        "ci_hi": hier["ci_hi"],
        "positive_scenario_frac": float(np.mean(delta > 0)),
        "positive_seed_frac": float(np.mean(seed_means >= 0)),
        "n_pos_seeds": int(np.sum(seed_means >= 0)),
        "per_seed_mean": seed_means.tolist(),
    }


def write_report(payload: Mapping[str, Any]) -> None:
    r = payload["regimes"]
    lines = [
        "# HTW Phase-9.4.2 Conditional Graph Advantage and Decoder-Order Sensitivity Audit",
        "",
        "本阶段为冻结模型的 **inference-only** 机制验证。未训练、未 PPO、未改 HGNN/DeepSets/AR decoder/L3-B/SINR/低层/Task-Waypoint。",
        "未根据模型 payload 挑选 scenario。I(s)/C(s) 在 confirmation 之前冻结。Primary 只使用 canonical decode order **O0**。",
        "heuristic reference = shadow-search `(n_near, n_far, idle)`。负 NHRR 不表示超过全局最优。",
        "",
        f"1. 新独立 scenario 数量：**{payload['n_scenarios']}**（seeds={payload['seeds']}）",
        f"2. R1–R4 数量：R1={r['R1']['n']}，R2={r['R2']['n']}，R3={r['R3']['n']}，R4={r['R4']['n']}",
        f"3. interference 阈值：q33={payload['inter']['interference_q33']:.6g}，q66={payload['inter']['interference_q66']:.6g}（复用 Phase-9.4.1）",
        f"4. competition 阈值：q33={payload['comp']['competition_q33']:.6g}，q67={payload['comp']['competition_q67']:.6g}（COLLECT_SEEDS tertiles）",
        f"5. R1 delta mean/CI = **{r['R1']['delta']['mean']:.0f}** / [{r['R1']['delta']['ci_lo']:.0f}, {r['R1']['delta']['ci_hi']:.0f}]",
        f"6. R2 delta mean/CI = **{r['R2']['delta']['mean']:.0f}** / [{r['R2']['delta']['ci_lo']:.0f}, {r['R2']['delta']['ci_hi']:.0f}]",
        f"7. R3 delta mean/CI = **{r['R3']['delta']['mean']:.0f}** / [{r['R3']['delta']['ci_lo']:.0f}, {r['R3']['delta']['ci_hi']:.0f}]",
        f"8. R4 delta mean/CI = **{r['R4']['delta']['mean']:.0f}** / [{r['R4']['delta']['ci_lo']:.0f}, {r['R4']['delta']['ci_hi']:.0f}]",
        f"9. R4 正向 training seed 比例 = **{r['R4']['delta']['positive_seed_frac']:.2f}**（{r['R4']['delta']['n_pos_seeds']}/5）",
        f"10. Graph×Interference interaction (highI−lowI) = **{payload['inter_ix']['mean']:.0f}** CI=[{payload['inter_ix']['ci_lo']:.0f}, {payload['inter_ix']['ci_hi']:.0f}]",
        f"11. Graph×Competition interaction (highC−lowC) = **{payload['comp_ix']['mean']:.0f}** CI=[{payload['comp_ix']['ci_lo']:.0f}, {payload['comp_ix']['ci_hi']:.0f}]",
        f"12. relation-complexity interaction (R4−R1) = **{payload['rel_ix']['mean']:.0f}** CI=[{payload['rel_ix']['ci_lo']:.0f}, {payload['rel_ix']['ci_hi']:.0f}]",
        f"13. HGNN 优势是否随关系复杂度增强：**{payload['advantage_grows_with_complexity']}**",
        f"14. low-complexity (R1) 下 HGNN 是否反而较差：**{payload['r1_hgnn_worse']}**",
        f"15. DeepSets OSR (mean) = **{payload['osr_deep']:.4f}**",
        f"16. HGNN OSR (mean) = **{payload['osr_hgnn']:.4f}**",
        f"17. 模型优劣是否随 order 反转：**{payload['order_reverses']}**（反转率={payload['reversal_rate']:.3f}）",
        f"18. ORDER Gate = **{payload['order_gate']['gate']}**",
        f"19. 负 training seed 失败原因：**{payload['neg_seed_reason']}**",
        f"20. CGRAPH Gate = **{payload['cgraph']['gate']}**",
        f"21. conditional_graph_advantage_confirmed = **{payload['cgraph']['conditional_graph_advantage_confirmed']}**",
        f"22. 高关系复杂度稳定优势？**{payload['answer_q2']}**",
        f"23. HGNN 普遍优于 DeepSets？**{payload['answer_q1']}**",
        f"24. 下一阶段选择：**{payload['next_step']}**",
        "",
        f"CGRAPH 原因：{payload['cgraph']['reason']}",
        f"ORDER 原因：{payload['order_gate']['reason']}",
        "",
        "JSON/CSV：`src/wt/htw_phase942/outputs/conditional_graph/`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proto = {
        "phase": "9.4.2",
        "inference_only": True,
        "ppo": False,
        "train": False,
        "modified_encoder": False,
        "modified_decoder": False,
        "modified_l3b": False,
        "primary_order": "O0",
        "max_over_order_forbidden": True,
        "model_seeds": list(MODEL_SEEDS),
        "blocked_scenario_seeds": list(BLOCKED_SCENARIO_SEEDS),
        "n_per_regime": N_PER_REGIME,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "order_seed": ORDER_SEED,
    }
    dump_frozen(OUTPUT / "phase942_protocol.json", proto)
    inter, comp = freeze_thresholds()
    orders = freeze_orders(ORDER_SEED)
    dump_frozen(OUTPUT / "phase942_decoder_order_protocol.json", {"orders": orders, "seed": ORDER_SEED, "primary": "O0"})
    print("select scenarios by I(s)/C(s) only", flush=True)
    selected = select_scenarios(inter, comp)
    conf_sha = dump_frozen(
        OUTPUT / "phase942_relational_confirmation_manifest.json",
        {
            "name": "phase942_relational_confirmation_v1",
            "seeds": selected["seeds"],
            "counts": selected["counts"],
            "blocked": selected["blocked"],
            "scanned_candidates": selected["scanned_candidates"],
            "registered_before_model_outcomes": True,
        },
    )
    _write_csv(OUTPUT / "phase942_regime_manifest.csv", selected["rows"])
    seeds = tuple(selected["seeds"])
    seed_to_meta = {int(row["scenario"]): row for row in selected["rows"]}
    print("heuristic reference (not used for scenario selection)", flush=True)
    href = heuristic_reference_rollout(seeds)
    print("load frozen checkpoints", flush=True)
    models = load_models()
    n_r, n_s = len(MODEL_SEEDS), len(seeds)
    pay_h = np.zeros((n_r, n_s))
    pay_d = np.zeros((n_r, n_s))
    scenario_rows = []
    entropy = {"deepsets": defaultdict(list), "hgnn": defaultdict(list)}
    for r_i, init_seed in enumerate(MODEL_SEEDS):
        print(f"O0 eval model_seed={init_seed}", flush=True)
        deep_rows = evaluate_detailed(models["deepsets"][init_seed], seeds, href["per_seed_payload"])
        hgnn_rows = evaluate_detailed(models["hgnn"][init_seed], seeds, href["per_seed_payload"])
        for s_i, (drow, hrow) in enumerate(zip(deep_rows, hgnn_rows)):
            pay_d[r_i, s_i] = drow["payload"]
            pay_h[r_i, s_i] = hrow["payload"]
            meta = seed_to_meta[int(drow["scenario"])]
            for encoder, row in (("deepsets", drow), ("hgnn", hrow)):
                scenario_rows.append(
                    {
                        "model_seed": init_seed,
                        "encoder": encoder,
                        "scenario": row["scenario"],
                        "regime": meta["regime"],
                        "i_label": meta["i_label"],
                        "c_label": meta["c_label"],
                        "interference": meta["interference"],
                        "competition": meta["competition"],
                        "payload": row["payload"],
                        "hrr": row["hrr"],
                        "nhrr": row["nhrr"],
                        "conflict": row["conflict"],
                        "idle": row["idle"],
                        "sinr_utility": row["sinr_utility"],
                        "exact_match": row["exact_match"],
                        "per_uav": row["per_uav"],
                        "tf_acc": row["tf_acc"],
                        "fr_acc": row["fr_acc"],
                        "sequence_nll": row["sequence_nll"],
                        "order": "O0",
                    }
                )
    delta = pay_h - pay_d
    _write_csv(OUTPUT / "phase942_per_seed_scenario_metrics.csv", scenario_rows)
    regime_metrics = {}
    regime_csv = []
    for name in ("R1", "R2", "R3", "R4"):
        mask = np.asarray([seed_to_meta[int(s)]["regime"] == name for s in seeds], dtype=bool)
        sub = delta[:, mask] if mask.any() else np.zeros((n_r, 0))
        stats = regime_stats(sub, N_BOOT, BOOT_SEED + {"R1": 1, "R2": 2, "R3": 3, "R4": 4}[name])
        d_mean = float(np.mean(pay_d[:, mask])) if mask.any() else float("nan")
        h_mean = float(np.mean(pay_h[:, mask])) if mask.any() else float("nan")
        idle_d = float(np.mean([row["idle"] for row in scenario_rows if row["encoder"] == "deepsets" and row["regime"] == name])) if mask.any() else float("nan")
        idle_h = float(np.mean([row["idle"] for row in scenario_rows if row["encoder"] == "hgnn" and row["regime"] == name])) if mask.any() else float("nan")
        conf_d = float(np.mean([row["conflict"] for row in scenario_rows if row["encoder"] == "deepsets" and row["regime"] == name])) if mask.any() else float("nan")
        conf_h = float(np.mean([row["conflict"] for row in scenario_rows if row["encoder"] == "hgnn" and row["regime"] == name])) if mask.any() else float("nan")
        sinr_d = float(np.mean([row["sinr_utility"] for row in scenario_rows if row["encoder"] == "deepsets" and row["regime"] == name])) if mask.any() else float("nan")
        sinr_h = float(np.mean([row["sinr_utility"] for row in scenario_rows if row["encoder"] == "hgnn" and row["regime"] == name])) if mask.any() else float("nan")
        regime_metrics[name] = {
            "n": int(mask.sum()),
            "deepsets_payload": d_mean,
            "hgnn_payload": h_mean,
            "delta": stats,
            "idle_deep": idle_d,
            "idle_hgnn": idle_h,
            "conflict_deep": conf_d,
            "conflict_hgnn": conf_h,
            "sinr_deep": sinr_d,
            "sinr_hgnn": sinr_h,
        }
        regime_csv.append(
            {
                "regime": name,
                "n": int(mask.sum()),
                "deepsets_payload": d_mean,
                "hgnn_payload": h_mean,
                "delta_mean": stats["mean"],
                "delta_median": stats["median"],
                "ci_lo": stats["ci_lo"],
                "ci_hi": stats["ci_hi"],
                "positive_scenario_frac": stats["positive_scenario_frac"],
                "positive_seed_frac": stats["positive_seed_frac"],
            }
        )
    _write_csv(OUTPUT / "phase942_regime_metrics.csv", regime_csv)
    high_i = np.asarray([seed_to_meta[int(s)]["i_label"] == "high" for s in seeds])
    low_i = np.asarray([seed_to_meta[int(s)]["i_label"] == "low" for s in seeds])
    high_c = np.asarray([seed_to_meta[int(s)]["c_label"] == "high" for s in seeds])
    low_c = np.asarray([seed_to_meta[int(s)]["c_label"] == "low" for s in seeds])
    inter_ix = interaction_bootstrap(delta[:, high_i], delta[:, low_i], N_BOOT, BOOT_SEED + 11)
    comp_ix = interaction_bootstrap(delta[:, high_c], delta[:, low_c], N_BOOT, BOOT_SEED + 12)
    r4_mask = np.asarray([seed_to_meta[int(s)]["regime"] == "R4" for s in seeds])
    r1_mask = np.asarray([seed_to_meta[int(s)]["regime"] == "R1" for s in seeds])
    rel_ix = interaction_bootstrap(delta[:, r4_mask], delta[:, r1_mask], N_BOOT, BOOT_SEED + 13)
    dump_frozen(OUTPUT / "phase942_interference_interaction.json", inter_ix)
    dump_frozen(OUTPUT / "phase942_competition_interaction.json", comp_ix)
    dump_frozen(OUTPUT / "phase942_relation_interaction.json", rel_ix)
    r4 = regime_metrics["R4"]
    idle_collapse = r4["idle_hgnn"] > 0.85 or (r4["idle_hgnn"] > r4["idle_deep"] + 0.35 and r4["hgnn_payload"] <= r4["deepsets_payload"])
    sinr_worse = r4["sinr_hgnn"] < 0.90 * r4["sinr_deep"]
    ix_not_neg = float(rel_ix["ci_hi"]) >= 0.0
    if (
        r4["hgnn_payload"] > r4["deepsets_payload"]
        and float(r4["delta"]["ci_lo"]) > 0
        and int(r4["delta"]["n_pos_seeds"]) >= 4
        and float(r4["delta"]["median"]) >= 0
        and r4["conflict_hgnn"] <= r4["conflict_deep"] + 1.0e-12
        and (not idle_collapse)
        and (not sinr_worse)
        and float(rel_ix["mean"]) > 0
        and ix_not_neg
    ):
        cgraph = {
            "gate": "CGRAPH-A",
            "conditional_graph_advantage_confirmed": True,
            "conditional_graph_advantage_suggestive": False,
            "reason": "R4 均值、CI、seed 方向与 idle/conflict/SINR/interaction 约束均满足。",
        }
    elif r4["hgnn_payload"] > r4["deepsets_payload"] and (float(r4["delta"]["ci_lo"]) <= 0 or int(r4["delta"]["n_pos_seeds"]) < 4):
        cgraph = {
            "gate": "CGRAPH-B",
            "conditional_graph_advantage_confirmed": False,
            "conditional_graph_advantage_suggestive": True,
            "reason": "R4 均值 HGNN>DeepSets，但 CI 跨 0 或 training-seed 不稳定。",
        }
    else:
        cgraph = {
            "gate": "CGRAPH-C",
            "conditional_graph_advantage_confirmed": False,
            "conditional_graph_advantage_suggestive": False,
            "reason": "高关系复杂度 regime 中 HGNN 无稳定优势。",
        }
    dump_frozen(OUTPUT / "phase942_conditional_graph_gate.json", cgraph)
    print("decoder order sensitivity", flush=True)
    order_rows = []
    payloads_order = {enc: np.zeros((n_r, n_s, len(orders))) for enc in ("deepsets", "hgnn")}
    order_names = list(orders.keys())
    for r_i, init_seed in enumerate(MODEL_SEEDS):
        for encoder in ("deepsets", "hgnn"):
            for s_i, seed in enumerate(seeds):
                pays, idles, sinrs, assigns = [], [], [], []
                for o_i, oname in enumerate(order_names):
                    packed = evaluate_payload_order(models[encoder][init_seed], int(seed), np.asarray(orders[oname]))
                    payloads_order[encoder][r_i, s_i, o_i] = packed["payload"]
                    pays.append(packed["payload"])
                    idles.append(packed["idle"])
                    sinrs.append(packed["sinr_utility"])
                    assigns.append(tuple(packed["assignments"][0].tolist()) if packed["assignments"].size else tuple())
                    if oname == "O0":
                        entropy[encoder][init_seed].append(assignment_entropy(packed["assignments"]))
                std = float(np.std(pays))
                mean_p = float(np.mean(pays))
                disagree = 0
                for a in assigns[1:]:
                    disagree += int(a != assigns[0])
                order_rows.append(
                    {
                        "model_seed": init_seed,
                        "encoder": encoder,
                        "scenario": int(seed),
                        "regime": seed_to_meta[int(seed)]["regime"],
                        "payload_mean": mean_p,
                        "payload_std": std,
                        "payload_range": float(max(pays) - min(pays)),
                        "osr": std / (abs(mean_p) + 1.0e-9),
                        "idle_std": float(np.std(idles)),
                        "sinr_std": float(np.std(sinrs)),
                        "assignment_disagreement_vs_O0": disagree,
                    }
                )
    _write_csv(OUTPUT / "phase942_decoder_order_sensitivity.csv", order_rows)
    osr_d = float(np.mean([row["osr"] for row in order_rows if row["encoder"] == "deepsets"]))
    osr_h = float(np.mean([row["osr"] for row in order_rows if row["encoder"] == "hgnn"]))
    o0_i = order_names.index("O0")
    reversals = 0
    comparable = 0
    mean_delta_by_order = []
    for o_i, oname in enumerate(order_names):
        dlt = payloads_order["hgnn"][:, :, o_i] - payloads_order["deepsets"][:, :, o_i]
        mean_delta_by_order.append({"order": oname, "mean_delta": float(np.mean(dlt))})
        if o_i == o0_i:
            continue
        sign0 = np.sign(payloads_order["hgnn"][:, :, o0_i] - payloads_order["deepsets"][:, :, o0_i])
        signo = np.sign(dlt)
        mask = np.abs(payloads_order["hgnn"][:, :, o0_i] - payloads_order["deepsets"][:, :, o0_i]) > 1.0
        comparable += int(mask.sum())
        reversals += int(((sign0 != signo) & mask).sum())
    reversal_rate = float(reversals / max(comparable, 1))
    o0_sign = np.sign(mean_delta_by_order[0]["mean_delta"])
    n_flip = int(sum(np.sign(item["mean_delta"]) != o0_sign and abs(item["mean_delta"]) > 1.0 for item in mean_delta_by_order[1:]))
    order_reverses = reversal_rate > 0.30 or n_flip >= 2
    if osr_d < 0.10 and osr_h < 0.10 and reversal_rate < 0.20 and n_flip == 0:
        order_gate = {"gate": "ORDER-A", "reason": "payload OSR 低，且 graph 方向在 orders 间一致。"}
    elif order_reverses:
        order_gate = {"gate": "ORDER-C", "reason": "HGNN vs DeepSets 优劣随 UAV decode order 频繁反转。"}
    else:
        order_gate = {"gate": "ORDER-B", "reason": "存在一定 order sensitivity，但 conditional graph 方向大体一致。"}
    dump_frozen(
        OUTPUT / "phase942_order_confound_gate.json",
        {**order_gate, "osr_deepsets": osr_d, "osr_hgnn": osr_h, "reversal_rate": reversal_rate, "n_mean_sign_flips": n_flip, "mean_delta_by_order": mean_delta_by_order},
    )
    pos_seeds = [int(MODEL_SEEDS[i]) for i, val in enumerate(delta.mean(axis=1)) if val >= 0]
    neg_seeds = [int(MODEL_SEEDS[i]) for i, val in enumerate(delta.mean(axis=1)) if val < 0]
    mech_rows = []
    for r_i, init_seed in enumerate(MODEL_SEEDS):
        r4_d = float(np.mean(delta[r_i, r4_mask])) if r4_mask.any() else float("nan")
        r1_d = float(np.mean(delta[r_i, r1_mask])) if r1_mask.any() else float("nan")
        idle_h = float(np.mean([row["idle"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "hgnn"]))
        idle_d = float(np.mean([row["idle"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "deepsets"]))
        fr_h = float(np.mean([row["fr_acc"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "hgnn"]))
        nll_h = float(np.mean([row["sequence_nll"] for row in scenario_rows if row["model_seed"] == init_seed and row["encoder"] == "hgnn"]))
        mech_rows.append(
            {
                "model_seed": init_seed,
                "group": "positive" if init_seed in pos_seeds else "negative",
                "mean_delta_all": float(np.mean(delta[r_i])),
                "mean_delta_R1": r1_d,
                "mean_delta_R4": r4_d,
                "hgnn_idle": idle_h,
                "deepsets_idle": idle_d,
                "hgnn_fr_acc": fr_h,
                "hgnn_sequence_nll": nll_h,
                "hgnn_assignment_entropy": float(np.mean(entropy["hgnn"][init_seed])),
                "deepsets_assignment_entropy": float(np.mean(entropy["deepsets"][init_seed])),
                "fails_only_low_complexity": bool(r1_d < 0 and r4_d >= 0),
                "global_fail": bool(r1_d < 0 and r4_d < 0),
            }
        )
    _write_csv(OUTPUT / "phase942_model_seed_mechanism.csv", mech_rows)
    if neg_seeds and all(row["global_fail"] for row in mech_rows if row["model_seed"] in neg_seeds):
        neg_reason = f"负 seed {neg_seeds} 在 R1 与 R4 均失败，偏全局训练失败而非单一 regime。"
    elif neg_seeds and all(row["fails_only_low_complexity"] for row in mech_rows if row["model_seed"] in neg_seeds):
        neg_reason = f"负 seed {neg_seeds} 主要在低复杂度失败，高复杂度仍非负。"
    else:
        neg_reason = f"负 seed {neg_seeds}（正 seed {pos_seeds}）；见 phase942_model_seed_mechanism.csv。未删除负 seed。"
    overall_mean = float(np.mean(delta))
    answer_q1 = "否。总体 paired delta 未形成稳定普遍优势。"
    if overall_mean > 0 and hierarchical_bootstrap(delta, N_BOOT, BOOT_SEED)["ci_lo"] > 0:
        answer_q1 = "是（本阶段 confirmation 上 CI 下界>0）。"
    answer_q2 = "否，CGRAPH 未达 A。"
    if cgraph["gate"] == "CGRAPH-A":
        answer_q2 = "是，仅限高干扰+高竞争（R4），不是普遍优势。"
    elif cgraph["gate"] == "CGRAPH-B":
        answer_q2 = "suggestive：R4 均值正向但不稳定。"
    if order_gate["gate"] == "ORDER-C":
        next_step = "fix decoder order dependence"
    elif cgraph["gate"] == "CGRAPH-A" and order_gate["gate"] in {"ORDER-A", "ORDER-B"}:
        next_step = "retain HGNN conditionally"
    elif cgraph["gate"] == "CGRAPH-B":
        next_step = "retain HGNN conditionally (exploratory only; no PPO)"
    else:
        next_step = "use DeepSets+AR"
    summary = {
        "n_scenarios": len(seeds),
        "seeds": list(seeds),
        "confirmation_sha256": conf_sha,
        "inter": inter,
        "comp": comp,
        "regimes": regime_metrics,
        "inter_ix": inter_ix,
        "comp_ix": comp_ix,
        "rel_ix": rel_ix,
        "advantage_grows_with_complexity": bool(rel_ix["mean"] > 0),
        "r1_hgnn_worse": bool(regime_metrics["R1"]["hgnn_payload"] < regime_metrics["R1"]["deepsets_payload"]),
        "osr_deep": osr_d,
        "osr_hgnn": osr_h,
        "order_reverses": order_reverses,
        "reversal_rate": reversal_rate,
        "order_gate": order_gate,
        "neg_seed_reason": neg_reason,
        "cgraph": cgraph,
        "answer_q1": answer_q1,
        "answer_q2": answer_q2,
        "next_step": next_step,
        "overall_mean_delta": overall_mean,
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
                "n": result["n_scenarios"],
                "cgraph": result["cgraph"]["gate"],
                "order": result["order_gate"]["gate"],
                "r4_mean": result["regimes"]["R4"]["delta"]["mean"],
                "next": result["next_step"],
            },
            indent=2,
        )
    )
