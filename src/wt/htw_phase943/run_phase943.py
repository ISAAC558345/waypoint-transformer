"""Phase-9.4.3 order-free decoder: structure tests then paired supervised confirmation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase9.features import build_graph, feature_dict_l3b, stack_feature_dicts
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase94.run_phase94 import collect_oracle_dataset
from wt.htw_phase941.bootstrap import hierarchical_bootstrap
from wt.htw_phase942.pressures import label_three, regime_of, scenario_pressures
from wt.l3b.env import L3BEnv
from wt.round30.ppo import seed_everything

from .evaluation import decode_assignment, encode_env, evaluate_model, heuristic_reference_rollout
from .metrics import ppv
from .order_free_decoder import OrderFreeAssignmentModel, decoder_parameter_count
from .permutation_audit import audit_model, candidate_order_invariance, permute_uav_batch
from .protocol import (
    AUDIT_SEEDS,
    AUX_RANK_WEIGHT,
    BATCH,
    BLOCKED_SCENARIO_SEEDS,
    BOOT_SEED,
    EPOCHS,
    LR,
    MODEL_SEEDS,
    N_BOOT,
    N_PER_REGIME,
    OUTPUT,
    PACK_REPORT,
    PHASE942_COMP,
    PHASE942_INTER,
    REPORT,
    SCAN_START,
    dump_frozen,
)
from .set_context import make_snap
from .set_likelihood_dp import exhaustive_logprob, subset_dp_logprob
from .training import train_set_model


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _batch(env, info):
    feats = stack_feature_dicts([feature_dict_l3b(info, env)])
    edge, edge_uu = build_graph(info, env)
    feats["edge"] = edge[None, ...]
    feats["edge_uu"] = edge_uu[None, ...]
    return feats


def dp_equivalence() -> dict[str, Any]:
    env = L3BEnv()
    info = env.reset(seed=0)
    batch = _batch(env, info)
    model = OrderFreeAssignmentModel("deepsets")
    model.eval()
    from wt.l3b.policies import oracle_option_assignment

    target = np.asarray(oracle_option_assignment(env), dtype=np.int64)
    encoded = encode_env(model, batch)
    snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
    with torch.no_grad():
        dp = float(subset_dp_logprob(model, encoded, snap, target, batch["buoy_mask"][0]).cpu())
        ex = float(exhaustive_logprob(model, encoded, snap, target, batch["buoy_mask"][0]).cpu())
    ok = abs(dp - ex) < 1.0e-4
    return {"dp": dp, "exhaustive_720": ex, "abs_diff": abs(dp - ex), "consistent": ok}


def select_confirmation(inter: Mapping[str, Any], comp: Mapping[str, Any]) -> dict[str, Any]:
    blocked = set(BLOCKED_SCENARIO_SEEDS)
    buckets = {key: [] for key in ("R1", "R2", "R3", "R4")}
    seed = int(SCAN_START)
    scanned = 0
    while any(len(buckets[key]) < N_PER_REGIME for key in buckets) and seed < 30000:
        if seed not in blocked:
            scanned += 1
            press = scenario_pressures(int(seed))
            i_lab = label_three(press["interference"], float(inter["interference_q33"]), float(inter["interference_q66"]))
            c_lab = label_three(press["competition"], float(comp["competition_q33"]), float(comp["competition_q67"]))
            regime = regime_of(i_lab, c_lab)
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
    rows = []
    for key in ("R1", "R2", "R3", "R4"):
        rows.extend(buckets[key])
    return {"rows": rows, "seeds": [int(row["scenario"]) for row in rows], "counts": {k: len(v) for k, v in buckets.items()}, "scanned": scanned}


def regime_pack(delta: np.ndarray, mask: np.ndarray, name: str) -> dict[str, Any]:
    sub = delta[:, mask] if mask.any() else np.zeros((delta.shape[0], 0))
    if sub.size == 0:
        return {"name": name, "n": 0, "mean": float("nan"), "median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pos_seeds": 0, "positive_seed_frac": float("nan")}
    hier = hierarchical_bootstrap(sub, N_BOOT, BOOT_SEED + {"R1": 1, "R2": 2, "R3": 3, "R4": 4}[name])
    means = sub.mean(axis=1)
    return {
        "name": name,
        "n": int(mask.sum()),
        "mean": float(np.mean(sub)),
        "median": float(np.median(sub)),
        "ci_lo": hier["ci_lo"],
        "ci_hi": hier["ci_hi"],
        "n_pos_seeds": int(np.sum(means >= 0)),
        "positive_seed_frac": float(np.mean(means >= 0)),
        "per_seed": means.tolist(),
    }


def write_report(p: Mapping[str, Any]) -> None:
    lines = [
        "# HTW Phase-9.4.3 Permutation-Equivariant Order-Free Global Assignment Decoder",
        "",
        "未改 HGNN/DeepSets encoder、L3-B、SINR、低层、Task-Waypoint。未做 PPO。只替换高层 assignment decoder 与 set-level supervised loss。",
        "",
        f"1. 旧 decoder 问题：固定 UAV 顺序 AR 造成 ORDER-C（OSR≈0.39/0.55，优劣反转≈0.50）。",
        "2. 新 decoder 每步在 remaining UAV×合法 buoy∪idle 上共享打分，选择 (m,k)；谁先分配由模型决定，不是外部 index 循环。",
        "3. 是否 combination-aware：是，每步条件于 partial assignment set。",
        "4. 是否保留 partial SINR context：是，且只依赖已选 assignment **集合**。",
        f"5. DP vs 6!：**{p['dp']['consistent']}**（diff={p['dp']['abs_diff']:.3e}）",
        f"6. UAV permutation equivariance = **{p['perm']['deepsets']['uav_rate']:.4f}** / HGNN **{p['perm']['hgnn']['uav_rate']:.4f}**",
        f"7. Buoy = **{p['perm']['deepsets']['buoy_rate']:.4f}** / **{p['perm']['hgnn']['buoy_rate']:.4f}**",
        f"8. Joint = **{p['perm']['deepsets']['joint_rate']:.4f}** / **{p['perm']['hgnn']['joint_rate']:.4f}**",
        f"9. candidate enumeration invariance = **{p['cand']['candidate_order_invariance_rate']:.4f}**",
        f"10. PPV (DeepSets/HGNN) = **{p.get('ppv_deep', float('nan')):.4f}** / **{p.get('ppv_hgnn', float('nan')):.4f}**",
        f"11. ORDERFIX Gate = **{p['orderfix']['gate']}**",
        f"12. decoder_order_dependence_fixed = **{p['orderfix']['decoder_order_dependence_fixed']}**",
        f"13. DeepSets mean payload = **{p.get('mean_deep', float('nan')):.0f}**",
        f"14. HGNN mean payload = **{p.get('mean_hgnn', float('nan')):.0f}**",
        f"15. mean paired delta = **{p.get('mean_delta', float('nan')):.0f}**",
        f"16. median delta = **{p.get('median_delta', float('nan')):.0f}**",
        f"17. hierarchical 95% CI = **{p.get('hier', {})}**",
        f"18. 5 model-seed delta = **{p.get('per_seed_delta', [])}**",
        f"19. 正向 seed 比例 = **{p.get('pos_frac', float('nan'))}**",
        f"20. R1 = **{p.get('R1', {})}**",
        f"21. R2 = **{p.get('R2', {})}**",
        f"22. R3 = **{p.get('R3', {})}**",
        f"23. R4 = **{p.get('R4', {})}**",
        f"24. CGRAPH Gate = **{p.get('cgraph', {}).get('gate', 'n/a')}**",
        f"25. GRAPH Gate = **{p.get('graph', {}).get('gate', 'n/a')}**",
        f"26. seed 2/3/4 是否仍失败：**{p.get('neg_seeds_still', 'n/a')}**",
        f"27. 先前 graph instability 可归因于 decoder order 的部分：**{p.get('order_attribution', 'n/a')}**",
        f"28. HGNN 普遍优于 DeepSets？**{p.get('q_universal', 'n/a')}**",
        f"29. 仅高关系复杂度条件优势？**{p.get('q_conditional', 'n/a')}**",
        f"30. 下一阶段：**{p.get('next_step', 'n/a')}**",
        "",
        f"ORDERFIX 原因：{p['orderfix'].get('reason', '')}",
        f"训练={'已完成' if p.get('trained') else '未训练（结构门未过）'}。confirmation seeds={p.get('seeds', [])}。",
        "",
        "JSON：`src/wt/htw_phase943/outputs/order_free_decoder/`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proto = {
        "phase": "9.4.3",
        "ppo": False,
        "train_encoders_frozen": True,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "aux_rank_weight": AUX_RANK_WEIGHT,
        "collect_seeds": list(COLLECT_SEEDS),
        "blocked_scenarios": list(BLOCKED_SCENARIO_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "checkpoint_rule": "last_epoch_fixed_phase94",
        "decoder_params": decoder_parameter_count(),
    }
    dump_frozen(OUTPUT / "phase943_protocol.json", proto)
    dump_frozen(OUTPUT / "phase943_training_manifest.json", {"collect_seeds": list(COLLECT_SEEDS), "no_241_242_confirmation": True})
    dump_frozen(OUTPUT / "phase943_model_seed_manifest.json", {"model_seeds": list(MODEL_SEEDS), "paired": True})
    print("DP vs 6! exhaustive", flush=True)
    dp = dp_equivalence()
    dump_frozen(OUTPUT / "phase943_order_marginalization_equivalence.json", dp)
    if not dp["consistent"]:
        payload = {"dp": dp, "perm": {"deepsets": {}, "hgnn": {}}, "cand": {}, "orderfix": {"gate": "ORDERFIX-C", "decoder_order_dependence_fixed": False, "reason": "DP 与 6! 不一致，停止训练。"}, "trained": False}
        write_report(payload)
        return payload
    print("structural permutation audit (random init)", flush=True)
    perm = {}
    cand = {}
    for encoder in ("deepsets", "hgnn"):
        model = OrderFreeAssignmentModel(encoder)
        model.eval()
        perm[encoder] = audit_model(model, AUDIT_SEEDS, n_uav=20, n_buoy=20, n_joint=10)
        cand[encoder] = candidate_order_invariance(model, AUDIT_SEEDS, n_shuffle=100)
    dump_frozen(OUTPUT / "phase943_structural_permutation_audit.json", perm)
    dump_frozen(OUTPUT / "phase943_candidate_order_audit.json", cand)
    min_eq = min(perm[e][k] for e in perm for k in ("uav_rate", "buoy_rate", "joint_rate"))
    cand_rate = min(cand[e]["candidate_order_invariance_rate"] for e in cand)
    if min_eq < 0.999 or cand_rate < 1.0 - 1.0e-9:
        orderfix = {
            "gate": "ORDERFIX-C" if min_eq < 0.95 else "ORDERFIX-B",
            "decoder_order_dependence_fixed": False,
            "reason": f"结构审计未达 0.999（min_eq={min_eq:.4f}, cand={cand_rate:.4f}），停止训练。",
        }
        dump_frozen(OUTPUT / "phase943_orderfix_gate.json", orderfix)
        payload = {"dp": dp, "perm": perm, "cand": cand["deepsets"], "orderfix": orderfix, "trained": False}
        write_report(payload)
        return payload
    inter, comp = _load(PHASE942_INTER), _load(PHASE942_COMP)
    selected = select_confirmation(inter, comp)
    dump_frozen(
        OUTPUT / "phase943_confirmation_manifest.json",
        {"name": "phase943_confirmation_v1", "seeds": selected["seeds"], "counts": selected["counts"], "blocked": list(BLOCKED_SCENARIO_SEEDS), "registered_before_payloads": True},
    )
    print("collect train data (historical collect seeds only)", flush=True)
    seed_everything(0)
    data = collect_oracle_dataset(COLLECT_SEEDS)
    models: dict[str, dict[int, OrderFreeAssignmentModel]] = {"deepsets": {}, "hgnn": {}}
    for init_seed in MODEL_SEEDS:
        for encoder in ("deepsets", "hgnn"):
            print(f"train {encoder} seed={init_seed}", flush=True)
            model = train_set_model(OrderFreeAssignmentModel(encoder), data, f"{encoder}_set", init_seed)
            torch.save(model.state_dict(), OUTPUT / f"{encoder}_seed{init_seed}.pt")
            models[encoder][init_seed] = model
    seeds = tuple(selected["seeds"])
    meta = {int(row["scenario"]): row for row in selected["rows"]}
    href = heuristic_reference_rollout(seeds)
    n_r, n_s = len(MODEL_SEEDS), len(seeds)
    pay_h = np.zeros((n_r, n_s))
    pay_d = np.zeros((n_r, n_s))
    scenario_rows = []
    seed_rows = []
    for r_i, init_seed in enumerate(MODEL_SEEDS):
        print(f"eval confirmation seed={init_seed}", flush=True)
        deep_rows = evaluate_model(models["deepsets"][init_seed], seeds, href["per_seed_payload"])
        hgnn_rows = evaluate_model(models["hgnn"][init_seed], seeds, href["per_seed_payload"])
        for s_i, (drow, hrow) in enumerate(zip(deep_rows, hgnn_rows)):
            pay_d[r_i, s_i] = drow["payload"]
            pay_h[r_i, s_i] = hrow["payload"]
            m = meta[int(drow["scenario"])]
            for encoder, row in (("deepsets", drow), ("hgnn", hrow)):
                scenario_rows.append({"model_seed": init_seed, "encoder": encoder, **row, "regime": m["regime"], "i_label": m["i_label"], "c_label": m["c_label"]})
        seed_rows.append({"model_seed": init_seed, "deepsets": float(np.mean(pay_d[r_i])), "hgnn": float(np.mean(pay_h[r_i])), "delta": float(np.mean(pay_h[r_i] - pay_d[r_i]))})
    delta = pay_h - pay_d
    _write_csv(OUTPUT / "phase943_per_scenario_metrics.csv", scenario_rows)
    _write_csv(OUTPUT / "phase943_per_seed_metrics.csv", seed_rows)
    hier = hierarchical_bootstrap(delta, N_BOOT, BOOT_SEED)
    dump_frozen(OUTPUT / "phase943_hierarchical_bootstrap.json", hier)
    _write_csv(OUTPUT / "phase943_hierarchical_bootstrap.csv", [{"kind": "hierarchical", **hier}])
    rpacks = {}
    for name in ("R1", "R2", "R3", "R4"):
        mask = np.asarray([meta[int(s)]["regime"] == name for s in seeds])
        rpacks[name] = regime_pack(delta, mask, name)
    _write_csv(OUTPUT / "phase943_regime_metrics.csv", [{"regime": k, **{kk: vv for kk, vv in v.items() if kk != "per_seed"}} for k, v in rpacks.items()])
    print("PPV on held confirmation (encoder seed 0, 4 UAV perms, 8 scenarios)", flush=True)
    from wt.htw_phase942.decode_order import inverse_assignment
    import copy

    ppv_rows = []
    rng = np.random.default_rng(9433)
    probe = list(seeds)[:8]
    for encoder in ("deepsets", "hgnn"):
        model = models[encoder][0]
        for seed in probe:
            env = L3BEnv()
            info = env.reset(seed=int(seed))
            batch = _batch(env, info)
            with torch.no_grad():
                base = decode_assignment(model, batch, env)
            from wt.htw_phase9.const import HOLD
            from wt.l3b.policies import fly_toward

            def _roll(assign):
                e = L3BEnv()
                inf = e.reset(seed=int(seed))
                pending = 0
                z = assign
                while not inf.get("terminated", False):
                    if pending <= 0:
                        pending = int(HOLD)
                    inf = e.step(fly_toward(e, z), z)
                    pending -= 1
                return float(inf["cumulative_payload_bits"])

            local = [_roll(base)]
            for _ in range(4):
                uav_perm = rng.permutation(6)
                env_u = copy.deepcopy(env)
                env_u.uav_positions = np.asarray(env.uav_positions, dtype=np.float64)[uav_perm]
                with torch.no_grad():
                    pred = decode_assignment(model, permute_uav_batch(batch, uav_perm), env_u)
                mapped = inverse_assignment(pred, uav_perm, env.config.num_uavs)
                local.append(_roll(mapped))
            ppv_rows.append({"encoder": encoder, "scenario": int(seed), "ppv": ppv(np.asarray(local)), "payloads": str(local)})
    _write_csv(OUTPUT / "phase943_permutation_sensitivity.csv", ppv_rows)
    ppv_deep = float(np.mean([row["ppv"] for row in ppv_rows if row["encoder"] == "deepsets"]))
    ppv_hgnn = float(np.mean([row["ppv"] for row in ppv_rows if row["encoder"] == "hgnn"]))
    conflict_h = float(np.mean([row["conflict"] for row in scenario_rows if row["encoder"] == "hgnn"]))
    conflict_d = float(np.mean([row["conflict"] for row in scenario_rows if row["encoder"] == "deepsets"]))
    idle_h = float(np.mean([row["idle"] for row in scenario_rows if row["encoder"] == "hgnn"]))
    idle_d = float(np.mean([row["idle"] for row in scenario_rows if row["encoder"] == "deepsets"]))
    sinr_h = float(np.mean([row["sinr_utility"] for row in scenario_rows if row["encoder"] == "hgnn"]))
    sinr_d = float(np.mean([row["sinr_utility"] for row in scenario_rows if row["encoder"] == "deepsets"]))
    orderfix = {
        "gate": "ORDERFIX-A" if ppv_deep < 0.05 and ppv_hgnn < 0.05 and conflict_h <= 1e-12 and conflict_d <= 1e-12 else "ORDERFIX-B",
        "decoder_order_dependence_fixed": True,
        "reason": f"结构审计通过；PPV deep={ppv_deep:.4f} hgnn={ppv_hgnn:.4f}。",
    }
    if ppv_deep > 0.15 or ppv_hgnn > 0.15:
        orderfix = {"gate": "ORDERFIX-C", "decoder_order_dependence_fixed": False, "reason": "PPV 仍高。"}
    elif ppv_deep > 0.05 or ppv_hgnn > 0.05:
        orderfix = {"gate": "ORDERFIX-B", "decoder_order_dependence_fixed": False, "reason": "PPV 下降但仍可测。"}
    dump_frozen(OUTPUT / "phase943_orderfix_gate.json", orderfix)
    mean_d, mean_h = float(np.mean(pay_d)), float(np.mean(pay_h))
    mean_delta = float(np.mean(delta))
    median_delta = float(np.median(delta))
    per_seed = delta.mean(axis=1)
    n_pos = int(np.sum(per_seed >= 0))
    graph = {"gate": "n/a", "reason": "ORDERFIX 未达 A，不正式讨论 Graph Value。"}
    cgraph = {"gate": "n/a"}
    if orderfix["gate"] == "ORDERFIX-A":
        idle_collapse = idle_h > 0.85 or (idle_h > idle_d + 0.35 and mean_h <= mean_d)
        sinr_worse = sinr_h < 0.9 * sinr_d
        if mean_h > mean_d and hier["ci_lo"] > 0 and n_pos >= 4 and median_delta >= 0 and conflict_h <= conflict_d + 1e-12 and not idle_collapse and not sinr_worse:
            graph = {"gate": "GRAPH-A", "reason": "配对 payload 稳定正优势。"}
        elif mean_h > mean_d:
            graph = {"gate": "GRAPH-B", "reason": "均值正但不稳定。"}
        else:
            graph = {"gate": "GRAPH-C", "reason": "均值非正或多数 seed 不支持。"}
        r4 = rpacks["R4"]
        if r4["mean"] > 0 and r4["ci_lo"] > 0 and r4["n_pos_seeds"] >= 4 and r4["median"] >= 0:
            cgraph = {"gate": "CGRAPH-A", "conditional_graph_advantage_confirmed": True}
        elif r4["mean"] > 0:
            cgraph = {"gate": "CGRAPH-B", "conditional_graph_advantage_confirmed": False, "conditional_graph_advantage_suggestive": True}
        else:
            cgraph = {"gate": "CGRAPH-C", "conditional_graph_advantage_confirmed": False}
    dump_frozen(OUTPUT / "phase943_graph_gate.json", graph)
    dump_frozen(OUTPUT / "phase943_conditional_graph_gate.json", cgraph)
    neg_still = [int(MODEL_SEEDS[i]) for i, val in enumerate(per_seed) if val < 0]
    if orderfix["gate"] != "ORDERFIX-A":
        next_step = "continue decoder (ORDERFIX not A); do not discuss HGNN performance"
        q_uni, q_cond = "不作正式结论", "不作正式结论"
        attr = "尚未在修复后的干净 decoder 上量化。"
    elif graph["gate"] == "GRAPH-A":
        next_step = "PPO fine-tune (later independent phase; not now)"
        q_uni, q_cond = "是（需 GRAPH-A）", "同时可报告 R4 条件结果"
        attr = "order 修复后若 seed 稳定性提高，则部分不稳定来自旧 decoder。"
    elif graph["gate"] == "GRAPH-B":
        next_step = "investigate remaining encoder instability"
        q_uni, q_cond = "否", "见 CGRAPH"
        attr = "order 已修，剩余不稳定更可能来自 encoder/训练。"
    else:
        next_step = "retain DeepSets" if graph["gate"] == "GRAPH-C" else "stop HGNN"
        q_uni, q_cond = "否", "否" if cgraph.get("gate") == "CGRAPH-C" else str(cgraph)
        attr = "order 已修仍无图优势，则不稳定主要不是 decode order。"
    payload = {
        "dp": dp,
        "perm": perm,
        "cand": cand["deepsets"],
        "orderfix": orderfix,
        "trained": True,
        "seeds": list(seeds),
        "mean_deep": mean_d,
        "mean_hgnn": mean_h,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "hier": hier,
        "per_seed_delta": per_seed.tolist(),
        "pos_frac": n_pos / 5.0,
        "R1": rpacks["R1"],
        "R2": rpacks["R2"],
        "R3": rpacks["R3"],
        "R4": rpacks["R4"],
        "cgraph": cgraph,
        "graph": graph,
        "neg_seeds_still": neg_still,
        "order_attribution": attr,
        "q_universal": q_uni,
        "q_conditional": q_cond,
        "next_step": next_step,
        "ppv_deep": ppv_deep,
        "ppv_hgnn": ppv_hgnn,
        "decoder_params": decoder_parameter_count(),
    }
    dump_frozen(OUTPUT / "summary.json", _plain(payload))
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps({"orderfix": result["orderfix"]["gate"], "trained": result.get("trained"), "graph": result.get("graph", {}).get("gate"), "next": result.get("next_step")}, indent=2))
