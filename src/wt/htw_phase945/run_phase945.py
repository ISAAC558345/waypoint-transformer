"""Phase-9.4.5 utility-aligned objective: train on collect/eval hold states, then new confirmation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase9.const import EVAL_SEEDS
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase941.bootstrap import hierarchical_bootstrap
from wt.htw_phase941.protocol import MODEL_SEEDS
from wt.htw_phase942.pressures import label_three, regime_of, scenario_pressures
from wt.htw_phase943.evaluation import evaluate_model, heuristic_reference_rollout
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel, decoder_parameter_count
from wt.htw_phase943.permutation_audit import audit_model, candidate_order_invariance
from wt.htw_phase944.tables import write_csv
from wt.round30.ppo import seed_everything

from .dataset import collect_utility_split
from .metrics import mean_metrics
from .protocol import (
    BATCH,
    BLOCKED_SCENARIO_SEEDS,
    BOOT_SEED,
    DELTA_RANGE_FRAC,
    EPOCHS,
    HOLD,
    LAMBDA_RANK,
    LAMBDA_REF,
    LR,
    N_BOOT,
    N_PER_REGIME,
    NORM_EPS,
    OUTPUT,
    PACK_REPORT,
    PHASE942_COMP,
    PHASE942_INTER,
    PHASE943_CKPT,
    POOL_SEED,
    POOL_SIZE,
    REPORT,
    SCAN_START,
    TAU,
    VAL_EVERY,
    dump_frozen,
)
from .training import split_metrics, train_utility_model


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split(name: str) -> dict[str, Any]:
    packed = np.load(OUTPUT / f"{name}.npz")
    data = {key: packed[key] for key in packed.files}
    man = _load(OUTPUT / f"{name}_manifest.json")
    data["meta"] = man["meta"]
    data["hashes"] = man["hashes"]
    data["split"] = man["split"]
    data["seeds"] = man["seeds"]
    data["pool_size"] = man["pool_size"]
    return data


def save_split(data: dict[str, Any], name: str) -> None:
    np.savez_compressed(
        OUTPUT / f"{name}.npz",
        **{key: data[key] for key in data if key not in {"meta", "hashes", "split", "seeds", "pool_size"}},
    )
    dump_frozen(
        OUTPUT / f"{name}_manifest.json",
        {"split": data["split"], "seeds": data["seeds"], "n": int(data["assignment"].shape[0]), "pool_size": data["pool_size"], "hashes": data["hashes"], "meta": data["meta"]},
    )


def load_model(encoder: str, seed: int, util: bool) -> OrderFreeAssignmentModel:
    model = OrderFreeAssignmentModel(encoder)
    path = (OUTPUT / f"{encoder}_seed{seed}.pt") if util else (PHASE943_CKPT / f"{encoder}_seed{seed}.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def select_confirmation(inter: Mapping[str, Any], comp: Mapping[str, Any]) -> dict[str, Any]:
    blocked = set(BLOCKED_SCENARIO_SEEDS)
    buckets = {key: [] for key in ("R1", "R2", "R3", "R4")}
    seed = int(SCAN_START)
    scanned = 0
    while any(len(buckets[key]) < N_PER_REGIME for key in buckets) and seed < 40000:
        if seed not in blocked:
            scanned += 1
            press = scenario_pressures(int(seed))
            i_lab = label_three(press["interference"], float(inter["interference_q33"]), float(inter["interference_q66"]))
            c_lab = label_three(press["competition"], float(comp["competition_q33"]), float(comp["competition_q67"]))
            regime = regime_of(i_lab, c_lab)
            if regime is not None and len(buckets[regime]) < N_PER_REGIME:
                buckets[regime].append({"scenario": int(seed), "regime": regime, "interference": press["interference"], "competition": press["competition"], "i_label": i_lab, "c_label": c_lab})
        seed += 1
    rows = []
    for key in ("R1", "R2", "R3", "R4"):
        rows.extend(buckets[key])
    leak = set(int(r["scenario"]) for r in rows) & blocked
    if leak:
        raise RuntimeError(f"confirmation leakage: {sorted(leak)}")
    return {"rows": rows, "seeds": [int(row["scenario"]) for row in rows], "counts": {k: len(v) for k, v in buckets.items()}, "scanned": scanned}


def regime_pack(delta: np.ndarray, mask: np.ndarray, name: str) -> dict[str, Any]:
    sub = delta[:, mask] if mask.any() else np.zeros((delta.shape[0], 0))
    if sub.size == 0:
        return {"name": name, "n": 0, "mean": float("nan"), "median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pos_seeds": 0}
    hier = hierarchical_bootstrap(sub, N_BOOT, BOOT_SEED + {"R1": 1, "R2": 2, "R3": 3, "R4": 4}[name])
    means = sub.mean(axis=1)
    return {"name": name, "n": int(mask.sum()), "mean": float(np.mean(sub)), "median": float(np.median(sub)), "ci_lo": hier["ci_lo"], "ci_hi": hier["ci_hi"], "n_pos_seeds": int(np.sum(means >= 0)), "positive_seed_frac": float(np.mean(means >= 0)), "per_seed": means.tolist()}


def obj_gate(old: dict[str, float], new: dict[str, float], conflict_old: float, conflict_new: float, seed_std_old: float, seed_std_new: float) -> dict[str, Any]:
    better_s = float(new["spearman"]) > float(old["spearman"]) + 0.05
    better_r = float(new["top1_regret"]) < float(old["top1_regret"]) - 1.0
    better_e = float(new["expected_utility"]) > float(old["expected_utility"])
    legal = conflict_new <= conflict_old + 1.0e-6
    stable = seed_std_new <= seed_std_old * 1.25 + 1.0e-6
    n_ok = int(better_s) + int(better_r) + int(better_e) + int(legal) + int(stable)
    if n_ok >= 4 and better_s and better_r:
        gate = "OBJ-A"
    elif n_ok >= 2:
        gate = "OBJ-B"
    else:
        gate = "OBJ-C"
    return {"gate": gate, "better_spearman": better_s, "better_regret": better_r, "better_expected": better_e, "conflict_ok": legal, "seed_var_ok": stable}


def write_report(p: Mapping[str, Any]) -> None:
    lines = [
        "# HTW Phase-9.4.5 Utility-Aligned Set Supervision for Global Assignment",
        "",
        "只改 supervised objective。未改 HGNN/DeepSets encoder、Order-Free Decoder、L3-B、SINR、hold=15、低层、Task-Waypoint。未做 PPO。",
        "phase943/941/942 confirmation 未进入 candidate/utility/tau/lambda/checkpoint。",
        "",
        f"1. candidate pool size = **{p['pool_size']}**",
        f"2. 生成：model-independent 合法池（heuristic / exclusive / 1-2 edge / competition / interference / idle / seeded diverse）seed={p['pool_seed']}",
        f"3. utility = 冻结 L3-B + fly-toward 下 **H={p['hold']} payload increment**",
        f"4. 是否 actual H15 payload：**是**",
        f"5. 旧 reference 仍是 heuristic/shadow-search：**是**（L_ref 权重={p['lambda_ref']}）",
        f"6. train near-optimal multiplicity (within 5% best-observed) mean n=**{p.get('multiplicity', float('nan')):.2f}**",
        f"7. utility-logP Spearman old/new DeepSets=**{p['spearman_ds_old']:.4f}/{p['spearman_ds_new']:.4f}** HGNN=**{p['spearman_hg_old']:.4f}/{p['spearman_hg_new']:.4f}**",
        f"8. candidate regret old/new DeepSets=**{p['regret_ds_old']:.1f}/{p['regret_ds_new']:.1f}** HGNN=**{p['regret_hg_old']:.1f}/{p['regret_hg_new']:.1f}**",
        f"9. NOM_5 old/new DeepSets=**{p['nom_ds_old']:.4f}/{p['nom_ds_new']:.4f}** HGNN=**{p['nom_hg_old']:.4f}/{p['nom_hg_new']:.4f}**",
        f"10. DeepSets old payload = **{p['d_old']:.0f}**",
        f"11. DeepSets util payload = **{p['d_util']:.0f}**",
        f"12. HGNN old payload = **{p['g_old']:.0f}**",
        f"13. HGNN util payload = **{p['g_util']:.0f}**",
        f"14. DeepSets objective delta = **{p['d_delta']:.0f}** CI={p['d_ci']}",
        f"15. HGNN objective delta = **{p['g_delta']:.0f}** CI={p['g_ci']}",
        f"16. 对应 95% CI 见上。",
        f"17. seed0-4 DeepSets delta={p['d_per_seed']}；HGNN delta={p['g_per_seed']}",
        f"18. seed3/4 HGNN-DeepSets util delta={p.get('g_minus_d_util_34')}（old {p.get('g_minus_d_old_34')}）",
        f"19. conflict util DeepSets/HGNN=**{p['conflict_d']:.4f}/{p['conflict_g']:.4f}**",
        f"20. idle util DeepSets/HGNN=**{p['idle_d']:.4f}/{p['idle_g']:.4f}**",
        f"21. SINR utility util DeepSets/HGNN=**{p['sinr_d']:.4f}/{p['sinr_g']:.4f}**",
        f"22. UOBJ Gate=**{p['uobj']['gate']}**",
        f"23. objective_alignment_confirmed=**{p['uobj'].get('objective_alignment_confirmed', False)}**",
        f"24. HGNN-util vs DeepSets-util delta=**{p['graph_delta']:.0f}**",
        f"25. GRAPH-U Gate=**{p['graph']['gate']}**",
        f"26. 稳定 Graph Value=**{p['graph']['gate']=='GRAPH-U-A'}**",
        f"27. R4={p['R4']}",
        "28. 本阶段是否支持进入 PPO：**否。本阶段仍不 PPO。**",
        f"29. 下一阶段：**{p['next_step']}**",
        "",
        f"OBJ (validation)={p['obj']['gate']}。checkpoint=best validation candidate regret，每 {VAL_EVERY} epoch。",
        f"confirmation seeds={p['seeds']}。",
        "",
        "JSON：`src/wt/htw_phase945/outputs/utility_aligned/`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proto = {
        "phase": "9.4.5",
        "ppo": False,
        "modify_encoder": False,
        "modify_decoder": False,
        "pool_size": POOL_SIZE,
        "pool_seed": POOL_SEED,
        "hold": HOLD,
        "utility": "h15_payload_increment",
        "normalization": "median_mad",
        "norm_eps": NORM_EPS,
        "tau": TAU,
        "lambda_rank": LAMBDA_RANK,
        "lambda_ref": LAMBDA_REF,
        "delta_range_frac": DELTA_RANGE_FRAC,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "val_every": VAL_EVERY,
        "checkpoint_rule": "best_validation_mean_candidate_regret",
        "checkpoint_reads_confirmation_payload": False,
        "collect_seeds": list(COLLECT_SEEDS),
        "val_seeds": list(EVAL_SEEDS),
        "blocked": list(BLOCKED_SCENARIO_SEEDS),
        "scan_start": SCAN_START,
        "model_seeds": list(MODEL_SEEDS),
        "decoder_params": decoder_parameter_count(),
        "phase943_confirmation_consumed": True,
        "phase945_model_tuning_on_prior_confirmation": False,
    }
    dump_frozen(OUTPUT / "phase945_protocol.json", proto)
    dump_frozen(OUTPUT / "phase945_loss_spec.json", {"L": "L_utility + 0.25 L_rank + 0.10 L_ref", "L_utility": "soft CE q||p", "tau": TAU, "lambda_rank": LAMBDA_RANK, "lambda_ref": LAMBDA_REF})
    dump_frozen(OUTPUT / "phase945_utility_normalization.json", {"method": "median_mad", "eps": NORM_EPS, "frozen_before_val_outcomes": True})
    dump_frozen(OUTPUT / "phase945_training_seed_manifest.json", {"model_seeds": list(MODEL_SEEDS), "paired": True, "retrain_required": True})
    dump_frozen(OUTPUT / "phase945_candidate_pool_manifest.json", {"pool_size": POOL_SIZE, "pool_seed": POOL_SEED, "model_independent": True, "no_utility_filtering": True})

    if (OUTPUT / "phase945_train.npz").exists() and (OUTPUT / "phase945_val.npz").exists():
        print("reuse frozen hold-aligned utility dataset from disk", flush=True)
        train = load_split("phase945_train")
        val = load_split("phase945_val")
    else:
        print("unit-level structural reuse: collect hold-aligned utility dataset", flush=True)
        seed_everything(0)
        train = collect_utility_split(COLLECT_SEEDS, "train_collect")
        val = collect_utility_split(tuple(EVAL_SEEDS), "val_eval815")
        save_split(train, "phase945_train")
        save_split(val, "phase945_val")
    within5 = []
    for row in train["utility"]:
        best = float(np.max(row))
        within5.append(int(np.sum(np.asarray(row) >= 0.95 * best if best > 0 else np.abs(np.asarray(row) - best) <= 1.0e-9)))
    dump_frozen(
        OUTPUT / "phase945_candidate_utility_dataset_manifest.json",
        {"train_n": int(train["assignment"].shape[0]), "val_n": int(val["assignment"].shape[0]), "pool_size": POOL_SIZE, "mean_within_5pct": float(np.mean(within5)), "primary_utility": "h15_payload_increment", "best_is_global_oracle": False},
    )

    old_csv = OUTPUT / "phase945_validation_metrics_old.csv"
    if old_csv.exists():
        print("reuse frozen 9.4.3 validation metrics from disk", flush=True)
        old_rows = [{**row, "model_seed": int(row["model_seed"])} for row in _csv(old_csv)]
        for row in old_rows:
            for key, value in list(row.items()):
                if key not in {"encoder", "objective"}:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
    else:
        print("validation metrics of frozen 9.4.3 (old objective)", flush=True)
        old_rows = []
        for encoder in ("deepsets", "hgnn"):
            for seed in MODEL_SEEDS:
                model = load_model(encoder, seed, util=False)
                mets = split_metrics(model, val)
                old_rows.append({"encoder": encoder, "model_seed": seed, "objective": "old", **mets})
        write_csv(old_csv, old_rows)

    dump_frozen(
        OUTPUT / "phase945_prefinal_freeze.json",
        {
            "frozen_before_confirmation": True,
            "pool_size": POOL_SIZE,
            "pool_seed": POOL_SEED,
            "tau": TAU,
            "lambda_rank": LAMBDA_RANK,
            "lambda_ref": LAMBDA_REF,
            "normalization": "median_mad",
            "checkpoint_rule": "best_validation_mean_candidate_regret",
            "model_seeds": list(MODEL_SEEDS),
            "hold": HOLD,
            "no_confirmation_payload": True,
        },
    )

    ckpt_rows = []
    models: dict[str, dict[int, OrderFreeAssignmentModel]] = {"deepsets": {}, "hgnn": {}}
    new_rows = []
    for encoder in ("deepsets", "hgnn"):
        for seed in MODEL_SEEDS:
            ckpt = OUTPUT / f"{encoder}_seed{seed}.pt"
            hist_path = OUTPUT / f"phase945_train_history_{encoder}_{seed}.csv"
            if ckpt.exists():
                print(f"skip finished {encoder} seed={seed}", flush=True)
                model = load_model(encoder, seed, util=True)
                info = {"best_epoch": -1, "best_val_regret": float("nan"), "checkpoint": str(ckpt), "history": []}
            else:
                print(f"train {encoder} seed={seed}", flush=True)
                model, info = train_utility_model(OrderFreeAssignmentModel(encoder), train, val, encoder, seed)
                write_csv(hist_path, info["history"])
            models[encoder][seed] = model
            ckpt_rows.append({"encoder": encoder, "model_seed": seed, **{k: info[k] for k in ("best_epoch", "best_val_regret", "checkpoint")}})
            mets = split_metrics(model, val)
            new_rows.append({"encoder": encoder, "model_seed": seed, "objective": "util", **mets})
    write_csv(OUTPUT / "phase945_checkpoint_selection.csv", ckpt_rows)
    write_csv(OUTPUT / "phase945_validation_metrics.csv", old_rows + new_rows)

    def avg(rows, encoder, key):
        return float(np.nanmean([float(r[key]) for r in rows if r["encoder"] == encoder]))

    obj = obj_gate(
        {"spearman": avg(old_rows, "hgnn", "spearman"), "top1_regret": avg(old_rows, "hgnn", "top1_regret"), "expected_utility": avg(old_rows, "hgnn", "expected_utility")},
        {"spearman": avg(new_rows, "hgnn", "spearman"), "top1_regret": avg(new_rows, "hgnn", "top1_regret"), "expected_utility": avg(new_rows, "hgnn", "expected_utility")},
        0.0,
        0.0,
        float(np.std([float(r["top1_regret"]) for r in old_rows if r["encoder"] == "hgnn"])),
        float(np.std([float(r["top1_regret"]) for r in new_rows if r["encoder"] == "hgnn"])),
    )
    dump_frozen(OUTPUT / "phase945_validation_obj_gate.json", obj)

    nom_rows = []
    for encoder in ("deepsets", "hgnn"):
        nom_rows.append({"encoder": encoder, "objective": "old", "nom5": avg(old_rows, encoder, "nom5"), "regret": avg(old_rows, encoder, "top1_regret"), "spearman": avg(old_rows, encoder, "spearman")})
        nom_rows.append({"encoder": encoder, "objective": "util", "nom5": avg(new_rows, encoder, "nom5"), "regret": avg(new_rows, encoder, "top1_regret"), "spearman": avg(new_rows, encoder, "spearman")})
    write_csv(OUTPUT / "phase945_near_optimal_mass.csv", nom_rows)

    print("new confirmation (blocked prior confirmations)", flush=True)
    inter, comp = _load(PHASE942_INTER), _load(PHASE942_COMP)
    selected = select_confirmation(inter, comp)
    dump_frozen(OUTPUT / "phase945_confirmation_manifest.json", {"name": "phase945_confirmation_v1", "seeds": selected["seeds"], "counts": selected["counts"], "blocked": list(BLOCKED_SCENARIO_SEEDS), "registered_before_payloads": True})
    seeds = tuple(selected["seeds"])
    meta = {int(row["scenario"]): row for row in selected["rows"]}
    href = heuristic_reference_rollout(seeds)
    scenario_rows = []
    n_r, n_s = len(MODEL_SEEDS), len(seeds)
    pay = {key: np.zeros((n_r, n_s)) for key in ("d_old", "g_old", "d_util", "g_util")}
    for util, tag, dkey, gkey in ((False, "old", "d_old", "g_old"), (True, "util", "d_util", "g_util")):
        for r_i, init_seed in enumerate(MODEL_SEEDS):
            print(f"eval {tag} seed={init_seed}", flush=True)
            for encoder, key in (("deepsets", dkey), ("hgnn", gkey)):
                model = load_model(encoder, init_seed, util=util)
                out = evaluate_model(model, seeds, href["per_seed_payload"])
                for s_i, row in enumerate(out):
                    pay[key][r_i, s_i] = row["payload"]
                    m = meta[int(row["scenario"])]
                    scenario_rows.append({"tag": tag, "model_seed": init_seed, "encoder": encoder, **row, "regime": m["regime"], "i_label": m["i_label"], "c_label": m["c_label"]})
    write_csv(OUTPUT / "phase945_per_scenario_metrics.csv", scenario_rows)

    d_delta = pay["d_util"] - pay["d_old"]
    g_delta = pay["g_util"] - pay["g_old"]
    gd_util = pay["g_util"] - pay["d_util"]
    gd_old = pay["g_old"] - pay["d_old"]
    d_ci = hierarchical_bootstrap(d_delta, N_BOOT, BOOT_SEED)
    g_ci = hierarchical_bootstrap(g_delta, N_BOOT, BOOT_SEED + 9)
    gd_ci = hierarchical_bootstrap(gd_util, N_BOOT, BOOT_SEED + 11)
    dump_frozen(OUTPUT / "phase945_bootstrap.json", {"deepsets_util_minus_old": d_ci, "hgnn_util_minus_old": g_ci, "hgnn_util_minus_deepsets_util": gd_ci})
    write_csv(OUTPUT / "phase945_hierarchical_bootstrap.csv", [{"kind": "d_util-d_old", **d_ci}, {"kind": "g_util-g_old", **g_ci}, {"kind": "g_util-d_util", **gd_ci}])

    seed_rows = []
    for i, s in enumerate(MODEL_SEEDS):
        seed_rows.append(
            {
                "model_seed": s,
                "d_old": float(np.mean(pay["d_old"][i])),
                "d_util": float(np.mean(pay["d_util"][i])),
                "g_old": float(np.mean(pay["g_old"][i])),
                "g_util": float(np.mean(pay["g_util"][i])),
                "d_delta": float(np.mean(d_delta[i])),
                "g_delta": float(np.mean(g_delta[i])),
                "g_minus_d_util": float(np.mean(gd_util[i])),
                "g_minus_d_old": float(np.mean(gd_old[i])),
            }
        )
    write_csv(OUTPUT / "phase945_per_seed_metrics.csv", seed_rows)
    rpacks = {}
    for name in ("R1", "R2", "R3", "R4"):
        mask = np.asarray([meta[int(s)]["regime"] == name for s in seeds])
        rpacks[name] = regime_pack(gd_util, mask, name)
    write_csv(OUTPUT / "phase945_regime_metrics.csv", [{"regime": k, **{kk: vv for kk, vv in v.items() if kk != "per_seed"}} for k, v in rpacks.items()])

    def mean_scen(tag, encoder, field):
        return float(np.mean([float(r[field]) for r in scenario_rows if r["tag"] == tag and r["encoder"] == encoder]))

    conflict_g = mean_scen("util", "hgnn", "conflict")
    idle_g = mean_scen("util", "hgnn", "idle")
    idle_old_g = mean_scen("old", "hgnn", "idle")
    idle_collapse = idle_g > 0.85 or (idle_g > idle_old_g + 0.35)
    g_pos = int(np.sum(g_delta.mean(axis=1) >= 0))
    d_pos = int(np.sum(d_delta.mean(axis=1) >= 0))
    align_ok = avg(new_rows, "hgnn", "spearman") > avg(old_rows, "hgnn", "spearman") + 0.05 or avg(new_rows, "deepsets", "spearman") > avg(old_rows, "deepsets", "spearman") + 0.05
    payload_up = (float(np.mean(pay["g_util"])) > float(np.mean(pay["g_old"])) and g_ci["ci_lo"] > 0) or (float(np.mean(pay["d_util"])) > float(np.mean(pay["d_old"])) and d_ci["ci_lo"] > 0)
    if payload_up and conflict_g <= 1e-12 and not idle_collapse and align_ok and max(g_pos, d_pos) >= 4:
        uobj = {"gate": "UOBJ-A", "objective_alignment_confirmed": True}
    elif float(np.mean(pay["g_util"])) > float(np.mean(pay["g_old"])) or float(np.mean(pay["d_util"])) > float(np.mean(pay["d_old"])):
        uobj = {"gate": "UOBJ-B", "objective_alignment_confirmed": False}
    else:
        uobj = {"gate": "UOBJ-C", "objective_alignment_confirmed": False}
    dump_frozen(OUTPUT / "phase945_objective_repair_gate.json", uobj)
    dump_frozen(OUTPUT / "phase945_objective_gate.json", uobj)

    graph = {"gate": "n/a", "reason": "UOBJ-A required before GRAPH-U"}
    if uobj["gate"] == "UOBJ-A":
        n_pos = int(np.sum(gd_util.mean(axis=1) >= 0))
        r4_bad = rpacks["R4"]["mean"] < -5.0e5
        if float(np.mean(gd_util)) > 0 and gd_ci["ci_lo"] > 0 and n_pos >= 4 and float(np.median(gd_util)) >= 0 and not r4_bad:
            graph = {"gate": "GRAPH-U-A", "stable_graph_value": True}
        elif float(np.mean(gd_util)) > 0:
            graph = {"gate": "GRAPH-U-B", "stable_graph_value": False}
        else:
            graph = {"gate": "GRAPH-U-C", "stable_graph_value": False}
    dump_frozen(OUTPUT / "phase945_graph_gate.json", graph)

    if uobj["gate"] == "UOBJ-A" and graph["gate"] == "GRAPH-U-A":
        nxt = "supervised objective finalized"
    elif uobj["gate"] == "UOBJ-A":
        nxt = "retain DeepSets" if graph.get("gate") == "GRAPH-U-C" else "supervised objective finalized"
    elif uobj["gate"] == "UOBJ-B":
        nxt = "investigate residual objective mismatch"
    else:
        nxt = "investigate residual objective mismatch"

    payload = {
        "pool_size": POOL_SIZE,
        "pool_seed": POOL_SEED,
        "hold": HOLD,
        "lambda_ref": LAMBDA_REF,
        "multiplicity": float(np.mean(within5)),
        "spearman_ds_old": avg(old_rows, "deepsets", "spearman"),
        "spearman_ds_new": avg(new_rows, "deepsets", "spearman"),
        "spearman_hg_old": avg(old_rows, "hgnn", "spearman"),
        "spearman_hg_new": avg(new_rows, "hgnn", "spearman"),
        "regret_ds_old": avg(old_rows, "deepsets", "top1_regret"),
        "regret_ds_new": avg(new_rows, "deepsets", "top1_regret"),
        "regret_hg_old": avg(old_rows, "hgnn", "top1_regret"),
        "regret_hg_new": avg(new_rows, "hgnn", "top1_regret"),
        "nom_ds_old": avg(old_rows, "deepsets", "nom5"),
        "nom_ds_new": avg(new_rows, "deepsets", "nom5"),
        "nom_hg_old": avg(old_rows, "hgnn", "nom5"),
        "nom_hg_new": avg(new_rows, "hgnn", "nom5"),
        "d_old": float(np.mean(pay["d_old"])),
        "d_util": float(np.mean(pay["d_util"])),
        "g_old": float(np.mean(pay["g_old"])),
        "g_util": float(np.mean(pay["g_util"])),
        "d_delta": float(np.mean(d_delta)),
        "g_delta": float(np.mean(g_delta)),
        "d_ci": d_ci,
        "g_ci": g_ci,
        "d_per_seed": d_delta.mean(1).tolist(),
        "g_per_seed": g_delta.mean(1).tolist(),
        "g_minus_d_util_34": [float(np.mean(gd_util[3])), float(np.mean(gd_util[4]))],
        "g_minus_d_old_34": [float(np.mean(gd_old[3])), float(np.mean(gd_old[4]))],
        "conflict_d": mean_scen("util", "deepsets", "conflict"),
        "conflict_g": conflict_g,
        "idle_d": mean_scen("util", "deepsets", "idle"),
        "idle_g": idle_g,
        "sinr_d": mean_scen("util", "deepsets", "sinr_utility"),
        "sinr_g": mean_scen("util", "hgnn", "sinr_utility"),
        "uobj": uobj,
        "graph_delta": float(np.mean(gd_util)),
        "graph": graph,
        "R4": rpacks["R4"],
        "next_step": nxt,
        "obj": obj,
        "seeds": list(seeds),
        "ppo": False,
    }
    dump_frozen(OUTPUT / "summary.json", _plain(payload))
    write_report(payload)
    print(json.dumps({"obj": obj["gate"], "uobj": uobj["gate"], "graph": graph["gate"], "next": nxt, "ppo": False}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
