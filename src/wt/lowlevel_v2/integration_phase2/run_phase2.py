"""HL-RL-ALIGN runner. Retrain DeepSets on frozen 5-seed RL H=15 mean utility."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase9.features import conflict_rate
from wt.htw_phase941.protocol import sha256_bytes
from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase945.candidate_pool import assignment_hash
from wt.l3a.env import competition_pairs
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.integration_phase1.stats import spearman, summarize
from wt.lowlevel_v2.integration_phase1.worker import CompatWorker
from wt.lowlevel_v2.integration_phase2.protocol import (
    BATCH,
    CHECKPOINT_RULE,
    COLLECT_SEEDS,
    EPOCHS,
    EVAL_SEEDS,
    EXPECTED_HIGH_SHA,
    HOLD,
    INTEG2,
    LAMBDA_RANK,
    LAMBDA_REF,
    LOW_SEEDS,
    LR,
    MODEL_SEEDS,
    NEAR_OPT_FRAC,
    OLD_HL_CKPT,
    OUTPUT,
    PHASE945_OUTPUT,
    POOL_SEED,
    POOL_SIZE,
    ROOT,
    SHARD_DIR,
    TAU,
    UTILITY_EPS,
    VAL_EVERY,
    ckpt_path,
    file_sha,
    high_sha,
    load_compat_gate,
    load_phase3_gate,
    load_phase4a_gate,
    load_phase4b_gate,
    load_pools,
)
from wt.lowlevel_v2.integration_phase2.train_hl import split_metrics
from wt.lowlevel_v2.integration_phase2.utility_gen import generate_split_utilities, reconstruct_official_split, replay_to_hold
from wt.lowlevel_v2.phase1b.controller import make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase4b.snapshot import restore_env


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def load_phase945(name: str) -> dict:
    return reconstruct_official_split(name)


def save_split(data: dict, name: str) -> None:
    np.savez_compressed(
        OUTPUT / f"{name}.npz",
        **{key: data[key] for key in data if key not in {"meta", "hashes", "split", "seeds", "pool_size", "utility_per_seed"}},
    )
    _dump(
        OUTPUT / f"{name}_manifest.json",
        {"split": data["split"], "seeds": data["seeds"], "n": int(data["assignment"].shape[0]), "pool_size": data["pool_size"], "hashes": data["hashes"], "meta": data["meta"]},
    )


def load_high(path: Path) -> OrderFreeAssignmentModel:
    model = OrderFreeAssignmentModel("deepsets")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_stack(high, ll_seed: int) -> CompatWorker:
    actor, critic = make_networks()
    payload = torch.load(ckpt_path(int(ll_seed)), map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return CompatWorker(high, actor, critic, service_lock=True, trust_region=True)


_HELD = {}
_LL_WORKERS = None
_OOD_U = {}


def _ll_workers():
    global _LL_WORKERS
    if _LL_WORKERS is None:
        from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker

        _LL_WORKERS = {int(seed): load_ll_worker(int(seed)) for seed in LOW_SEEDS}
    return _LL_WORKERS


def decoded_policy_utility(data: dict, index: int, assignment: np.ndarray) -> tuple[float, bool]:
    from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15

    u = np.asarray(data["utility"][index], dtype=np.float64)
    hashes = list(data["hashes"][index])
    digest = assignment_hash(assignment)
    if digest in hashes:
        return float(u[hashes.index(digest)]), True
    meta = data["meta"][index]
    cache_key = (int(meta["scenario"]), int(meta["t"]), digest)
    if cache_key not in _OOD_U:
        key = (int(meta["scenario"]), int(meta["t"]))
        if key not in _HELD:
            _HELD[key] = replay_to_hold(int(meta["scenario"]), int(meta["t"]))
        vals = []
        for seed, worker in _ll_workers().items():
            stats = shadow_h15(worker, _HELD[key], assignment, zero_residual=False)
            vals.append(float(stats["payload_increment"]))
        _OOD_U[cache_key] = float(np.mean(vals))
    return _OOD_U[cache_key], False


def policy_and_probs(model, data: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
    from wt.htw_phase943.set_context import make_snap
    from wt.htw_phase943.training import _encode_one
    from wt.htw_phase945.utility_loss import candidate_logprobs

    encoded = _encode_one(model, data, index, "cpu")
    snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
    with torch.no_grad():
        logp = candidate_logprobs(model, encoded, snap, data["candidates"][index], data["buoy_mask"][index])
    lp = logp.detach().cpu().numpy()
    meta = data["meta"][index]
    key = (int(meta["scenario"]), int(meta["t"]))
    if key not in _HELD:
        _HELD[key] = replay_to_hold(int(meta["scenario"]), int(meta["t"]))
    held = _HELD[key]
    env = L3BEnv()
    env.reset(seed=int(meta["scenario"]))
    restore_env(env, held["env"])
    info = env.info()
    batch = _batch_from_env(info, env, held["last_arrival"], held["last_positions"])
    with torch.no_grad():
        assignment = decode_assignment(model, batch, env)
    return np.asarray(assignment, dtype=np.int64), lp


def nr_rows(model, data: dict, name: str, seed) -> list[dict]:
    rows = []
    for index in range(int(data["assignment"].shape[0])):
        u = np.asarray(data["utility"][index], dtype=np.float64)
        hashes = list(data["hashes"][index])
        assignment, lp = policy_and_probs(model, data, index)
        u_pol, in_pool = decoded_policy_utility(data, index, assignment)
        best = float(np.max(u))
        regret = best - u_pol
        nr = regret / (abs(best) + UTILITY_EPS)
        top = int(np.argmax(lp))
        u_inpool = float(u[top])
        nr_inpool = (best - u_inpool) / (abs(best) + UTILITY_EPS)
        p = np.exp(lp - np.max(lp))
        p = p / max(float(p.sum()), 1.0e-12)
        nom = float(p[u >= NEAR_OPT_FRAC * best].sum()) if best > 0 else float(p[np.abs(u - best) <= 1.0e-9].sum())
        rows.append(
            {
                "model": name,
                "high_seed": seed,
                "state_id": index,
                "scenario": data["meta"][index]["scenario"],
                "t": data["meta"][index]["t"],
                "U_policy": u_pol,
                "U_best": best,
                "regret": regret,
                "NR": nr,
                "NR_inpool_top1": nr_inpool,
                "U_inpool_top1": u_inpool,
                "spearman": spearman(u, lp),
                "nom5": nom,
                "idle_fraction": float(np.mean(assignment < 0)),
                "conflict_rate": float(conflict_rate(assignment, 12)),
                "in_pool": in_pool,
            }
        )
    return rows


def run() -> dict:
    compat = load_compat_gate()
    p3 = load_phase3_gate()
    p4a = load_phase4a_gate()
    p4b = load_phase4b_gate()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("OLD-HL SHA mismatch")
    train_raw = load_phase945("phase945_train")
    val_raw = load_phase945("phase945_val")
    if int(train_raw["pool_size"]) != int(POOL_SIZE) or int(val_raw["pool_size"]) != int(POOL_SIZE):
        raise RuntimeError("candidate pool size mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "HL-RL-ALIGN",
        "training_high_level": True,
        "training_low_level": False,
        "architecture_changed": False,
        "hold": HOLD,
        "pool_size": POOL_SIZE,
        "pool_seed": POOL_SEED,
        "tau": TAU,
        "lambda_rank": LAMBDA_RANK,
        "lambda_ref": LAMBDA_REF,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "val_every": VAL_EVERY,
        "checkpoint_rule": CHECKPOINT_RULE,
        "utility": "mean of five frozen Residual MAPPO seeds H15 payload",
        "train_seeds": list(COLLECT_SEEDS),
        "val_seeds": list(EVAL_SEEDS),
        "low_seeds": list(LOW_SEEDS),
        "old_hl": str(OLD_HL_CKPT),
        "old_hl_sha": high_sha(),
        "phase3": p3.get("LLV2_MS2"),
        "COLREL": p4a.get("COLREL"),
        "COMM": p4b.get("COMM"),
        "COMPAT": compat.get("COMPAT"),
        "collision_enhancement": False,
        "interference_enhancement": False,
        "hgnn": False,
    }
    _dump(OUTPUT / "HL_RL_ALIGN_PROTOCOL.json", protocol)

    per_rows = []
    mean_rows = []
    for split, data in (("train", train_raw), ("val", val_raw)):
        generate_split_utilities(split, data, workers=4)
        n = int(data["assignment"].shape[0])
        for index in range(n):
            for slot in range(POOL_SIZE):
                mean_rows.append({"split": split, "index": index, "slot": slot, "hash": data["hashes"][index][slot], "U_new": float(data["utility"][index, slot])})
                for seed in LOW_SEEDS:
                    per_rows.append({"split": split, "index": index, "slot": slot, "model_seed": int(seed), "U": float(data["utility_per_seed"][int(seed)][index, slot])})
        save_split(data, f"hl_rl_align_{split}")
    _csv(OUTPUT / "HL_RL_ALIGN_UTILITY_PER_LOWLEVEL_SEED.csv", per_rows)
    _csv(OUTPUT / "HL_RL_ALIGN_UTILITY_MEAN.csv", mean_rows)

    unique = {h for data in (train_raw, val_raw) for row in data["hashes"] for h in row}
    util_bytes = (OUTPUT / "hl_rl_align_train.npz").read_bytes() + (OUTPUT / "hl_rl_align_val.npz").read_bytes()
    manifest = {
        "train_state_count": int(train_raw["assignment"].shape[0]),
        "validation_state_count": int(val_raw["assignment"].shape[0]),
        "candidate_count_per_state": POOL_SIZE,
        "unique_candidate_count": len(unique),
        "H": HOLD,
        "low_level_seed_count": 5,
        "low_level_checkpoint_sha": {str(seed): file_sha(ckpt_path(int(seed))) for seed in LOW_SEEDS},
        "candidate_generator_sha": file_sha(ROOT / "src" / "wt" / "htw_phase945" / "candidate_pool.py"),
        "environment_sha": file_sha(ROOT / "src" / "wt" / "l3b" / "env.py"),
        "utility_code_sha": file_sha(ROOT / "src" / "wt" / "lowlevel_v2" / "integration_phase1" / "shadow.py"),
        "utility_dataset_sha": sha256_bytes(util_bytes),
        "primary_utility": "mean_five_frozen_rl_h15_payload",
        "from_scratch": True,
        "shared_by_all_high_seeds": True,
    }
    _dump(OUTPUT / "HL_RL_ALIGN_UTILITY_MANIFEST.json", manifest)

    old_model = load_high(OLD_HL_CKPT)
    from concurrent.futures import ProcessPoolExecutor
    from wt.lowlevel_v2.integration_phase2.train_hl import train_seed_process

    missing = [int(seed) for seed in MODEL_SEEDS if not (OUTPUT / f"deepsets_rlalign_seed{int(seed)}.pt").is_file()]
    infos = {}
    if missing:
        print(f"train high-level seeds from scratch in parallel: {missing}", flush=True)
        with ProcessPoolExecutor(max_workers=len(missing)) as pool:
            for seed, info in zip(missing, pool.map(train_seed_process, missing)):
                infos[int(seed)] = info
    ckpt_rows = []
    hist_rows = []
    new_models = {}
    for seed in MODEL_SEEDS:
        ckpt = OUTPUT / f"deepsets_rlalign_seed{int(seed)}.pt"
        new_models[int(seed)] = load_high(ckpt)
        if int(seed) in infos:
            hist_rows.extend(infos[int(seed)]["history"])
            ckpt_rows.append(
                {
                    "high_seed": int(seed),
                    "best_epoch": infos[int(seed)]["best_epoch"],
                    "best_val_regret": infos[int(seed)]["best_val_regret"],
                    "checkpoint": infos[int(seed)]["checkpoint"],
                    "from_old_checkpoint": False,
                }
            )
        else:
            ckpt_rows.append({"high_seed": int(seed), "best_epoch": "reused", "best_val_regret": float("nan"), "checkpoint": str(ckpt), "from_old_checkpoint": False})
    existing_ckpt = OUTPUT / "HL_RL_ALIGN_HIGHLEVEL_CHECKPOINTS.json"
    if not infos and existing_ckpt.is_file():
        ckpt_rows = json.loads(existing_ckpt.read_text(encoding="utf-8")).get("items", ckpt_rows)
    if hist_rows:
        _csv(OUTPUT / "HL_RL_ALIGN_TRAINING_HISTORY.csv", hist_rows)
    _dump(OUTPUT / "HL_RL_ALIGN_HIGHLEVEL_CHECKPOINTS.json", {"rule": CHECKPOINT_RULE, "old_hl": str(OLD_HL_CKPT), "items": ckpt_rows})

    val_metric_rows = []
    old_val = split_metrics(old_model, val_raw)
    val_metric_rows.append({"model": "OLD-HL", "high_seed": "old0", **old_val})
    new_val = {}
    for seed, model in new_models.items():
        metrics = split_metrics(model, val_raw)
        new_val[int(seed)] = metrics
        val_metric_rows.append({"model": "NEW-HL", "high_seed": int(seed), **metrics})
    _csv(OUTPUT / "HL_RL_ALIGN_VALIDATION_METRICS.csv", val_metric_rows)

    regret_rows = nr_rows(old_model, val_raw, "OLD-HL", "old0")
    for seed, model in new_models.items():
        regret_rows.extend(nr_rows(model, val_raw, "NEW-HL", int(seed)))
    _csv(OUTPUT / "HL_RL_ALIGN_POLICY_REGRET.csv", regret_rows)

    def seed_stats(name, seed):
        sub = [r for r in regret_rows if r["model"] == name and str(r["high_seed"]) == str(seed)]
        return summarize([r["NR"] for r in sub]), summarize([r["spearman"] for r in sub]), summarize([r["nom5"] for r in sub])

    old_nr, old_sp, old_nom = seed_stats("OLD-HL", "old0")
    new_nr = {}
    new_sp = {}
    new_nom = {}
    for seed in MODEL_SEEDS:
        new_nr[int(seed)], new_sp[int(seed)], new_nom[int(seed)] = seed_stats("NEW-HL", int(seed))

    align_rows = [{"model": "OLD-HL", "high_seed": "old0", "median_spearman": old_sp["median"], "mean_spearman": old_sp["mean"]}]
    nom_rows = [{"model": "OLD-HL", "high_seed": "old0", "median_nom5": old_nom["median"], "mean_nom5": old_nom["mean"]}]
    for seed in MODEL_SEEDS:
        align_rows.append({"model": "NEW-HL", "high_seed": int(seed), "median_spearman": new_sp[int(seed)]["median"], "mean_spearman": new_sp[int(seed)]["mean"]})
        nom_rows.append({"model": "NEW-HL", "high_seed": int(seed), "median_nom5": new_nom[int(seed)]["median"], "mean_nom5": new_nom[int(seed)]["mean"]})
    _csv(OUTPUT / "HL_RL_ALIGN_UTILITY_PROBABILITY.csv", align_rows)
    _csv(OUTPUT / "HL_RL_ALIGN_NOM5.csv", nom_rows)

    inter_rows = []
    for index in range(int(val_raw["assignment"].shape[0])):
        delta = np.asarray(val_raw["utility"][index], dtype=np.float64) - np.asarray(val_raw["utility_old"][index], dtype=np.float64)
        comps = [float(len(competition_pairs(np.asarray(row, dtype=np.int64), 12))) for row in val_raw["candidates"][index]]
        items = [("old0", old_model)] + [(int(s), new_models[int(s)]) for s in MODEL_SEEDS]
        for seed, model in items:
            _, lp = policy_and_probs(model, val_raw, index)
            p = np.exp(lp - np.max(lp))
            p = p / max(float(p.sum()), 1.0e-12)
            name = "OLD-HL" if seed == "old0" else "NEW-HL"
            inter_rows.append(
                {
                    "model": name,
                    "high_seed": seed,
                    "state_id": index,
                    "spearman_p_delta": spearman(p, delta),
                    "spearman_p_competition": spearman(p, comps),
                    "mean_p_weighted_competition": float(np.dot(p, comps)),
                }
            )
    _csv(OUTPUT / "HL_RL_ALIGN_EXECUTOR_INTERACTION.csv", inter_rows)

    loop_path = OUTPUT / "HL_RL_ALIGN_CLOSED_LOOP_DESIGN.csv"
    if loop_path.is_file():
        print("reuse frozen closed-loop design CSV", flush=True)
        with loop_path.open(encoding="utf-8", newline="") as handle:
            loop_rows = []
            for row in csv.DictReader(handle):
                out = dict(row)
                for key, val in list(out.items()):
                    if key in {"high", "high_seed"}:
                        continue
                    try:
                        out[key] = float(val)
                    except (TypeError, ValueError):
                        pass
                loop_rows.append(out)
    else:
        _, design = load_pools()
        loop_rows = []
        print("closed-loop OLD-HL x 5 LL", flush=True)
        for ll in LOW_SEEDS:
            metrics = evaluate_design(make_stack(old_model, int(ll)), design, zero_residual=False)
            loop_rows.append({"high": "OLD-HL", "high_seed": "old0", "low_seed": int(ll), **{k: v for k, v in metrics.items() if k != "scenarios"}})
        print("closed-loop NEW-HL 5 x 5", flush=True)
        for hs, model in new_models.items():
            for ll in LOW_SEEDS:
                metrics = evaluate_design(make_stack(model, int(ll)), design, zero_residual=False)
                loop_rows.append({"high": "NEW-HL", "high_seed": int(hs), "low_seed": int(ll), **{k: v for k, v in metrics.items() if k != "scenarios"}})
        _csv(loop_path, loop_rows)

    new_pay = np.array(
        [[next(r["payload"] for r in loop_rows if r["high"] == "NEW-HL" and int(r["high_seed"]) == hs and int(r["low_seed"]) == ll) for ll in LOW_SEEDS] for hs in MODEL_SEEDS],
        dtype=np.float64,
    )
    old_pay = np.array([r["payload"] for r in loop_rows if r["high"] == "OLD-HL"], dtype=np.float64)
    _csv(
        OUTPUT / "HL_RL_ALIGN_SEED_VARIANCE.csv",
        [
            {"axis": "high-level", "std": float(np.std(new_pay.mean(axis=1))), "mean": float(np.mean(new_pay))},
            {"axis": "low-level", "std": float(np.std(new_pay.mean(axis=0))), "mean": float(np.mean(new_pay))},
            {"axis": "OLD-HL-low-level", "std": float(np.std(old_pay)), "mean": float(np.mean(old_pay))},
        ],
    )

    old_med_nr = old_nr["median"]
    n_better = sum(1 for seed in MODEL_SEEDS if new_nr[int(seed)]["median"] < old_med_nr - 1.0e-12)
    pooled_new_nr = summarize([r["NR"] for r in regret_rows if r["model"] == "NEW-HL"])
    seed_medians = [new_nr[int(seed)]["median"] for seed in MODEL_SEEDS]
    agg_med = float(np.median(seed_medians))
    new_sp_mean = float(np.mean([new_sp[int(seed)]["median"] for seed in MODEL_SEEDS]))
    new_nom_mean = float(np.mean([new_nom[int(seed)]["mean"] for seed in MODEL_SEEDS]))
    new_conf = float(np.mean([new_val[int(seed)]["conflict_rate"] for seed in MODEL_SEEDS]))
    old_conf = float(old_val["conflict_rate"])
    loop_ok = float(np.mean(new_pay)) >= 0.95 * float(np.mean(old_pay))
    ll_fail = any(float(np.mean(new_pay[:, j])) < 0.90 * float(old_pay[j]) for j in range(5))
    spearman_up = new_sp_mean > old_sp["median"] + 0.05
    nom_up = new_nom_mean > old_nom["mean"]
    legal_ok = new_conf <= old_conf + 1.0e-6
    if n_better >= 4 and agg_med <= 0.10 and spearman_up and nom_up and legal_ok and loop_ok and not ll_fail:
        gate, confirmed = "A", True
        text = "RL-executor utility labels realign DeepSets policy with frozen Residual MAPPO."
    elif n_better == 0 or float(np.mean(new_pay)) < 0.90 * float(np.mean(old_pay)) or agg_med >= old_med_nr:
        gate, confirmed = "C", False
        text = "New utility supervision did not reduce decoded policy regret, or design closed-loop payload dropped systematically."
    else:
        gate, confirmed = "B", False
        text = "Regret improved only partially, or closed-loop/seed stability is insufficient."
    next_step = (
        "Stop method development. Next allowed phase: End-to-End Independent Validation with new scenarios (S0 old FINAL-HIER, S1 old HL+RL LL, S2 new HL+RL LL)."
        if gate == "A"
        else "Stop. Do not retune tau/lambda/network. Diagnose high-level seed instability or utility objective issues."
    )
    gate_payload = {
        "HLALIGN": gate,
        "high_level_rl_alignment_confirmed": confirmed,
        "support_e2e_independent_validation": bool(gate == "A"),
        "training_high_level": True,
        "training_low_level": False,
        "architecture_changed": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "COLREL": "C",
        "COMM": "B",
        "communication_trajectory_headroom": False,
        "n_better_than_old": n_better,
        "OLD_median_NR": old_med_nr,
        "NEW_per_seed_median_NR": {str(s): new_nr[int(s)]["median"] for s in MODEL_SEEDS},
        "NEW_aggregate_median_NR": agg_med,
        "NEW_pooled_median_NR": pooled_new_nr["median"],
        "OLD_spearman": old_sp["median"],
        "NEW_spearman_mean": new_sp_mean,
        "OLD_NOM5": old_nom["mean"],
        "NEW_NOM5": new_nom_mean,
        "OLD_conflict": old_conf,
        "NEW_conflict": new_conf,
        "OLD_closed_loop_payload": float(np.mean(old_pay)),
        "NEW_closed_loop_payload": float(np.mean(new_pay)),
        "OLD_closed_loop_per_ll": old_pay.tolist(),
        "NEW_closed_loop_matrix": new_pay.tolist(),
        "high_level_payload_std": float(np.std(new_pay.mean(axis=1))),
        "low_level_payload_std": float(np.std(new_pay.mean(axis=0))),
        "utility_dataset_sha": manifest["utility_dataset_sha"],
        "checkpoint_rule": CHECKPOINT_RULE,
        "text": text,
        "next_step": next_step,
        "old_val_metrics": old_val,
        "new_val_metrics": {str(s): new_val[int(s)] for s in MODEL_SEEDS},
        "old_nr_summary": old_nr,
        "new_nr_summary": {str(s): new_nr[int(s)] for s in MODEL_SEEDS},
        "new_sp_summary": {str(s): new_sp[int(s)] for s in MODEL_SEEDS},
        "new_nom_summary": {str(s): new_nom[int(s)] for s in MODEL_SEEDS},
        "train_n": manifest["train_state_count"],
        "val_n": manifest["validation_state_count"],
        "best_epochs": ckpt_rows,
    }
    _dump(OUTPUT / "HL_RL_ALIGN_GATE.json", gate_payload)
    plot_figures(regret_rows, align_rows, nom_rows, new_pay, old_pay, OUTPUT / "figures")
    report = write_report(gate_payload, manifest, ckpt_rows, old_pay, new_pay)
    (OUTPUT / "HTW_HighLevel_RLExecutor_Realignment_CN.md").write_text(report, encoding="utf-8")
    publish(report)
    return gate_payload


def plot_figures(regret_rows, align_rows, nom_rows, new_pay, old_pay, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    labels = ["OLD"] + [f"N{s}" for s in MODEL_SEEDS]
    meds = []
    for name, seed in [("OLD-HL", "old0")] + [("NEW-HL", s) for s in MODEL_SEEDS]:
        vals = [r["NR"] for r in regret_rows if r["model"] == name and str(r["high_seed"]) == str(seed) and np.isfinite(r["NR"])]
        meds.append(float(np.median(vals)) if vals else float("nan"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, meds)
    ax.set_ylabel("median NR")
    ax.set_title("old vs new policy regret")
    fig.tight_layout()
    fig.savefig(out / "hl_rl_policy_regret.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([str(r["high_seed"]) for r in align_rows], [r["median_spearman"] for r in align_rows])
    ax.set_title("utility-probability Spearman")
    fig.tight_layout()
    fig.savefig(out / "hl_rl_spearman.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([str(r["high_seed"]) for r in nom_rows], [r["mean_nom5"] for r in nom_rows])
    ax.set_title("NOM5")
    fig.tight_layout()
    fig.savefig(out / "hl_rl_nom5.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([f"N{s}" for s in MODEL_SEEDS], [float(new_pay[i].mean()) / 1e6 for i in range(5)])
    ax.axhline(float(np.mean(old_pay)) / 1e6, color="k", linestyle="--", label="OLD mean")
    ax.set_ylabel("design payload (Mbit)")
    ax.set_title("high-level seed comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hl_rl_high_seed.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(new_pay / 1e6, cmap="viridis")
    ax.set_xticks(range(5), [str(s) for s in LOW_SEEDS])
    ax.set_yticks(range(5), [str(s) for s in MODEL_SEEDS])
    ax.set_xlabel("low-level seed")
    ax.set_ylabel("high-level seed")
    ax.set_title("5x5 closed-loop payload")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "hl_rl_payload_matrix.png", dpi=130)
    plt.close(fig)


def write_report(gate: dict, manifest: dict, ckpts: list, old_pay, new_pay) -> str:
    per = gate["NEW_per_seed_median_NR"]
    per_txt = ", ".join(f"seed{k}={float(v):.4f}" for k, v in per.items())
    sp_txt = ", ".join(f"seed{s}={gate['new_sp_summary'][str(s)]['median']:.4f}" for s in MODEL_SEEDS)
    nom_txt = ", ".join(f"seed{s}={gate['new_nom_summary'][str(s)]['mean']:.4f}" for s in MODEL_SEEDS)
    pay_txt = "; ".join("H" + str(hs) + "=" + "/".join(f"{new_pay[i, j]:.3e}" for j in range(5)) for i, hs in enumerate(MODEL_SEEDS))
    ckpt_txt = "; ".join(f"seed{r['high_seed']} epoch={r['best_epoch']}" for r in ckpts)
    return f"""# HTW Integration Phase-2：RL 执行器对齐的高层重训练

本阶段只改高层 supervision 的 utility label。未改 DeepSets / Order-Free Decoder / Task Option / hold=15 / 低层 RL / SafetyProjector / reward / candidate generator / L=24。未恢复 HGNN。未做 collision / interference 增强。未访问 confirmation / untouched / `106000–106063` / `222000–222063` / `223000–223063`。OLD-HL checkpoint 未覆盖。

恢复：COMPAT-B，`high_level_retraining_needed=true`。LLV2-MS2-A。COLREL-C。COMM-B。五低层 seed 全部 `update100`。U_new = 五冻结 Residual MAPPO 的 H=15 payload 均值。

---

## 24 问正式回答

1. **训练 utility state 数？** {manifest['train_state_count']}（validation {manifest['validation_state_count']}）
2. **每 state candidate 数？** {manifest['candidate_count_per_state']}
3. **是否使用 5 低层 seed mean？** 是。禁止 best-seed。
4. **utility dataset SHA？** `{manifest['utility_dataset_sha']}`
5. **高层架构是否改变？否。**
6. **训练了几个 high-level seed？** 5（全部从头初始化）
7. **checkpoint selection rule？** `{gate['checkpoint_rule']}`（{ckpt_txt}）
8. **OLD-HL median normalized regret？** {gate['OLD_median_NR']:.4f}
9. **NEW-HL 五 seed 各自 median regret？** {per_txt}
10. **NEW 聚合 median regret？** {gate['NEW_aggregate_median_NR']:.4f}
11. **OLD probability-U_new Spearman？** {gate['OLD_spearman']:.4f}
12. **NEW 五 seed Spearman？** {sp_txt}（mean={gate['NEW_spearman_mean']:.4f}）
13. **OLD NOM5？** {gate['OLD_NOM5']:.4f}
14. **NEW NOM5？** {nom_txt}（mean={gate['NEW_NOM5']:.4f}）
15. **legality？** OLD conflict={gate['OLD_conflict']:.4f}；NEW conflict={gate['NEW_conflict']:.4f}
16. **OLD-HL + RL low-level closed-loop payload？** mean={gate['OLD_closed_loop_payload']:.6e}；per-LL={', '.join(f'{x:.3e}' for x in old_pay)}
17. **NEW-HL × RL low-level closed-loop payload？** mean={gate['NEW_closed_loop_payload']:.6e}；{pay_txt}
18. **high-level seed variance？** std={gate['high_level_payload_std']:.6e}
19. **low-level seed variance？** std={gate['low_level_payload_std']:.6e}
20. **HLALIGN Gate？** {gate['HLALIGN']}
21. **high_level_rl_alignment_confirmed？** {str(gate['high_level_rl_alignment_confirmed']).lower()}
22. **是否支持进入新的 End-to-End Independent Validation？** {str(gate['support_e2e_independent_validation']).lower()}
23. **本阶段是否修改低层？否。**
24. **是否重新开启 collision/interference 增强？否。**

{gate['text']}

{gate['next_step']}

本阶段停止。不要自动进入下一阶段。
"""


def publish(report: str) -> None:
    names = [
        "HL_RL_ALIGN_PROTOCOL.json",
        "HL_RL_ALIGN_UTILITY_MANIFEST.json",
        "HL_RL_ALIGN_UTILITY_PER_LOWLEVEL_SEED.csv",
        "HL_RL_ALIGN_UTILITY_MEAN.csv",
        "HL_RL_ALIGN_TRAINING_HISTORY.csv",
        "HL_RL_ALIGN_HIGHLEVEL_CHECKPOINTS.json",
        "HL_RL_ALIGN_VALIDATION_METRICS.csv",
        "HL_RL_ALIGN_POLICY_REGRET.csv",
        "HL_RL_ALIGN_UTILITY_PROBABILITY.csv",
        "HL_RL_ALIGN_NOM5.csv",
        "HL_RL_ALIGN_CLOSED_LOOP_DESIGN.csv",
        "HL_RL_ALIGN_SEED_VARIANCE.csv",
        "HL_RL_ALIGN_GATE.json",
        "HTW_HighLevel_RLExecutor_Realignment_CN.md",
    ]
    for name in names:
        src = OUTPUT / name
        if src.is_file():
            shutil.copy2(src, INTEG2 / name)
            shutil.copy2(src, ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name)
    fig_dst = INTEG2 / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    fig_src = OUTPUT / "figures"
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)
    (ROOT / "docs/archive/phase_reports/HTW_HighLevel_RLExecutor_Realignment_CN.md").write_text(report, encoding="utf-8")
    (INTEG2 / "HTW_HighLevel_RLExecutor_Realignment_CN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    gate = run()
    print(json.dumps({"HLALIGN": gate["HLALIGN"], "confirmed": gate["high_level_rl_alignment_confirmed"]}, indent=2), flush=True)
