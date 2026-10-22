"""HL-RL-SUPPORT-AUDIT runner. Frozen checkpoints only. No training."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.set_likelihood_dp import subset_dp_logprob
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.candidate_pool import assignment_hash
from wt.htw_phase945.metrics import pool_metrics
from wt.htw_phase945.utility_loss import candidate_logprobs, utility_losses
from wt.htw_phase945.utility_targets import soft_targets
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase1.stats import spearman, summarize
from wt.lowlevel_v2.integration_phase2.train_hl import load_align_split
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker, replay_to_hold
from wt.lowlevel_v2.phase4b.snapshot import restore_env

from .beam import beam_assignments
from .coverage import coverage_label, idle_count, nearest_pool
from .protocol import (
    BEAM_UTILITY_K,
    BEAM_WIDTH,
    CHECKPOINT_RULE,
    EXPECTED_HIGH_SHA,
    HOLD,
    INTEG2A,
    LAMBDA_RANK,
    LAMBDA_REF,
    LOW_SEEDS,
    MODEL_SEEDS,
    NEAR_OPT_FRAC,
    OLD_HL_CKPT,
    OUTPUT,
    PHASE2_OUTPUT,
    POOL_SIZE,
    ROOT,
    TAU,
    UTILITY_EPS,
    ckpt_path,
    file_sha,
    high_sha,
    load_compat_gate,
    load_phase2_gate,
    new_ckpt,
)

HELD: dict[tuple[int, int], dict] = {}
WORKERS = None
U_CACHE: dict[str, float] = {}
CACHE_PATH = OUTPUT / "utility_cache.json"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


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


def load_high(path: Path) -> OrderFreeAssignmentModel:
    model = OrderFreeAssignmentModel("deepsets")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def held_state(meta: dict) -> dict:
    key = (int(meta["scenario"]), int(meta["t"]))
    if key not in HELD:
        HELD[key] = replay_to_hold(int(meta["scenario"]), int(meta["t"]))
    return HELD[key]


def workers():
    global WORKERS
    if WORKERS is None:
        WORKERS = {int(seed): load_ll_worker(int(seed)) for seed in LOW_SEEDS}
    return WORKERS


def load_u_cache() -> None:
    global U_CACHE
    if CACHE_PATH.is_file():
        U_CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_u_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(U_CACHE) + "\n", encoding="utf-8")


def u_new(data: dict, index: int, assignment: np.ndarray) -> tuple[float, bool]:
    u = np.asarray(data["utility"][index], dtype=np.float64)
    hashes = list(data["hashes"][index])
    digest = assignment_hash(assignment)
    if digest in hashes:
        return float(u[hashes.index(digest)]), True
    meta = data["meta"][index]
    cache_key = f"{int(meta['scenario'])}|{int(meta['t'])}|{digest}"
    if cache_key not in U_CACHE:
        state = held_state(meta)
        vals = [float(shadow_h15(worker, state, assignment, zero_residual=False)["payload_increment"]) for worker in workers().values()]
        U_CACHE[cache_key] = float(np.mean(vals))
        if len(U_CACHE) % 8 == 0:
            save_u_cache()
    return float(U_CACHE[cache_key]), False


def restore_env_at(data: dict, index: int) -> tuple[L3BEnv, dict]:
    meta = data["meta"][index]
    held = held_state(meta)
    env = L3BEnv()
    env.reset(seed=int(meta["scenario"]))
    restore_env(env, held["env"])
    info = env.info()
    batch = _batch_from_env(info, env, held["last_arrival"], held["last_positions"])
    return env, batch


@torch.no_grad()
def absolute_logp(model, data, index, assignment) -> float:
    encoded = _encode_one(model, data, index, "cpu")
    snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
    return float(subset_dp_logprob(model, encoded, snap, np.asarray(assignment, dtype=np.int64), data["buoy_mask"][index]).cpu())


def shift_invariance(model, data) -> dict:
    encoded = _encode_one(model, data, 0, "cpu")
    snap = make_snap(data["uav_positions"][0], data["buoy_positions"][0])
    packed = utility_losses(
        model,
        encoded,
        snap,
        data["candidates"][0],
        data["utility"][0],
        data["assignment"][0],
        data["buoy_mask"][0],
    )
    logp = packed["logp"].detach()
    q = torch.as_tensor(soft_targets(data["utility"][0], TAU), dtype=torch.float32)
    base_util = float((-(q * torch.log_softmax(logp, dim=0)).sum()).cpu())
    rows = []
    ok = True
    for const in (-5.0, 0.0, 5.0):
        shifted = logp + const
        util = float((-(q * torch.log_softmax(shifted, dim=0)).sum()).cpu())
        rank_delta = float(torch.max(torch.abs((logp[0] - logp[1]) - (shifted[0] - shifted[1]))).cpu())
        rows.append({"c": const, "L_utility": util, "delta_vs_0": util - base_util, "rank_logit_delta": rank_delta})
        if const != 0.0 and abs(util - base_util) > 1.0e-5:
            ok = False
    return {"candidate_uniform_logprob_shift_invariant": ok, "base_L_utility": base_util, "rows": rows, "ref_uses_absolute_nll": True}


def model_specs() -> list[tuple[str, object, Path]]:
    items = [("OLD-HL", "old0", OLD_HL_CKPT)]
    for seed in MODEL_SEEDS:
        items.append(("NEW-HL", int(seed), new_ckpt(int(seed))))
    return items


def run() -> dict:
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("OLD-HL SHA mismatch")
    p2 = load_phase2_gate()
    compat = load_compat_gate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    load_u_cache()
    val = load_align_split("hl_rl_align_val")
    if int(val["pool_size"]) != int(POOL_SIZE):
        raise RuntimeError("pool size mismatch")
    protocol = {
        "name": "HL-RL-SUPPORT-AUDIT",
        "training": False,
        "training_low_level": False,
        "architecture_changed": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "hold": HOLD,
        "pool_size": POOL_SIZE,
        "beam_width": BEAM_WIDTH,
        "beam_utility_k": BEAM_UTILITY_K,
        "tau": TAU,
        "lambda_rank": LAMBDA_RANK,
        "lambda_ref": LAMBDA_REF,
        "checkpoint_rule": CHECKPOINT_RULE,
        "old_hl": str(OLD_HL_CKPT),
        "old_hl_sha": high_sha(),
        "new_hl_sha": {str(s): file_sha(new_ckpt(int(s))) for s in MODEL_SEEDS},
        "low_level_sha": {str(s): file_sha(ckpt_path(int(s))) for s in LOW_SEEDS},
        "HLALIGN": p2.get("HLALIGN"),
        "COMPAT": compat.get("COMPAT"),
        "val_n": int(val["assignment"].shape[0]),
        "set_probability_space": "order-marginalized complete legal assignments; PoolMass sums unique candidate P_theta",
    }
    _dump(OUTPUT / "HL_RL_SUPPORT_PROTOCOL.json", protocol)
    shutil.copy2(INTEG2A / "HL_RL_SUPPORT_LOSS_EQUATIONS.md", OUTPUT / "HL_RL_SUPPORT_LOSS_EQUATIONS.md")

    models = []
    for name, seed, path in model_specs():
        print(f"load {name} {seed}", flush=True)
        models.append({"name": name, "seed": seed, "model": load_high(path), "path": str(path)})

    shift = shift_invariance(models[0]["model"], val)
    _dump(OUTPUT / "HL_RL_SUPPORT_SHIFT_INVARIANCE.json", shift)

    mass_rows = []
    mem_rows = []
    dist_rows = []
    dec_u_rows = []
    beam32_rows = []
    beam8_rows = []
    idle_rows = []
    card_rows = []
    cov_rows = []
    pool_beam_rows = []
    corr_src = []
    decoded = {f"{item['name']}|{item['seed']}": [] for item in models}

    for item in models:
        model = item["model"]
        tag = f"{item['name']}|{item['seed']}"
        print(f"audit {tag}", flush=True)
        masses = []
        in_pool_flags = []
        decode_nrs = []
        idle_dec = []
        idle_pool = []
        idle_beam = []
        logp_vs_idle = []
        u_vs_idle = []
        card_logp = {k: [] for k in range(7)}
        card_u = {k: [] for k in range(7)}
        for index in range(int(val["assignment"].shape[0])):
            encoded = _encode_one(model, val, index, "cpu")
            snap = make_snap(val["uav_positions"][index], val["buoy_positions"][index])
            buoy_mask = val["buoy_mask"][index]
            pool = np.asarray(val["candidates"][index], dtype=np.int64)
            u_pool = np.asarray(val["utility"][index], dtype=np.float64)
            hashes = list(val["hashes"][index])
            with torch.no_grad():
                lp_t = candidate_logprobs(model, encoded, snap, pool, buoy_mask)
            lp = lp_t.detach().cpu().numpy()
            unique = {}
            for slot, digest in enumerate(hashes):
                pabs = float(np.exp(float(lp[slot])))
                unique[digest] = max(unique.get(digest, 0.0), pabs)
            pool_mass = float(sum(unique.values()))
            masses.append(pool_mass)
            metrics = pool_metrics(u_pool, lp)
            env, batch = restore_env_at(val, index)
            with torch.no_grad():
                decoded_assn = np.asarray(decode_assignment(model, batch, env), dtype=np.int64)
            decoded[tag].append(decoded_assn.copy())
            digest = assignment_hash(decoded_assn)
            in_pool = digest in set(hashes)
            in_pool_flags.append(float(in_pool))
            logp_dec = absolute_logp(model, val, index, decoded_assn)
            u_dec, _ = u_new(val, index, decoded_assn)
            best_pool = float(np.max(u_pool))
            decode_regret = best_pool - u_dec
            decode_nr = decode_regret / (abs(best_pool) + UTILITY_EPS)
            decode_nrs.append(decode_nr)
            max_pool_lp = float(np.max(lp))
            best_u_lp = float(lp[int(np.argmax(u_pool))])
            nom = float(metrics["nom5"])
            near = u_pool >= NEAR_OPT_FRAC * best_pool if best_pool > 0 else np.abs(u_pool - best_pool) <= 1.0e-9
            near_mass_cond = float(np.exp(lp[near] - np.max(lp)).sum() / max(float(np.exp(lp - np.max(lp)).sum()), 1.0e-12))
            idle_d = idle_count(decoded_assn)
            idle_dec.append(idle_d / 6.0)
            idle_pool.append(float(np.mean([idle_count(row) / 6.0 for row in pool])))
            logp_vs_idle.append((float(logp_dec), idle_d))
            u_vs_idle.append((u_dec, idle_d))
            nearest = nearest_pool(decoded_assn, pool)
            cov = coverage_label(decoded_assn, env) if not in_pool else "in_pool"
            mass_rows.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "state_id": index,
                    "scenario": val["meta"][index]["scenario"],
                    "t": val["meta"][index]["t"],
                    "PoolMass": pool_mass,
                    "OutsideMass": 1.0 - pool_mass,
                    "unique_candidates": len(unique),
                    "within_spearman": metrics["spearman"],
                    "nom5": nom,
                }
            )
            mem_rows.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "state_id": index,
                    "in_pool": in_pool,
                    "decode_hash": digest,
                    "coverage": cov,
                    "idle_count": idle_d,
                    "active_uav": 6 - idle_d,
                }
            )
            dist_rows.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "state_id": index,
                    "in_pool": in_pool,
                    **nearest,
                }
            )
            dec_u_rows.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "state_id": index,
                "U_decode": u_dec,
                "U_best_pool": best_pool,
                "DecodeRegret": decode_regret,
                "DecodeNR": decode_nr,
                "idle_count": idle_d,
                    "logP_decode": logp_dec,
                    "max_pool_logP": max_pool_lp,
                    "best_utility_logP": best_u_lp,
                    "logP_decode_minus_max_pool": logp_dec - max_pool_lp,
                    "near_opt_conditional_mass": near_mass_cond,
                    "in_pool": in_pool,
                    "P_decode": float(np.exp(logp_dec)),
                }
            )
            corr_src.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "state_id": index,
                    "PoolMass": pool_mass,
                    "DecodeNR": decode_nr,
                    "within_spearman": metrics["spearman"],
                    "nom5": nom,
                }
            )
            beams = beam_assignments(model, encoded, snap, buoy_mask, BEAM_WIDTH)
            pool_set = set(hashes)
            beam_in = 0
            beam_idles = []
            for rank, (path_lp, assn) in enumerate(beams):
                bhash = assignment_hash(assn)
                member = bhash in pool_set
                beam_in += int(member)
                bidle = idle_count(assn)
                beam_idles.append(bidle / 6.0)
                beam32_rows.append(
                    {
                        "model": item["name"],
                        "high_seed": item["seed"],
                        "state_id": index,
                        "rank": rank,
                        "path_logP": path_lp,
                        "in_pool": member,
                        "idle_count": bidle,
                        "hash": bhash,
                    }
                )
            idle_beam.append(float(np.mean(beam_idles)) if beam_idles else float("nan"))
            top8 = beams[:BEAM_UTILITY_K]
            u_beam = []
            n_ood_high = n_ood_low = n_in = 0
            for rank, (path_lp, assn) in enumerate(top8):
                uu, member = u_new(val, index, assn)
                u_beam.append(uu)
                high = uu >= 0.95 * best_pool if best_pool > 0 else uu >= best_pool - 1.0e-9
                if member:
                    n_in += 1
                elif high:
                    n_ood_high += 1
                else:
                    n_ood_low += 1
                beam8_rows.append(
                    {
                        "model": item["name"],
                        "high_seed": item["seed"],
                        "state_id": index,
                        "rank": rank,
                        "U_new": uu,
                        "in_pool": member,
                        "high_utility": high,
                        "idle_count": idle_count(assn),
                        "path_logP": path_lp,
                    }
                )
            u_best_beam = float(np.max(u_beam)) if u_beam else float("nan")
            gain = u_best_beam - best_pool
            pool_beam_rows.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "state_id": index,
                    "U_best_pool": best_pool,
                    "U_best_beam8": u_best_beam,
                    "BeamBestGain": gain,
                    "beam32_in_pool": beam_in,
                    "top8_in_pool": n_in,
                    "top8_ood_high": n_ood_high,
                    "top8_ood_low": n_ood_low,
                }
            )
            cov_rows.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "state_id": index,
                    "in_pool": in_pool,
                    "coverage": cov,
                    "DecodeNR": decode_nr,
                    "U_decode": u_dec,
                    "U_best_pool": best_pool,
                }
            )
            for slot, row in enumerate(pool):
                active = 6 - idle_count(row)
                card_logp[int(active)].append(float(lp[slot]))
                card_u[int(active)].append(float(u_pool[slot]))
            print(f"  {tag} state {index + 1}/{val['assignment'].shape[0]} PoolMass={pool_mass:.4g} in_pool={in_pool} NR={decode_nr:.3f}", flush=True)
        idle_lp = spearman([a for a, _ in logp_vs_idle], [b for _, b in logp_vs_idle])
        idle_u = spearman([a for a, _ in u_vs_idle], [b for _, b in u_vs_idle])
        idle_rows.append(
            {
                "model": item["name"],
                "high_seed": item["seed"],
                "decoded_idle_fraction": float(np.mean(idle_dec)),
                "pool_idle_fraction": float(np.mean(idle_pool)),
                "beam32_idle_fraction": float(np.nanmean(idle_beam)),
                "logP_vs_idle_spearman": idle_lp,
                "U_new_vs_idle_spearman": idle_u,
                "decoded_in_pool_rate": float(np.mean(in_pool_flags)),
                "PoolMass_mean": float(np.mean(masses)),
                "PoolMass_median": float(np.median(masses)),
                "DecodeNR_median": float(np.median(decode_nrs)),
            }
        )
        for active, lps in card_logp.items():
            card_rows.append(
                {
                    "model": item["name"],
                    "high_seed": item["seed"],
                    "active_uav": active,
                    "n": len(lps),
                    "mean_logP": float(np.mean(lps)) if lps else float("nan"),
                    "mean_U": float(np.mean(card_u[active])) if card_u[active] else float("nan"),
                }
            )
        save_u_cache()

    agree_rows = []
    new_tags = [f"NEW-HL|{int(s)}" for s in MODEL_SEEDS]
    n_state = int(val["assignment"].shape[0])
    for i, a in enumerate(new_tags):
        for b in new_tags[i + 1 :]:
            exact = []
            per = []
            idle_ag = []
            ham = []
            for s in range(n_state):
                x, y = decoded[a][s], decoded[b][s]
                exact.append(float(np.array_equal(x, y)))
                per.append(float(np.mean(x == y)))
                idle_ag.append(float(idle_count(x) == idle_count(y)))
                ham.append(float(np.sum(x != y)))
            agree_rows.append(
                {
                    "seed_a": a,
                    "seed_b": b,
                    "exact_agreement": float(np.mean(exact)),
                    "per_uav_agreement": float(np.mean(per)),
                    "idle_agreement": float(np.mean(idle_ag)),
                    "mean_hamming": float(np.mean(ham)),
                }
            )

    _csv(OUTPUT / "HL_RL_SUPPORT_POOL_MASS.csv", mass_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_DECODE_MEMBERSHIP.csv", mem_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_DECODE_DISTANCE.csv", dist_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_DECODE_UTILITY.csv", dec_u_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_BEAM32.csv", beam32_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_BEAM8_UTILITY.csv", beam8_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_IDLE_BIAS.csv", idle_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_CARDINALITY_BIAS.csv", card_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_SEED_AGREEMENT.csv", agree_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_CANDIDATE_COVERAGE.csv", cov_rows)
    _csv(OUTPUT / "HL_RL_SUPPORT_POOL_VS_BEAM.csv", pool_beam_rows)

    def model_slice(rows, name, seed):
        return [r for r in rows if r["model"] == name and str(r["high_seed"]) == str(seed)]

    corr_rows = []
    payload_by = {"OLD-HL|old0": float(np.mean(p2["OLD_closed_loop_per_ll"]))}
    for i, seed in enumerate(MODEL_SEEDS):
        payload_by[f"NEW-HL|{int(seed)}"] = float(np.mean(p2["NEW_closed_loop_matrix"][i]))
    for item in models:
        sub = model_slice(corr_src, item["name"], item["seed"])
        pm = [r["PoolMass"] for r in sub]
        nr = [r["DecodeNR"] for r in sub]
        sp = [r["within_spearman"] for r in sub]
        corr_rows.append(
            {
                "model": item["name"],
                "high_seed": item["seed"],
                "PoolMass_vs_DecodeNR": spearman(pm, nr),
                "Spearman_vs_DecodeNR": spearman(sp, nr),
                "closed_loop_payload": payload_by[f"{item['name']}|{item['seed']}"],
                "mean_PoolMass": float(np.mean(pm)),
                "median_DecodeNR": float(np.median(nr)),
            }
        )
    _csv(OUTPUT / "HL_RL_SUPPORT_CORRELATIONS.csv", corr_rows)

    def stats_for(name, seed, key, rows):
        return summarize([r[key] for r in model_slice(rows, name, seed)])

    old_mass = stats_for("OLD-HL", "old0", "PoolMass", mass_rows)
    old_in = float(np.mean([r["in_pool"] for r in model_slice(mem_rows, "OLD-HL", "old0")]))
    old_nr = stats_for("OLD-HL", "old0", "DecodeNR", dec_u_rows)
    new_mass = {int(s): stats_for("NEW-HL", int(s), "PoolMass", mass_rows) for s in MODEL_SEEDS}
    new_in = {int(s): float(np.mean([r["in_pool"] for r in model_slice(mem_rows, "NEW-HL", int(s))])) for s in MODEL_SEEDS}
    new_nr = {int(s): stats_for("NEW-HL", int(s), "DecodeNR", dec_u_rows) for s in MODEL_SEEDS}
    new_mass_mean = float(np.mean([new_mass[int(s)]["mean"] for s in MODEL_SEEDS]))
    new_in_mean = float(np.mean(list(new_in.values())))
    new_nr_med = float(np.median([new_nr[int(s)]["median"] for s in MODEL_SEEDS]))
    gains = [r["BeamBestGain"] for r in pool_beam_rows]
    new_gains = [r["BeamBestGain"] for r in pool_beam_rows if r["model"] == "NEW-HL"]
    frac_pos_gain = float(np.mean([g > 1.0 for g in new_gains])) if new_gains else float("nan")
    ood_new = [r for r in dec_u_rows if r["model"] == "NEW-HL" and not r["in_pool"]]
    in_new = [r for r in dec_u_rows if r["model"] == "NEW-HL" and r["in_pool"]]
    ood_u = summarize([r["U_decode"] for r in ood_new]) if ood_new else summarize([])
    in_u = summarize([r["U_decode"] for r in in_new]) if in_new else summarize([])
    ood_high_frac = float(np.mean([r["top8_ood_high"] for r in pool_beam_rows if r["model"] == "NEW-HL"]))
    ood_low_frac = float(np.mean([r["top8_ood_low"] for r in pool_beam_rows if r["model"] == "NEW-HL"]))
    beam_in_new = float(np.mean([r["beam32_in_pool"] / BEAM_WIDTH for r in pool_beam_rows if r["model"] == "NEW-HL"]))
    beam_in_old = float(np.mean([r["beam32_in_pool"] / BEAM_WIDTH for r in pool_beam_rows if r["model"] == "OLD-HL"]))
    idle_old = next(r for r in idle_rows if r["model"] == "OLD-HL")
    idle_new = [r for r in idle_rows if r["model"] == "NEW-HL"]
    idle_up = float(np.mean([r["decoded_idle_fraction"] for r in idle_new])) > idle_old["decoded_idle_fraction"] + 0.05
    logp_gap_new = summarize([r["logP_decode_minus_max_pool"] for r in dec_u_rows if r["model"] == "NEW-HL"])
    logp_gap_old = summarize([r["logP_decode_minus_max_pool"] for r in dec_u_rows if r["model"] == "OLD-HL"])
    cov_ood = [r["coverage"] for r in cov_rows if r["model"] == "NEW-HL" and not r["in_pool"]]
    uncovered = float(np.mean([c == "uncovered_structure" for c in cov_ood])) if cov_ood else float("nan")
    seed3_vs_04 = [r for r in agree_rows if ("NEW-HL|3" in r["seed_a"] + r["seed_b"]) and (("NEW-HL|0" in r["seed_a"] + r["seed_b"]) or ("NEW-HL|4" in r["seed_a"] + r["seed_b"]))]

    within_better = float(p2["NEW_spearman_mean"]) > float(p2["OLD_spearman"]) + 0.05 and float(p2["NEW_NOM5"]) > float(p2["OLD_NOM5"])
    in_pool_drop = new_in_mean + 0.10 < old_in
    mass_drop = new_mass_mean + 1.0e-3 < old_mass["mean"] or logp_gap_new["median"] > logp_gap_old["median"] + 0.5
    ood_worse = (not ood_new) or (float(ood_u["median"]) + 1.0 < float(in_u["median"]) if in_new else True) or new_nr_med > old_nr["median"] + 0.05
    pool_has_good = frac_pos_gain < 0.25

    labels = []
    if within_better and in_pool_drop and mass_drop and ood_worse and pool_has_good:
        labels.append("SUPPORT")
    if frac_pos_gain >= 0.25:
        labels.append("COVERAGE")
    if idle_up:
        labels.append("IDLE")
    if logp_gap_new["median"] > 0.5 and new_in_mean < 0.5:
        labels.append("DECODER")
    seed_exact = [r["exact_agreement"] for r in agree_rows]
    if seed_exact and float(np.min(seed_exact)) < 0.4:
        labels.append("SEED")
    if shift["candidate_uniform_logprob_shift_invariant"] and within_better and new_nr_med > old_nr["median"]:
        labels.append("OBJECTIVE")
    labels = list(dict.fromkeys(labels)) or ["NONE"]
    if labels == ["SUPPORT"] or (labels[0] == "SUPPORT" and "COVERAGE" not in labels and "IDLE" not in labels[:1]):
        primary = "SUPPORT"
    elif "COVERAGE" in labels and frac_pos_gain >= 0.25 and not pool_has_good:
        primary = "COVERAGE"
    elif "IDLE" in labels and idle_up and not (within_better and in_pool_drop and pool_has_good):
        primary = "IDLE"
    elif len(labels) >= 2:
        primary = labels[0]
    else:
        primary = labels[0]
    secondary = labels[1] if len(labels) > 1 else "NONE"
    if primary == "SUPPORT" and pool_has_good and in_pool_drop and within_better:
        gate = "A"
        gate_name = "HLSUP-A"
        nxt = "Next allowed change only: replace candidate-conditional softmax utility CE with absolute set-likelihood utility supervision L_abs=-sum q_j log P_theta(Gamma_j|s). Do not change ranking/reference yet. Do not train in this phase."
    elif primary == "COVERAGE":
        gate = "B"
        gate_name = "HLCOV-B"
        nxt = "Next: model-independent coverage augmentation. Do not add validation decoder outputs to the train pool."
    elif primary == "IDLE" and "SUPPORT" not in labels:
        gate = "C"
        gate_name = "HLIDLE-C"
        nxt = "Next: idle/cardinality bias in decoder/set likelihood only."
    elif labels == ["NONE"]:
        gate = "E"
        gate_name = "HLNONE-E"
        nxt = "Evidence does not explain HLALIGN-C."
    else:
        gate = "D"
        gate_name = "HLMIX-D"
        nxt = "Multiple mechanisms; ranked primary/secondary. Minimal next change follows primary."
        if primary == "SUPPORT":
            nxt = "HLMIX-D with primary SUPPORT. Next minimal change still absolute set-likelihood utility supervision, then re-check idle/seed."

    gate_payload = {
        "MECHANISM_GATE": gate,
        "gate_name": gate_name,
        "primary_mechanism": primary,
        "secondary_mechanism": secondary,
        "labels": labels,
        "training": False,
        "training_low_level": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "candidate_uniform_logprob_shift_invariant": shift["candidate_uniform_logprob_shift_invariant"],
        "set_probability_globally_normalized": True,
        "OLD_PoolMass": old_mass,
        "NEW_PoolMass": {str(s): new_mass[int(s)] for s in MODEL_SEEDS},
        "OLD_decoded_in_pool": old_in,
        "NEW_decoded_in_pool": {str(s): new_in[int(s)] for s in MODEL_SEEDS},
        "OLD_DecodeNR": old_nr,
        "NEW_DecodeNR": {str(s): new_nr[int(s)] for s in MODEL_SEEDS},
        "NEW_ood_U_decode": ood_u,
        "NEW_inpool_U_decode": in_u,
        "beam32_in_pool_rate_OLD": beam_in_old,
        "beam32_in_pool_rate_NEW": beam_in_new,
        "NEW_frac_positive_BeamBestGain": frac_pos_gain,
        "NEW_mean_top8_ood_high": ood_high_frac,
        "NEW_mean_top8_ood_low": ood_low_frac,
        "uncovered_among_ood": uncovered,
        "logP_decode_minus_max_pool_OLD": logp_gap_old,
        "logP_decode_minus_max_pool_NEW": logp_gap_new,
        "idle": {"OLD": idle_old, "NEW": idle_new},
        "seed3_vs_0_4": seed3_vs_04,
        "within_pool_improved": within_better,
        "pool_still_has_good_solutions": pool_has_good,
        "next_minimal_change": nxt,
        "phase2_closed_loop": {"OLD": p2["OLD_closed_loop_payload"], "NEW": p2["NEW_closed_loop_payload"], "NEW_matrix": p2["NEW_closed_loop_matrix"]},
        "shift": shift,
    }
    _dump(OUTPUT / "HL_RL_SUPPORT_MECHANISM_GATE.json", gate_payload)
    plot_figures(mass_rows, mem_rows, dec_u_rows, idle_rows, pool_beam_rows, corr_src, decoded, OUTPUT / "figures")
    report = write_report(gate_payload, protocol, shift, idle_rows, agree_rows, p2)
    (OUTPUT / "HTW_HighLevel_SupportMismatch_Audit_CN.md").write_text(report, encoding="utf-8")
    publish(report)
    save_u_cache()
    return gate_payload


def plot_figures(mass_rows, mem_rows, dec_u_rows, idle_rows, pool_beam_rows, corr_src, decoded, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    labels = ["OLD"] + [f"N{s}" for s in MODEL_SEEDS]
    keys = [("OLD-HL", "old0")] + [("NEW-HL", s) for s in MODEL_SEEDS]

    def col(rows, key):
        vals = []
        for name, seed in keys:
            sub = [r[key] for r in rows if r["model"] == name and str(r["high_seed"]) == str(seed)]
            vals.append(float(np.mean(sub)) if key != "DecodeNR" else float(np.median(sub)))
        return vals

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, col(mass_rows, "PoolMass"))
    ax.set_title("PoolMass old vs new")
    fig.tight_layout()
    fig.savefig(out / "support_poolmass.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rates = []
    for name, seed in keys:
        sub = [float(r["in_pool"]) for r in mem_rows if r["model"] == name and str(r["high_seed"]) == str(seed)]
        rates.append(float(np.mean(sub)))
    ax.bar(labels, rates)
    ax.set_title("decoded-in-pool rate")
    fig.tight_layout()
    fig.savefig(out / "support_in_pool.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, col(dec_u_rows, "DecodeNR"))
    ax.set_title("decoded regret")
    fig.tight_layout()
    fig.savefig(out / "support_decode_nr.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, [r["decoded_idle_fraction"] for r in idle_rows])
    ax.set_title("idle count / decoded idle fraction")
    fig.tight_layout()
    fig.savefig(out / "support_idle.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    xs = [r["U_best_pool"] / 1e6 for r in pool_beam_rows]
    ys = [r["U_best_beam8"] / 1e6 for r in pool_beam_rows]
    ax.scatter(xs, ys, s=12, alpha=0.6)
    lim = max(max(xs), max(ys), 0.1)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1)
    ax.set_xlabel("pool best (Mbit)")
    ax.set_ylabel("beam8 best (Mbit)")
    ax.set_title("beam utility vs pool utility")
    fig.tight_layout()
    fig.savefig(out / "support_beam_vs_pool.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter([r["PoolMass"] for r in corr_src], [r["DecodeNR"] for r in corr_src], s=12, alpha=0.6)
    ax.set_xlabel("PoolMass")
    ax.set_ylabel("DecodeNR")
    ax.set_title("PoolMass vs DecodeNR")
    fig.tight_layout()
    fig.savefig(out / "support_mass_vs_nr.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    n = len(next(iter(decoded.values())))
    d03, d34 = [], []
    for i in range(n):
        d03.append(float(np.mean(decoded["NEW-HL|0"][i] == decoded["NEW-HL|3"][i])))
        d34.append(float(np.mean(decoded["NEW-HL|4"][i] == decoded["NEW-HL|3"][i])))
    ax.plot(d03, label="seed0 vs seed3 per-UAV")
    ax.plot(d34, label="seed4 vs seed3 per-UAV")
    ax.set_title("NEW seed free-decode disagreement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "support_seed_disagree.png", dpi=130)
    plt.close(fig)


def write_report(gate: dict, protocol: dict, shift: dict, idle_rows: list, agree_rows: list, p2: dict) -> str:
    nm = gate["NEW_PoolMass"]
    ni = gate["NEW_decoded_in_pool"]
    nnr = gate["NEW_DecodeNR"]
    mass_txt = ", ".join(f"seed{s} mean={nm[str(s)]['mean']:.4g} median={nm[str(s)]['median']:.4g}" for s in MODEL_SEEDS)
    in_txt = ", ".join(f"seed{s}={ni[str(s)]:.3f}" for s in MODEL_SEEDS)
    nr_txt = ", ".join(f"seed{s} median={nnr[str(s)]['median']:.4f}" for s in MODEL_SEEDS)
    idle_txt = "; ".join(f"{r['model']} {r['high_seed']} idle={r['decoded_idle_fraction']:.3f}" for r in idle_rows)
    agr = "; ".join(f"{r['seed_a']} vs {r['seed_b']} exact={r['exact_agreement']:.3f} ham={r['mean_hamming']:.2f}" for r in agree_rows)
    return f"""# HTW Integration Phase-2A：Candidate-Support 与自由解码错配审计

本阶段 **training=false**。未改 tau/lambda/utility/DeepSets/decoder/candidate generator/L=24/低层/高层 checkpoint。未开 HGNN。未碰 collision/interference。未访问 confirmation / untouched / 独立验证。OLD-HL 与 NEW-HL seed0–4 只读，checkpoint 为 Phase-2 正式 selection。

Phase-2 冻结：HLALIGN-C。COMPAT-B。

---

## 23 问正式回答

1. **当前 utility loss 真实数学式？** \(L=L_{{utility}}+0.25 L_{{rank}}+0.10 L_{{ref}}\)，其中 \(L_{{utility}}=-\\sum_j q_j \\log p_j\)，\(p=\\mathrm{{softmax}}_{{L=24}}(\\log P_\\theta(\\Gamma_j|s))\)，\(q=\\mathrm{{softmax}}(\\mathrm{{median\\text{{-}}MAD}}(U)/\\tau)\)，\(\\tau=1\)。详见 `HL_RL_SUPPORT_LOSS_EQUATIONS.md`。
2. **是否 candidate-conditional？** 是。\(L_{{utility}}\) 在 L=24 上 softmax。
3. **candidate 统一 logP 平移后 utility loss 是否不变？** {str(shift['candidate_uniform_logprob_shift_invariant']).lower()}（c=-5 与 +5）。`candidate_uniform_logprob_shift_invariant={shift['candidate_uniform_logprob_shift_invariant']}`。
4. **OLD PoolMass？** mean={gate['OLD_PoolMass']['mean']:.6g}，median={gate['OLD_PoolMass']['median']:.6g}，p10={gate['OLD_PoolMass']['p10']:.6g}，p90={gate['OLD_PoolMass']['p90']:.6g}
5. **NEW 五 seed PoolMass？** {mass_txt}
6. **OLD decoded-in-pool rate？** {gate['OLD_decoded_in_pool']:.4f}
7. **NEW 五 seed decoded-in-pool rate？** {in_txt}
8. **NEW pool 外 decoded assignment utility？** median={gate['NEW_ood_U_decode']['median']:.4g}；池内解码 median={gate['NEW_inpool_U_decode']['median']:.4g}
9. **Decode normalized regret？** OLD median={gate['OLD_DecodeNR']['median']:.4f}；NEW {nr_txt}
10. **free decode 相对 pool candidate 的 logP？** OLD logP_decode-max_pool median={gate['logP_decode_minus_max_pool_OLD']['median']:.4f}；NEW median={gate['logP_decode_minus_max_pool_NEW']['median']:.4f}
11. **beam32 有多少 assignment 在 pool 内？** OLD 比例={gate['beam32_in_pool_rate_OLD']:.4f}；NEW={gate['beam32_in_pool_rate_NEW']:.4f}（相对 width=32）
12. **beam top8 中 pool 外高 utility 多不多？** NEW 每 state 均值 top8_ood_high={gate['NEW_mean_top8_ood_high']:.3f}，ood_low={gate['NEW_mean_top8_ood_low']:.3f}
13. **candidate pool 是否缺好解？** BeamBestGain>0 的 NEW 比例={gate['NEW_frac_positive_BeamBestGain']:.3f}；pool_still_has_good_solutions={gate['pool_still_has_good_solutions']}
14. **idle/cardinality bias？** {idle_txt}
15. **为什么 seed3 较好而 seed0/4 崩？** 闭环 payload Phase-2：seed3≈1.02e7，seed0/4≈2e6。自由解码一致：{agr}
16. **OLD-HL 为什么更稳定？** OLD decoded-in-pool={gate['OLD_decoded_in_pool']:.3f}，PoolMass mean={gate['OLD_PoolMass']['mean']:.4g}，DecodeNR median={gate['OLD_DecodeNR']['median']:.3f}。是否因更高绝对 candidate mass：比较 NEW PoolMass 与 in-pool rate。
17. **primary mechanism？** {gate['primary_mechanism']}
18. **secondary mechanism？** {gate['secondary_mechanism']}
19. **Mechanism Gate？** {gate['gate_name']} / {gate['MECHANISM_GATE']}
20. **下一步唯一 minimal change？** {gate['next_minimal_change']}
21. **本阶段是否训练？否。**
22. **是否修改低层？否。**
23. **是否重新碰 collision/interference？否。**

标签：{', '.join(gate['labels'])}。平移不变性说明 utility CE 不约束 pool 绝对质量。本阶段停止。不要自动训练。
"""


def publish(report: str) -> None:
    names = [
        "HL_RL_SUPPORT_PROTOCOL.json",
        "HL_RL_SUPPORT_LOSS_EQUATIONS.md",
        "HL_RL_SUPPORT_POOL_MASS.csv",
        "HL_RL_SUPPORT_DECODE_MEMBERSHIP.csv",
        "HL_RL_SUPPORT_DECODE_DISTANCE.csv",
        "HL_RL_SUPPORT_DECODE_UTILITY.csv",
        "HL_RL_SUPPORT_BEAM32.csv",
        "HL_RL_SUPPORT_BEAM8_UTILITY.csv",
        "HL_RL_SUPPORT_IDLE_BIAS.csv",
        "HL_RL_SUPPORT_CARDINALITY_BIAS.csv",
        "HL_RL_SUPPORT_SEED_AGREEMENT.csv",
        "HL_RL_SUPPORT_CANDIDATE_COVERAGE.csv",
        "HL_RL_SUPPORT_POOL_VS_BEAM.csv",
        "HL_RL_SUPPORT_CORRELATIONS.csv",
        "HL_RL_SUPPORT_MECHANISM_GATE.json",
        "HTW_HighLevel_SupportMismatch_Audit_CN.md",
        "HL_RL_SUPPORT_SHIFT_INVARIANCE.json",
    ]
    for name in names:
        src = OUTPUT / name
        if src.is_file():
            shutil.copy2(src, INTEG2A / name)
            shutil.copy2(src, ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name)
    fig_src = OUTPUT / "figures"
    fig_dst = INTEG2A / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)
    (ROOT / "docs/archive/phase_reports/HTW_HighLevel_SupportMismatch_Audit_CN.md").write_text(report, encoding="utf-8")
    (INTEG2A / "HTW_HighLevel_SupportMismatch_Audit_CN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    gate = run()
    print(json.dumps({"MECHANISM_GATE": gate["MECHANISM_GATE"], "primary": gate["primary_mechanism"]}, indent=2), flush=True)
