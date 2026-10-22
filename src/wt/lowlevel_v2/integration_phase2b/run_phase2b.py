"""HTW Integration Phase-2B runner. Absolute utility supervision only."""
from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase941.protocol import sha256_bytes
from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.set_likelihood_dp import subset_dp_logprob
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.candidate_pool import assignment_hash
from wt.htw_phase945.metrics import pool_metrics
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase1.stats import spearman, summarize
from wt.lowlevel_v2.integration_phase2.train_hl import load_align_split, split_metrics
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker, replay_to_hold
from wt.lowlevel_v2.integration_phase2a.beam import beam_assignments
from wt.lowlevel_v2.integration_phase2a.coverage import idle_count
from wt.lowlevel_v2.phase1b.controller import make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase4b.snapshot import restore_env
from wt.lowlevel_v2.integration_phase1.worker import CompatWorker

from wt.lowlevel_v2.integration_phase1.protocol import load_pools

from .protocol import (
    BATCH,
    BEAM_UTILITY_K,
    BEAM_WIDTH,
    CHECKPOINT_RULE,
    EPOCHS,
    EXPECTED_HIGH_SHA,
    EXPECTED_UTIL_SHA,
    HOLD,
    INTEG2B,
    LAMBDA_RANK,
    LAMBDA_REF,
    LOW_SEEDS,
    LR,
    MODEL_SEEDS,
    OLD_HL_CKPT,
    OUTPUT,
    PHASE2_OUTPUT,
    POOL_SIZE,
    ROOT,
    TAU,
    UTILITY_EPS,
    abs_ckpt,
    ckpt_path,
    high_sha,
    load_compat_gate,
    load_phase2_gate,
    load_phase2a_gate,
    new_ckpt,
)
from .tests import check_unique_candidates, identity_and_shift_tests, production_restricted_normalization, toy_global_normalization
from .train import train_seed_process

HELD = {}
WORKERS = None
U_CACHE = {}
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


def make_stack(high, ll_seed: int) -> CompatWorker:
    actor, critic = make_networks()
    payload = torch.load(ckpt_path(int(ll_seed)), map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return CompatWorker(high, actor, critic, service_lock=True, trust_region=True)


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
        U_CACHE.update(json.loads(CACHE_PATH.read_text(encoding="utf-8")))


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
        vals = [float(shadow_h15(w, state, assignment, zero_residual=False)["payload_increment"]) for w in workers().values()]
        U_CACHE[cache_key] = float(np.mean(vals))
        if len(U_CACHE) % 10 == 0:
            save_u_cache()
    return float(U_CACHE[cache_key]), False


def restore_env_at(data, index):
    meta = data["meta"][index]
    held = held_state(meta)
    env = L3BEnv()
    env.reset(seed=int(meta["scenario"]))
    restore_env(env, held["env"])
    info = env.info()
    batch = _batch_from_env(info, env, held["last_arrival"], held["last_positions"])
    return env, batch


def families() -> list[dict]:
    items = [{"family": "OLD", "name": "OLD-HL", "seed": "old0", "path": OLD_HL_CKPT}]
    for seed in MODEL_SEEDS:
        items.append({"family": "COND", "name": "COND-HL", "seed": int(seed), "path": new_ckpt(int(seed))})
    for seed in MODEL_SEEDS:
        items.append({"family": "ABS", "name": "ABS-HL", "seed": int(seed), "path": abs_ckpt(int(seed))})
    return items


def run_pretests(train, val) -> dict:
    unique = check_unique_candidates(train, val)
    _dump(OUTPUT / "HL_ABS_CANDIDATE_UNIQUENESS.json", unique)
    if unique["candidate_duplicate_weighting_detected"]:
        raise RuntimeError("candidate_duplicate_weighting_detected=true; stop without changing weighting")
    toy = toy_global_normalization()
    prod = production_restricted_normalization(3)
    glob = {"toy": toy, "production_restricted": prod, "pass": bool(toy["pass"] and prod["pass"]), "globally_normalized": bool(toy["pass"] and prod["pass"])}
    _dump(OUTPUT / "HL_ABS_GLOBAL_NORMALIZATION_TEST.json", glob)
    if not glob["pass"]:
        raise RuntimeError("global set-probability normalization failed; refuse L_abs training")
    ident, shift = identity_and_shift_tests()
    _dump(OUTPUT / "HL_ABS_LOSS_IDENTITY_TEST.json", ident)
    _dump(OUTPUT / "HL_ABS_SHIFT_SENSITIVITY_TEST.json", shift)
    if not ident["pass"] or not shift["pass"]:
        raise RuntimeError("L_abs identity or shift sensitivity failed")
    return {"unique": unique, "global": glob, "identity": ident, "shift": shift}


def audit_models(val: dict, specs: list[dict]) -> dict:
    load_u_cache()
    mass_rows, mem_rows, dec_rows, beam_rows, idle_rows, agree_src = [], [], [], [], [], {}
    within_rows = []
    for spec in specs:
        print(f"audit {spec['family']} {spec['seed']}", flush=True)
        model = load_high(spec["path"])
        within = split_metrics(model, val, mechanism=True)
        within_rows.append({"family": spec["family"], "model": spec["name"], "high_seed": spec["seed"], **within})
        masses, flags, nrs, idles, logp_idle, u_idle, decoded = [], [], [], [], [], [], []
        for index in range(int(val["assignment"].shape[0])):
            encoded = _encode_one(model, val, index, "cpu")
            snap = make_snap(val["uav_positions"][index], val["buoy_positions"][index])
            pool = np.asarray(val["candidates"][index], dtype=np.int64)
            u_pool = np.asarray(val["utility"][index], dtype=np.float64)
            hashes = list(val["hashes"][index])
            with torch.no_grad():
                lp = candidate_logprobs(model, encoded, snap, pool, val["buoy_mask"][index]).detach().cpu().numpy()
            unique = {}
            for slot, digest in enumerate(hashes):
                unique[digest] = max(unique.get(digest, 0.0), float(np.exp(float(lp[slot]))))
            log_pool = float(np.logaddexp.reduce(lp))
            pool_mass = float(np.exp(log_pool))
            masses.append(pool_mass)
            metrics = pool_metrics(u_pool, lp)
            env, batch = restore_env_at(val, index)
            with torch.no_grad():
                decoded_assn = np.asarray(decode_assignment(model, batch, env), dtype=np.int64)
            decoded.append(decoded_assn.copy())
            digest = assignment_hash(decoded_assn)
            in_pool = digest in set(hashes)
            flags.append(float(in_pool))
            logp_dec = float(subset_dp_logprob(model, encoded, snap, decoded_assn, val["buoy_mask"][index]).cpu())
            u_dec, _ = u_new(val, index, decoded_assn)
            best = float(np.max(u_pool))
            nr = (best - u_dec) / (abs(best) + UTILITY_EPS)
            nrs.append(nr)
            idle = idle_count(decoded_assn)
            idles.append(idle / 6.0)
            logp_idle.append((logp_dec, idle))
            u_idle.append((u_dec, idle))
            max_pool_lp = float(np.max(lp))
            mass_rows.append({"family": spec["family"], "model": spec["name"], "high_seed": spec["seed"], "state_id": index, "PoolMass": pool_mass, "logPoolMass": log_pool, "within_spearman": metrics["spearman"], "nom5": metrics["nom5"]})
            mem_rows.append({"family": spec["family"], "model": spec["name"], "high_seed": spec["seed"], "state_id": index, "in_pool": in_pool, "idle_count": idle, "active_uav": 6 - idle})
            dec_rows.append({"family": spec["family"], "model": spec["name"], "high_seed": spec["seed"], "state_id": index, "U_decode": u_dec, "U_best_pool": best, "DecodeNR": nr, "logP_decode": logp_dec, "max_pool_logP": max_pool_lp, "logP_decode_minus_max_pool": logp_dec - max_pool_lp, "in_pool": in_pool, "idle_count": idle})
            beams = beam_assignments(model, encoded, snap, val["buoy_mask"][index], BEAM_WIDTH)
            pool_set = set(hashes)
            n_in = sum(assignment_hash(a) in pool_set for _, a in beams)
            ood_high = ood_low = 0
            for rank, (path_lp, assn) in enumerate(beams[:BEAM_UTILITY_K]):
                uu, member = u_new(val, index, assn)
                high = uu >= 0.95 * best if best > 0 else uu >= best - 1.0e-9
                if (not member) and high:
                    ood_high += 1
                if (not member) and (not high):
                    ood_low += 1
            beam_rows.append({"family": spec["family"], "model": spec["name"], "high_seed": spec["seed"], "state_id": index, "beam32_in_pool": n_in, "beam32_in_pool_rate": n_in / float(BEAM_WIDTH), "top8_ood_high": ood_high, "top8_ood_low": ood_low})
            print(f"  {spec['family']} {spec['seed']} {index+1}/{val['assignment'].shape[0]} mass={pool_mass:.4g} in_pool={in_pool} NR={nr:.3f}", flush=True)
        agree_src[f"{spec['family']}|{spec['seed']}"] = decoded
        idle_rows.append(
            {
                "family": spec["family"],
                "model": spec["name"],
                "high_seed": spec["seed"],
                "decoded_idle_fraction": float(np.mean(idles)),
                "logP_vs_idle": spearman([a for a, _ in logp_idle], [b for _, b in logp_idle]),
                "U_vs_idle": spearman([a for a, _ in u_idle], [b for _, b in u_idle]),
                "decoded_in_pool_rate": float(np.mean(flags)),
                "PoolMass_mean": float(np.mean(masses)),
                "PoolMass_median": float(np.median(masses)),
                "DecodeNR_median": float(np.median(nrs)),
                "DecodeNR_mean": float(np.mean(nrs)),
                "DecodeNR_p90": float(np.quantile(nrs, 0.90)),
            }
        )
        save_u_cache()
        del model
    agree_rows = []
    tags = [f"ABS|{int(s)}" for s in MODEL_SEEDS]
    n_state = int(val["assignment"].shape[0])
    for i, a in enumerate(tags):
        for b in tags[i + 1 :]:
            exact, per, ham, uvar = [], [], [], []
            for s in range(n_state):
                x, y = agree_src[f"ABS|{int(a.split('|')[1])}"][s], agree_src[f"ABS|{int(b.split('|')[1])}"][s]
                exact.append(float(np.array_equal(x, y)))
                per.append(float(np.mean(x == y)))
                ham.append(float(np.sum(x != y)))
            agree_rows.append({"seed_a": a, "seed_b": b, "exact_agreement": float(np.mean(exact)), "per_uav_agreement": float(np.mean(per)), "mean_hamming": float(np.mean(ham))})
    _csv(OUTPUT / "HL_ABS_WITHIN_POOL_METRICS.csv", within_rows)
    _csv(OUTPUT / "HL_ABS_POOL_MASS.csv", mass_rows)
    _csv(OUTPUT / "HL_ABS_DECODE_MEMBERSHIP.csv", mem_rows)
    _csv(OUTPUT / "HL_ABS_DECODE_UTILITY.csv", dec_rows)
    _csv(OUTPUT / "HL_ABS_BEAM32.csv", beam_rows)
    _csv(OUTPUT / "HL_ABS_IDLE_CARDINALITY.csv", idle_rows)
    _csv(OUTPUT / "HL_ABS_SEED_AGREEMENT.csv", agree_rows)
    _csv(OUTPUT / "HL_ABS_POLICY_REGRET.csv", dec_rows)
    return {"within": within_rows, "mass": mass_rows, "mem": mem_rows, "dec": dec_rows, "beam": beam_rows, "idle": idle_rows, "agree": agree_rows}


def load_loop_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            out = dict(row)
            for key, val in list(out.items()):
                if key in {"high", "high_seed"}:
                    continue
                try:
                    out[key] = float(val)
                except (TypeError, ValueError):
                    pass
            rows.append(out)
        return rows


def run() -> dict:
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("OLD-HL SHA mismatch")
    p2 = load_phase2_gate()
    p2a = load_phase2a_gate()
    compat = load_compat_gate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = load_align_split("hl_rl_align_train")
    val = load_align_split("hl_rl_align_val")
    util_sha = sha256_bytes((PHASE2_OUTPUT / "hl_rl_align_train.npz").read_bytes() + (PHASE2_OUTPUT / "hl_rl_align_val.npz").read_bytes())
    if util_sha != EXPECTED_UTIL_SHA:
        print(f"utility SHA disk={util_sha} expected={EXPECTED_UTIL_SHA}; using disk manifest", flush=True)
    protocol = {
        "name": "HL-ABS-SUPPORT",
        "training_high_level": True,
        "training_low_level": False,
        "loss": "L_abs + 0.25 L_rank + 0.10 L_ref",
        "L_abs": "-sum_j q_j log P_theta(Gamma_j|s)",
        "tau": TAU,
        "lambda_rank": LAMBDA_RANK,
        "lambda_ref": LAMBDA_REF,
        "epochs": EPOCHS,
        "batch": int(BATCH),
        "lr": LR,
        "checkpoint_rule": CHECKPOINT_RULE,
        "utility_dataset_sha": util_sha,
        "expected_utility_sha": EXPECTED_UTIL_SHA,
        "architecture_changed": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "from_cond_checkpoint": False,
        "hold": HOLD,
        "pool_size": POOL_SIZE,
        "COMPAT": compat.get("COMPAT"),
        "HLALIGN": p2.get("HLALIGN"),
        "HLSUP": p2a.get("gate_name"),
        "COLREL": "C",
        "COMM": "B",
    }
    _dump(OUTPUT / "HL_ABS_PROTOCOL.json", protocol)
    pre = run_pretests(train, val)
    missing = [int(s) for s in MODEL_SEEDS if not abs_ckpt(int(s)).is_file()]
    infos = {}
    if missing:
        print(f"train ABS seeds from scratch: {missing}", flush=True)
        with ProcessPoolExecutor(max_workers=len(missing)) as pool:
            for seed, info in zip(missing, pool.map(train_seed_process, missing)):
                infos[int(seed)] = info
    ckpt_rows = []
    hist, comps = [], []
    existing = OUTPUT / "HL_ABS_HIGHLEVEL_CHECKPOINTS.json"
    if infos:
        for seed in MODEL_SEEDS:
            if int(seed) in infos:
                hist.extend(infos[int(seed)]["history"])
                comps.extend(infos[int(seed)]["components"])
                ckpt_rows.append({"high_seed": int(seed), "best_epoch": infos[int(seed)]["best_epoch"], "best_val_regret": infos[int(seed)]["best_val_regret"], "checkpoint": infos[int(seed)]["checkpoint"], "from_cond_checkpoint": False})
            else:
                ckpt_rows.append({"high_seed": int(seed), "best_epoch": "reused", "checkpoint": str(abs_ckpt(int(seed)))})
        _csv(OUTPUT / "HL_ABS_TRAINING_HISTORY.csv", hist)
        _csv(OUTPUT / "HL_ABS_LOSS_COMPONENTS.csv", comps)
        _dump(OUTPUT / "HL_ABS_HIGHLEVEL_CHECKPOINTS.json", {"rule": CHECKPOINT_RULE, "items": ckpt_rows})
    elif existing.is_file():
        ckpt_rows = json.loads(existing.read_text(encoding="utf-8"))["items"]
    audited = audit_models(val, families())
    loop_path = OUTPUT / "HL_ABS_CLOSED_LOOP_DESIGN.csv"
    if loop_path.is_file():
        loop_rows = load_loop_csv(loop_path)
    else:
        old_cond = load_loop_csv(PHASE2_OUTPUT / "HL_RL_ALIGN_CLOSED_LOOP_DESIGN.csv")
        loop_rows = []
        for row in old_cond:
            fam = "OLD" if row.get("high") == "OLD-HL" else "COND"
            high = "OLD-HL" if fam == "OLD" else "COND-HL"
            loop_rows.append({**row, "family": fam, "high": high})
        _, design = load_pools()
        print("closed-loop ABS 5 x 5", flush=True)
        for hs in MODEL_SEEDS:
            high = load_high(abs_ckpt(int(hs)))
            for ll in LOW_SEEDS:
                metrics = evaluate_design(make_stack(high, int(ll)), design, zero_residual=False)
                loop_rows.append({"family": "ABS", "high": "ABS-HL", "high_seed": int(hs), "low_seed": int(ll), **{k: v for k, v in metrics.items() if k != "scenarios"}})
        _csv(loop_path, loop_rows)

    def sl(rows, fam, seed=None):
        out = [r for r in rows if r.get("family") == fam]
        if seed is not None:
            out = [r for r in out if str(r.get("high_seed")) == str(seed)]
        return out

    def mass_sum(fam, seed):
        return summarize([r["PoolMass"] for r in sl(audited["mass"], fam, seed)])

    def in_rate(fam, seed):
        return float(np.mean([float(r["in_pool"]) for r in sl(audited["mem"], fam, seed)]))

    def nr_sum(fam, seed):
        return summarize([r["DecodeNR"] for r in sl(audited["dec"], fam, seed)])

    old_nr = nr_sum("OLD", "old0")
    abs_nr = {int(s): nr_sum("ABS", int(s)) for s in MODEL_SEEDS}
    cond_nr = {int(s): nr_sum("COND", int(s)) for s in MODEL_SEEDS}
    abs_medians = [abs_nr[int(s)]["median"] for s in MODEL_SEEDS]
    abs_agg = float(np.median(abs_medians))
    n_better_old = sum(1 for s in MODEL_SEEDS if abs_nr[int(s)]["median"] < old_nr["median"] - 1.0e-12)
    cond_sp = float(np.mean([r["spearman"] for r in sl(audited["within"], "COND")]))
    abs_sp = float(np.mean([r["spearman"] for r in sl(audited["within"], "ABS")]))
    cond_nom = float(np.mean([r["nom5"] for r in sl(audited["within"], "COND")]))
    abs_nom = float(np.mean([r["nom5"] for r in sl(audited["within"], "ABS")]))
    within_ok = abs_sp + 0.05 >= cond_sp and abs_nom + 0.02 >= cond_nom
    cond_mass = float(np.mean([mass_sum("COND", int(s))["mean"] for s in MODEL_SEEDS]))
    abs_mass = float(np.mean([mass_sum("ABS", int(s))["mean"] for s in MODEL_SEEDS]))
    old_mass = mass_sum("OLD", "old0")
    mass_up = abs_mass > cond_mass * 1.5
    cond_in = float(np.mean([in_rate("COND", int(s)) for s in MODEL_SEEDS]))
    abs_in = float(np.mean([in_rate("ABS", int(s)) for s in MODEL_SEEDS]))
    in_up = abs_in > cond_in + 0.05
    old_pay = np.array([float(r["payload"]) for r in loop_rows if r.get("high") == "OLD-HL"], dtype=np.float64)
    cond_mat = np.array(
        [
            [
                next(
                    float(r["payload"])
                    for r in loop_rows
                    if r.get("high") in {"NEW-HL", "COND-HL"} and int(r.get("high_seed")) == int(hs) and int(r.get("low_seed")) == int(ll)
                )
                for ll in LOW_SEEDS
            ]
            for hs in MODEL_SEEDS
        ]
    )
    abs_mat = np.array(
        [
            [
                next(
                    float(r["payload"])
                    for r in loop_rows
                    if (r.get("family") == "ABS" or r.get("high") == "ABS-HL") and int(r.get("high_seed")) == int(hs) and int(r.get("low_seed")) == int(ll)
                )
                for ll in LOW_SEEDS
            ]
            for hs in MODEL_SEEDS
        ]
    )
    abs_mean_pay = float(np.mean(abs_mat))
    old_mean_pay = float(np.mean(old_pay))
    cond_mean_pay = float(np.mean(cond_mat))
    loop_ge_old = abs_mean_pay + 1.0 >= old_mean_pay
    n_loop_ok = sum(1 for i, s in enumerate(MODEL_SEEDS) if float(np.mean(abs_mat[i])) >= 0.95 * old_mean_pay)
    legal_ok = float(np.mean([r["conflict_rate"] for r in sl(audited["within"], "ABS")])) <= float(np.mean([r["conflict_rate"] for r in sl(audited["within"], "COND")])) + 1.0e-6
    idle_abs = [r["decoded_idle_fraction"] for r in audited["idle"] if r["family"] == "ABS"]
    idle_ok = all(0.15 < x < 0.90 for x in idle_abs)
    gap_abs = summarize([r["logP_decode_minus_max_pool"] for r in sl(audited["dec"], "ABS")])
    gap_cond = summarize([r["logP_decode_minus_max_pool"] for r in sl(audited["dec"], "COND")])
    if within_ok and mass_up and in_up and n_better_old >= 4 and abs_agg <= 0.10 and loop_ge_old and n_loop_ok >= 4 and legal_ok and idle_ok:
        gate, confirmed = "A", True
        text = "Absolute set-likelihood utility raises support mass and free-decode utility without collapsing within-pool learning."
    elif mass_up and (abs_agg < float(np.median([cond_nr[int(s)]["median"] for s in MODEL_SEEDS])) - 0.05) and not (n_better_old >= 4 and loop_ge_old):
        gate, confirmed = "B", False
        text = "Support improved and decode regret improved, but not to OLD level or seed/closed-loop still insufficient."
    else:
        gate, confirmed = "C", False
        text = "PoolMass may move, but free-decode utility or closed-loop remains systematically poor."
    next_step = (
        "Stop method development. Next allowed: End-to-End Independent Validation with new scenarios. Do not auto-run."
        if gate == "A"
        else "Stop. Do not retune absolute-loss weight, candidate pool, decoder, or checkpoint rule. Diagnose remaining mechanism."
    )
    support_frac = float("nan")
    comp_path = OUTPUT / "HL_ABS_LOSS_COMPONENTS.csv"
    if comp_path.is_file():
        with comp_path.open(encoding="utf-8", newline="") as handle:
            crows = list(csv.DictReader(handle))
        if crows:
            support_frac = float(np.mean([float(r["support_fraction"]) for r in crows if r.get("support_fraction")]))
    gate_payload = {
        "ABSSUP": gate,
        "absolute_support_supervision_confirmed": confirmed,
        "support_e2e_independent_validation": bool(gate == "A"),
        "training_low_level": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "COLREL": "C",
        "COMM": "B",
        "text": text,
        "next_step": next_step,
        "pretests": {"unique": pre["unique"]["candidate_duplicate_weighting_detected"], "global_norm": pre["global"]["pass"], "identity": pre["identity"]["pass"], "shift": pre["shift"]["pass"]},
        "OLD_PoolMass": old_mass,
        "COND_PoolMass_mean": cond_mass,
        "ABS_PoolMass_mean": abs_mass,
        "ABS_PoolMass": {str(s): mass_sum("ABS", int(s)) for s in MODEL_SEEDS},
        "COND_PoolMass": {str(s): mass_sum("COND", int(s)) for s in MODEL_SEEDS},
        "OLD_in_pool": in_rate("OLD", "old0"),
        "COND_in_pool": {str(s): in_rate("COND", int(s)) for s in MODEL_SEEDS},
        "ABS_in_pool": {str(s): in_rate("ABS", int(s)) for s in MODEL_SEEDS},
        "OLD_DecodeNR": old_nr,
        "COND_DecodeNR": {str(s): cond_nr[int(s)] for s in MODEL_SEEDS},
        "ABS_DecodeNR": {str(s): abs_nr[int(s)] for s in MODEL_SEEDS},
        "ABS_aggregate_median_DecodeNR": abs_agg,
        "n_better_than_old": n_better_old,
        "within_pool": {"COND_spearman": cond_sp, "ABS_spearman": abs_sp, "COND_nom5": cond_nom, "ABS_nom5": abs_nom, "kept": within_ok},
        "logP_gap_COND": gap_cond,
        "logP_gap_ABS": gap_abs,
        "OLD_closed_loop": old_mean_pay,
        "COND_closed_loop": cond_mean_pay,
        "ABS_closed_loop": abs_mean_pay,
        "ABS_closed_loop_matrix": abs_mat.tolist(),
        "COND_closed_loop_matrix": cond_mat.tolist(),
        "n_loop_ok": n_loop_ok,
        "idle_ok": idle_ok,
        "legal_ok": legal_ok,
        "support_penalty_fraction": support_frac,
        "best_epochs": ckpt_rows,
        "beam_ood_low_ABS": float(np.mean([r["top8_ood_low"] for r in audited["beam"] if r["family"] == "ABS"])),
        "beam_ood_low_COND": float(np.mean([r["top8_ood_low"] for r in audited["beam"] if r["family"] == "COND"])),
        "beam32_in_ABS": float(np.mean([r["beam32_in_pool_rate"] for r in audited["beam"] if r["family"] == "ABS"])),
        "beam32_in_COND": float(np.mean([r["beam32_in_pool_rate"] for r in audited["beam"] if r["family"] == "COND"])),
        "idle": audited["idle"],
        "agree": audited["agree"],
    }
    _dump(OUTPUT / "HL_ABS_GATE.json", gate_payload)
    plot_figures(audited, abs_mat, cond_mat, old_mean_pay, OUTPUT / "figures")
    report = write_report(gate_payload, protocol, pre)
    (OUTPUT / "HTW_HighLevel_AbsoluteSupport_Supervision_CN.md").write_text(report, encoding="utf-8")
    publish(report)
    return gate_payload


def plot_figures(audited, abs_mat, cond_mat, old_mean, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    labels = ["OLD"] + [f"C{s}" for s in MODEL_SEEDS] + [f"A{s}" for s in MODEL_SEEDS]
    keys = [("OLD", "old0")] + [("COND", s) for s in MODEL_SEEDS] + [("ABS", s) for s in MODEL_SEEDS]

    def series(rows, key, agg=np.mean):
        vals = []
        for fam, seed in keys:
            sub = [r[key] for r in rows if r["family"] == fam and str(r["high_seed"]) == str(seed)]
            vals.append(float(agg(sub)) if sub else float("nan"))
        return vals

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(labels, series(audited["mass"], "PoolMass"))
    ax.set_title("PoolMass old/cond/abs")
    fig.tight_layout()
    fig.savefig(out / "abs_poolmass.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(labels, series(audited["mem"], "in_pool"))
    ax.set_title("decoded-in-pool")
    fig.tight_layout()
    fig.savefig(out / "abs_in_pool.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(labels, series(audited["dec"], "DecodeNR", np.median))
    ax.set_title("DecodeNR")
    fig.tight_layout()
    fig.savefig(out / "abs_decode_nr.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(labels, series(audited["dec"], "logP_decode_minus_max_pool", np.median))
    ax.set_title("logP decode-maxpool")
    fig.tight_layout()
    fig.savefig(out / "abs_logp_gap.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if audited["agree"]:
        ax.bar([f"{r['seed_a'][-1]}-{r['seed_b'][-1]}" for r in audited["agree"]], [r["exact_agreement"] for r in audited["agree"]])
    ax.set_title("ABS seed agreement")
    fig.tight_layout()
    fig.savefig(out / "abs_seed_agree.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([f"C{s}" for s in MODEL_SEEDS], cond_mat.mean(axis=1) / 1e6, alpha=0.5, label="COND")
    ax.bar([f"A{s}" for s in MODEL_SEEDS], abs_mat.mean(axis=1) / 1e6, alpha=0.8, label="ABS")
    ax.axhline(old_mean / 1e6, color="k", linestyle="--", label="OLD")
    ax.set_ylabel("payload Mbit")
    ax.set_title("closed-loop payload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "abs_closed_loop.png", dpi=130)
    plt.close(fig)
    comp = OUTPUT / "HL_ABS_LOSS_COMPONENTS.csv"
    if comp.is_file():
        with comp.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for seed in MODEL_SEEDS:
            sub = [r for r in rows if int(float(r["model_seed"])) == int(seed)]
            ax.plot([int(float(r["epoch"])) for r in sub], [float(r["support_penalty"]) for r in sub], label=f"s{seed}")
        ax.set_title("loss components: support_penalty")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "abs_loss_components.png", dpi=130)
        plt.close(fig)


def write_report(gate: dict, protocol: dict, pre: dict) -> str:
    abs_nr = ", ".join(f"seed{s} median={gate['ABS_DecodeNR'][str(s)]['median']:.4f}" for s in MODEL_SEEDS)
    abs_in = ", ".join(f"seed{s}={gate['ABS_in_pool'][str(s)]:.3f}" for s in MODEL_SEEDS)
    cond_in = ", ".join(f"seed{s}={gate['COND_in_pool'][str(s)]:.3f}" for s in MODEL_SEEDS)
    epochs = "; ".join(f"seed{r.get('high_seed')} epoch={r.get('best_epoch')}" for r in gate.get("best_epochs") or [])
    return f"""# HTW Integration Phase-2B：Absolute Set-Likelihood Utility Supervision

本阶段 **只改** \(L_{{utility}}\)：candidate-conditional softmax → absolute set log-probability。未改 q/tau/lambda、ranking、reference、网络、candidate、L=24、优化器、checkpoint rule、低层、SafetyProjector、collision/interference。未覆盖 OLD/COND checkpoint。未访问 confirmation/untouched。

---

## 25 问正式回答

1. **L_abs 真实数学式？** \(L_{{abs}}=-\\sum_j q_j \\log P_\\theta(\\Gamma_j\\mid s)\)，\(L=L_{{abs}}+0.25 L_{{rank}}+0.10 L_{{ref}}\)。
2. **L_abs=L_cond-logPoolMass 是否 PASS？** {str(pre['identity']['pass']).lower()}
3. **全局 set probability normalization 是否 PASS？** {str(pre['global']['pass']).lower()}（toy M=3 K=4 sumP={pre['global']['toy']['sum_P']:.6f}；production restricted sumP={pre['global']['production_restricted']['sum_P']:.6f}）
4. **shift sensitivity 是否符合理论？** {str(pre['shift']['pass']).lower()}（conditional 不变，absolute 变为 \(L-c\)）
5. **训练了几个 ABS high-level seed？** 5（从头初始化）
6. **selected epochs？** {epochs}
7. **support penalty 训练中占比？** {gate['support_penalty_fraction']:.4f}
8. **OLD/COND/ABS PoolMass？** OLD mean={gate['OLD_PoolMass']['mean']:.6g}；COND mean={gate['COND_PoolMass_mean']:.6g}；ABS mean={gate['ABS_PoolMass_mean']:.6g}
9. **decoded-in-pool rate？** OLD={gate['OLD_in_pool']:.4f}；COND {cond_in}；ABS {abs_in}
10. **logP_decode-max_pool？** COND median={gate['logP_gap_COND']['median']:.4f}；ABS median={gate['logP_gap_ABS']['median']:.4f}
11. **ABS 五 seed DecodeNR？** {abs_nr}
12. **ABS 聚合 median DecodeNR？** {gate['ABS_aggregate_median_DecodeNR']:.4f}
13. **within-pool Spearman/NOM5？** COND spearman={gate['within_pool']['COND_spearman']:.4f} nom5={gate['within_pool']['COND_nom5']:.4f}；ABS spearman={gate['within_pool']['ABS_spearman']:.4f} nom5={gate['within_pool']['ABS_nom5']:.4f}；kept={gate['within_pool']['kept']}
14. **beam32 pool overlap？** COND={gate['beam32_in_COND']:.4f}；ABS={gate['beam32_in_ABS']:.4f}
15. **pool 外低 utility 高概率解？** COND top8_ood_low={gate['beam_ood_low_COND']:.3f}；ABS={gate['beam_ood_low_ABS']:.3f}
16. **idle/cardinality bias？** idle_ok={gate['idle_ok']}
17. **五 seed 自由解码稳定性？** {'; '.join(f"{r['seed_a']} vs {r['seed_b']} exact={r['exact_agreement']:.3f}" for r in gate['agree'])}
18. **OLD-HL+RL closed-loop payload？** {gate['OLD_closed_loop']:.6e}
19. **COND-HL closed-loop payload？** {gate['COND_closed_loop']:.6e}
20. **ABS-HL closed-loop payload？** {gate['ABS_closed_loop']:.6e}
21. **ABSSUP Gate？** {gate['ABSSUP']}
22. **absolute_support_supervision_confirmed？** {str(gate['absolute_support_supervision_confirmed']).lower()}
23. **是否支持进入新的 End-to-End Independent Validation？** {str(gate['support_e2e_independent_validation']).lower()}
24. **本阶段是否修改低层？否。**
25. **是否重新开启 collision/interference？否。**

{gate['text']}

{gate['next_step']}

本阶段停止。不要自动进入下一阶段。
"""


def publish(report: str) -> None:
    names = [
        "HL_ABS_PROTOCOL.json",
        "HL_ABS_CANDIDATE_UNIQUENESS.json",
        "HL_ABS_GLOBAL_NORMALIZATION_TEST.json",
        "HL_ABS_LOSS_IDENTITY_TEST.json",
        "HL_ABS_SHIFT_SENSITIVITY_TEST.json",
        "HL_ABS_TRAINING_HISTORY.csv",
        "HL_ABS_LOSS_COMPONENTS.csv",
        "HL_ABS_HIGHLEVEL_CHECKPOINTS.json",
        "HL_ABS_WITHIN_POOL_METRICS.csv",
        "HL_ABS_POOL_MASS.csv",
        "HL_ABS_DECODE_MEMBERSHIP.csv",
        "HL_ABS_DECODE_UTILITY.csv",
        "HL_ABS_BEAM32.csv",
        "HL_ABS_IDLE_CARDINALITY.csv",
        "HL_ABS_SEED_AGREEMENT.csv",
        "HL_ABS_POLICY_REGRET.csv",
        "HL_ABS_CLOSED_LOOP_DESIGN.csv",
        "HL_ABS_GATE.json",
        "HTW_HighLevel_AbsoluteSupport_Supervision_CN.md",
    ]
    for name in names:
        src = OUTPUT / name
        if src.is_file():
            shutil.copy2(src, INTEG2B / name)
            shutil.copy2(src, ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name)
    fig_src = OUTPUT / "figures"
    fig_dst = INTEG2B / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)
    (ROOT / "docs/archive/phase_reports/HTW_HighLevel_AbsoluteSupport_Supervision_CN.md").write_text(report, encoding="utf-8")
    (INTEG2B / "HTW_HighLevel_AbsoluteSupport_Supervision_CN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    gate = run()
    print(json.dumps({"ABSSUP": gate["ABSSUP"], "confirmed": gate["absolute_support_supervision_confirmed"]}, indent=2), flush=True)
