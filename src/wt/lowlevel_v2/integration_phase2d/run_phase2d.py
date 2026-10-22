"""HTW Integration Phase-2D. Prefix-consistent utility pilot. Two seeds. No low-level training."""
from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase941.protocol import sha256_bytes
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.candidate_pool import assignment_hash
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.lowlevel_v2.integration_phase1.protocol import load_pools
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase1.stats import summarize
from wt.lowlevel_v2.integration_phase1.worker import CompatWorker
from wt.lowlevel_v2.integration_phase2.train_hl import load_align_split, split_metrics
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker, replay_to_hold
from wt.lowlevel_v2.integration_phase2b.abs_loss import utility_losses_abs
from wt.lowlevel_v2.integration_phase2c.search_core import greedy_trace, prefix_audit, set_path_stats, target_cache
from wt.lowlevel_v2.phase1b.controller import make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design

from wt.round30.ppo import seed_everything

from .prefix_loss import utility_losses_prefix
from .protocol import (
    ABS_AGG_DECODE_NR,
    BATCH,
    CHECKPOINT_RULE,
    EPOCHS,
    EXPECTED_HIGH_SHA,
    EXPECTED_UTIL_SHA,
    INTEG2D,
    LAMBDA_RANK,
    LAMBDA_REF,
    LOW_SEEDS,
    LR,
    OLD_HL_CKPT,
    OUTPUT,
    PHASE2B_OUTPUT,
    PHASE2C_OUTPUT,
    PHASE2_OUTPUT,
    PILOT_SEEDS,
    ROOT,
    SCALE_N,
    UTILITY_EPS,
    abs_ckpt,
    ckpt_path,
    high_sha,
    load_compat_gate,
    load_phase2_gate,
    load_phase2a_gate,
    load_phase2c_gate,
    prefix_ckpt,
)
from .tests import enum_equivalence, gradient_sanity, permutation_tests, target_audit_on_split
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
    for seed in (PHASE2C_OUTPUT / "utility_cache.json", PHASE2B_OUTPUT / "utility_cache.json"):
        if seed.is_file() and not CACHE_PATH.is_file():
            OUTPUT.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
            break
    if CACHE_PATH.is_file():
        U_CACHE.update(json.loads(CACHE_PATH.read_text(encoding="utf-8")))


def save_u_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(U_CACHE) + "\n", encoding="utf-8")


def u_new(data: dict, index: int, assignment: np.ndarray) -> float:
    u = np.asarray(data["utility"][index], dtype=np.float64)
    hashes = list(data["hashes"][index])
    digest = assignment_hash(assignment)
    if digest in hashes:
        return float(u[hashes.index(digest)])
    meta = data["meta"][index]
    cache_key = f"{int(meta['scenario'])}|{int(meta['t'])}|{digest}"
    if cache_key not in U_CACHE:
        state = held_state(meta)
        vals = [float(shadow_h15(w, state, assignment, zero_residual=False)["payload_increment"]) for w in workers().values()]
        U_CACHE[cache_key] = float(np.mean(vals))
        if len(U_CACHE) % 8 == 0:
            save_u_cache()
    return float(U_CACHE[cache_key])


def make_stack(high, ll_seed: int) -> CompatWorker:
    actor, critic = make_networks()
    payload = torch.load(ckpt_path(int(ll_seed)), map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return CompatWorker(high, actor, critic, service_lock=True, trust_region=True)


def decode_nr(best: float, u: float) -> float:
    return (float(best) - float(u)) / (abs(float(best)) + UTILITY_EPS)


def pretests(train, val) -> dict:
    enum = enum_equivalence(100)
    _dump(OUTPUT / "HL_PREFIX_ENUM_EQUIVALENCE.json", enum)
    if not enum["pass"]:
        raise RuntimeError("enum vs subset L_prefix failed")
    perm = permutation_tests(20)
    _dump(OUTPUT / "HL_PREFIX_PERMUTATION_TEST.json", perm)
    if not perm["pass"]:
        raise RuntimeError("permutation tests failed; refuse training")
    audits = []
    illegal = 0
    for index in range(min(16, int(val["assignment"].shape[0]))):
        rec = target_audit_on_split(val["candidates"][index], val["utility"][index], val["buoy_mask"][index])
        audits.append(rec)
        illegal += rec["illegal_target_edges"]
    target = {"pass": illegal == 0, "illegal_total": illegal, "items": audits[:4], "q_sum_ok": all(abs(r["q_sum"] - 1.0) < 1.0e-6 for r in audits)}
    _dump(OUTPUT / "HL_PREFIX_TARGET_AUDIT.json", target)
    if not target["pass"]:
        raise RuntimeError("illegal prefix target mass")
    grad = gradient_sanity()
    if not grad["pass"]:
        raise RuntimeError("prefix gradient does not raise target root edges")
    _dump(OUTPUT / "HL_PREFIX_GRADIENT_SANITY.json", grad)
    return {"enum": enum, "perm": perm, "target": target, "grad": grad}


def loss_scale(train) -> list[dict]:
    seed_everything(0)
    model = OrderFreeAssignmentModel("deepsets")
    rows = []
    for index in range(int(SCALE_N)):
        encoded = _encode_one(model, train, index, "cpu")
        snap = make_snap(train["uav_positions"][index], train["buoy_positions"][index])
        with torch.no_grad():
            pref = utility_losses_prefix(model, encoded, snap, train["candidates"][index], train["utility"][index], train["assignment"][index], train["buoy_mask"][index])
            abso = utility_losses_abs(model, encoded, snap, train["candidates"][index], train["utility"][index], train["assignment"][index], train["buoy_mask"][index])
        rows.append(
            {
                "index": index,
                "L_prefix": float(pref["utility_prefix"].cpu()),
                "L_abs": float(abso["utility_abs"].cpu()),
                "L_rank": float(pref["rank"].cpu()),
                "L_ref": float(pref["ref"].cpu()),
            }
        )
    _csv(OUTPUT / "HL_PREFIX_LOSS_SCALE.csv", rows)
    return rows


def eval_model(spec: dict, val: dict, rows: dict) -> None:
    print(f"eval {spec['family']} {spec['seed']}", flush=True)
    model = load_high(spec["path"])
    within = split_metrics(model, val, mechanism=False)
    rows["within"].append({"family": spec["family"], "high_seed": spec["seed"], **within})
    for index in range(int(val["assignment"].shape[0])):
        encoded = _encode_one(model, val, index, "cpu")
        snap = make_snap(val["uav_positions"][index], val["buoy_positions"][index])
        buoy_mask = val["buoy_mask"][index]
        pool = np.asarray(val["candidates"][index], dtype=np.int64)
        u_pool = np.asarray(val["utility"][index], dtype=np.float64)
        hashes = list(val["hashes"][index])
        best_u = float(np.max(u_pool))
        best_pool = pool[int(np.argmax(u_pool))]
        with torch.no_grad():
            greedy = greedy_trace(model, encoded, snap, buoy_mask)
            lp = candidate_logprobs(model, encoded, snap, pool, buoy_mask).detach().cpu().numpy()
        g_assn = greedy["assignment"]
        u_g = u_new(val, index, g_assn)
        nr = decode_nr(best_u, u_g)
        pref = prefix_audit(model, encoded, snap, buoy_mask, greedy["steps"], g_assn, best_pool)
        g_st = set_path_stats(target_cache(model, encoded, snap, g_assn, buoy_mask), g_assn)
        b_st = set_path_stats(target_cache(model, encoded, snap, best_pool, buoy_mask), best_pool)
        mass_s1 = next((r["compatible_prefix_mass"] for r in pref["teacher_forced_mass"] if r.get("order") == "mean_all_perms" and int(r["step"]) == 1), float("nan"))
        mass_s2 = next((r["compatible_prefix_mass"] for r in pref["teacher_forced_mass"] if r.get("order") == "mean_all_perms" and int(r["step"]) == 2), float("nan"))
        pool_mass = float(np.exp(float(np.logaddexp.reduce(lp))))
        rank = 999
        margin = float("nan")
        if pref["margins"]:
            m0 = pref["margins"][0]
            rank = 1 if m0.get("still_compatible") else 999
            margin = m0.get("prefix_margin")
        rows["prefix"].append(
            {
                "family": spec["family"],
                "high_seed": spec["seed"],
                "state_id": index,
                "step1_mass": mass_s1,
                "step2_mass": mass_s2,
                "first_divergence_step": pref["first_divergence_step"],
                "best_pool_edge_rank": rank,
                "prefix_margin": margin,
            }
        )
        rows["decode"].append(
            {
                "family": spec["family"],
                "high_seed": spec["seed"],
                "state_id": index,
                "U_decode": u_g,
                "U_best_pool": best_u,
                "DecodeNR": nr,
                "in_pool": assignment_hash(g_assn) in set(hashes),
                "PoolMass": pool_mass,
                "idle_fraction": float(np.mean(g_assn[:6] < 0)),
            }
        )
        rows["setpath"].append(
            {
                "family": spec["family"],
                "high_seed": spec["seed"],
                "state_id": index,
                "omb_greedy": g_st["omb"],
                "omb_best_pool": b_st["omb"],
                "n_eff_greedy": g_st["n_eff"],
                "n_eff_best_pool": b_st["n_eff"],
                "logp_set_greedy": g_st["logp_set"],
                "max_path_greedy": g_st["max_order_logp_path"],
            }
        )
        print(f"  {spec['family']} {spec['seed']} {index+1}/{val['assignment'].shape[0]} NR={nr:.3f} s1={mass_s1:.3f}", flush=True)
        save_u_cache()
    del model


def toy_map_payload() -> dict:
    from wt.lowlevel_v2.integration_phase2c.toy_map import toy_map_recovery

    before = toy_map_recovery()
    return {"before_random_logits": before, "note": "architecture-level toy; PREFIX checkpoint is not used to pick this metric"}


def run() -> dict:
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("OLD-HL SHA mismatch")
    p2 = load_phase2_gate()
    p2a = load_phase2a_gate()
    p2c = load_phase2c_gate()
    compat = load_compat_gate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    load_u_cache()
    train = load_align_split("hl_rl_align_train")
    val = load_align_split("hl_rl_align_val")
    util_sha = sha256_bytes((PHASE2_OUTPUT / "hl_rl_align_train.npz").read_bytes() + (PHASE2_OUTPUT / "hl_rl_align_val.npz").read_bytes())
    protocol = {
        "name": "HL-PREFIX-PILOT",
        "training_high_level": True,
        "training_low_level": False,
        "seeds": list(PILOT_SEEDS),
        "loss": "L_prefix + 0.25 L_rank + 0.10 L_ref",
        "replaces": "L_abs",
        "no_L_abs_plus_prefix": True,
        "official_decode": "joint greedy argmax",
        "beam_official": False,
        "pool_map_official": False,
        "tau": 1.0,
        "lambda_rank": LAMBDA_RANK,
        "lambda_ref": LAMBDA_REF,
        "epochs": EPOCHS,
        "batch": int(BATCH),
        "lr": LR,
        "checkpoint_rule": CHECKPOINT_RULE,
        "utility_dataset_sha": util_sha,
        "expected_utility_sha": EXPECTED_UTIL_SHA,
        "HLSETPATH": p2c.get("gate_key"),
        "HLSUP": p2a.get("MECHANISM_GATE"),
        "HLALIGN": p2.get("HLALIGN"),
        "COMPAT": compat.get("COMPAT"),
        "COLREL": "C",
        "COMM": "B",
        "from_abs_checkpoint": False,
    }
    _dump(OUTPUT / "HL_PREFIX_PROTOCOL.json", protocol)
    pre = pretests(train, val)
    scale = loss_scale(train)
    missing = [int(s) for s in PILOT_SEEDS if not prefix_ckpt(int(s)).is_file()]
    infos = {}
    if missing:
        print(f"train PREFIX seeds from scratch: {missing}", flush=True)
        with ProcessPoolExecutor(max_workers=len(missing)) as pool:
            for seed, info in zip(missing, pool.map(train_seed_process, missing)):
                infos[int(seed)] = info
    hist = []
    ckpt_rows = []
    if infos:
        for seed in PILOT_SEEDS:
            hist.extend(infos[int(seed)]["history"])
            ckpt_rows.append({"high_seed": int(seed), "best_epoch": infos[int(seed)]["best_epoch"], "best_val_regret": infos[int(seed)]["best_val_regret"], "checkpoint": infos[int(seed)]["checkpoint"]})
        _csv(OUTPUT / "HL_PREFIX_TRAINING_HISTORY.csv", hist)
        _dump(OUTPUT / "HL_PREFIX_HIGHLEVEL_CHECKPOINTS.json", {"rule": CHECKPOINT_RULE, "items": ckpt_rows})
    elif (OUTPUT / "HL_PREFIX_HIGHLEVEL_CHECKPOINTS.json").is_file():
        ckpt_rows = json.loads((OUTPUT / "HL_PREFIX_HIGHLEVEL_CHECKPOINTS.json").read_text(encoding="utf-8"))["items"]
    specs = [{"family": "OLD", "seed": "old0", "path": OLD_HL_CKPT}, {"family": "ABS", "seed": 3, "path": abs_ckpt(3)}, {"family": "ABS", "seed": 4, "path": abs_ckpt(4)}]
    for seed in PILOT_SEEDS:
        specs.append({"family": "PREFIX", "seed": int(seed), "path": prefix_ckpt(int(seed))})
    # ABS seed3/4 for prefix-metric contrast; PREFIX both seeds; OLD greedy baseline
    rows = {"within": [], "prefix": [], "decode": [], "setpath": []}
    for spec in specs:
        eval_model(spec, val, rows)
    _csv(OUTPUT / "HL_PREFIX_WITHIN_POOL.csv", rows["within"])
    _csv(OUTPUT / "HL_PREFIX_PREFIX_METRICS.csv", rows["prefix"])
    _csv(OUTPUT / "HL_PREFIX_DECODE_UTILITY.csv", rows["decode"])
    _csv(OUTPUT / "HL_PREFIX_SET_PATH_ALIGNMENT.csv", rows["setpath"])
    toy = toy_map_payload()
    _dump(OUTPUT / "HL_PREFIX_TOY_MAP_RECOVERY.json", toy)
    loop_path = OUTPUT / "HL_PREFIX_CLOSED_LOOP_DESIGN.csv"
    if loop_path.is_file():
        loop_rows = list(csv.DictReader(loop_path.open(encoding="utf-8", newline="")))
        for row in loop_rows:
            for k, v in list(row.items()):
                if k in {"high", "high_seed"}:
                    continue
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
    else:
        old_abs = []
        with (PHASE2B_OUTPUT / "HL_ABS_CLOSED_LOOP_DESIGN.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("high") in {"OLD-HL", "ABS-HL"}:
                    rec = dict(row)
                    rec["family"] = "OLD" if rec["high"] == "OLD-HL" else "ABS"
                    for k, v in list(rec.items()):
                        if k in {"high", "high_seed", "family"}:
                            continue
                        try:
                            rec[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                    old_abs.append(rec)
        _, design = load_pools()
        loop_rows = list(old_abs)
        print("closed-loop PREFIX 2 x 5", flush=True)
        for hs in PILOT_SEEDS:
            high = load_high(prefix_ckpt(int(hs)))
            for ll in LOW_SEEDS:
                metrics = evaluate_design(make_stack(high, int(ll)), design, zero_residual=False)
                loop_rows.append({"family": "PREFIX", "high": "PREFIX-HL", "high_seed": int(hs), "low_seed": int(ll), **{k: v for k, v in metrics.items() if k != "scenarios"}})
        _csv(loop_path, loop_rows)
    gate = classify(rows, loop_rows, pre, scale, toy, ckpt_rows)
    _dump(OUTPUT / "HL_PREFIX_GATE.json", gate)
    plot_figures(rows, loop_rows, OUTPUT / "figures")
    report = write_report(gate, protocol, pre)
    (OUTPUT / "HTW_HighLevel_PrefixConsistent_Pilot_CN.md").write_text(report, encoding="utf-8")
    publish(report)
    return gate


def _fam(rows, family, seed=None, key=None):
    out = [r for r in rows if r.get("family") == family]
    if seed is not None:
        out = [r for r in out if str(r.get("high_seed")) == str(seed)]
    if key is None:
        return out
    return [r[key] for r in out]


def classify(rows, loop_rows, pre, scale, toy, ckpt_rows) -> dict:
    abs_s1 = summarize(_fam(rows["prefix"], "ABS", None, "step1_mass"))
    abs_s2 = summarize(_fam(rows["prefix"], "ABS", None, "step2_mass"))
    pref_s1 = {int(s): summarize(_fam(rows["prefix"], "PREFIX", s, "step1_mass")) for s in PILOT_SEEDS}
    pref_s2 = {int(s): summarize(_fam(rows["prefix"], "PREFIX", s, "step2_mass")) for s in PILOT_SEEDS}
    pref_nr = {int(s): summarize(_fam(rows["decode"], "PREFIX", s, "DecodeNR")) for s in PILOT_SEEDS}
    abs_nr = summarize(_fam(rows["decode"], "ABS", None, "DecodeNR"))
    old_nr = summarize(_fam(rows["decode"], "OLD", None, "DecodeNR"))
    abs_div = [int(x) for x in _fam(rows["prefix"], "ABS", None, "first_divergence_step") if x is not None]
    pref_div = [int(x) for x in _fam(rows["prefix"], "PREFIX", None, "first_divergence_step") if x is not None]
    early_abs = float(np.mean([d <= 2 for d in abs_div])) if abs_div else float("nan")
    early_pref = float(np.mean([d <= 2 for d in pref_div])) if pref_div else float("nan")

    def pay(fam):
        sub = [r for r in loop_rows if r.get("family") == fam or r.get("high") == {"OLD": "OLD-HL", "ABS": "ABS-HL", "PREFIX": "PREFIX-HL"}.get(fam)]
        return np.array([float(r["payload"]) for r in sub], dtype=np.float64)

    old_pay = float(np.mean(pay("OLD")))
    abs_pay = float(np.mean(pay("ABS")))
    pref_pay = float(np.mean(pay("PREFIX")))
    pref_pay_seed = {int(s): float(np.mean([float(r["payload"]) for r in loop_rows if (r.get("family") == "PREFIX" or r.get("high") == "PREFIX-HL") and int(r.get("high_seed")) == int(s)])) for s in PILOT_SEEDS}
    s1_up = all(pref_s1[int(s)]["median"] > abs_s1["median"] + 0.03 for s in PILOT_SEEDS)
    s2_up = all(pref_s2[int(s)]["median"] > abs_s2["median"] + 0.02 for s in PILOT_SEEDS)
    nr_down = all(pref_nr[int(s)]["median"] < ABS_AGG_DECODE_NR - 1.0e-6 for s in PILOT_SEEDS)
    nr10 = any(pref_nr[int(s)]["median"] <= 0.10 for s in PILOT_SEEDS)
    loop_ok = all(pref_pay_seed[int(s)] >= 0.95 * old_pay for s in PILOT_SEEDS)
    idle = _fam(rows["decode"], "PREFIX", None, "idle_fraction")
    idle_ok = all(0.15 < float(x) < 0.90 for x in idle)
    legal_ok = all(float(r.get("conflict_rate", 0.0)) <= 1.0e-8 for r in rows["within"] if r["family"] == "PREFIX")
    if s1_up and s2_up and nr_down and nr10 and loop_ok and idle_ok and legal_ok:
        gate, supported = "A", True
        text = "Prefix-consistent utility improves local compatible mass and greedy DecodeNR without collapsing legality."
        nxt = "Stop this phase. Next allowed: 5-seed formal PREFIX training. Do not auto-run."
    elif (s1_up or s2_up) and (nr_down or any(pref_nr[int(s)]["median"] < abs_nr["median"] - 0.02 for s in PILOT_SEEDS)) and not (nr10 and loop_ok):
        gate, supported = "B", False
        text = "Prefix metrics and DecodeNR moved the right way, but closed-loop or seed stability is insufficient."
        nxt = "Stop. Do not add L_abs, retune prefix weight, change pool, decoder, beam, or checkpoint rule."
    else:
        gate, supported = "C", False
        text = "Prefix objective did not improve prefix mass or free-decode utility enough."
        nxt = "Stop. Report only. Do not auto-adjust."
    scale_sum = {
        "L_prefix": summarize([r["L_prefix"] for r in scale]),
        "L_abs": summarize([r["L_abs"] for r in scale]),
        "L_rank": summarize([r["L_rank"] for r in scale]),
        "L_ref": summarize([r["L_ref"] for r in scale]),
    }
    return {
        "PREFIX": gate,
        "prefix_consistent_supervision_supported": supported,
        "support_5seed_formal": bool(gate == "A"),
        "training_low_level": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "COLREL": "C",
        "COMM": "B",
        "text": text,
        "next_step": nxt,
        "pretests": {"enum": pre["enum"]["pass"], "perm": pre["perm"]["pass"], "target": pre["target"]["pass"], "grad": pre["grad"]["pass"]},
        "ABS_step1": abs_s1,
        "ABS_step2": abs_s2,
        "PREFIX_step1": {str(s): pref_s1[int(s)] for s in PILOT_SEEDS},
        "PREFIX_step2": {str(s): pref_s2[int(s)] for s in PILOT_SEEDS},
        "PREFIX_DecodeNR": {str(s): pref_nr[int(s)] for s in PILOT_SEEDS},
        "ABS_DecodeNR_eval": abs_nr,
        "OLD_DecodeNR_eval": old_nr,
        "early_div_ABS": early_abs,
        "early_div_PREFIX": early_pref,
        "OLD_closed_loop": old_pay,
        "ABS_closed_loop": abs_pay,
        "PREFIX_closed_loop": pref_pay,
        "PREFIX_closed_loop_seed": {str(s): pref_pay_seed[int(s)] for s in PILOT_SEEDS},
        "scale": scale_sum,
        "best_epochs": ckpt_rows,
        "idle_ok": idle_ok,
        "legal_ok": legal_ok,
        "in_pool_PREFIX": {str(s): float(np.mean(_fam(rows["decode"], "PREFIX", s, "in_pool"))) for s in PILOT_SEEDS},
        "PoolMass_PREFIX": {str(s): summarize(_fam(rows["decode"], "PREFIX", s, "PoolMass")) for s in PILOT_SEEDS},
        "PoolMass_ABS": summarize(_fam(rows["decode"], "ABS", None, "PoolMass")),
        "within_PREFIX": {str(s): next(r for r in rows["within"] if r["family"] == "PREFIX" and int(r["high_seed"]) == int(s)) for s in PILOT_SEEDS},
        "within_ABS": [r for r in rows["within"] if r["family"] == "ABS"],
        "omb_PREFIX": summarize(_fam(rows["setpath"], "PREFIX", None, "omb_greedy")),
        "omb_ABS": summarize(_fam(rows["setpath"], "ABS", None, "omb_greedy")),
        "n_eff_PREFIX": summarize(_fam(rows["setpath"], "PREFIX", None, "n_eff_greedy")),
        "n_eff_ABS": summarize(_fam(rows["setpath"], "ABS", None, "n_eff_greedy")),
        "toy_before": toy["before_random_logits"]["greedy_map_recovery"],
    }


def plot_figures(rows, loop_rows, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    labels, s1, s2 = [], [], []
    for fam, seed in [("ABS", 3), ("ABS", 4), ("PREFIX", 0), ("PREFIX", 1)]:
        labels.append(f"{fam}{seed}")
        s1.append(float(np.median(_fam(rows["prefix"], fam, seed, "step1_mass"))))
        s2.append(float(np.median(_fam(rows["prefix"], fam, seed, "step2_mass"))))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, s1, 0.4, label="step1")
    ax.bar(x + 0.2, s2, 0.4, label="step2")
    ax.set_xticks(x, labels)
    ax.set_title("compatible prefix mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prefix_mass.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for fam in ("ABS", "PREFIX"):
        counts = [int(np.sum([int(r["first_divergence_step"] or 0) == k for r in rows["prefix"] if r["family"] == fam])) for k in range(1, 7)]
        ax.plot(range(1, 7), counts, label=fam)
    ax.set_title("first divergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prefix_divergence.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labs, nrs = [], []
    for fam, seed in [("OLD", "old0"), ("ABS", 3), ("ABS", 4), ("PREFIX", 0), ("PREFIX", 1)]:
        labs.append(f"{fam}{seed}")
        nrs.append(float(np.median(_fam(rows["decode"], fam, seed, "DecodeNR"))))
    ax.bar(labs, nrs)
    ax.set_title("DecodeNR")
    fig.tight_layout()
    fig.savefig(out / "prefix_decode_nr.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for fam in ("OLD", "ABS", "PREFIX"):
        sub = [float(r["payload"]) for r in loop_rows if r.get("family") == fam or r.get("high") == {"OLD": "OLD-HL", "ABS": "ABS-HL", "PREFIX": "PREFIX-HL"}[fam]]
        if fam == "PREFIX":
            means = [float(np.mean([float(r["payload"]) for r in loop_rows if (r.get("family") == "PREFIX" or r.get("high") == "PREFIX-HL") and int(r.get("high_seed")) == int(s)])) / 1e6 for s in PILOT_SEEDS]
            ax.bar([f"P{s}" for s in PILOT_SEEDS], means, label="PREFIX")
        elif fam == "OLD":
            ax.axhline(float(np.mean(sub)) / 1e6, color="k", linestyle="--", label="OLD")
        else:
            ax.axhline(float(np.mean(sub)) / 1e6, color="C1", linestyle=":", label="ABS")
    ax.set_ylabel("payload Mbit")
    ax.set_title("closed-loop payload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prefix_closed_loop.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(["ABS omb", "PREFIX omb"], [float(np.median(_fam(rows["setpath"], "ABS", None, "omb_greedy"))), float(np.median(_fam(rows["setpath"], "PREFIX", None, "omb_greedy")))])
    ax.set_title("set/path OMB greedy")
    fig.tight_layout()
    fig.savefig(out / "prefix_setpath.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(["random toy greedy MAP"], [0.5])
    ax.set_title("toy MAP recovery (mechanism)")
    fig.tight_layout()
    fig.savefig(out / "prefix_toy_map.png", dpi=130)
    plt.close(fig)


def write_report(gate: dict, protocol: dict, pre: dict) -> str:
    epochs = "; ".join(f"seed{r.get('high_seed')} epoch={r.get('best_epoch')}" for r in gate.get("best_epochs") or [])
    nr = ", ".join(f"seed{s} median={gate['PREFIX_DecodeNR'][str(s)]['median']:.4f}" for s in PILOT_SEEDS)
    s1 = ", ".join(f"seed{s}={gate['PREFIX_step1'][str(s)]['median']:.4f}" for s in PILOT_SEEDS)
    s2 = ", ".join(f"seed{s}={gate['PREFIX_step2'][str(s)]['median']:.4f}" for s in PILOT_SEEDS)
    sp0 = gate["within_PREFIX"]["0"]
    sp1 = gate["within_PREFIX"]["1"]
    return f"""# HTW Integration Phase-2D：Prefix-Consistent Utility Supervision Pilot

本阶段 **只把** \(L_{{\\mathrm{{abs}}}}\) **换成** \(L_{{\\mathrm{{prefix}}}}\)。未改 q/tau/lambda、ranking、reference、网络、L=24、优化器、checkpoint、低层。正式推理仍是 joint greedy。未用 beam / Pool-MAP。训练 PREFIX seed0/1 从头初始化。

---

## 27 问正式回答

1. **L_prefix？** \(L_{{\\mathrm{{prefix}}}}=-\\sum_j q_j \\mathbb{{E}}_{{\\pi\\sim\\mathrm{{Unif}}(S_M)}}[\\log P_\\theta(\\pi(\\Gamma_j)\\mid s)]\)，等价 subset 公式 \(\\sum_{{k,S,e}} q_j\\frac{{1}}{{C(M,k)}}\\frac{{1}}{{M-k}}\\log\\pi_\\theta(e\\mid S,s)\) 的负值。\(L=L_{{\\mathrm{{prefix}}}}+0.25 L_{{\\mathrm{{rank}}}}+0.10 L_{{\\mathrm{{ref}}}}\)。
2. **enum vs subset？** {str(pre['enum']['pass']).lower()}（worst={pre['enum']['worst_delta']:.3e}）
3. **permutation tests？** {str(pre['perm']['pass']).lower()}
4. **target 合法？** {str(pre['target']['pass']).lower()}
5. **PREFIX seed 数？** 2（seed0, seed1）
6. **selected epochs？** {epochs}
7. **L_prefix vs L_abs 尺度？** prefix mean={gate['scale']['L_prefix']['mean']:.3f}；abs mean={gate['scale']['L_abs']['mean']:.3f}；未手工缩放
8. **ABS step1 mass？** median={gate['ABS_step1']['median']:.4f}
9. **PREFIX step1？** {s1}
10. **ABS step2？** median={gate['ABS_step2']['median']:.4f}
11. **PREFIX step2？** {s2}
12. **first divergence 是否后移？** ABS early(<=2)={gate['early_div_ABS']:.3f}；PREFIX={gate['early_div_PREFIX']:.3f}
13. **PREFIX DecodeNR？** {nr}
14. **是否有 seed <=0.10？** {str(any(gate['PREFIX_DecodeNR'][str(s)]['median']<=0.10 for s in PILOT_SEEDS)).lower()}
15. **decoded-in-pool？** seed0={gate['in_pool_PREFIX']['0']:.3f} seed1={gate['in_pool_PREFIX']['1']:.3f}
16. **PoolMass？** PREFIX seed0 mean={gate['PoolMass_PREFIX']['0']['mean']:.6g} seed1={gate['PoolMass_PREFIX']['1']['mean']:.6g}；ABS mean={gate['PoolMass_ABS']['mean']:.6g}
17. **within-pool？** seed0 spearman={sp0['spearman']:.4f} nom5={sp0['nom5']:.4f}；seed1 spearman={sp1['spearman']:.4f} nom5={sp1['nom5']:.4f}
18. **toy greedy MAP？** before random logits={gate['toy_before']:.3f}（机制指标，不选 checkpoint）
19. **set/path gap？** ABS OMB greedy median={gate['omb_ABS']['median']:.4f}；PREFIX={gate['omb_PREFIX']['median']:.4f}
20. **OLD-HL+RL closed-loop？** {gate['OLD_closed_loop']:.6e}
21. **ABS-HL+RL closed-loop？** {gate['ABS_closed_loop']:.6e}
22. **PREFIX-HL+RL closed-loop？** {gate['PREFIX_closed_loop']:.6e}（seed0={gate['PREFIX_closed_loop_seed']['0']:.6e} seed1={gate['PREFIX_closed_loop_seed']['1']:.6e}）
23. **PREFIX Gate？** {gate['PREFIX']}
24. **prefix_consistent_supervision_supported？** {str(gate['prefix_consistent_supervision_supported']).lower()}
25. **是否支持 5-seed 正式训练？** {str(gate['support_5seed_formal']).lower()}
26. **本阶段是否修改低层？否。**
27. **是否重新碰 collision/interference？否。**

{gate['text']}

{gate['next_step']}

本阶段停止。不要自动进入下一阶段。
"""


def publish(report: str) -> None:
    names = [
        "HL_PREFIX_PROTOCOL.json",
        "HL_PREFIX_ENUM_EQUIVALENCE.json",
        "HL_PREFIX_PERMUTATION_TEST.json",
        "HL_PREFIX_TARGET_AUDIT.json",
        "HL_PREFIX_LOSS_SCALE.csv",
        "HL_PREFIX_TRAINING_HISTORY.csv",
        "HL_PREFIX_HIGHLEVEL_CHECKPOINTS.json",
        "HL_PREFIX_WITHIN_POOL.csv",
        "HL_PREFIX_PREFIX_METRICS.csv",
        "HL_PREFIX_DECODE_UTILITY.csv",
        "HL_PREFIX_SET_PATH_ALIGNMENT.csv",
        "HL_PREFIX_TOY_MAP_RECOVERY.json",
        "HL_PREFIX_CLOSED_LOOP_DESIGN.csv",
        "HL_PREFIX_GATE.json",
        "HTW_HighLevel_PrefixConsistent_Pilot_CN.md",
    ]
    for name in names:
        src = OUTPUT / name
        if src.is_file():
            shutil.copy2(src, INTEG2D / name)
            shutil.copy2(src, ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name)
    fig_src = OUTPUT / "figures"
    fig_dst = INTEG2D / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)
    (ROOT / "docs/archive/phase_reports/HTW_HighLevel_PrefixConsistent_Pilot_CN.md").write_text(report, encoding="utf-8")
    (INTEG2D / "HTW_HighLevel_PrefixConsistent_Pilot_CN.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    gate = run()
    print(json.dumps({"PREFIX": gate["PREFIX"], "supported": gate["prefix_consistent_supervision_supported"]}, indent=2), flush=True)
