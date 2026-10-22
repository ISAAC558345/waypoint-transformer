"""HTW Integration Phase-2C runner. Frozen search audit. No training."""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase941.protocol import sha256_bytes
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.candidate_pool import assignment_hash
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase1.stats import summarize
from wt.lowlevel_v2.integration_phase2.train_hl import load_align_split
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker, replay_to_hold
from wt.lowlevel_v2.phase4b.snapshot import restore_env

from .protocol import (
    BEAM_WIDTHS,
    EXPECTED_HIGH_SHA,
    EXPECTED_UTIL_SHA,
    INTEG2C,
    LOW_SEEDS,
    MODEL_SEEDS,
    OLD_HL_CKPT,
    OUTPUT,
    PHASE2B_OUTPUT,
    PHASE2_OUTPUT,
    RECOVER_NR,
    ROOT,
    UTILITY_EPS,
    abs_ckpt,
    high_sha,
    load_compat_gate,
    load_phase2_gate,
    load_phase2a_gate,
    load_phase2b_gate,
    new_ckpt,
)
from .search_core import (
    beam_top,
    greedy_trace,
    idle_active,
    prefix_audit,
    set_path_stats,
    target_cache,
)
from .toy_map import toy_map_recovery

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
    seed = PHASE2B_OUTPUT / "utility_cache.json"
    if seed.is_file() and not CACHE_PATH.is_file():
        OUTPUT.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
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
        if len(U_CACHE) % 8 == 0:
            save_u_cache()
    return float(U_CACHE[cache_key]), False


def families() -> list[dict]:
    items = [{"family": "OLD", "name": "OLD-HL", "seed": "old0", "path": OLD_HL_CKPT}]
    for seed in MODEL_SEEDS:
        items.append({"family": "COND", "name": "COND-HL", "seed": int(seed), "path": new_ckpt(int(seed))})
    for seed in MODEL_SEEDS:
        items.append({"family": "ABS", "name": "ABS-HL", "seed": int(seed), "path": abs_ckpt(int(seed))})
    return items


def decode_nr(best: float, u: float) -> float:
    return (float(best) - float(u)) / (abs(float(best)) + UTILITY_EPS)


def stats_of(model, encoded, snap, buoy_mask, assignment: np.ndarray, cache: dict) -> dict:
    digest = assignment_hash(assignment)
    if digest not in cache:
        tcache = target_cache(model, encoded, snap, assignment, buoy_mask)
        cache[digest] = set_path_stats(tcache, assignment)
    return cache[digest]


def run() -> dict:
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("OLD-HL SHA mismatch")
    p2 = load_phase2_gate()
    p2a = load_phase2a_gate()
    p2b = load_phase2b_gate()
    compat = load_compat_gate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    load_u_cache()
    val = load_align_split("hl_rl_align_val")
    util_sha = sha256_bytes((PHASE2_OUTPUT / "hl_rl_align_train.npz").read_bytes() + (PHASE2_OUTPUT / "hl_rl_align_val.npz").read_bytes())
    protocol = {
        "name": "HL-SEARCH-AUDIT",
        "training": False,
        "training_low_level": False,
        "beam_widths": list(BEAM_WIDTHS),
        "utility_rerank": False,
        "checkpoint_change": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "COLREL": "C",
        "COMM": "B",
        "COMPAT": compat.get("COMPAT"),
        "HLALIGN": p2.get("HLALIGN"),
        "HLSUP": p2a.get("MECHANISM_GATE") or p2a.get("gate_name"),
        "ABSSUP": p2b.get("ABSSUP"),
        "utility_dataset_sha": util_sha,
        "expected_utility_sha": EXPECTED_UTIL_SHA,
        "split": "hl_rl_align_val",
        "n_states": int(val["assignment"].shape[0]),
        "abs_checkpoints": [str(abs_ckpt(int(s))) for s in MODEL_SEEDS],
        "selected_epochs_frozen": True,
    }
    _dump(OUTPUT / "HL_SEARCH_PROTOCOL.json", protocol)
    shutil.copy2(INTEG2C / "HL_SEARCH_DECODER_EQUATIONS.md", OUTPUT / "HL_SEARCH_DECODER_EQUATIONS.md")
    toy = toy_map_recovery()
    _dump(OUTPUT / "HL_SEARCH_TOY_MAP_RECOVERY.json", toy)
    print(f"toy greedy MAP recovery={toy['greedy_map_recovery']:.3f}", flush=True)

    greedy_rows, beam_rows, util_rows, setpath_rows, entropy_rows = [], [], [], [], []
    div_rows, margin_rows, mass_rows, pool_rows, vs_rows, idle_rows, s34_rows = [], [], [], [], [], [], []

    for spec in families():
        print(f"audit {spec['family']} {spec['seed']}", flush=True)
        model = load_high(spec["path"])
        for index in range(int(val["assignment"].shape[0])):
            encoded = _encode_one(model, val, index, "cpu")
            snap = make_snap(val["uav_positions"][index], val["buoy_positions"][index])
            buoy_mask = val["buoy_mask"][index]
            pool = np.asarray(val["candidates"][index], dtype=np.int64)
            u_pool = np.asarray(val["utility"][index], dtype=np.float64)
            hashes = list(val["hashes"][index])
            best_u = float(np.max(u_pool))
            best_pool = pool[int(np.argmax(u_pool))]
            stat_cache: dict[str, dict] = {}
            with torch.no_grad():
                greedy = greedy_trace(model, encoded, snap, buoy_mask)
            g_assn = greedy["assignment"]
            g_stats = stats_of(model, encoded, snap, buoy_mask, g_assn, stat_cache)
            bp_stats = stats_of(model, encoded, snap, buoy_mask, best_pool, stat_cache)
            u_g, g_in = u_new(val, index, g_assn)
            nr_g = decode_nr(best_u, u_g)
            idle_g, act_g = idle_active(g_assn)
            pref = prefix_audit(model, encoded, snap, buoy_mask, greedy["steps"], g_assn, best_pool)
            pool_set = []
            for slot, row in enumerate(pool):
                st = stats_of(model, encoded, snap, buoy_mask, row, stat_cache)
                pool_set.append((float(st["logp_set"]), int(slot), row, st))
            pool_set.sort(key=lambda item: item[0], reverse=True)
            map_lp, map_slot, map_assn, map_st = pool_set[0]
            u_map = float(u_pool[map_slot])
            nr_map = decode_nr(best_u, u_map)
            greedy_rows.append(
                {
                    "family": spec["family"],
                    "model": spec["name"],
                    "high_seed": spec["seed"],
                    "state_id": index,
                    "logp_path_greedy": greedy["logp_path"],
                    "logp_set_greedy": g_stats["logp_set"],
                    "omb_greedy": g_stats["omb"],
                    "in_pool": assignment_hash(g_assn) in set(hashes),
                    "idle": idle_g,
                    "active": act_g,
                    "U_greedy": u_g,
                    "DecodeNR_greedy": nr_g,
                    "from_pool_label": g_in,
                }
            )
            width_pack = {}
            setbest64 = None
            for width in BEAM_WIDTHS:
                beams = beam_top(model, encoded, snap, buoy_mask, width)
                path_assn = beams[0][1]
                path_lp = beams[0][0]
                p_stats = stats_of(model, encoded, snap, buoy_mask, path_assn, stat_cache)
                best_set = None
                best_set_lp = -1.0e30
                best_set_path = None
                n_in = 0
                for plp, assn in beams:
                    n_in += int(assignment_hash(assn) in set(hashes))
                    st = stats_of(model, encoded, snap, buoy_mask, assn, stat_cache)
                    if st["logp_set"] > best_set_lp:
                        best_set_lp = float(st["logp_set"])
                        best_set = assn
                        best_set_path = float(plp)
                        best_set_st = st
                u_w, _ = u_new(val, index, path_assn)
                nr_w = decode_nr(best_u, u_w)
                idle_w, act_w = idle_active(path_assn)
                u_sb, _ = u_new(val, index, best_set)
                nr_sb = decode_nr(best_u, u_sb)
                rec = {
                    "family": spec["family"],
                    "model": spec["name"],
                    "high_seed": spec["seed"],
                    "state_id": index,
                    "beam_width": int(width),
                    "logp_path_top1": float(path_lp),
                    "logp_set_top1": p_stats["logp_set"],
                    "in_pool_top1": assignment_hash(path_assn) in set(hashes),
                    "idle_top1": idle_w,
                    "active_top1": act_w,
                    "U_top1": u_w,
                    "DecodeNR_top1": nr_w,
                    "logp_set_beambest": float(best_set_lp),
                    "logp_path_beambest": float(best_set_path),
                    "U_setbest": u_sb,
                    "DecodeNR_setbest": nr_sb,
                    "set_minus_greedy": float(best_set_lp - g_stats["logp_set"]),
                    "beam_in_pool_rate": n_in / float(len(beams)),
                }
                beam_rows.append(rec)
                util_rows.append(rec)
                width_pack[int(width)] = rec
                if int(width) == 64:
                    setbest64 = (best_set, best_set_st, u_sb, nr_sb, rec)
            gap = width_pack[64]["set_minus_greedy"]
            greedy_rows[-1]["logp_set_beambest64"] = width_pack[64]["logp_set_beambest"]
            greedy_rows[-1]["set_minus_greedy64"] = gap
            sb_assn, sb_st, u_sb, nr_sb, rec64 = setbest64
            for tag, assn, st, uu, nr in (
                ("best_pool", best_pool, bp_stats, best_u, 0.0),
                ("greedy", g_assn, g_stats, u_g, nr_g),
                ("beam64_setbest", sb_assn, sb_st, u_sb, nr_sb),
                ("pool_map", map_assn, map_st, u_map, nr_map),
            ):
                setpath_rows.append(
                    {
                        "family": spec["family"],
                        "high_seed": spec["seed"],
                        "state_id": index,
                        "tag": tag,
                        "logp_set": st["logp_set"],
                        "max_order_logp_path": st["max_order_logp_path"],
                        "mean_order_logp_path": st["mean_order_logp_path"],
                        "omb": st["omb"],
                        "U": uu,
                        "DecodeNR": nr,
                    }
                )
                entropy_rows.append(
                    {
                        "family": spec["family"],
                        "high_seed": spec["seed"],
                        "state_id": index,
                        "tag": tag,
                        "order_entropy": st["order_entropy"],
                        "n_eff": st["n_eff"],
                        "n_valid_orders": st["n_valid_orders"],
                        "U": uu,
                    }
                )
            div_rows.append(
                {
                    "family": spec["family"],
                    "high_seed": spec["seed"],
                    "state_id": index,
                    "first_divergence_step": pref["first_divergence_step"],
                    "DecodeNR_greedy": nr_g,
                }
            )
            for row in pref["margins"]:
                margin_rows.append({"family": spec["family"], "high_seed": spec["seed"], "state_id": index, **row})
            for row in pref["teacher_forced_mass"]:
                mass_rows.append({"family": spec["family"], "high_seed": spec["seed"], "state_id": index, **row})
            pool_rows.append(
                {
                    "family": spec["family"],
                    "high_seed": spec["seed"],
                    "state_id": index,
                    "U_pool_map": u_map,
                    "DecodeNR_pool_map": nr_map,
                    "logp_set_pool_map": map_lp,
                    "U_greedy": u_g,
                    "DecodeNR_greedy": nr_g,
                    "logp_set_greedy": g_stats["logp_set"],
                    "pool_map_minus_greedy_U": u_map - u_g,
                    "in_pool_greedy": assignment_hash(g_assn) in set(hashes),
                }
            )
            vs_rows.append(
                {
                    "family": spec["family"],
                    "high_seed": spec["seed"],
                    "state_id": index,
                    "DecodeNR_greedy": nr_g,
                    "DecodeNR_beam64_path": width_pack[64]["DecodeNR_top1"],
                    "DecodeNR_beam64_setbest": nr_sb,
                    "DecodeNR_pool_map": nr_map,
                    "logp_set_greedy": g_stats["logp_set"],
                    "logp_set_beam64_setbest": rec64["logp_set_beambest"],
                    "logp_set_pool_map": map_lp,
                    "U_greedy": u_g,
                    "U_beam64_setbest": u_sb,
                    "U_pool_map": u_map,
                }
            )
            for rec in greedy["steps"]:
                idle_rows.append({"family": spec["family"], "high_seed": spec["seed"], "state_id": index, **rec})
            recovered = None
            for width in BEAM_WIDTHS:
                if width_pack[int(width)]["DecodeNR_top1"] <= RECOVER_NR:
                    recovered = int(width)
                    break
            if spec["family"] == "ABS" and spec["seed"] in (3, 4):
                s34_rows.append(
                    {
                        "high_seed": spec["seed"],
                        "state_id": index,
                        "DecodeNR_greedy": nr_g,
                        "DecodeNR_pool_map": nr_map,
                        "DecodeNR_beam64_path": width_pack[64]["DecodeNR_top1"],
                        "DecodeNR_beam64_setbest": nr_sb,
                        "set_minus_greedy64": gap,
                        "first_divergence_step": pref["first_divergence_step"],
                        "n_eff_best_pool": bp_stats["n_eff"],
                        "n_eff_greedy": g_stats["n_eff"],
                        "omb_best_pool": bp_stats["omb"],
                        "omb_greedy": g_stats["omb"],
                        "min_recover_width": recovered if recovered is not None else 999,
                    }
                )
            print(
                f"  {spec['family']} {spec['seed']} {index+1}/{val['assignment'].shape[0]} NRg={nr_g:.3f} NRmap={nr_map:.3f} NRb64={width_pack[64]['DecodeNR_top1']:.3f} gap={gap:.3f}",
                flush=True,
            )
            save_u_cache()
        del model

    _csv(OUTPUT / "HL_SEARCH_GREEDY_SET_GAP.csv", greedy_rows)
    _csv(OUTPUT / "HL_SEARCH_BEAM_WIDTH.csv", beam_rows)
    _csv(OUTPUT / "HL_SEARCH_BEAM_UTILITY.csv", util_rows)
    _csv(OUTPUT / "HL_SEARCH_SET_PATH_PROB.csv", setpath_rows)
    _csv(OUTPUT / "HL_SEARCH_ORDER_ENTROPY.csv", entropy_rows)
    _csv(OUTPUT / "HL_SEARCH_PREFIX_DIVERGENCE.csv", div_rows)
    _csv(OUTPUT / "HL_SEARCH_PREFIX_MARGIN.csv", margin_rows)
    _csv(OUTPUT / "HL_SEARCH_COMPATIBLE_PREFIX_MASS.csv", mass_rows)
    _csv(OUTPUT / "HL_SEARCH_POOL_MAP.csv", pool_rows)
    _csv(OUTPUT / "HL_SEARCH_BEAM_VS_POOL.csv", vs_rows)
    _csv(OUTPUT / "HL_SEARCH_IDLE_PATH.csv", idle_rows)
    _csv(OUTPUT / "HL_SEARCH_SEED3_SEED4.csv", s34_rows)
    save_u_cache()
    gate = classify(greedy_rows, beam_rows, vs_rows, entropy_rows, setpath_rows, div_rows, mass_rows, idle_rows, toy, s34_rows)
    _dump(OUTPUT / "HL_SEARCH_MECHANISM_GATE.json", gate)
    plot_figures(beam_rows, setpath_rows, entropy_rows, div_rows, vs_rows, idle_rows, s34_rows, OUTPUT / "figures")
    report = write_report(gate, protocol, toy)
    (OUTPUT / "HTW_HighLevel_SearchMismatch_Audit_CN.md").write_text(report, encoding="utf-8")
    publish(report)
    return gate


def _sl(rows, family, seed=None):
    out = [r for r in rows if r.get("family") == family]
    if seed is not None:
        out = [r for r in out if str(r.get("high_seed")) == str(seed)]
    return out


def classify(greedy_rows, beam_rows, vs_rows, entropy_rows, setpath_rows, div_rows, mass_rows, idle_rows, toy, s34_rows) -> dict:
    def nr_of(rows, key):
        return summarize([r[key] for r in rows])

    abs_g = _sl(greedy_rows, "ABS")
    cond_g = _sl(greedy_rows, "COND")
    old_g = _sl(greedy_rows, "OLD")
    abs_vs = _sl(vs_rows, "ABS")
    gaps = summarize([r["set_minus_greedy64"] for r in greedy_rows if r["family"] == "ABS"])
    frac_pos = float(np.mean([r["set_minus_greedy64"] > 1.0e-6 for r in _sl(greedy_rows, "ABS")]))
    beam_nr = {}
    recover = {}
    for width in BEAM_WIDTHS:
        sub = [r for r in beam_rows if r["family"] == "ABS" and int(r["beam_width"]) == int(width)]
        beam_nr[str(width)] = summarize([r["DecodeNR_top1"] for r in sub])
        recover[str(width)] = float(np.mean([r["DecodeNR_top1"] <= RECOVER_NR for r in sub]))
    minw = []
    for r in abs_vs:
        got = None
        for width in BEAM_WIDTHS:
            hit = next(b for b in beam_rows if b["family"] == "ABS" and str(b["high_seed"]) == str(r["high_seed"]) and int(b["state_id"]) == int(r["state_id"]) and int(b["beam_width"]) == int(width))
            if hit["DecodeNR_top1"] <= RECOVER_NR:
                got = int(width)
                break
        minw.append(got if got is not None else 999)
    rec32 = float(np.mean([w <= 32 for w in minw]))
    rec64 = float(np.mean([w <= 64 for w in minw]))
    g_nr = nr_of(abs_g, "DecodeNR_greedy")
    pool_nr = nr_of(abs_vs, "DecodeNR_pool_map")
    b64_nr = nr_of(abs_vs, "DecodeNR_beam64_path")
    b64s_nr = nr_of(abs_vs, "DecodeNR_beam64_setbest")
    old_nr = nr_of(old_g, "DecodeNR_greedy")
    omb_pool = summarize([r["omb"] for r in setpath_rows if r["family"] == "ABS" and r["tag"] == "best_pool"])
    omb_g = summarize([r["omb"] for r in setpath_rows if r["family"] == "ABS" and r["tag"] == "greedy"])
    neff_pool = summarize([r["n_eff"] for r in entropy_rows if r["family"] == "ABS" and r["tag"] == "best_pool"])
    neff_g = summarize([r["n_eff"] for r in entropy_rows if r["family"] == "ABS" and r["tag"] == "greedy"])
    div = [r["first_divergence_step"] for r in div_rows if r["family"] == "ABS" and r["first_divergence_step"] is not None]
    div_hist = {str(k): float(np.mean([int(x) == k for x in div])) for k in range(1, 7)}
    early = float(np.mean([int(x) <= 2 for x in div])) if div else float("nan")
    mass_s1 = summarize([r["compatible_prefix_mass"] for r in mass_rows if r["family"] == "ABS" and r.get("order") == "mean_all_perms" and int(r["step"]) == 1])
    idle_choice = _sl(idle_rows, "ABS")
    idle_when_worse = [r for r in idle_choice if r["chose_idle"] and r["best_nonidle_logp"] > r["idle_logp"] + 1.0e-9]
    idle_close = [r for r in idle_choice if r["chose_idle"] and abs(r["idle_logp"] - r["best_nonidle_logp"]) < 0.2]
    frac_idle = float(np.mean([r["chose_idle"] for r in idle_choice])) if idle_choice else float("nan")
    s3 = [r for r in s34_rows if int(r["high_seed"]) == 3]
    s4 = [r for r in s34_rows if int(r["high_seed"]) == 4]
    s3g = summarize([r["DecodeNR_greedy"] for r in s3]) if s3 else {}
    s4g = summarize([r["DecodeNR_greedy"] for r in s4]) if s4 else {}
    s4map = summarize([r["DecodeNR_pool_map"] for r in s4]) if s4 else {}
    s4b = summarize([r["DecodeNR_beam64_path"] for r in s4]) if s4 else {}
    s4bs = summarize([r["DecodeNR_beam64_setbest"] for r in s4]) if s4 else {}
    toy_g = float(toy["greedy_map_recovery"])
    mismatch = bool(toy_g < 0.8 or frac_pos > 0.3)
    search = bool(b64_nr["median"] + 0.05 < g_nr["median"] or pool_nr["median"] + 0.05 < g_nr["median"] or rec32 >= 0.4)
    setpath = bool(neff_pool.get("median", 0) > neff_g.get("median", 0) + 0.5 or omb_pool.get("median", 0) > omb_g.get("median", 0) + 0.05)
    prefix = bool(early >= 0.4 or mass_s1.get("median", 1.0) < 0.35)
    objective = bool(pool_nr["median"] > old_nr["median"] + 0.15 and b64s_nr["median"] > old_nr["median"] + 0.15)
    seed_issue = bool(s4map and s4g and s4map["median"] > 0.5 and abs(s4map["median"] - s3g.get("median", 0)) > 0.3)
    poolprior = bool(pool_nr["median"] + 0.08 < b64_nr["median"] and pool_nr["median"] + 0.05 < g_nr["median"])
    idle_mech = bool(frac_idle > 0.35 and early >= 0.3)
    tags = []
    if search:
        tags.append("SEARCH")
    if setpath:
        tags.append("SETPATH")
    if prefix:
        tags.append("PREFIX")
    if objective:
        tags.append("OBJECTIVE")
    if seed_issue:
        tags.append("SEED")
    if poolprior:
        tags.append("POOLPRIOR")
    if idle_mech:
        tags.append("IDLE")
    if not tags:
        tags = ["NONE"]
    primary = tags[0]
    secondary = tags[1] if len(tags) > 1 else "NONE"
    near_old_map = pool_nr["median"] <= old_nr["median"] + 0.08
    near_old_beam = b64s_nr["median"] <= old_nr["median"] + 0.08
    if search and rec32 >= 0.5 and (near_old_map or near_old_beam) and (pool_nr["median"] + 0.05 < g_nr["median"] or b64s_nr["median"] + 0.05 < g_nr["median"]):
        gate_name, key = "A", "HLSEARCH-A"
        confirmed = True
        if b64s_nr["median"] <= pool_nr["median"] + 0.03:
            nxt = "Next allowed change: beam-set-MAP decoding only. Do not retrain. Do not constrain to the L=24 pool."
        else:
            nxt = "Next allowed change: candidate-constrained set-MAP inference only. Do not retrain."
        text = "Greedy free decode fails to realize trained set probability; search recovery is the primary lever."
    elif (not rec32 >= 0.5) and setpath:
        gate_name, key, confirmed = "B", "HLSETPATH-B", False
        nxt = "Next allowed: prefix-consistent utility supervision research. Do not train in this phase."
        text = "Wider path beam is not enough; set probability mass is split across orders/prefixes."
    elif objective:
        gate_name, key, confirmed = "C", "HLOBJ-C", False
        nxt = "Stop. Absolute support still does not align set-MAP with utility. Do not change decoder this phase."
        text = "Even pool-MAP / beam set-MAP remain low-utility."
    elif len(tags) > 1:
        gate_name, key, confirmed = "D", "HLMIX-D", False
        nxt = "Stop. Diagnose the remaining mixture; do not combine decoder change and retraining."
        text = "Multiple mechanisms coexist; search is not a single sufficient fix."
    else:
        gate_name, key, confirmed = "E", "HLNONE-E", False
        nxt = "Stop. Cannot localize search mismatch from these diagnostics."
        text = "Unable to locate a primary mechanism."
    if primary == "SEARCH" and gate_name != "A" and rec32 < 0.5:
        if setpath or prefix:
            gate_name, key, confirmed = "B", "HLSETPATH-B", False
            nxt = "Next allowed: prefix-consistent utility supervision research. Do not train in this phase."
            text = "Beam expansion does not recover most states; prefix/set-path inconsistency remains."
    return {
        "SEARCH_GATE": gate_name,
        "gate_key": key,
        "search_mismatch_confirmed": confirmed,
        "primary_mechanism": primary,
        "secondary_mechanism": secondary,
        "mechanisms": tags,
        "training_inference_objective_mismatch": mismatch,
        "training": False,
        "training_low_level": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "COLREL": "C",
        "COMM": "B",
        "text": text,
        "next_step": nxt,
        "toy_greedy_map_recovery": toy_g,
        "toy_beam_contains_map": toy.get("beam_contains_map"),
        "ABS_greedy_DecodeNR": g_nr,
        "ABS_pool_map_DecodeNR": pool_nr,
        "ABS_beam64_path_DecodeNR": b64_nr,
        "ABS_beam64_setbest_DecodeNR": b64s_nr,
        "OLD_greedy_DecodeNR": old_nr,
        "COND_greedy_DecodeNR": nr_of(cond_g, "DecodeNR_greedy"),
        "ABS_set_minus_greedy64": gaps,
        "frac_beam_setbest_gt_greedy": frac_pos,
        "DecodeNR_vs_width": beam_nr,
        "fraction_recovered": recover,
        "fraction_recovered_by_min_width_le32": rec32,
        "fraction_recovered_by_min_width_le64": rec64,
        "omb_best_pool": omb_pool,
        "omb_greedy": omb_g,
        "n_eff_best_pool": neff_pool,
        "n_eff_greedy": neff_g,
        "divergence_hist": div_hist,
        "early_div_le2": early,
        "compatible_mass_step1": mass_s1,
        "idle_choice_fraction": frac_idle,
        "idle_against_nonidle": len(idle_when_worse),
        "idle_close_calls": len(idle_close),
        "ABS3_greedy": s3g,
        "ABS4_greedy": s4g,
        "ABS4_pool_map": s4map,
        "ABS4_beam64_path": s4b,
        "ABS4_beam64_setbest": s4bs,
        "per_seed_greedy": {str(s): nr_of(_sl(greedy_rows, "ABS", s), "DecodeNR_greedy") for s in MODEL_SEEDS},
        "per_seed_pool_map": {str(s): nr_of(_sl(vs_rows, "ABS", s), "DecodeNR_pool_map") for s in MODEL_SEEDS},
        "per_seed_beam64": {str(s): nr_of(_sl(vs_rows, "ABS", s), "DecodeNR_beam64_path") for s in MODEL_SEEDS},
    }


def plot_figures(beam_rows, setpath_rows, entropy_rows, div_rows, vs_rows, idle_rows, s34_rows, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for fam, style in (("OLD", "-k"), ("COND", "--"), ("ABS", "-")):
        xs, ys = [], []
        for width in BEAM_WIDTHS:
            sub = [r["DecodeNR_top1"] for r in beam_rows if r["family"] == fam and int(r["beam_width"]) == int(width)]
            xs.append(int(width))
            ys.append(float(np.median(sub)) if sub else float("nan"))
        ax.plot(xs, ys, style, label=fam)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("beam width")
    ax.set_ylabel("median DecodeNR")
    ax.set_title("DecodeNR vs beam width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "search_decode_nr_vs_width.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for tag, color in (("best_pool", "C0"), ("greedy", "C1"), ("beam64_setbest", "C2")):
        sub = [r for r in setpath_rows if r["family"] == "ABS" and r["tag"] == tag]
        ax.scatter([r["max_order_logp_path"] for r in sub], [r["logp_set"] for r in sub], s=12, alpha=0.6, label=tag, c=color)
    ax.set_xlabel("max-order path logP")
    ax.set_ylabel("set logP")
    ax.set_title("set logP vs path logP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "search_set_vs_path.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    data = []
    labs = []
    for tag in ("best_pool", "greedy", "beam64_setbest"):
        data.append([r["n_eff"] for r in entropy_rows if r["family"] == "ABS" and r["tag"] == tag])
        labs.append(tag)
    ax.boxplot(data, tick_labels=labs)
    ax.set_title("order entropy N_eff")
    fig.tight_layout()
    fig.savefig(out / "search_order_entropy.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    counts = [int(np.sum([int(r["first_divergence_step"] or 0) == k for r in div_rows if r["family"] == "ABS"])) for k in range(1, 7)]
    ax.bar(range(1, 7), counts)
    ax.set_xlabel("first divergence step")
    ax.set_title("ABS first divergence")
    fig.tight_layout()
    fig.savefig(out / "search_divergence.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = ["Greedy", "Beam64 path", "Beam64 set", "Pool-MAP"]
    meds = [
        float(np.median([r["DecodeNR_greedy"] for r in vs_rows if r["family"] == "ABS"])),
        float(np.median([r["DecodeNR_beam64_path"] for r in vs_rows if r["family"] == "ABS"])),
        float(np.median([r["DecodeNR_beam64_setbest"] for r in vs_rows if r["family"] == "ABS"])),
        float(np.median([r["DecodeNR_pool_map"] for r in vs_rows if r["family"] == "ABS"])),
    ]
    ax.bar(labels, meds)
    ax.set_title("Pool-MAP vs Beam-MAP vs Greedy")
    fig.tight_layout()
    fig.savefig(out / "search_pool_vs_beam_vs_greedy.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for seed, color in ((3, "C0"), (4, "C1")):
        xs, ys = [], []
        for width in BEAM_WIDTHS:
            sub = [r["DecodeNR_top1"] for r in beam_rows if r["family"] == "ABS" and str(r["high_seed"]) == str(seed) and int(r["beam_width"]) == int(width)]
            xs.append(int(width))
            ys.append(float(np.median(sub)) if sub else float("nan"))
        ax.plot(xs, ys, label=f"ABS seed{seed}", color=color)
    ax.set_xscale("log", base=2)
    ax.set_title("seed3 vs seed4 recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "search_seed3_seed4.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for step in range(1, 7):
        sub = [r["idle_prob"] for r in idle_rows if r["family"] == "ABS" and int(r["step"]) == step]
        ax.plot([step], [float(np.mean(sub)) if sub else float("nan")], "o")
    means = [float(np.mean([r["idle_prob"] for r in idle_rows if r["family"] == "ABS" and int(r["step"]) == step])) for step in range(1, 7)]
    ax.plot(range(1, 7), means, "-C0", label="idle p")
    chosen = [float(np.mean([r["chose_idle"] for r in idle_rows if r["family"] == "ABS" and int(r["step"]) == step])) for step in range(1, 7)]
    ax.plot(range(1, 7), chosen, "-C1", label="P(choose idle)")
    ax.set_xlabel("decode step")
    ax.set_title("idle probability by decode step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "search_idle_step.png", dpi=130)
    plt.close(fig)


def write_report(gate: dict, protocol: dict, toy: dict) -> str:
    rec = gate["fraction_recovered"]
    rec_txt = ", ".join(f"w={w}:{rec[str(w)]:.3f}" for w in BEAM_WIDTHS)
    width_nr = ", ".join(f"w={w}:{gate['DecodeNR_vs_width'][str(w)]['median']:.3f}" for w in BEAM_WIDTHS)
    seed_g = ", ".join(f"s{s}={gate['per_seed_greedy'][str(s)]['median']:.3f}" for s in MODEL_SEEDS)
    seed_m = ", ".join(f"s{s}={gate['per_seed_pool_map'][str(s)]['median']:.3f}" for s in MODEL_SEEDS)
    seed_b = ", ".join(f"s{s}={gate['per_seed_beam64'][str(s)]['median']:.3f}" for s in MODEL_SEEDS)
    return f"""# HTW Integration Phase-2C：Set-Likelihood vs Free-Decoding Search Mismatch Audit

本阶段 **不训练**。不改 \(L_{{\\mathrm{{abs}}}}\)、网络、checkpoint、低层、collision/interference。只诊断冻结 OLD / COND / ABS 在 development val states 上的 search。

---

## 23 问正式回答

1. **free decoder 实际是什么 search？** 联合 greedy \(\\arg\\max\) over legal \((m,k)\) logits，平局用 `canonical_keys`。不是 beam，也不是按 UAV 下标的顺序 softmax。
2. **是否最大化 set probability？** toy greedy MAP recovery={gate['toy_greedy_map_recovery']:.3f}；ABS 中 beam-set-best \(>\) greedy set-logP 的比例={gate['frac_beam_setbest_gt_greedy']:.3f}。`training_inference_objective_mismatch`={str(gate['training_inference_objective_mismatch']).lower()}
3. **greedy set logP 比 beam-set-best 低多少？** ABS median gap={gate['ABS_set_minus_greedy64']['median']:.4f}（mean={gate['ABS_set_minus_greedy64']['mean']:.4f}）
4. **beam 变宽后 DecodeNR 是否下降？** ABS median DecodeNR by width: {width_nr}
5. **多少 state 在 beam<=32 恢复（DecodeNR<=0.10）？** {gate['fraction_recovered_by_min_width_le32']:.3f}；各 width 即时恢复率：{rec_txt}
6. **set vs path？** ABS best-pool OMB median={gate['omb_best_pool']['median']:.4f}；greedy OMB median={gate['omb_greedy']['median']:.4f}
7. **高 utility candidate 的 order entropy 是否更大？** best-pool \(N_{{\\mathrm{{eff}}}}\) median={gate['n_eff_best_pool']['median']:.3f}；greedy {gate['n_eff_greedy']['median']:.3f}
8. **first divergence？** 分布 {gate['divergence_hist']}；step<=2 比例={gate['early_div_le2']:.3f}
9. **best-pool-compatible prefix mass（step1, 全排列平均）？** median={gate['compatible_mass_step1']['median']:.4f}
10. **Pool-MAP DecodeNR？** median={gate['ABS_pool_map_DecodeNR']['median']:.4f} mean={gate['ABS_pool_map_DecodeNR']['mean']:.4f}
11. **Beam64 path DecodeNR？** median={gate['ABS_beam64_path_DecodeNR']['median']:.4f}；set-best median={gate['ABS_beam64_setbest_DecodeNR']['median']:.4f}
12. **Greedy DecodeNR？** median={gate['ABS_greedy_DecodeNR']['median']:.4f}（OLD greedy median={gate['OLD_greedy_DecodeNR']['median']:.4f}）
13. **Pool-MAP 是否恢复 seed4？** seed4 greedy median={gate['ABS4_greedy'].get('median', float('nan')):.4f}；pool-MAP {gate['ABS4_pool_map'].get('median', float('nan')):.4f}
14. **Beam 是否恢复 seed4？** path {gate['ABS4_beam64_path'].get('median', float('nan')):.4f}；set-best {gate['ABS4_beam64_setbest'].get('median', float('nan')):.4f}
15. **idle 是否主错误分支？** idle choice fraction={gate['idle_choice_fraction']:.3f}；close idle calls={gate['idle_close_calls']}
16. **toy exact set-MAP recovery？** greedy={toy['greedy_map_recovery']:.3f}；beam contains MAP {toy['beam_contains_map']}
17. **training-inference mismatch？** {str(gate['training_inference_objective_mismatch']).lower()}
18. **primary mechanism？** {gate['primary_mechanism']}
19. **secondary mechanism？** {gate['secondary_mechanism']}
20. **Search Gate？** {gate['SEARCH_GATE']} ({gate['gate_key']})
21. **下一步唯一 minimal change？** {gate['next_step']}
22. **本阶段是否训练？否。**
23. **是否修改低层？否。**

ABS greedy per seed: {seed_g}
ABS pool-MAP per seed: {seed_m}
ABS beam64-path per seed: {seed_b}

{gate['text']}

{gate['next_step']}

本阶段停止。不要自动进入下一阶段。
"""


def publish(report: str) -> None:
    names = [
        "HL_SEARCH_PROTOCOL.json",
        "HL_SEARCH_DECODER_EQUATIONS.md",
        "HL_SEARCH_GREEDY_SET_GAP.csv",
        "HL_SEARCH_BEAM_WIDTH.csv",
        "HL_SEARCH_BEAM_UTILITY.csv",
        "HL_SEARCH_SET_PATH_PROB.csv",
        "HL_SEARCH_ORDER_ENTROPY.csv",
        "HL_SEARCH_PREFIX_DIVERGENCE.csv",
        "HL_SEARCH_PREFIX_MARGIN.csv",
        "HL_SEARCH_COMPATIBLE_PREFIX_MASS.csv",
        "HL_SEARCH_POOL_MAP.csv",
        "HL_SEARCH_BEAM_VS_POOL.csv",
        "HL_SEARCH_IDLE_PATH.csv",
        "HL_SEARCH_SEED3_SEED4.csv",
        "HL_SEARCH_TOY_MAP_RECOVERY.json",
        "HL_SEARCH_MECHANISM_GATE.json",
        "HTW_HighLevel_SearchMismatch_Audit_CN.md",
    ]
    for name in names:
        src = OUTPUT / name
        if src.is_file():
            shutil.copy2(src, INTEG2C / name)
            shutil.copy2(src, ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name)
    fig_src = OUTPUT / "figures"
    fig_dst = INTEG2C / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)
    (ROOT / "docs/archive/phase_reports/HTW_HighLevel_SearchMismatch_Audit_CN.md").write_text(report, encoding="utf-8")
    (INTEG2C / "HTW_HighLevel_SearchMismatch_Audit_CN.md").write_text(report, encoding="utf-8")


def _load_csv(name: str) -> list[dict]:
    path = OUTPUT / name
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out = dict(row)
            for key, val in list(out.items()):
                if key in {"family", "model", "high_seed", "tag", "order"}:
                    continue
                if val in {"", "None", "nan"}:
                    out[key] = None
                    continue
                if val in {"True", "False"}:
                    out[key] = val == "True"
                    continue
                try:
                    num = float(val)
                    out[key] = int(num) if num.is_integer() and key in {"state_id", "beam_width", "step", "uav", "token", "idle", "active", "n_valid_orders", "min_recover_width", "first_divergence_step", "n_compatible_edges"} else num
                except (TypeError, ValueError):
                    pass
            rows.append(out)
    return rows


def finish_from_outputs() -> dict:
    greedy_rows = _load_csv("HL_SEARCH_GREEDY_SET_GAP.csv")
    beam_rows = _load_csv("HL_SEARCH_BEAM_WIDTH.csv")
    vs_rows = _load_csv("HL_SEARCH_BEAM_VS_POOL.csv")
    entropy_rows = _load_csv("HL_SEARCH_ORDER_ENTROPY.csv")
    setpath_rows = _load_csv("HL_SEARCH_SET_PATH_PROB.csv")
    div_rows = _load_csv("HL_SEARCH_PREFIX_DIVERGENCE.csv")
    mass_rows = _load_csv("HL_SEARCH_COMPATIBLE_PREFIX_MASS.csv")
    idle_rows = _load_csv("HL_SEARCH_IDLE_PATH.csv")
    s34_rows = _load_csv("HL_SEARCH_SEED3_SEED4.csv")
    toy = json.loads((OUTPUT / "HL_SEARCH_TOY_MAP_RECOVERY.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUTPUT / "HL_SEARCH_PROTOCOL.json").read_text(encoding="utf-8"))
    gate = classify(greedy_rows, beam_rows, vs_rows, entropy_rows, setpath_rows, div_rows, mass_rows, idle_rows, toy, s34_rows)
    _dump(OUTPUT / "HL_SEARCH_MECHANISM_GATE.json", gate)
    plot_figures(beam_rows, setpath_rows, entropy_rows, div_rows, vs_rows, idle_rows, s34_rows, OUTPUT / "figures")
    report = write_report(gate, protocol, toy)
    (OUTPUT / "HTW_HighLevel_SearchMismatch_Audit_CN.md").write_text(report, encoding="utf-8")
    publish(report)
    return gate


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--finish":
        gate = finish_from_outputs()
    else:
        gate = run()
    print(json.dumps({"SEARCH_GATE": gate["SEARCH_GATE"], "primary": gate["primary_mechanism"], "secondary": gate["secondary_mechanism"]}, indent=2), flush=True)
