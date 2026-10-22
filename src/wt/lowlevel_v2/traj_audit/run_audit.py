"""Offline trajectory-utility audit. Frozen checkpoints only. No training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase945.candidate_pool import assignment_hash, generate_pool
from wt.lowlevel_v2.integration_phase1.shadow import shadow_h15
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, summarize
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker
from wt.lowlevel_v2.integration_phase3.protocol import PHASE2B_OUTPUT, PHASE2D_OUTPUT
from wt.lowlevel_v2.integration_phase3.run_phase3 import make_stack
from wt.lowlevel_v2.phase1b.controller import make_networks
from wt.lowlevel_v2.phase4b.snapshot import snapshot_env
from wt.lowlevel_v2.traj_audit.protocol import HL_SPECS, LL_RUN, LOW_SEEDS, ORACLE_FAMILIES, OUTPUT, ROOT, UTILITY_EPS, ckpt_path, load_pools
from wt.lowlevel_v2.traj_audit.worker import FirstHoldOverrideWorker


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


def capture_state(worker, scenario: int) -> dict:
    return {
        "scenario": int(scenario),
        "t": int(worker.env.t),
        "env": snapshot_env(worker.env),
        "last_arrival": np.asarray(worker.last_arrival, dtype=np.float32).copy(),
        "last_positions": np.asarray(worker.last_positions, dtype=np.float32).copy(),
        "last_movement": np.asarray(worker.last_movement, dtype=np.float64).copy(),
    }


def state_key(state: dict, assignment: np.ndarray) -> str:
    env = state["env"]
    raw = np.asarray(env["uav_positions"], dtype=np.float64).tobytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f"{int(state['scenario'])}|{int(state['t'])}|{float(env['payload_bits'])}|{digest}|{assignment_hash(assignment)}"


def make_override_stack(high, ll_seed: int) -> FirstHoldOverrideWorker:
    actor, critic = make_networks()
    payload = torch.load(ckpt_path(int(ll_seed)), map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return FirstHoldOverrideWorker(high, actor, critic, service_lock=True, trust_region=True)


def pearson(x, y) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class Auditor:
    def __init__(self) -> None:
        self.shadows = {int(s): load_ll_worker(int(s)) for s in LOW_SEEDS}
        self.ull_cache: dict[str, float] = {}
        cache_path = OUTPUT / "ull_cache.json"
        if cache_path.is_file():
            self.ull_cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def save_cache(self) -> None:
        _dump(OUTPUT / "ull_cache.json", self.ull_cache)

    def u_ll(self, state: dict, assignment: np.ndarray, *, seeds=None) -> float:
        used = tuple(int(s) for s in (LOW_SEEDS if seeds is None else seeds))
        key = state_key(state, assignment) + f"|n{len(used)}"
        if key in self.ull_cache:
            return float(self.ull_cache[key])
        vals = [
            float(shadow_h15(self.shadows[int(s)], state, assignment, zero_residual=False)["payload_increment"])
            for s in used
        ]
        value = float(np.mean(vals))
        self.ull_cache[key] = value
        if len(self.ull_cache) % 16 == 0:
            self.save_cache()
        return value

    def u_ll_fast(self, state: dict, assignment: np.ndarray) -> float:
        return self.u_ll(state, assignment, seeds=(LL_RUN,))


def run_episode(worker, scenario: int, auditor: Auditor, *, compute_ull: bool) -> dict:
    info = worker.reset(int(scenario))
    payload0 = float(worker.env.payload_bits)
    holds = []
    current = None
    while True:
        if int(worker.pending) <= 0:
            if current is not None:
                current["u_hold_actual"] = float(worker.env.payload_bits) - float(current["pay0"])
                holds.append(current)
            state = capture_state(worker, scenario)
            worker._refresh_option(info)
            gamma = np.asarray(worker.option.association, dtype=np.int64).copy()
            current = {
                "hold": len(holds),
                "t": int(worker.env.t),
                "gamma": gamma,
                "hash": assignment_hash(gamma),
                "idle": float(np.mean(gamma[:6] < 0)),
                "pay0": float(worker.env.payload_bits),
                "state": state,
            }
        info, rec = worker.step(info, deterministic=True, zero_residual=False)
        if rec.done or worker.env.t >= worker.env.config.episode_horizon:
            break
    if current is not None:
        current["u_hold_actual"] = float(worker.env.payload_bits) - float(current["pay0"])
        holds.append(current)
    u_traj = float(worker.env.payload_bits - payload0)
    rows = []
    u_singles = []
    for item in holds:
        u_single = auditor.u_ll(item["state"], item["gamma"]) if compute_ull else float("nan")
        u_singles.append(u_single)
        rows.append(
            {
                "hold": int(item["hold"]),
                "t": int(item["t"]),
                "hash": item["hash"],
                "gamma": [int(x) for x in item["gamma"][:6]],
                "idle": float(item["idle"]),
                "U_single": u_single,
                "U_hold_actual": float(item["u_hold_actual"]),
            }
        )
    return {
        "scenario": int(scenario),
        "n_hold": len(rows),
        "U_traj": u_traj,
        "sum_U_single": float(np.nansum(u_singles)),
        "interaction_gap": float(u_traj - np.nansum(u_singles)),
        "holds": rows,
    }


def pool_at_reset(scenario: int) -> np.ndarray:
    env = auditor_env_reset(scenario)
    return generate_pool(env, seed_key=int(scenario) * 1009 + 0)


def auditor_env_reset(scenario: int):
    from wt.l3b.env import L3BEnv

    env = L3BEnv()
    env.reset(seed=int(scenario))
    return env


def reset_state(scenario: int) -> dict:
    from wt.l3b.env import L3BEnv

    env = L3BEnv()
    env.reset(seed=int(scenario))
    return {
        "scenario": int(scenario),
        "t": int(env.t),
        "env": snapshot_env(env),
        "last_arrival": np.zeros(env.config.num_buoys, dtype=np.float32),
        "last_positions": np.asarray(env.uav_positions, dtype=np.float32).copy(),
        "last_movement": np.zeros((env.config.num_uavs, 2), dtype=np.float64),
    }


def candidate_type(gamma: list[int]) -> str:
    idle = int(np.sum(np.asarray(gamma, dtype=np.int64)[:6] < 0))
    if idle >= 5:
        return "mostly_idle"
    if idle >= 3:
        return "half_idle"
    if idle <= 1:
        return "busy"
    return "mixed"


def official_payloads() -> dict:
    return {"OLD": 10912500.0, "ABS": 8440000.0, "PREFIX": 11287500.0, "EXEC": 8692500.0}


def analyze(natural: list[dict], counter: list[dict], first_pool: list[dict], later_oracle: list[dict]) -> dict:
    by_fam = {}
    for fam in ("OLD", "ABS", "PREFIX", "EXEC"):
        nat = [r for r in natural if r["family"] == fam]
        ctf = [r for r in counter if r["family"] == fam]
        fp = [r for r in first_pool if r["family"] == fam]
        if not nat:
            continue
        sum_u = [float(r["sum_U_single"]) for r in nat]
        traj = [float(r["U_traj"]) for r in nat]
        gap = [float(r["interaction_gap"]) for r in nat]
        ratio = [float(r["U_traj"]) / max(abs(float(r["sum_U_single"])), 1.0) for r in nat]
        abs_err = [abs(float(r["U_traj"]) - float(r["sum_U_single"])) for r in nat]
        rel_err = [abs(float(r["U_traj"]) - float(r["sum_U_single"])) / max(abs(float(r["U_traj"])), 1.0) for r in nat]
        hold_u = []
        hold_act = []
        types = {}
        for r in nat:
            for h in r["holds"]:
                hold_u.append({"hold": h["hold"], "U_single": h["U_single"], "U_hold_actual": h["U_hold_actual"]})
                hold_act.append(h)
                types.setdefault(candidate_type(h["gamma"]), []).append(float(h["U_single"]))
        hold_groups = {}
        for idx in range(4):
            vals = [x["U_single"] for x in hold_u if int(x["hold"]) == idx]
            acts = [x["U_hold_actual"] for x in hold_u if int(x["hold"]) == idx]
            hold_groups[str(idx)] = {"U_single": summarize(vals), "U_hold_actual": summarize(acts)}
        first = [r for r in nat]
        first_agree = []
        first_regret = []
        first_pct = []
        first_nr = []
        for r in first:
            h0 = r["holds"][0]
            pool_rows = [p for p in fp if int(p["scenario"]) == int(r["scenario"])]
            if not pool_rows:
                continue
            u_pool = np.array([float(p["U_LL"]) for p in pool_rows], dtype=np.float64)
            hashes = [p["hash"] for p in pool_rows]
            best = float(np.max(u_pool))
            iu = int(np.argmax(u_pool))
            chosen = h0["hash"]
            in_pool = chosen in hashes
            u_ch = float(h0["U_single"])
            first_agree.append(float(in_pool and hashes.index(chosen) == iu))
            first_regret.append((best - u_ch) / max(abs(best), 1.0))
            first_nr.append((best - u_ch) / (abs(best) + UTILITY_EPS))
            first_pct.append(float(np.mean(u_pool <= u_ch)))
        ctf_corr = {}
        if ctf:
            by_sc = {}
            for row in ctf:
                by_sc.setdefault(int(row["scenario"]), []).append(row)
            spears, kends, top1, regs = [], [], [], []
            for rows in by_sc.values():
                u = np.array([float(x["U_LL_first"]) for x in rows], dtype=np.float64)
                t = np.array([float(x["U_traj"]) for x in rows], dtype=np.float64)
                spears.append(spearman(u, t))
                kends.append(kendall_tau(u, t))
                top1.append(float(int(np.argmax(u)) == int(np.argmax(t))))
                best_t = float(np.max(t))
                regs.append((best_t - float(t[int(np.argmax(u))])) / max(abs(best_t), 1.0))
            ctf_corr = {
                "spearman": summarize(spears),
                "kendall": summarize(kends),
                "top1_agree": summarize(top1),
                "regret": summarize(regs),
                "n_scenario": len(by_sc),
            }
        later = [r for r in later_oracle if r["family"] == fam]
        later_stats = {}
        if later:
            later_stats = {
                "top1_agree": summarize([float(r["top1_agree"]) for r in later]),
                "top3_overlap": summarize([float(r["top3_overlap"]) for r in later]),
                "regret": summarize([float(r["regret"]) for r in later]),
                "percentile": summarize([float(r["percentile"]) for r in later]),
            }
        by_fam[fam] = {
            "n_traj": len(nat),
            "U_traj": summarize(traj),
            "sum_U_single": summarize(sum_u),
            "pearson": pearson(sum_u, traj),
            "spearman": spearman(sum_u, traj),
            "kendall": kendall_tau(sum_u, traj),
            "ratio": summarize(ratio),
            "abs_error": summarize(abs_err),
            "rel_error": summarize(rel_err),
            "interaction_gap": summarize(gap),
            "gap_std": float(np.std(gap, ddof=1)) if len(gap) > 1 else float("nan"),
            "by_hold": hold_groups,
            "by_candidate_type_U_single": {k: summarize(v) for k, v in types.items()},
            "first_hold_top1_agree": summarize(first_agree),
            "first_hold_regret": summarize(first_regret),
            "first_hold_DecodeNR": summarize(first_nr),
            "first_hold_percentile": summarize(first_pct),
            "later_hold_oracle": later_stats,
            "counterfactual": ctf_corr,
            "official_closed_loop": official_payloads()[fam],
        }
    return by_fam


def classify_case(summary: dict) -> dict:
    exec_s = summary.get("EXEC", {})
    spear = float(exec_s.get("spearman", float("nan")))
    gap = exec_s.get("interaction_gap", {})
    gap_mean = float(gap.get("mean", float("nan"))) if isinstance(gap, dict) else float("nan")
    traj_mean = float(exec_s.get("U_traj", {}).get("mean", float("nan"))) if isinstance(exec_s.get("U_traj"), dict) else float("nan")
    rel_gap = abs(gap_mean) / max(abs(traj_mean), 1.0)
    ctf = exec_s.get("counterfactual", {})
    ctf_sp = float(ctf.get("spearman", {}).get("mean", float("nan"))) if isinstance(ctf.get("spearman"), dict) else float("nan")
    cases = []
    if np.isfinite(spear) and spear >= 0.80 and rel_gap < 0.15:
        cases.append("A")
    if np.isfinite(spear) and 0.30 <= spear < 0.80 and rel_gap >= 0.10:
        cases.append("B")
    if np.isfinite(spear) and spear < 0.30:
        cases.append("C")
    if np.isfinite(ctf_sp) and ctf_sp < 0.40:
        cases.append("D")
    if not cases:
        if np.isfinite(ctf_sp) and ctf_sp < 0.55:
            cases.append("D")
        elif np.isfinite(spear) and spear >= 0.50:
            cases.append("B")
        else:
            cases.append("C")
    return {"primary": cases[0], "all": cases, "exec_spearman": spear, "rel_gap": rel_gap, "ctf_spearman": ctf_sp}


def write_report(summary: dict, case: dict) -> str:
    lines = []
    lines.append("# HTW Phase-3 Trajectory Utility Consistency Audit")
    lines.append("")
    lines.append("本报告只做离线评估。未训练、未改模型、未改 checkpoint、未改 pool / decoder / Low-Level / reward / collision、未改 Phase-3。")
    lines.append("")
    lines.append("定义：`U_traj = payload_bits(end) - payload_bits(reset)`，与 `evaluate_design` 相同。`U_LL` 为官方 `shadow_h15` 的 `payload_increment`（5 个 Low-Level seed 均值）。设计集 16 个 scenario，High-Level 各取正式 seed0（OLD 为冻结 util seed0），闭环执行 Low-Level seed=0。")
    lines.append("")
    lines.append(f"判定：**主结论为 Case {case['primary']}**（同时触发 {case['all']}）。EXEC 的 sum-U_single vs U_traj Spearman={case['exec_spearman']:.3f}，|interaction_gap|/U_traj={case['rel_gap']:.3f}，第一 hold counterfactual Spearman={case['ctf_spearman']:.3f}。")
    lines.append("")
    lines.append("## 方法对照")
    lines.append("")
    lines.append("| method | first-hold DecodeNR | first-hold Spearman (U_LL vs U_traj) | trajectory correlation | interaction gap mean | closed-loop payload (official) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for fam in ("OLD", "ABS", "PREFIX", "EXEC"):
        s = summary[fam]
        nr = s["first_hold_DecodeNR"]["median"]
        ctf_sp = s["counterfactual"].get("spearman", {}).get("mean", float("nan"))
        lines.append(
            f"| {fam} | {nr:.4f} | {ctf_sp:.4f} | {s['spearman']:.4f} | {s['interaction_gap']['mean']:.3e} | {s['official_closed_loop']:.3e} |"
        )
    lines.append("")
    for fam, s in summary.items():
        lines.append(f"## {fam}")
        lines.append("")
        lines.append(f"- n_traj={s['n_traj']}；U_traj mean={s['U_traj']['mean']:.3e}；sum U_single mean={s['sum_U_single']['mean']:.3e}")
        lines.append(f"- sum vs U_traj：Pearson={s['pearson']:.3f}，Spearman={s['spearman']:.3f}，Kendall={s['kendall']:.3f}")
        lines.append(f"- ratio median={s['ratio']['median']:.3f}；abs error median={s['abs_error']['median']:.3e}；rel error median={s['rel_error']['median']:.3f}")
        g = s["interaction_gap"]
        lines.append(f"- interaction_gap mean={g['mean']:.3e} median={g['median']:.3e} std={s['gap_std']:.3e} p10={g['p10']:.3e} p90={g['p90']:.3e}")
        lines.append(f"- first-hold vs pool oracle：top-1={s['first_hold_top1_agree']['mean']:.3f}，DecodeNR median={s['first_hold_DecodeNR']['median']:.3f}，percentile median={s['first_hold_percentile']['median']:.3f}")
        if s["counterfactual"]:
            c = s["counterfactual"]
            lines.append(f"- counterfactual 第一 hold：Spearman={c['spearman']['mean']:.3f}，Kendall={c['kendall']['mean']:.3f}，top-1={c['top1_agree']['mean']:.3f}，regret={c['regret']['mean']:.3f}")
        if s["later_hold_oracle"]:
            L = s["later_hold_oracle"]
            lines.append(f"- later-hold oracle：top-1={L['top1_agree']['mean']:.3f}，top-3={L['top3_overlap']['mean']:.3f}，regret={L['regret']['mean']:.3f}，percentile={L['percentile']['mean']:.3f}")
        lines.append("")
    lines.append("## 五个问题")
    lines.append("")
    lines.append("见文末结论段。本阶段停止。不实现新方法，不训练。")
    lines.append("")
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, design = load_pools()
    auditor = Auditor()
    highs = {f"{s['family']}:{s['seed']}": load_high(s["path"]) for s in HL_SPECS}

    first_pool_path = OUTPUT / "first_hold_pool.csv"
    first_pool = []
    if first_pool_path.is_file():
        with first_pool_path.open(encoding="utf-8", newline="") as handle:
            first_pool = list(csv.DictReader(handle))
            for row in first_pool:
                row["U_LL"] = float(row["U_LL"])
                row["scenario"] = int(float(row["scenario"]))
    else:
        print("first-hold U_LL on official L=24 pool at t=0", flush=True)
        for scenario in design:
            state = reset_state(int(scenario))
            env = auditor_env_reset(int(scenario))
            pool = generate_pool(env, seed_key=int(scenario) * 1009 + 0)
            for index, gamma in enumerate(pool):
                u = auditor.u_ll(state, gamma)
                row = {
                    "family": "SHARED",
                    "scenario": int(scenario),
                    "cand": int(index),
                    "hash": assignment_hash(gamma),
                    "gamma": json.dumps([int(x) for x in gamma[:6]]),
                    "U_LL": u,
                    "idle": float(np.mean(gamma[:6] < 0)),
                }
                first_pool.append(row)
                print(f"  pool scenario={scenario} cand={index+1}/24 U={u:.1f}", flush=True)
            auditor.save_cache()
        _csv(first_pool_path, first_pool)

    shared_pool = {}
    for row in first_pool:
        if row.get("family") not in {None, "SHARED"} and row.get("family") != "SHARED":
            continue
        shared_pool.setdefault(int(row["scenario"]), []).append(row)
    if not shared_pool:
        for row in first_pool:
            shared_pool.setdefault(int(row["scenario"]), []).append(row)

    natural_path = OUTPUT / "natural_trajectories.json"
    if natural_path.is_file():
        natural = json.loads(natural_path.read_text(encoding="utf-8"))
    else:
        natural = []
        for spec in HL_SPECS:
            high = highs[f"{spec['family']}:{spec['seed']}"]
            worker = make_stack(high, LL_RUN)
            for scenario in design:
                print(f"natural {spec['family']} scenario={scenario}", flush=True)
                rec = run_episode(worker, int(scenario), auditor, compute_ull=True)
                rec["family"] = spec["family"]
                rec["high_seed"] = spec["seed"]
                rec["low_seed"] = LL_RUN
                h0 = rec["holds"][0]
                pool_rows = shared_pool[int(scenario)]
                hashes = [p["hash"] for p in pool_rows]
                rec["first_in_pool"] = bool(h0["hash"] in hashes)
                rec["first_hash"] = h0["hash"]
                natural.append(rec)
                auditor.save_cache()
            del worker
        _dump(natural_path, natural)

    # attach shared pool rows per family for first-hold oracle table
    first_for_family = []
    for spec in HL_SPECS:
        for row in first_pool:
            item = dict(row)
            item["family"] = spec["family"]
            first_for_family.append(item)

    later_path = OUTPUT / "later_hold_oracle.csv"
    if later_path.is_file():
        later_oracle = list(csv.DictReader(later_path.open(encoding="utf-8", newline="")))
        for row in later_oracle:
            for key in ("top1_agree", "top3_overlap", "regret", "percentile", "U_chosen", "U_best"):
                if key in row:
                    row[key] = float(row[key])
    else:
        later_oracle = []
        for spec in HL_SPECS:
            if spec["family"] not in ORACLE_FAMILIES:
                continue
            high = highs[f"{spec['family']}:{spec['seed']}"]
            worker = make_stack(high, LL_RUN)
            for scenario in design:
                info = worker.reset(int(scenario))
                hold_i = -1
                while True:
                    if int(worker.pending) <= 0:
                        state = capture_state(worker, int(scenario))
                        worker._refresh_option(info)
                        gamma = np.asarray(worker.option.association, dtype=np.int64).copy()
                        hold_i += 1
                        if hold_i >= 1:
                            pool = generate_pool(worker.env, seed_key=int(scenario) * 1009 + int(worker.env.t))
                            u_pool = [auditor.u_ll_fast(state, row) for row in pool]
                            u_ch = auditor.u_ll_fast(state, gamma)
                            u_arr = np.asarray(u_pool, dtype=np.float64)
                            best = float(np.max(u_arr))
                            top3 = set(np.argsort(-u_arr)[:3].tolist())
                            hashes = [assignment_hash(row) for row in pool]
                            chosen_idx = hashes.index(assignment_hash(gamma)) if assignment_hash(gamma) in hashes else -1
                            later_oracle.append(
                                {
                                    "family": spec["family"],
                                    "scenario": int(scenario),
                                    "hold": hold_i,
                                    "t": int(worker.env.t),
                                    "in_pool": chosen_idx >= 0,
                                    "top1_agree": float(chosen_idx == int(np.argmax(u_arr))),
                                    "top3_overlap": float(chosen_idx in top3) if chosen_idx >= 0 else 0.0,
                                    "regret": (best - u_ch) / max(abs(best), 1.0),
                                    "percentile": float(np.mean(u_arr <= u_ch)),
                                    "U_chosen": u_ch,
                                    "U_best": best,
                                }
                            )
                            print(f"  later oracle {spec['family']} sc={scenario} hold={hold_i} pct={later_oracle[-1]['percentile']:.2f}", flush=True)
                            auditor.save_cache()
                    info, rec = worker.step(info, deterministic=True, zero_residual=False)
                    if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                        break
            del worker
        _csv(later_path, later_oracle)

    counter_path = OUTPUT / "counterfactual.json"
    if counter_path.is_file():
        counter = json.loads(counter_path.read_text(encoding="utf-8"))
    else:
        counter = []
        for spec in HL_SPECS:
            high = highs[f"{spec['family']}:{spec['seed']}"]
            worker = make_override_stack(high, LL_RUN)
            for scenario in design:
                env = auditor_env_reset(int(scenario))
                pool = generate_pool(env, seed_key=int(scenario) * 1009 + 0)
                state0 = reset_state(int(scenario))
                for index, gamma in enumerate(pool):
                    worker.set_first_assignment(gamma)
                    rec = run_episode(worker, int(scenario), auditor, compute_ull=False)
                    u_ll = auditor.u_ll(state0, gamma)
                    counter.append(
                        {
                            "family": spec["family"],
                            "high_seed": spec["seed"],
                            "scenario": int(scenario),
                            "cand": int(index),
                            "hash": assignment_hash(gamma),
                            "U_LL_first": u_ll,
                            "U_traj": rec["U_traj"],
                            "n_hold": rec["n_hold"],
                            "gamma2": rec["holds"][1]["hash"] if len(rec["holds"]) > 1 else "",
                            "gamma3": rec["holds"][2]["hash"] if len(rec["holds"]) > 2 else "",
                            "gamma4": rec["holds"][3]["hash"] if len(rec["holds"]) > 3 else "",
                        }
                    )
                    print(f"cf {spec['family']} sc={scenario} cand={index+1}/24 U_LL={u_ll:.1f} U_traj={rec['U_traj']:.1f}", flush=True)
                    if len(counter) % 8 == 0:
                        _dump(counter_path, counter)
                        auditor.save_cache()
            del worker
        _dump(counter_path, counter)

    auditor.save_cache()
    summary = analyze(natural, counter, first_for_family, later_oracle)
    case = classify_case(summary)
    _dump(OUTPUT / "TRAJ_AUDIT_SUMMARY.json", {"summary": summary, "case": case})
    body = write_report(summary, case)
    # expand answers after numbers exist
    extra = _answers(summary, case)
    report = body.replace("见文末结论段。本阶段停止。不实现新方法，不训练。", extra)
    (OUTPUT / "HTW_Phase3_TrajectoryUtility_Audit_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_Phase3_TrajectoryUtility_Audit_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps({"case": case, "exec_spearman": summary["EXEC"]["spearman"], "exec_ctf": summary["EXEC"]["counterfactual"]}, indent=2), flush=True)
    return {"summary": summary, "case": case}


def _answers(summary: dict, case: dict) -> str:
    e = summary["EXEC"]
    p = summary["PREFIX"]
    ctf_e = e["counterfactual"]
    lines = [
        f"1. **U_LL 能否预测最终 4-hold closed-loop utility？** EXEC 上 sum_t U_single 与 U_traj 的 Spearman={e['spearman']:.3f}，Pearson={e['pearson']:.3f}。第一 hold 的 U_LL 对 counterfactual U_traj 的 Spearman={ctf_e.get('spearman', {}).get('mean', float('nan')):.3f}。",
        f"2. **是否存在显著 interaction？** EXEC interaction_gap mean={e['interaction_gap']['mean']:.3e}，median={e['interaction_gap']['median']:.3e}，p10={e['interaction_gap']['p10']:.3e}，p90={e['interaction_gap']['p90']:.3e}。相对 U_traj 的 |gap| 比例={case['rel_gap']:.3f}。",
        f"3. **第一 hold oracle 与最终 4-hold 收益是否一致？** counterfactual top-1={ctf_e.get('top1_agree', {}).get('mean', float('nan')):.3f}，regret={ctf_e.get('regret', {}).get('mean', float('nan')):.3f}。greedy 相对 pool 的 first-hold DecodeNR median={e['first_hold_DecodeNR']['median']:.3f}，percentile median={e['first_hold_percentile']['median']:.3f}。",
        f"4. **EXEC 失败机制：** 主 Case {case['primary']}。pool 内 Phase-3 已显示中等 U_LL 排序；本审计看的是单 hold 标签与真实 4-hold 轨迹。PREFIX 对照：trajectory Spearman={p['spearman']:.3f}，counterfactual Spearman={p['counterfactual'].get('spearman', {}).get('mean', float('nan')):.3f}，official payload {p['official_closed_loop']:.3e} vs EXEC {e['official_closed_loop']:.3e}。",
        "5. **下一阶段应优化什么：** 由 Case 决定。若 D/C 成立，应先验证 trajectory-level / first-hold credit，而不是再改 candidate-level q。若 A 成立，重点在 greedy decoder。若 B 成立，需要 trajectory-conditioned utility。",
        "",
        "**下一阶段只验证一个假设（不实现）：** 第一 hold 的官方 U_LL 排序与固定该 hold 后由 greedy 走完的 U_traj 不同序；因此 Phase-3 的 candidate-level 监督对象不是闭环目标。最小验证已由本 counterfactual 给出，下一阶段若继续，应在冻结模型上定义轨迹级标签，而不是新 loss。",
        "",
        "本阶段停止。不要训练。不要实现新方法。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
