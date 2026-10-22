"""Phase-4A: export frozen first-hold hindsight credit. No training. No reroll."""
from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.integration_phase1.stats import spearman, summarize
from wt.lowlevel_v2.htw_phase4a_hindsight_credit.protocol import (
    ALIGN_ATOL,
    CKPT_ID,
    EPISODE_HORIZON,
    FAMS,
    FIG,
    HOLD_STEPS,
    LL_RUN,
    LOCKED_EXEC,
    LOW_SEEDS,
    N_HOLD,
    N_SCENARIO,
    OUTPUT,
    POOL_SIZE,
    PRIMARY,
    REPORT_NAME,
    ROOT,
    SOURCE,
    UTILITY_EPS,
)

GEOM_KEYS = ("idle_count", "mean_uav_target_dist", "n_in_range", "n_unique_buoys", "assigned_queue_mean")


def reset_env(scenario: int) -> L3BEnv:
    env = L3BEnv()
    env.reset(seed=int(scenario))
    return env


def first_hold_geom(env: L3BEnv, gamma) -> dict[str, float]:
    chosen = np.asarray(gamma, dtype=np.int64)
    dists = []
    in_range = 0
    q_assigned = []
    for uav, buoy in enumerate(chosen[: env.config.num_uavs]):
        if int(buoy) >= 0:
            d = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
            dists.append(d)
            in_range += int(env._in_range(uav, int(buoy)))
            q_assigned.append(float(env.queue[int(buoy)]))
    return {
        "idle_count": float(np.sum(chosen[: env.config.num_uavs] < 0)),
        "mean_uav_target_dist": float(np.mean(dists)) if dists else 0.0,
        "n_in_range": float(in_range),
        "n_unique_buoys": float(len({int(b) for b in chosen[: env.config.num_uavs] if int(b) >= 0})),
        "assigned_queue_mean": float(np.mean(q_assigned)) if q_assigned else 0.0,
    }


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


def ext_sum(values) -> dict:
    out = summarize(values)
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    out["std"] = float(np.std(a, ddof=1)) if a.size > 1 else float("nan")
    out["min"] = float(np.min(a)) if a.size else float("nan")
    out["max"] = float(np.max(a)) if a.size else float("nan")
    return out


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_gamma(text: str) -> list[int]:
    return [int(x) for x in ast.literal_eval(text)]


def load_frozen() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    for key, path in SOURCE.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Phase-3 artifact {key}: {path}")
    cand = load_csv(SOURCE["candidates"])
    inter = load_csv(SOURCE["interventions"])
    pool = load_csv(SOURCE["pool"])
    cf = json.loads(SOURCE["counterfactual"].read_text(encoding="utf-8"))
    return cand, inter, pool, cf


def build_dataset(cand: list[dict], pool: list[dict], cf: list[dict]) -> list[dict]:
    pool_by: dict[tuple[int, str], dict] = {}
    for row in pool:
        sc = int(float(row["scenario"]))
        pool_by[(sc, row["hash"])] = row
    later: dict[tuple[str, int, int], dict] = {}
    for row in cf:
        later[(row["family"], int(row["scenario"]), int(row["cand"]))] = row

    env_cache: dict[int, object] = {}
    geom_cache: dict[tuple[int, str], dict] = {}
    rows = []
    for row in cand:
        fam = row["method"]
        sc = int(float(row["scenario"]))
        cid = int(float(row["cand"]))
        digest = row["first_hold"]
        pool_row = pool_by.get((sc, digest))
        if pool_row is None:
            raise RuntimeError(f"pool miss scenario={sc} hash={digest}")
        later_row = later.get((fam, sc, cid))
        if later_row is None:
            raise RuntimeError(f"counterfactual miss family={fam} scenario={sc} cand={cid}")
        u = float(row["first_hold_U_single"])
        t = float(row["final_U_traj"])
        f = float(row["F"])
        gamma = parse_gamma(pool_row["gamma"])
        if sc not in env_cache:
            env_cache[sc] = reset_env(sc)
        gkey = (sc, digest)
        if gkey not in geom_cache:
            geom_cache[gkey] = first_hold_geom(env_cache[sc], gamma)
        geom = geom_cache[gkey]
        c_abs = f
        c_ratio = c_abs / (abs(u) + float(UTILITY_EPS))
        rows.append(
            {
                "family": fam,
                "method": fam,
                "scenario_id": sc,
                "state_id": sc,
                "candidate_id": cid,
                "seed": int(LL_RUN),
                "u_single_seeds": ",".join(str(s) for s in LOW_SEEDS),
                "U_single": u,
                "U_traj": t,
                "C_abs": c_abs,
                "C_ratio": c_ratio,
                "F": f,
                "first_hold": digest,
                "hold2": later_row.get("gamma2", ""),
                "hold3": later_row.get("gamma3", ""),
                "hold4": later_row.get("gamma4", ""),
                "trajectory_length": int(later_row.get("n_hold", N_HOLD)),
                "assignment_hash": digest,
                "checkpoint_id": CKPT_ID,
                "gamma": json.dumps(gamma),
                "pool_U_LL": float(pool_row["U_LL"]),
                "cf_U_LL_first": float(later_row["U_LL_first"]),
                "cf_U_traj": float(later_row["U_traj"]),
                "is_local_oracle": int(float(row["is_local_oracle"])),
                "is_traj_oracle": int(float(row["is_traj_oracle"])),
                "is_natural": int(float(row["is_natural"])),
                "episode_horizon": EPISODE_HORIZON,
                "hold_steps": HOLD_STEPS,
                **{k: float(geom[k]) for k in GEOM_KEYS},
            }
        )
    rows.sort(key=lambda r: (FAMS.index(r["family"]) if r["family"] in FAMS else r["family"], r["scenario_id"], r["candidate_id"]))
    return rows


def unique_ok(rows: list[dict]) -> dict:
    keys = [(r["family"], r["scenario_id"], r["candidate_id"]) for r in rows]
    hashes = [(r["family"], r["scenario_id"], r["assignment_hash"]) for r in rows]
    return {
        "n": len(rows),
        "unique_family_scenario_cand": len(set(keys)) == len(keys),
        "unique_family_scenario_hash": len(set(hashes)) == len(hashes),
        "n_family": len({r["family"] for r in rows}),
        "n_scenario": len({r["scenario_id"] for r in rows}),
        "n_cand_per_state": sorted({sum(1 for r in rows if r["family"] == fam and r["scenario_id"] == sc) for fam in {x["family"] for x in rows} for sc in {x["scenario_id"] for x in rows if x["family"] == fam}}),
    }


def alignment(rows: list[dict], inter: list[dict]) -> dict:
    u_pool = [abs(r["U_single"] - r["pool_U_LL"]) for r in rows]
    u_cf = [abs(r["U_single"] - r["cf_U_LL_first"]) for r in rows]
    t_cf = [abs(r["U_traj"] - r["cf_U_traj"]) for r in rows]
    ident = [abs(r["U_traj"] - (r["U_single"] + r["C_abs"])) for r in rows]
    f_copy = [abs(r["C_abs"] - r["F"]) for r in rows]
    cond = {}
    for fam in FAMS:
        cond[fam] = {}
        for name in ("NATURAL", "LOCAL-ORACLE", "TRAJ-ORACLE"):
            vals = [float(r["final_U_traj"]) for r in inter if r["method"] == fam and r["condition"] == name]
            cond[fam][name] = ext_sum(vals)
    exec_ok = True
    exec_diff = {}
    for name, locked in LOCKED_EXEC.items():
        got = cond[PRIMARY][name]["mean"]
        exec_diff[name] = {"locked": locked, "got": got, "abs_diff": abs(got - locked)}
        if abs(got - locked) > ALIGN_ATOL:
            exec_ok = False
    local_from_table = []
    traj_from_table = []
    for fam in FAMS:
        by = defaultdict(list)
        for r in rows:
            if r["family"] == fam:
                by[r["scenario_id"]].append(r)
        for items in by.values():
            iu = max(items, key=lambda x: x["U_single"])
            it = max(items, key=lambda x: x["U_traj"])
            local_from_table.append((fam, iu["U_traj"]))
            traj_from_table.append((fam, it["U_traj"]))
    exec_local_rebuild = float(np.mean([v for fam, v in local_from_table if fam == PRIMARY]))
    exec_traj_rebuild = float(np.mean([v for fam, v in traj_from_table if fam == PRIMARY]))
    return {
        "max_abs_U_single_vs_pool_U_LL": float(np.max(u_pool)),
        "max_abs_U_single_vs_cf_U_LL_first": float(np.max(u_cf)),
        "max_abs_U_traj_vs_cf": float(np.max(t_cf)),
        "max_abs_U_traj_minus_U_single_C_abs": float(np.max(ident)),
        "max_abs_C_abs_minus_F": float(np.max(f_copy)),
        "identity_ok": float(np.max(ident)) <= ALIGN_ATOL,
        "phase3_table_ok": float(np.max(u_pool + u_cf + t_cf + f_copy)) <= ALIGN_ATOL,
        "conditions": cond,
        "exec_locked": exec_diff,
        "exec_locked_ok": exec_ok,
        "exec_local_rebuild": exec_local_rebuild,
        "exec_traj_rebuild": exec_traj_rebuild,
        "exec_local_rebuild_ok": abs(exec_local_rebuild - LOCKED_EXEC["LOCAL-ORACLE"]) <= ALIGN_ATOL,
        "exec_traj_rebuild_ok": abs(exec_traj_rebuild - LOCKED_EXEC["TRAJ-ORACLE"]) <= ALIGN_ATOL,
    }


def grouped(rows: list[dict], family: str) -> dict[int, list[dict]]:
    by = defaultdict(list)
    for r in rows:
        if r["family"] == family:
            by[r["scenario_id"]].append(r)
    return dict(sorted(by.items()))


def rank_block(items: list[dict], xkey: str, ykey: str) -> float:
    return spearman([r[xkey] for r in items], [r[ykey] for r in items])


def selection_stats(rows: list[dict], family: str) -> dict:
    by = grouped(rows, family)
    overlap = []
    r_local = []
    u_loc = []
    u_trj = []
    for items in by.values():
        loc = max(items, key=lambda r: r["U_single"])
        trj = max(items, key=lambda r: r["U_traj"])
        overlap.append(float(loc["assignment_hash"] == trj["assignment_hash"]))
        best = float(trj["U_traj"])
        r_local.append(float(loc["U_traj"]) / best if abs(best) > 0 else float("nan"))
        u_loc.append(float(loc["U_traj"]))
        u_trj.append(best)
    return {
        "n_state": len(by),
        "argmax_overlap": ext_sum(overlap),
        "R_local": ext_sum(r_local),
        "U_traj_argmax_U_single": ext_sum(u_loc),
        "max_U_traj": ext_sum(u_trj),
    }


def variance_split(rows: list[dict], family: str, key: str = "C_abs") -> dict:
    by = grouped(rows, family)
    vals = np.array([r[key] for r in rows if r["family"] == family], dtype=np.float64)
    means = []
    within = []
    for items in by.values():
        a = np.array([r[key] for r in items], dtype=np.float64)
        means.append(float(np.mean(a)))
        if a.size > 1:
            within.append(float(np.var(a, ddof=1)))
    between = float(np.var(np.asarray(means, dtype=np.float64), ddof=1)) if len(means) > 1 else float("nan")
    within_mean = float(np.mean(within)) if within else float("nan")
    total = float(np.var(vals, ddof=1)) if vals.size > 1 else float("nan")
    return {
        "total_var": total,
        "within_state_var_mean": within_mean,
        "between_state_var": between,
        "within_share": within_mean / total if total and total > 0 else float("nan"),
        "between_share": between / total if total and total > 0 else float("nan"),
        "n_state": len(by),
    }


def descriptive(rows: list[dict], family: str) -> dict:
    items = [r for r in rows if r["family"] == family]
    by = grouped(rows, family)
    pooled = {
        "spearman_U_single_U_traj": spearman([r["U_single"] for r in items], [r["U_traj"] for r in items]),
        "spearman_C_abs_U_traj": spearman([r["C_abs"] for r in items], [r["U_traj"] for r in items]),
        "spearman_C_abs_identity": spearman([r["C_abs"] for r in items], [r["U_traj"] - r["U_single"] for r in items]),
    }
    within = {
        "spearman_U_single_U_traj": ext_sum([rank_block(v, "U_single", "U_traj") for v in by.values()]),
        "spearman_C_abs_U_traj": ext_sum([rank_block(v, "C_abs", "U_traj") for v in by.values()]),
        "spearman_C_abs_U_single": ext_sum([rank_block(v, "C_abs", "U_single") for v in by.values()]),
    }
    geom = {}
    for key in GEOM_KEYS:
        geom[key] = {
            "pooled_spearman_C": spearman([r[key] for r in items], [r["C_abs"] for r in items]),
            "within_spearman_C": ext_sum([rank_block(v, key, "C_abs") for v in by.values()]),
        }
    return {
        "n": len(items),
        "C_abs": ext_sum([r["C_abs"] for r in items]),
        "C_ratio": ext_sum([r["C_ratio"] for r in items]),
        "U_single": ext_sum([r["U_single"] for r in items]),
        "U_traj": ext_sum([r["U_traj"] for r in items]),
        "pooled": pooled,
        "within_state": within,
        "geometry_vs_C": geom,
        "var_C": variance_split(rows, family, "C_abs"),
        "var_U_traj": variance_split(rows, family, "U_traj"),
        "selection": selection_stats(rows, family),
    }


def figures(rows: list[dict], desc: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    exec_rows = [r for r in rows if r["family"] == PRIMARY]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist([r["C_abs"] / 1e6 for r in exec_rows], bins=16, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("EXEC C_abs (1e6 bits)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIG / "c_abs_hist_exec.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.scatter(
        [r["U_single"] / 1e6 for r in exec_rows],
        [r["U_traj"] / 1e6 for r in exec_rows],
        s=12,
        alpha=0.55,
        c="#4C78A8",
    )
    ax.set_xlabel("U_single (1e6)")
    ax.set_ylabel("U_traj (1e6)")
    fig.tight_layout()
    fig.savefig(FIG / "u_single_vs_u_traj_exec.png", dpi=140)
    plt.close(fig)

    by = grouped(rows, PRIMARY)
    names = list(by)
    w_u = [rank_block(by[s], "U_single", "U_traj") for s in names]
    w_c = [rank_block(by[s], "C_abs", "U_traj") for s in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = np.arange(len(names))
    ax.bar(xs - 0.18, w_u, width=0.36, label="within Spearman(U_single,U_traj)", color="#4C78A8")
    ax.bar(xs + 0.18, w_c, width=0.36, label="within Spearman(C_abs,U_traj)", color="#F58518")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s)[-3:] for s in names], rotation=60)
    ax.set_ylabel("Spearman")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "within_state_spearman_exec.png", dpi=140)
    plt.close(fig)


def _sci(x: float) -> str:
    return f"{float(x):.3e}"


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{float(x):.3f}"


def verdict(uniq: dict, align: dict, rows: list[dict]) -> dict:
    n_exec = sum(1 for r in rows if r["family"] == PRIMARY)
    ok = (
        uniq["unique_family_scenario_cand"]
        and uniq["unique_family_scenario_hash"]
        and align["identity_ok"]
        and align["phase3_table_ok"]
        and align["exec_locked_ok"]
        and align["exec_local_rebuild_ok"]
        and align["exec_traj_rebuild_ok"]
        and n_exec == N_SCENARIO * POOL_SIZE
        and uniq["n_scenario"] == N_SCENARIO
    )
    return {
        "gate": "PASS" if ok else "FAIL",
        "n_exec": n_exec,
        "ok": ok,
    }


def write_report(rows: list[dict], uniq: dict, align: dict, by_fam: dict, gate: dict) -> str:
    e = by_fam[PRIMARY]
    sel = e["selection"]
    varc = e["var_C"]
    seeds = ",".join(str(s) for s in LOW_SEEDS)
    lines = [
        "# HTW Phase-4A Hindsight Credit Dataset Audit",
        "",
        "本阶段只构建并审计 offline candidate-level hindsight credit。未训练，未改 Low-Level / Actor / Critic / reward / decoder / pool / checkpoint，未改 `U_single` / `U_traj`，未引入 predictor / timed subgoal / joint training / HIRO，未重跑 Phase-3C-3H，未调用 `traj_audit.run`。",
        "",
        f"**门控：{gate['gate']}**",
        "",
        "## 1. 数据来源",
        "",
        "优先复用已冻结 Phase-3 rollout，不重复闭环。",
        "",
        f"- pool: `{SOURCE['pool']}`",
        f"- counterfactual: `{SOURCE['counterfactual']}`",
        f"- Phase-3C candidates: `{SOURCE['candidates']}`",
        f"- Phase-3C interventions: `{SOURCE['interventions']}`",
        "",
        "`C_abs` 直接复制 Phase-3C 的 `F`。`U_single` 来自 `first_hold_U_single`（= `U_LL` / `U_LL_first`）。`U_traj` 来自 `final_U_traj`（= counterfactual `U_traj`）。",
        "",
        "## 2. Rollout protocol（冻结，未重跑）",
        "",
        "与 Phase-3 trajectory audit 同一条链：",
        "",
        "- first-hold: `generate_pool(env, seed_key=scenario*1009+0)` 的 24 个合法 Gamma",
        "- `U_single`: `Auditor.u_ll` 对 5 个 Low-Level seed 的 `shadow_h15` 取均值",
        "- 闭环: `FirstHoldOverrideWorker` 只覆盖 first-hold；hold 2/3/4 走正式 `decide()` / `decode_assignment()` joint greedy",
        "- Low-Level: `ckpt_seed0_update100.pt`，`deterministic=True`",
        "- `U_traj = payload_bits(end) - payload_bits(reset)`，horizon=48，hold=15，4 个 decision hold",
        "",
        "正式 NATURAL first-hold 是 High-Level joint greedy，不是 `argmax U_single`。EXEC 的 16 条 NATURAL 全部不在 24-pool 内。本数据集的 24 点是 pool 干预，不是 NATURAL 解码。",
        "",
        "## 3-5. 规模",
        "",
        f"- family: {uniq['n_family']}（{', '.join(FAMS)}）；主判断 **{PRIMARY}**",
        f"- state / scenario: {uniq['n_scenario']}（221000-221015）；`state_id = scenario_id`（t=0）",
        f"- 每 state candidate: {uniq['n_cand_per_state']}",
        f"- 总行数: {uniq['n']}；EXEC 行数: {gate['n_exec']}",
        f"- 闭环 seed: `seed={LL_RUN}`（`LL_RUN`）",
        f"- U_single seeds: {seeds}（均值，不另定义）",
        f"- checkpoint_id: `{CKPT_ID}`",
        "",
        "## 6. U_single / U_traj 对齐",
        "",
        f"- max |U_single - pool U_LL| = {align['max_abs_U_single_vs_pool_U_LL']:.3e}",
        f"- max |U_single - cf U_LL_first| = {align['max_abs_U_single_vs_cf_U_LL_first']:.3e}",
        f"- max |U_traj - cf U_traj| = {align['max_abs_U_traj_vs_cf']:.3e}",
        f"- max |C_abs - F| = {align['max_abs_C_abs_minus_F']:.3e}",
        f"- max |U_traj - (U_single+C_abs)| = {align['max_abs_U_traj_minus_U_single_C_abs']:.3e}",
        "",
        "| condition | locked EXEC | this table |",
        "|---|---:|---:|",
    ]
    for name, locked in LOCKED_EXEC.items():
        got = align["exec_locked"][name]["got"]
        lines.append(f"| {name} | {_sci(locked)} | {_sci(got)} |")
    lines.extend(
        [
            "",
            f"从表内 argmax 重建: LOCAL={_sci(align['exec_local_rebuild'])}，TRAJ={_sci(align['exec_traj_rebuild'])}。",
            "",
            "## 7. C_abs 分布（EXEC pool 384 点）",
            "",
            f"- mean={_sci(e['C_abs']['mean'])} median={_sci(e['C_abs']['median'])} std={_sci(e['C_abs']['std'])}",
            f"- p25={_sci(e['C_abs']['p25'])} p75={_sci(e['C_abs']['p75'])} min={_sci(e['C_abs']['min'])} max={_sci(e['C_abs']['max'])}",
            "",
            "## 8. C_ratio 分布（EXEC）",
            "",
            f"`C_ratio=(U_traj-U_single)/(|U_single|+{UTILITY_EPS})`，epsilon=Phase-3 `UTILITY_EPS`。",
            "",
            f"- mean={_fmt(e['C_ratio']['mean'])} median={_fmt(e['C_ratio']['median'])} std={_fmt(e['C_ratio']['std'])}",
            f"- p25={_fmt(e['C_ratio']['p25'])} p75={_fmt(e['C_ratio']['p75'])} min={_fmt(e['C_ratio']['min'])} max={_fmt(e['C_ratio']['max'])}",
            "",
            "C_ratio 的均值被 `U_single=0` 的行拉爆：此时分母=epsilon=1，C_ratio=C_abs。看分布请优先用 median / p25 / p75，不要用 mean 当主指标。",
            "",
            "## 9. Within-state 统计（EXEC）",
            "",
            f"- pooled Spearman(U_single, U_traj)={_fmt(e['pooled']['spearman_U_single_U_traj'])}",
            f"- pooled Spearman(C_abs, U_traj)={_fmt(e['pooled']['spearman_C_abs_U_traj'])}",
            f"- pooled Spearman(C_abs, U_traj-U_single)={_fmt(e['pooled']['spearman_C_abs_identity'])}（应为 1）",
            f"- within-state Spearman(U_single, U_traj) median={_fmt(e['within_state']['spearman_U_single_U_traj']['median'])} mean={_fmt(e['within_state']['spearman_U_single_U_traj']['mean'])}",
            f"- within-state Spearman(C_abs, U_traj) median={_fmt(e['within_state']['spearman_C_abs_U_traj']['median'])} mean={_fmt(e['within_state']['spearman_C_abs_U_traj']['mean'])}",
            f"- within-state Spearman(C_abs, U_single) median={_fmt(e['within_state']['spearman_C_abs_U_single']['median'])}",
            "",
            "这些是描述统计，不是模型精度。",
            "",
            "## 10. argmax U_single 到 U_traj recovery（EXEC）",
            "",
            "`R_local = U_traj(argmax U_single) / max_i U_traj(i)`，按 state 再平均。",
            "",
            f"- argmax overlap mean={_fmt(sel['argmax_overlap']['mean'])}",
            f"- R_local mean={_fmt(sel['R_local']['mean'])} median={_fmt(sel['R_local']['median'])}",
            f"- mean U_traj(argmax U_single)={_sci(sel['U_traj_argmax_U_single']['mean'])}",
            f"- mean max U_traj={_sci(sel['max_U_traj']['mean'])}",
            "",
            "这不是 predictor accuracy。它只回答：若仍按 U_single 选 first-hold，相对 pool 内最优 U_traj 丢掉多少。",
            "",
            "## 11. Between / within-state 方差（EXEC C_abs）",
            "",
            f"- total Var(C)={_sci(varc['total_var'])}",
            f"- mean within-state Var(C)={_sci(varc['within_state_var_mean'])}（share={_fmt(varc['within_share'])}）",
            f"- between-state Var(state-mean C)={_sci(varc['between_state_var'])}（share={_fmt(varc['between_share'])}）",
            "",
            "C 不是纯 state offset：组内仍有 candidate 变异。同时 C 也不是与 U_single 无关的独立量。",
            "",
            "## 7 补充：C 与 first-hold geometry（描述，非训练）",
            "",
        ]
    )
    for key in GEOM_KEYS:
        g = e["geometry_vs_C"][key]
        lines.append(
            f"- {key}: pooled Spearman(C)={_fmt(g['pooled_spearman_C'])}，within median={_fmt(g['within_spearman_C']['median'])}"
        )
    lines.extend(
        [
            "",
            "特征来自 t=0 `reset(seed=scenario)` 上的 first-hold 几何，不含未来 hold / U_traj / F。未拟合任何模型。",
            "",
            "## 12. 与 Phase-3 的一致性",
            "",
            "EXEC NATURAL / LOCAL / TRAJ 均值与 Phase-3 锁定值一致（允许显示精度）。`C_abs=F`。未改 utility。未覆写任何 Phase-3 文件。",
            "",
            "## 13. Rollout 随机性",
            "",
            "- Low-Level policy: `deterministic=True`，用 actor 均值，不采样。",
            "- High-Level decoder: `pick_candidate` 默认无 rng，平局走 `canonical_keys`。",
            "- 环境到达: `arrivals_at(t)` 由 t 决定，无逐步 RNG。reset 的 UAV y jitter 由 scenario seed 决定。",
            "- 本阶段没有重新 step。同一 candidate 的 U_traj 以冻结 JSON 为准。",
            "- 若将来重放: 必须固定 scenario seed、`LL_RUN=0`、同一 High-Level family；禁止与 shadow worker 共用同一个 env。",
            "",
            "## 14. 是否可直接用于下一阶段 credit learner",
            "",
            "可以作为 offline 监督标签: 键 `(family, scenario_id, candidate_id)` 唯一，`C_abs` 与 Phase-3 `F` 逐行相同，EXEC 三条件对齐。",
            "",
            "不能当作已经证明“学 C 一定比学 F / 学 U_traj 更容易”。本阶段没有训练。组内 Spearman(C, U_traj) 高，部分是因为 U_traj=U_single+C 的代数结构，不是新信息。下一阶段若做 learner，必须另开指令，且不能把 U_traj / 后续 hold 当 first-hold 特征。",
            "",
            "## PASS / FAIL",
            "",
            f"**{gate['gate']}**。判据: candidate → 冻结 Low-Level → 完整 4-hold trajectory → U_traj → `C_abs=U_traj-U_single` 的数据链由 Phase-3 冻结表稳定复现；EXEC NATURAL/LOCAL/TRAJ 与锁定值一致；identity 逐行成立。",
            "",
            "本阶段停止。不要训练。不要改 planner。不要进入 Phase-4B。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cand, inter, pool, cf = load_frozen()
    rows = build_dataset(cand, pool, cf)
    uniq = unique_ok(rows)
    align = alignment(rows, inter)
    by_fam = {fam: descriptive(rows, fam) for fam in FAMS}
    gate = verdict(uniq, align, rows)
    figures(rows, by_fam[PRIMARY])
    export_rows = []
    for r in rows:
        export_rows.append(
            {
                "family": r["family"],
                "scenario_id": r["scenario_id"],
                "state_id": r["state_id"],
                "candidate_id": r["candidate_id"],
                "seed": r["seed"],
                "U_single": r["U_single"],
                "U_traj": r["U_traj"],
                "C_abs": r["C_abs"],
                "C_ratio": r["C_ratio"],
                "first_hold": r["first_hold"],
                "hold2": r["hold2"],
                "hold3": r["hold3"],
                "hold4": r["hold4"],
                "trajectory_length": r["trajectory_length"],
                "assignment_hash": r["assignment_hash"],
                "checkpoint_id": r["checkpoint_id"],
                "gamma": r["gamma"],
                "u_single_seeds": r["u_single_seeds"],
                "F": r["F"],
                "is_local_oracle": r["is_local_oracle"],
                "is_traj_oracle": r["is_traj_oracle"],
                "is_natural": r["is_natural"],
            }
        )
    _csv(OUTPUT / "phase4a_hindsight_credit.csv", export_rows)
    summary = {
        "gate": gate,
        "unique": uniq,
        "alignment": {
            "max_abs_U_single_vs_pool_U_LL": align["max_abs_U_single_vs_pool_U_LL"],
            "max_abs_U_single_vs_cf_U_LL_first": align["max_abs_U_single_vs_cf_U_LL_first"],
            "max_abs_U_traj_vs_cf": align["max_abs_U_traj_vs_cf"],
            "max_abs_U_traj_minus_U_single_C_abs": align["max_abs_U_traj_minus_U_single_C_abs"],
            "max_abs_C_abs_minus_F": align["max_abs_C_abs_minus_F"],
            "identity_ok": align["identity_ok"],
            "phase3_table_ok": align["phase3_table_ok"],
            "exec_locked": align["exec_locked"],
            "exec_locked_ok": align["exec_locked_ok"],
            "exec_local_rebuild": align["exec_local_rebuild"],
            "exec_traj_rebuild": align["exec_traj_rebuild"],
        },
        "families": by_fam,
        "protocol": {
            "reused_phase3_rollouts": True,
            "reran_traj_audit": False,
            "trained": False,
            "ll_traj_seed": LL_RUN,
            "u_single_seeds": list(LOW_SEEDS),
            "checkpoint_id": CKPT_ID,
            "hold_steps": HOLD_STEPS,
            "episode_horizon": EPISODE_HORIZON,
            "utility_eps": UTILITY_EPS,
            "C_abs_definition": "Phase-3C F = U_traj - U_single",
            "policy": "deterministic actor mean; joint greedy decoder",
            "environment": "arrivals_at(t) deterministic; reset jitter from scenario seed",
        },
        "source": {k: str(v) for k, v in SOURCE.items()},
    }
    _dump(OUTPUT / "phase4a_hindsight_credit_summary.json", summary)
    body = write_report(rows, uniq, align, by_fam, gate)
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "exec_C_abs_mean": by_fam[PRIMARY]["C_abs"]["mean"], "R_local": by_fam[PRIMARY]["selection"]["R_local"]["mean"]}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run()
