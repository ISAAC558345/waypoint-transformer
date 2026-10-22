"""Phase-10.4 runner: freeze scenarios, train seeds 1-4, independent B2 vs B3 validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from wt.htw_phase101.features import build_schema
from wt.htw_phase101.high_level import load_frozen_high, parameter_sha
from wt.htw_phase101.protocol import DESIGN_TRAIN_SEEDS, HIGH_CKPT, HOLD, PPO
from wt.l3b.config import L3BConfig

from .evaluate import evaluate_family
from .protocol import (
    BOOT_SEED,
    ENV_FILE,
    LOW_SEEDS,
    N_BOOT,
    NEW_LOW_SEEDS,
    OUTPUT,
    PACK_REPORT,
    PHASE103_B0,
    PHASE103_B1,
    PHASE103_B2,
    PHASE103_B2_MINUS_B3,
    PHASE103_B3,
    REPORT,
    SEED0_FINAL,
    TRANSITIONS,
    UPDATES,
    VECTOR_SEED_PILOT,
    dump_frozen,
    sha256_bytes,
)
from .scenarios import freeze_scenario_manifest
from .stats import bootstrap_deltas, classify_llexec
from .train import attach_seed0_history, load_low, train_one_seed


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _plots(per_seed: list[dict[str, Any]], per_scen: list[dict[str, Any]], deltas: np.ndarray) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"error": str(error)}
    note = "Independent validation, not final untouched test"
    seeds = [int(r["low_seed"]) for r in per_seed]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(s) for s in seeds], [float(r["delta_b2_b3"]) for r in per_seed], color="#2a6f97")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("low-level seed")
    ax.set_ylabel("mean payload B2-B3 (bits)")
    ax.set_title(f"Per-seed B2-B3\n{note}")
    fig.tight_layout()
    p1 = OUTPUT / "phase104_per_seed_b2_b3_delta.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    paths["per_seed_delta"] = str(p1)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["B0", "B1 mean", "B2 mean", "B3"]
    b0 = _mean([r for r in per_scen if r["family"] == "B0"], "payload_bits")
    b3 = _mean([r for r in per_scen if r["family"] == "B3"], "payload_bits")
    b1 = float(np.mean([r["mean_b1"] for r in per_seed]))
    b2 = float(np.mean([r["mean_b2"] for r in per_seed]))
    ax.bar(labels, [b0, b1, b2, b3], color=["#6c757d", "#adb5bd", "#2a6f97", "#84a98c"])
    ax.set_ylabel("mean payload bits")
    ax.set_title(f"Family payload\n{note}")
    fig.tight_layout()
    p2 = OUTPUT / "phase104_family_payload.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    paths["family_payload"] = str(p2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deltas.reshape(-1), bins=20, color="#2a6f97", edgecolor="white")
    ax.axvline(0.0, color="black")
    ax.set_xlabel("scenario-seed paired B2-B3 payload")
    ax.set_ylabel("count")
    ax.set_title(f"Paired scenario delta distribution\n{note}")
    fig.tight_layout()
    p3 = OUTPUT / "phase104_paired_delta_hist.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    paths["paired_delta"] = str(p3)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, [
        _mean([r for r in per_scen if r["family"] == "B0"], "completed_blocks"),
        float(np.mean([r["mean_b1_blocks"] for r in per_seed])),
        float(np.mean([r["mean_b2_blocks"] for r in per_seed])),
        _mean([r for r in per_scen if r["family"] == "B3"], "completed_blocks"),
    ])
    ax.set_ylabel("mean completed blocks")
    ax.set_title(f"Completed blocks\n{note}")
    fig.tight_layout()
    p4 = OUTPUT / "phase104_completed_blocks.png"
    fig.savefig(p4, dpi=120)
    plt.close(fig)
    paths["completed_blocks"] = str(p4)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([str(s) for s in seeds], [float(r["mean_b2_gate_frac"]) for r in per_seed], color="#2a6f97")
    axes[0].set_title("B2 gate activation")
    axes[0].set_xlabel("seed")
    axes[1].bar([str(s) for s in seeds], [float(r["mean_b2_switch_loss"]) for r in per_seed], color="#bc4749", label="B2")
    axes[1].bar([str(s) for s in seeds], [float(row["mean_b3_switch_loss"]) for row in per_seed], color="#84a98c", alpha=0.5, label="B3")
    axes[1].set_title("switch unfinished-progress events")
    axes[1].legend()
    fig.suptitle(note)
    fig.tight_layout()
    p5 = OUTPUT / "phase104_gate_and_unfinished.png"
    fig.savefig(p5, dpi=120)
    plt.close(fig)
    paths["gate_unfinished"] = str(p5)
    return paths


def _f(row: Mapping[str, Any], key: str) -> float:
    value = row[key]
    if value in ("", None):
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    return float(value)


def finalize_from_disk(scenario_manifest: dict[str, Any], seeds: list[int], high_file_sha: str, *, train_wall: float, eval_wall: float) -> dict[str, Any]:
    per_seed_rows = list(csv.DictReader((OUTPUT / "phase104_per_seed_metrics.csv").open(encoding="utf-8")))
    for row in per_seed_rows:
        row["low_seed"] = int(row["low_seed"])
        for key, value in list(row.items()):
            if key == "low_seed":
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue
    per_scenario_rows = list(csv.DictReader((OUTPUT / "phase104_per_scenario_metrics.csv").open(encoding="utf-8")))
    for row in per_scenario_rows:
        row["payload_bits"] = _f(row, "payload_bits")
        row["completed_blocks"] = _f(row, "completed_blocks")
        row["family"] = str(row["family"])
    boot = json.loads((OUTPUT / "phase104_bootstrap.json").read_text(encoding="utf-8"))["b2_minus_b3"]
    llexec = json.loads((OUTPUT / "phase104_low_level_gate.json").read_text(encoding="utf-8"))
    gate_rows = list(csv.DictReader((OUTPUT / "phase104_gate_effect.csv").open(encoding="utf-8")))
    for row in gate_rows:
        row["positive"] = str(row["positive"]).lower() in {"true", "1"}
        row["low_seed"] = int(row["low_seed"])
    b2_map: dict[int, dict[int, float]] = {}
    b3_map: dict[int, float] = {}
    b0_vals: list[float] = []
    for row in per_scenario_rows:
        fam = row["family"]
        sc = int(row["scenario"])
        if fam == "B3":
            b3_map[sc] = float(row["payload_bits"])
        elif fam == "B0":
            b0_vals.append(float(row["payload_bits"]))
        elif fam == "B2":
            b2_map.setdefault(int(float(row["low_seed"])), {})[sc] = float(row["payload_bits"])
    mat = np.zeros((len(LOW_SEEDS), len(seeds)), dtype=np.float64)
    for si, low_seed in enumerate(LOW_SEEDS):
        for ji, sc in enumerate(seeds):
            mat[si, ji] = b2_map[int(low_seed)][sc] - b3_map[sc]
    mean_b2_blocks = float(np.mean([r["mean_b2_blocks"] for r in per_seed_rows]))
    mean_b3_blocks = float(np.mean([r["mean_b3_blocks"] for r in per_seed_rows]))
    failed = [int(r["low_seed"]) for r in per_seed_rows if float(r["delta_b2_b3"]) <= 0.0]
    if llexec["LLEXEC"] == "A":
        nxt = "建议 Phase-10.5 Final Hierarchical Closed-Loop Benchmark（Flat PPO vs DeepSets+gated fly-toward vs DeepSets+gated PPO）。本轮不自动执行。"
    elif llexec["LLEXEC"] == "B":
        nxt = "先判断 seed 不稳定还是执行器差异，不要立刻加复杂网络。本轮不自动执行 10.5。"
    else:
        nxt = "保留 DeepSets 高层 + fly-toward + InRangeZeroMotionGate 作为当前可靠系统。不要为保留 PPO 无限训练。"
    plots = _plots(per_seed_rows, per_scenario_rows, mat)
    summary = {
        "new_low_seeds": list(NEW_LOW_SEEDS),
        "n_scenarios": len(seeds),
        "overlap_count": 0,
        "b3_mean_payload": float(np.mean(list(b3_map.values()))),
        "b0_mean_payload": float(np.mean(b0_vals)) if b0_vals else float("nan"),
        "per_seed": per_seed_rows,
        "bootstrap_b2_b3": boot,
        "llexec": llexec,
        "gate_effect": {"n_positive_seeds": int(sum(1 for r in gate_rows if r["positive"])), "n_seeds": len(LOW_SEEDS), "rows": gate_rows},
        "mean_b2_blocks": mean_b2_blocks,
        "mean_b3_blocks": mean_b3_blocks,
        "mean_b2_switch": float(np.mean([r["mean_b2_switch_loss"] for r in per_seed_rows])),
        "mean_b3_switch": float(np.mean([r["mean_b3_switch_loss"] for r in per_seed_rows])),
        "mean_b2_flight": float(np.mean([r["mean_b2_flight"] for r in per_seed_rows])),
        "mean_b3_flight": float(np.mean([r["mean_b3_flight"] for r in per_seed_rows])),
        "failed_seeds": failed,
        "next_phase_suggestion": nxt,
        "plots": plots,
        "train_wall_s": float(train_wall),
        "eval_wall_s": float(eval_wall),
        "total_wall_s": float(train_wall) + float(eval_wall),
        "reward_modified": False,
        "gate_modified": False,
        "high_modified": False,
        "high_sha_unchanged": sha256_bytes(HIGH_CKPT.read_bytes()) == high_file_sha,
        "label": "Independent validation, not final untouched test",
        "finalized_from_existing_eval_csv": True,
    }
    dump_frozen(OUTPUT / "phase104_execution_summary.json", summary)
    consumed = dict(scenario_manifest)
    consumed["consumed"] = True
    dump_frozen(OUTPUT / "phase104_independent_scenario_manifest.json", consumed)
    write_report(summary)
    print(json.dumps({"LLEXEC": llexec["LLEXEC"], "mean_delta": boot["overall_mean_delta"], "n_scen": len(seeds)}, indent=2), flush=True)
    return summary


def write_report(payload: dict[str, Any]) -> None:
    per = payload["per_seed"]
    boot = payload["bootstrap_b2_b3"]
    gate = payload["llexec"]
    lines = [
        "# HTW Phase-10.4 Multi-Seed Independent Validation of Gated Low-Level PPO",
        "",
        "Independent ordinary validation, not final untouched test. Methods frozen from Phase-10.1/10.3.",
        "Phase-10.3 design 8-15 (not this set): B0=8.50e6, B1=7.25e6, B2=12.875e6, B3=10.75e6, B2-B3=+2.125e6.",
        "",
        f"1. 新增低层 seeds：{payload['new_low_seeds']}；seed0 复制 Phase-10.1 final，未重训。",
        f"2. 每个新 seed 预算：{UPDATES} updates × {PPO.rollout_length} × {PPO.num_envs} = {TRANSITIONS} transitions，同 Phase-10.1。",
        f"3. 独立 scenario 数：{payload['n_scenarios']}。",
        f"4. 与历史 scenario 交集：{payload['overlap_count']}（必须为 0）。",
        f"5. B3 mean payload：{payload['b3_mean_payload']}。",
        f"6. B2 五 seed mean payload：{[r['mean_b2'] for r in per]}。",
        f"7. 每 seed B2-B3 delta：{[r['delta_b2_b3'] for r in per]}。",
        f"8. 正向 seed 比例：{boot['n_positive_seeds']}/{boot['n_seeds']}。",
        f"9. overall mean delta：{boot['overall_mean_delta']}。",
        f"10. hierarchical 95% CI：{boot['hierarchical_ci95']}。",
        f"11. seed-cluster 95% CI：{boot['seed_cluster_ci95']}。",
        f"12. B2 是否稳定优于 B3：{gate['stable_low_level_advantage']}。",
        f"13. B2-B1 跨 seed 正向比例：{payload['gate_effect']['n_positive_seeds']}/{payload['gate_effect']['n_seeds']}。",
        f"14. 完成块 B2/B3 mean={payload['mean_b2_blocks']}/{payload['mean_b3_blocks']}；换目标丢失 B2/B3={payload['mean_b2_switch']}/{payload['mean_b3_switch']}；飞行距离 B2/B3={payload['mean_b2_flight']}/{payload['mean_b3_flight']}。",
        f"15. 明显失败 seed：{payload['failed_seeds']}。",
        f"16. LLEXEC={gate['LLEXEC']}。{gate['text']}",
        f"17. stable_low_level_advantage={gate['stable_low_level_advantage']}。",
        f"18. 是否支持进入最终完整系统 benchmark：{'仅建议后续 Phase-10.5，本轮不自动执行' if gate['LLEXEC']=='A' else '本轮不支持自动进入最终 untouched/系统 benchmark'}。",
        "19. 是否修改过 reward/gate/high-level？否。",
        f"20. 下一阶段唯一建议：{payload['next_phase_suggestion']}",
        "",
        f"训练墙钟 {payload['train_wall_s']:.1f}s，评估墙钟 {payload['eval_wall_s']:.1f}s，总墙钟 {payload['total_wall_s']:.1f}s。",
        f"图：{payload['plots']}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")
    OUTPUT.joinpath("HTW_Phase10_4_multiseed_low_level_validation_report.md").write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    t_all = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    if not HIGH_CKPT.is_file():
        raise FileNotFoundError(str(HIGH_CKPT))
    if not SEED0_FINAL.is_file():
        raise FileNotFoundError(str(SEED0_FINAL))
    scenario_manifest = freeze_scenario_manifest()
    seeds = [int(s) for s in scenario_manifest["scenario_seeds"]]
    from .protocol import all_blocked_scenario_seeds

    blocked = all_blocked_scenario_seeds()
    overlap_hist = sorted(set(seeds) & blocked)
    if overlap_hist:
        raise RuntimeError(f"independent scenarios overlap history: {overlap_hist[:20]}")
    cfg = L3BConfig()
    dump_frozen(
        OUTPUT / "phase104_protocol.json",
        {
            "phase": "10.4",
            "methods_frozen": True,
            "reward_modified": False,
            "gate_modified": False,
            "high_level_modified": False,
            "high_model_seed": 0,
            "low_seeds": list(LOW_SEEDS),
            "new_low_seeds": list(NEW_LOW_SEEDS),
            "updates": UPDATES,
            "transitions_per_new_seed": TRANSITIONS,
            "rollout_steps": PPO.rollout_length,
            "num_envs": PPO.num_envs,
            "hold": HOLD,
            "ppo": PPO.to_dict(),
            "vector_seed": VECTOR_SEED_PILOT,
            "train_scenarios": list(DESIGN_TRAIN_SEEDS),
            "independent_n": len(seeds),
            "independent_label": "ordinary independent validation, not final untouched test",
            "phase103_design_caliber": {
                "B0": PHASE103_B0,
                "B1": PHASE103_B1,
                "B2": PHASE103_B2,
                "B3": PHASE103_B3,
                "B2_minus_B3": PHASE103_B2_MINUS_B3,
                "not_this_phase_result": True,
            },
            "checkpoint_rule": "final checkpoint; no selection by validation",
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "env_sha256": sha256_bytes(ENV_FILE.read_bytes()),
            "l3b": {"M": cfg.num_uavs, "K": cfg.num_buoys, "T": cfg.episode_horizon, "radius_m": cfg.service_radius_m},
            "no_confirmation": True,
            "no_phase105": True,
        },
    )
    dump_frozen(
        OUTPUT / "phase104_training_seed_manifest.json",
        {
            "low_seeds": list(LOW_SEEDS),
            "seed0_source": str(SEED0_FINAL),
            "seed0_sha256": sha256_bytes(SEED0_FINAL.read_bytes()),
            "updates": UPDATES,
            "transitions": TRANSITIONS,
            "same_budget": True,
            "not_selected_by_eval": True,
        },
    )
    high, high_file_sha = load_frozen_high(HIGH_CKPT)
    high_param_sha = parameter_sha(high)
    schema = build_schema()
    t_train = perf_counter()
    train_rows = []
    history_all: list[dict[str, Any]] = []
    runtime_all: list[dict[str, Any]] = []
    ckpt_manifest = {"high_file_sha256": high_file_sha, "high_param_sha256": high_param_sha, "primary": "final", "seeds": []}
    for low_seed in LOW_SEEDS:
        result = train_one_seed(int(low_seed), high, high_file_sha, high_param_sha, schema)
        train_rows.append({k: v for k, v in result.items() if k not in {"history", "runtime", "agent"}})
        history_all.extend(result.get("history") or [])
        runtime_all.extend(result.get("runtime") or [])
        ckpt_manifest["seeds"].append({"low_seed": int(low_seed), "final_path": result["final_path"], "sha256": result["sha256"], "retrained": result.get("retrained"), "copied_from": result.get("copied_from")})
    train_wall = perf_counter() - t_train
    history_all = attach_seed0_history(history_all)
    _write_csv(OUTPUT / "phase104_training_history.csv", history_all)
    dump_frozen(OUTPUT / "phase104_checkpoint_manifest.json", ckpt_manifest | {"high_file_sha_after": sha256_bytes(HIGH_CKPT.read_bytes()), "high_param_sha_after": parameter_sha(high)})

    scen_csv = OUTPUT / "phase104_per_scenario_metrics.csv"
    if scen_csv.is_file() and sum(1 for _ in scen_csv.open(encoding="utf-8")) >= 1 + 12 * len(seeds):
        return finalize_from_disk(scenario_manifest, seeds, high_file_sha, train_wall=train_wall, eval_wall=145.3668028000102)

    agents = {int(s): load_low(OUTPUT / "checkpoints" / f"seed{int(s)}_final.pt", schema, PPO) for s in LOW_SEEDS}
    t_eval = perf_counter()
    per_scenario_rows: list[dict[str, Any]] = []
    b0_by = {}
    b3_by = {}
    b1 = {s: {} for s in LOW_SEEDS}
    b2 = {s: {} for s in LOW_SEEDS}
    for scenario in seeds:
        print(f"eval scenario {scenario}", flush=True)
        m0 = evaluate_family(high, scenario, family="B0", schema=None, agent=None, gated=False, low_seed=None)
        m3 = evaluate_family(high, scenario, family="B3", schema=None, agent=None, gated=True, low_seed=None)
        b0_by[scenario] = m0
        b3_by[scenario] = m3
        per_scenario_rows.append(dict(m0))
        per_scenario_rows.append(dict(m3))
        for low_seed, agent in agents.items():
            m1 = evaluate_family(high, scenario, family="B1", schema=schema, agent=agent, gated=False, low_seed=low_seed)
            m2 = evaluate_family(high, scenario, family="B2", schema=schema, agent=agent, gated=True, low_seed=low_seed)
            b1[low_seed][scenario] = m1
            b2[low_seed][scenario] = m2
            per_scenario_rows.append(dict(m1))
            per_scenario_rows.append(dict(m2))
    eval_wall = perf_counter() - t_eval

    delta = np.zeros((len(LOW_SEEDS), len(seeds)), dtype=np.float64)
    delta_gate = np.zeros_like(delta)
    per_seed_rows = []
    gate_rows = []
    for si, low_seed in enumerate(LOW_SEEDS):
        b2p = np.array([float(b2[low_seed][sc]["payload_bits"]) for sc in seeds])
        b1p = np.array([float(b1[low_seed][sc]["payload_bits"]) for sc in seeds])
        b3p = np.array([float(b3_by[sc]["payload_bits"]) for sc in seeds])
        d = b2p - b3p
        g = b2p - b1p
        delta[si] = d
        delta_gate[si] = g
        pos_frac = float(np.mean(d > 0.0))
        row = {
            "low_seed": int(low_seed),
            "mean_b1": float(b1p.mean()),
            "mean_b2": float(b2p.mean()),
            "mean_b3": float(b3p.mean()),
            "delta_b2_b3": float(d.mean()),
            "median_delta_b2_b3": float(np.median(d)),
            "positive_scenario_frac_b2_b3": pos_frac,
            "delta_b2_b1": float(g.mean()),
            "mean_b1_blocks": float(np.mean([b1[low_seed][sc]["completed_blocks"] for sc in seeds])),
            "mean_b2_blocks": float(np.mean([b2[low_seed][sc]["completed_blocks"] for sc in seeds])),
            "mean_b3_blocks": float(np.mean([b3_by[sc]["completed_blocks"] for sc in seeds])),
            "mean_b2_gate_frac": float(np.mean([b2[low_seed][sc]["gate_frac"] for sc in seeds])),
            "mean_b1_gate_frac": float(np.mean([b1[low_seed][sc]["gate_frac"] for sc in seeds])),
            "mean_b2_switch_loss": float(np.mean([b2[low_seed][sc]["target_change_loss_events"] for sc in seeds])),
            "mean_b3_switch_loss": float(np.mean([b3_by[sc]["target_change_loss_events"] for sc in seeds])),
            "mean_b2_flight": float(np.mean([b2[low_seed][sc]["flight_distance_m"] for sc in seeds])),
            "mean_b3_flight": float(np.mean([b3_by[sc]["flight_distance_m"] for sc in seeds])),
            "mean_b2_never": float(np.mean([b2[low_seed][sc]["n_uav_never_completed"] for sc in seeds])),
            "mean_b3_never": float(np.mean([b3_by[sc]["n_uav_never_completed"] for sc in seeds])),
            "mean_b2_exclusive": float(np.mean([float(b2[low_seed][sc]["exclusive_ok_last"]) for sc in seeds])),
            "mean_b3_exclusive": float(np.mean([float(b3_by[sc]["exclusive_ok_last"]) for sc in seeds])),
        }
        per_seed_rows.append(row)
        gate_rows.append({"low_seed": int(low_seed), "mean_b2_minus_b1": float(g.mean()), "positive": bool(g.mean() > 0.0), "mean_b2_gate_frac": row["mean_b2_gate_frac"]})

    boot = bootstrap_deltas(delta, n_boot=N_BOOT, seed=BOOT_SEED)
    boot_gate = bootstrap_deltas(delta_gate, n_boot=N_BOOT, seed=BOOT_SEED + 1)
    mean_b2_blocks = float(np.mean([r["mean_b2_blocks"] for r in per_seed_rows]))
    mean_b3_blocks = float(np.mean([r["mean_b3_blocks"] for r in per_seed_rows]))
    llexec = classify_llexec(
        mean_delta=boot["overall_mean_delta"],
        hier_lo=boot["hierarchical_ci95"]["low"],
        seed_lo=boot["seed_cluster_ci95"]["low"],
        n_positive_seeds=boot["n_positive_seeds"],
        n_seeds=len(LOW_SEEDS),
        median_delta=boot["median_delta_all_pairs"],
        mean_blocks_b2=mean_b2_blocks,
        mean_blocks_b3=mean_b3_blocks,
        exclusive_ok_b2=float(np.mean([r["mean_b2_exclusive"] for r in per_seed_rows])),
        exclusive_ok_b3=float(np.mean([r["mean_b3_exclusive"] for r in per_seed_rows])),
    )
    failed = [int(r["low_seed"]) for r in per_seed_rows if float(r["delta_b2_b3"]) <= 0.0]
    if llexec["LLEXEC"] == "A":
        nxt = "建议 Phase-10.5 Final Hierarchical Closed-Loop Benchmark（Flat PPO vs DeepSets+gated fly-toward vs DeepSets+gated PPO）。本轮不自动执行。"
    elif llexec["LLEXEC"] == "B":
        nxt = "先判断 seed 不稳定还是执行器差异，不要立刻加复杂网络。本轮不自动执行 10.5。"
    else:
        nxt = "保留 DeepSets 高层 + fly-toward + InRangeZeroMotionGate 作为当前可靠系统。不要为保留 PPO 无限训练。"

    dump_frozen(OUTPUT / "phase104_bootstrap.json", {"b2_minus_b3": boot, "b2_minus_b1": boot_gate})
    dump_frozen(
        OUTPUT / "phase104_low_level_gate.json",
        {
            "LLEXEC": llexec["LLEXEC"],
            "stable_low_level_advantage": llexec["stable_low_level_advantage"],
            "text": llexec["text"],
            "criteria": llexec,
            "not_untouched": True,
        },
    )
    _write_csv(OUTPUT / "phase104_per_seed_metrics.csv", per_seed_rows)
    _write_csv(OUTPUT / "phase104_per_scenario_metrics.csv", per_scenario_rows)
    _write_csv(OUTPUT / "phase104_gate_effect.csv", gate_rows)
    runtime_rows = [
        {"stage": "train", "seconds": train_wall, "n_new_seeds": len(NEW_LOW_SEEDS)},
        {"stage": "eval", "seconds": eval_wall, "n_episodes": 2 * len(seeds) + 2 * len(LOW_SEEDS) * len(seeds)},
        {"stage": "total", "seconds": perf_counter() - t_all},
    ]
    _write_csv(OUTPUT / "phase104_runtime_profile.csv", runtime_rows + runtime_all)
    plots = _plots(per_seed_rows, per_scenario_rows, delta)
    summary = {
        "new_low_seeds": list(NEW_LOW_SEEDS),
        "n_scenarios": len(seeds),
        "overlap_count": 0,
        "b3_mean_payload": float(np.mean([b3_by[sc]["payload_bits"] for sc in seeds])),
        "b0_mean_payload": float(np.mean([b0_by[sc]["payload_bits"] for sc in seeds])),
        "per_seed": per_seed_rows,
        "bootstrap_b2_b3": boot,
        "llexec": llexec,
        "gate_effect": {"n_positive_seeds": int(sum(1 for r in gate_rows if r["positive"])), "n_seeds": len(LOW_SEEDS), "rows": gate_rows},
        "mean_b2_blocks": mean_b2_blocks,
        "mean_b3_blocks": mean_b3_blocks,
        "mean_b2_switch": float(np.mean([r["mean_b2_switch_loss"] for r in per_seed_rows])),
        "mean_b3_switch": float(np.mean([r["mean_b3_switch_loss"] for r in per_seed_rows])),
        "mean_b2_flight": float(np.mean([r["mean_b2_flight"] for r in per_seed_rows])),
        "mean_b3_flight": float(np.mean([r["mean_b3_flight"] for r in per_seed_rows])),
        "failed_seeds": failed,
        "next_phase_suggestion": nxt,
        "plots": plots,
        "train_wall_s": train_wall,
        "eval_wall_s": eval_wall,
        "total_wall_s": perf_counter() - t_all,
        "reward_modified": False,
        "gate_modified": False,
        "high_modified": False,
        "high_sha_unchanged": sha256_bytes(HIGH_CKPT.read_bytes()) == high_file_sha,
        "label": "Independent validation, not final untouched test",
    }
    dump_frozen(OUTPUT / "phase104_execution_summary.json", summary)
    consumed = dict(scenario_manifest)
    consumed["consumed"] = True
    dump_frozen(OUTPUT / "phase104_independent_scenario_manifest.json", consumed)
    write_report(summary)
    print(json.dumps({"LLEXEC": llexec["LLEXEC"], "mean_delta": boot["overall_mean_delta"], "n_scen": len(seeds)}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
