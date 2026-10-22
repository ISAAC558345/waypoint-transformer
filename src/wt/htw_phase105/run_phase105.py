"""Phase-10.5 runner: freeze architecture, pre-final 64-scenario benchmark, stop."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from wt.htw_phase100.high_level import load_deepsets_util
from wt.htw_phase101.high_level import load_frozen_high, parameter_sha
from wt.htw_phase9.const import HOLD
from wt.l3b.config import L3BConfig

from .baselines import exclusive_assignment, legal_myopic_assignment
from .evaluate import learned_assigner, rule_assigner, run_episode
from .protocol import (
    BOOT_SEED,
    ENV_FILE,
    HIGH_CKPT,
    HIGH_SEED,
    LAMBDA_RANK,
    LAMBDA_REF,
    N_BOOT,
    N_SCENARIOS,
    NORM_EPS,
    OLD_DEEPSETS,
    OUTPUT,
    PACK_REPORT,
    POOL_SEED,
    POOL_SIZE,
    REPORT,
    SCENARIO_BASE,
    TAU,
    all_blocked_scenario_seeds,
    dump_frozen,
    sha256_bytes,
)
from .stats import classify_sys, paired_bootstrap


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


def select_seeds() -> list[int]:
    blocked = all_blocked_scenario_seeds()
    seeds: list[int] = []
    cursor = int(SCENARIO_BASE)
    while len(seeds) < int(N_SCENARIOS):
        if int(cursor) not in blocked:
            seeds.append(int(cursor))
        cursor += 1
    if set(seeds) & blocked:
        raise RuntimeError("pre-final seeds overlap history")
    return seeds


def freeze_scenarios(consumed: bool = False) -> dict[str, Any]:
    path = OUTPUT / "phase105_scenario_manifest.json"
    if path.is_file() and not consumed:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scenario_seeds") and not payload.get("consumed"):
            return payload
        if payload.get("scenario_seeds") and payload.get("consumed") and consumed:
            return payload
        if payload.get("scenario_seeds"):
            return payload
    cfg = L3BConfig()
    seeds = select_seeds()
    payload = {
        "label": "Pre-Final Independent Benchmark, not final untouched test",
        "consumed": bool(consumed),
        "n_scenarios": len(seeds),
        "scenario_seeds": seeds,
        "generation_rule": f"first {len(seeds)} unused integers from {SCENARIO_BASE}",
        "hold": HOLD,
        "env_sha256": sha256_bytes(ENV_FILE.read_bytes()),
        "l3b_config": asdict(cfg) | {"step_length_m": cfg.step_length_m},
        "not_confirmation": True,
        "not_protected": True,
        "not_untouched": True,
        "not_phase104_independent": True,
    }
    dump_frozen(path, payload)
    return payload


def architecture_manifest(high_file_sha: str, high_param_sha: str) -> dict[str, Any]:
    cfg = L3BConfig()
    return {
        "name": "FINAL-HIER",
        "architecture_frozen": True,
        "training_frozen": True,
        "hyperparameters_frozen": True,
        "high_level": {
            "encoder": "DeepSets-util",
            "decoder": "Permutation-Equivariant Order-Free Global Assignment Decoder",
            "objective": "utility_aligned_set_supervision",
            "checkpoint": str(HIGH_CKPT),
            "high_model_seed": HIGH_SEED,
            "file_sha256": high_file_sha,
            "param_sha256": high_param_sha,
            "utility": {"tau": TAU, "lambda_rank": LAMBDA_RANK, "lambda_ref": LAMBDA_REF, "pool_size": POOL_SIZE, "pool_seed": POOL_SEED},
            "normalization": {"method": "median_mad", "eps": NORM_EPS},
        },
        "task_option": {
            "interface": "per-UAV target buoy / idle=-1",
            "hold": HOLD,
            "idle_index": 12,
            "association_semantics": "env association int64; decoder idle token=12",
        },
        "executor": {
            "movement": "fly-toward",
            "gate": "InRangeZeroMotionGate",
            "gate_rule": "non-idle AND valid AND queue>0 AND env._in_range -> movement (0,0)",
            "not_ppo": True,
        },
        "environment": {
            "name": "L3BEnv",
            "M": cfg.num_uavs,
            "K": cfg.num_buoys,
            "T": cfg.episode_horizon,
            "hold": HOLD,
            "service_radius_m": cfg.service_radius_m,
            "sinr": {
                "bandwidth_hz": cfg.bandwidth_hz,
                "tx_power": cfg.tx_power,
                "noise_power": cfg.noise_power,
                "path_loss_exp": cfg.path_loss_exp,
                "kappa_same_cluster": cfg.kappa_same_cluster,
                "kappa_cross_cluster": cfg.kappa_cross_cluster,
            },
            "env_sha256": sha256_bytes(ENV_FILE.read_bytes()),
        },
        "retired_from_primary": ["HGNN", "low-level PPO", "old Task-Waypoint predictor", "spatial waypoint predictor", "Transformer", "Hungarian decoder", "fixed-order AR decoder"],
    }


def compatibility_report(old_ok: bool) -> dict[str, Any]:
    return {
        "M0": {
            "name": "Myopic",
            "raw_allows_collision": True,
            "legal_version": "independent greedy then keep best UAV per buoy, idle extras",
            "comparable": True,
        },
        "M1": {"name": "Exclusive matching", "future_information": False, "comparable": True},
        "M2": {
            "name": "DeepSets-Old",
            "checkpoint": str(OLD_DEEPSETS),
            "exists": bool(old_ok),
            "compatible_order_free_decoder": bool(old_ok),
            "comparable": bool(old_ok),
            "reason_if_missing": None if old_ok else "frozen Phase-9.4.3 DeepSets seed0 checkpoint is not on disk; will not fabricate an old baseline",
        },
        "M3": {"name": "FINAL-HIER", "comparable": True, "primary": True},
        "flat_ppo": {
            "strictly_comparable": False,
            "reason": "Existing state-only / GoalOption PPO (Phase-5.1/6.x) lives on L2-A or uses external association (oracle/myopic) and only outputs movement. Decision authority differs from FINAL-HIER, which chooses association. Phase-6.2 M=5 weights are forbidden on L3-B M=6. No new joint PPO is trained in this phase.",
            "used_as": "secondary historical baseline (not in primary table)",
        },
        "hold_ablation": {
            "comparable_l3b_nohold_h5_h10_h15": False,
            "historical": "Phase-6.4 hold=5/10/15 on L2-A is historical mechanism evidence only",
        },
        "trained_this_phase": False,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows]))


def _plots(by: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"error": str(error)}
    note = "Pre-Final Independent Benchmark\nNot Final Untouched Test"
    methods = [key for key in ("M0", "M1", "M2", "M3") if key in by]
    labels = {"M0": "Myopic", "M1": "Exclusive", "M2": "DeepSets-Old", "M3": "FINAL-HIER"}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([labels[m] for m in methods], [_mean(by[m], "payload_bits") for m in methods], color="#2a6f97")
    ax.set_ylabel("mean payload bits")
    ax.set_title(f"Main payload comparison\n{note}")
    fig.tight_layout()
    p = OUTPUT / "phase105_main_payload.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["payload"] = str(p)

    if "M3" in by:
        fig, ax = plt.subplots(figsize=(7, 4))
        m3p = np.array([float(r["payload_bits"]) for r in sorted(by["M3"], key=lambda x: x["scenario"])])
        for other, color in (("M0", "#6c757d"), ("M1", "#84a98c"), ("M2", "#bc4749")):
            if other not in by:
                continue
            op = np.array([float(r["payload_bits"]) for r in sorted(by[other], key=lambda x: x["scenario"])])
            ax.hist(m3p - op, bins=16, alpha=0.45, label=f"M3-{other}", color=color)
        ax.axvline(0.0, color="black")
        ax.set_xlabel("paired payload delta")
        ax.legend()
        ax.set_title(f"Paired payload delta\n{note}")
        fig.tight_layout()
        p = OUTPUT / "phase105_paired_delta.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["paired"] = str(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([labels[m] for m in methods], [_mean(by[m], "completed_blocks") for m in methods])
    ax.set_title(f"Completed blocks\n{note}")
    fig.tight_layout()
    p = OUTPUT / "phase105_completed_blocks.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["blocks"] = str(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([labels[m] for m in methods], [_mean(by[m], "residual_queue_blocks") for m in methods])
    ax.set_title(f"Residual queue\n{note}")
    fig.tight_layout()
    p = OUTPUT / "phase105_residual_queue.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["queue"] = str(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([labels[m] for m in methods], [_mean(by[m], "flight_distance_m") for m in methods])
    ax.set_title(f"Flight distance\n{note}")
    fig.tight_layout()
    p = OUTPUT / "phase105_flight_distance.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["flight"] = str(p)
    return paths


def write_ablation() -> str:
    text = """# Phase-10.5 ablation evidence map

| ID | Claim | Source | Status on current L3-B |
| --- | --- | --- | --- |
| A | Utility-aligned supervision vs old single-reference | Phase-9.4.5 UOBJ-A | Formal L3-B confirmation (historical confirmation set, not this pre-final set) |
| B | HGNN vs DeepSets | Phase-9.4.5 GRAPH-U-C | Formal: HGNN not stably better; encoder frozen to DeepSets |
| C | Order-free decoder | Phase-9.4.3 ORDERFIX-A | Formal architecture freeze; permutation dependence removed |
| D | Task Option temporal hold | Phase-6.4 hold=5/10/15 | Historical mechanism on L2-A, not a current L3-B primary table |
| E | Low-level PPO vs gated fly-toward | Phase-10.4 LLEXEC-C | Formal L3-B independent validation: no stable PPO advantage |
| F | InRangeZeroMotionGate | Phase-10.3 design + Phase-10.4 B2-B1 5/5 | Formal: keep as deterministic execution rule, not a PPO credit |

This pre-final 64-scenario table is a new ordinary benchmark of the frozen FINAL-HIER stack. It is not a final untouched test.
"""
    path = OUTPUT / "phase105_ablation_evidence.md"
    path.write_text(text, encoding="utf-8")
    return text


def write_report(payload: dict[str, Any]) -> None:
    main = {row["method_id"]: row for row in payload["main_table"]}
    lines = [
        "# HTW Phase-10.5 Final Architecture Freeze and Unified Hierarchical Benchmark",
        "",
        "Pre-Final Independent Benchmark. Not Final Untouched Test. No training.",
        "",
        "FINAL-HIER = DeepSets-util + Order-Free decoder + hold=15 + fly-toward + InRangeZeroMotionGate.",
        f"UOBJ-A / GRAPH-U-C / LLEXEC-C recorded. M2 legal={payload['m2_legal']}.",
        "",
        f"1. 最终冻结架构：FINAL-HIER（高层 seed0 SHA={payload['high_sha'][:12]}…）。",
        "2. 不再使用 HGNN：GRAPH-U-C，没有稳定优于 DeepSets。",
        "3. 不再使用低层 PPO：Phase-10.4 LLEXEC-C，stable_low_level_advantage=false。",
        "4. 最终 Task Option：每机目标 buoy/idle，hold=15，低层不改 association。",
        "5. Primary baselines：M0 合法 myopic、M1 exclusive、M2 DeepSets-old（若磁盘兼容）、M3 FINAL-HIER。",
        "6. Flat PPO 不可作为严格主 baseline：决策权限不同（只控 movement / 外部 association / 旧环境）。",
        f"7. 64 场景与历史交集：{payload['overlap_count']}。",
        f"8. payload M0/M1/M2/M3 = {main.get('M0', {}).get('payload')} / {main.get('M1', {}).get('payload')} / {main.get('M2', {}).get('payload')} / {main.get('M3', {}).get('payload')}。",
        f"9. FINAL-HIER delta vs M0/M1/M2 = {payload['delta_vs']}。",
        f"10. 95% CI = {payload['ci_vs']}。",
        f"11. completed blocks M3={main.get('M3', {}).get('completed_blocks')} vs M0={main.get('M0', {}).get('completed_blocks')} / M1={main.get('M1', {}).get('completed_blocks')}。",
        f"12. residual/drop M3={main.get('M3', {}).get('residual_queue')}/{main.get('M3', {}).get('dropped')}。",
        f"13. flight M3={main.get('M3', {}).get('flight_distance')}。",
        f"14. SYS={payload['sys']['SYS']}。{payload['sys']['text']}",
        f"15. system_advantage_confirmed={payload['sys']['system_advantage_confirmed']}。",
        "16. 当前 L3-B 支持：DeepSets encoder、utility-aligned 高层、order-free decoder、gated fly-toward 执行。",
        "17. 仅历史机制：Phase-6.4 hold 消融（L2-A）；Flat PPO 对照。",
        f"18. 是否支持进入 final untouched：{'仅建议后续 Phase-10.6，本轮不自动执行' if payload['sys']['SYS']=='A' else '本轮不支持自动打开 untouched'}。",
        "19. 本阶段有没有重新训练任何模型？否。",
        f"20. 下一阶段唯一建议：{payload['next']}",
        "",
        payload["markdown_table"],
        "",
        f"墙钟 {payload['wall_s']:.1f}s。图：{payload['plots']}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")
    OUTPUT.joinpath("HTW_Phase10_5_final_architecture_benchmark_report.md").write_text(text, encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Payload | Gain vs FINAL-HIER | Completed Blocks | Residual Queue | Drop | Flight Distance | Legality |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    m3 = next((r for r in rows if r["method_id"] == "M3"), None)
    m3p = None if m3 is None or r_na(m3) else float(m3["payload"])
    for row in rows:
        if r_na(row):
            lines.append(f"| {row['method']} | N/A | N/A | N/A | N/A | N/A | N/A | {row.get('na_reason', 'N/A')} |")
            continue
        gain = "0" if m3p is None or row["method_id"] == "M3" else f"{(m3p - float(row['payload'])) / max(abs(float(row['payload'])), 1.0):.4f}"
        if row["method_id"] == "M3":
            gain = "—"
        lines.append(
            f"| {row['method']} | {row['payload']:.0f} | {gain} | {row['completed_blocks']:.3f} | {row['residual_queue']:.3f} | {row['dropped']:.3f} | {row['flight_distance']:.1f} | {row['legality']:.3f} |"
        )
    return "\n".join(lines)


def r_na(row: dict[str, Any]) -> bool:
    return str(row.get("payload")) in {"N/A", "None"} or row.get("na") is True


def main() -> dict[str, Any]:
    t0 = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not HIGH_CKPT.is_file():
        raise FileNotFoundError(str(HIGH_CKPT))
    scen = freeze_scenarios(consumed=False)
    seeds = [int(s) for s in scen["scenario_seeds"]]
    overlap = sorted(set(seeds) & all_blocked_scenario_seeds())
    if overlap:
        raise RuntimeError(f"overlap {overlap[:10]}")
    high, high_file_sha = load_frozen_high(HIGH_CKPT)
    high_param_sha = parameter_sha(high)
    dump_frozen(OUTPUT / "phase105_final_architecture_manifest.json", architecture_manifest(high_file_sha, high_param_sha))
    old_ok = OLD_DEEPSETS.is_file()
    dump_frozen(OUTPUT / "phase105_baseline_compatibility.json", compatibility_report(old_ok))
    dump_frozen(
        OUTPUT / "phase105_protocol.json",
        {
            "phase": "10.5",
            "training": False,
            "architecture_frozen": True,
            "n_scenarios": len(seeds),
            "hold": HOLD,
            "families": ["M0", "M1", "M2", "M3"],
            "label": "Pre-Final Independent Benchmark, not final untouched test",
            "no_phase106": True,
            "historical": {
                "UOBJ": "UOBJ-A",
                "GRAPH_U": "GRAPH-U-C",
                "LLEXEC": "C",
                "stable_low_level_advantage": False,
                "gate_b2_minus_b1_seeds": "5/5",
            },
        },
    )
    write_ablation()
    methods: dict[str, Any] = {
        "M0": rule_assigner(legal_myopic_assignment),
        "M1": rule_assigner(exclusive_assignment),
        "M3": learned_assigner(high),
    }
    old_model = None
    if old_ok:
        old_model = load_deepsets_util(0, OLD_DEEPSETS)
        for p in old_model.parameters():
            p.requires_grad_(False)
        old_model.eval()
        methods["M2"] = learned_assigner(old_model)
    rows: list[dict[str, Any]] = []
    by: dict[str, list[dict[str, Any]]] = {key: [] for key in methods}
    for scenario in seeds:
        print(f"eval {scenario}", flush=True)
        for fam, fn in methods.items():
            met = run_episode(scenario, family=fam, assign_fn=fn)
            rows.append(met)
            by[fam].append(met)
    _write_csv(OUTPUT / "phase105_per_scenario_metrics.csv", rows)

    def pack(fam: str, name: str) -> dict[str, Any]:
        if fam not in by:
            return {"method_id": fam, "method": name, "na": True, "payload": "N/A", "na_reason": "checkpoint missing or incompatible"}
        return {
            "method_id": fam,
            "method": name,
            "na": False,
            "payload": _mean(by[fam], "payload_bits"),
            "completed_blocks": _mean(by[fam], "completed_blocks"),
            "residual_queue": _mean(by[fam], "residual_queue_blocks"),
            "dropped": _mean(by[fam], "dropped_blocks"),
            "flight_distance": _mean(by[fam], "flight_distance_m"),
            "legality": _mean(by[fam], "exclusive_ok_frac"),
            "in_range": _mean(by[fam], "active_queued_in_range_frac"),
            "idle": _mean(by[fam], "idle_frac"),
            "switch_loss": _mean(by[fam], "target_change_loss_events"),
            "terminal_rho": _mean(by[fam], "terminal_unfinished_rho"),
            "high_decisions": _mean(by[fam], "high_decisions"),
            "mean_sinr": _mean(by[fam], "mean_sinr"),
            "mean_rate": _mean(by[fam], "mean_active_rate_bps"),
        }

    main_table = [pack("M0", "Myopic"), pack("M1", "Exclusive"), pack("M2", "DeepSets-Old"), pack("M3", "FINAL-HIER")]
    _write_csv(OUTPUT / "phase105_main_results.csv", main_table)
    m3p = np.array([float(r["payload_bits"]) for r in sorted(by["M3"], key=lambda x: x["scenario"])])
    boot = {}
    for other in ("M0", "M1", "M2"):
        if other not in by:
            continue
        op = np.array([float(r["payload_bits"]) for r in sorted(by[other], key=lambda x: x["scenario"])])
        d = m3p - op
        item = paired_bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED + {"M0": 0, "M1": 1, "M2": 2}[other])
        item["relative_gain"] = float(d.mean() / max(float(np.mean(op)), 1.0e-9))
        boot[f"M3_minus_{other}"] = item
    dump_frozen(OUTPUT / "phase105_paired_bootstrap.json", boot)
    sys = classify_sys(
        vs_m0=boot["M3_minus_M0"],
        vs_m1=boot["M3_minus_M1"],
        vs_m2=boot.get("M3_minus_M2"),
        blocks_m3=_mean(by["M3"], "completed_blocks"),
        blocks_m0=_mean(by["M0"], "completed_blocks"),
        blocks_m1=_mean(by["M1"], "completed_blocks"),
        legal_m3=_mean(by["M3"], "exclusive_ok_frac"),
        legal_m0=_mean(by["M0"], "exclusive_ok_frac"),
        legal_m1=_mean(by["M1"], "exclusive_ok_frac"),
        drop_m3=_mean(by["M3"], "dropped_blocks"),
        drop_m0=_mean(by["M0"], "dropped_blocks"),
        drop_m1=_mean(by["M1"], "dropped_blocks"),
        idle_m3=_mean(by["M3"], "idle_frac"),
        idle_ref=max(_mean(by["M0"], "idle_frac"), _mean(by["M1"], "idle_frac")),
    )
    dump_frozen(OUTPUT / "phase105_system_gate.json", sys | {"not_untouched": True, "trained": False})
    if sys["SYS"] == "A":
        dump_frozen(
            OUTPUT / "phase105_final_test_protocol_freeze.json",
            {
                "next_phase": "10.6 Final Untouched Evaluation only, after explicit instruction",
                "architecture": "FINAL-HIER",
                "baselines": ["M0 legal myopic", "M1 exclusive", "M2 DeepSets-old if legal", "M3 FINAL-HIER"],
                "metrics": ["payload", "completed_blocks", "residual_queue", "dropped", "flight", "legality", "paired bootstrap"],
                "scenario_generation": f"unused integers from a new base; never reuse {SCENARIO_BASE} pre-final set",
                "statistics": "scenario-paired percentile bootstrap; no invented model-seed CI unless multiple frozen high seeds are pre-registered",
                "table_structure": "Method | Payload | Gain | Completed Blocks | Residual Queue | Drop | Flight Distance | Legality",
                "one_shot_untouched": True,
                "auto_run_now": False,
            },
        )
        nxt = "建议 Phase-10.6 Final Untouched Evaluation 一次。本轮不自动执行。"
    elif sys["SYS"] == "B":
        nxt = "有趋势但 CI 跨 0。不要调模型，不要打开 HGNN/PPO/Transformer。本轮不进入 untouched。"
    else:
        nxt = "相对主基线未确认优势。停止。不要重开 HGNN/PPO/Transformer。"
    plots = _plots(by)
    md = markdown_table(main_table)
    summary = {
        "high_sha": high_file_sha,
        "overlap_count": 0,
        "m2_legal": old_ok,
        "main_table": main_table,
        "delta_vs": {k: v["mean_delta"] for k, v in boot.items()},
        "ci_vs": {k: v["ci95"] for k, v in boot.items()},
        "sys": sys,
        "next": nxt,
        "plots": plots,
        "markdown_table": md,
        "wall_s": perf_counter() - t0,
        "trained": False,
        "high_sha_unchanged": sha256_bytes(HIGH_CKPT.read_bytes()) == high_file_sha and parameter_sha(high) == high_param_sha,
    }
    dump_frozen(OUTPUT / "phase105_execution_summary.json", summary)
    consumed = dict(scen)
    consumed["consumed"] = True
    dump_frozen(OUTPUT / "phase105_scenario_manifest.json", consumed)
    write_report(summary)
    print(json.dumps({"SYS": sys["SYS"], "M3": main_table[3]["payload"], "trained": False}, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    main()
