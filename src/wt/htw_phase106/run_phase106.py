"""Phase-10.6 one-shot final untouched evaluation. No training."""
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
from wt.htw_phase105.baselines import exclusive_assignment, legal_myopic_assignment
from wt.htw_phase105.evaluate import learned_assigner, rule_assigner, run_episode
from wt.htw_phase9.const import HOLD
from wt.l3b.config import L3BConfig

from .protocol import (
    ARCH_MANIFEST,
    BOOT_SEED,
    ENV_FILE,
    EXPECTED_ENV_SHA,
    EXPECTED_HIGH_SHA,
    FREEZE_FILE,
    HIGH_CKPT,
    N_BOOT,
    N_SCENARIOS,
    OLD_DEEPSETS,
    OUTPUT,
    PACK_REPORT,
    REPORT,
    SCENARIO_BASE,
    candidate_untouched_seeds,
    dump_frozen,
    historical_blocked_seeds,
    sha256_bytes,
)
from .stats import classify_final, paired_bootstrap

BASELINE_PY = Path(__file__).resolve().parents[1] / "htw_phase105" / "baselines.py"
EVAL_PY = Path(__file__).resolve().parents[1] / "htw_phase105" / "evaluate.py"
STATS105_PY = Path(__file__).resolve().parents[1] / "htw_phase105" / "stats.py"


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
    return float(np.mean([float(r[key]) for r in rows]))


def restore_and_check() -> dict[str, Any]:
    for path in (ARCH_MANIFEST, FREEZE_FILE, PHASE105_PROTOCOL := Path(ARCH_MANIFEST).parent / "phase105_protocol.json", Path(ARCH_MANIFEST).parent / "phase105_baseline_compatibility.json", Path(ARCH_MANIFEST).parent / "phase105_system_gate.json"):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    arch = json.loads(ARCH_MANIFEST.read_text(encoding="utf-8"))
    sys105 = json.loads((ARCH_MANIFEST.parent / "phase105_system_gate.json").read_text(encoding="utf-8"))
    if not arch.get("architecture_frozen") or not arch.get("training_frozen") or not arch.get("hyperparameters_frozen"):
        raise RuntimeError("Phase-10.5 architecture is not frozen")
    if sys105.get("SYS") != "A" or not sys105.get("system_advantage_confirmed"):
        raise RuntimeError("Phase-10.6 requires SYS-A from Phase-10.5")
    file_sha = sha256_bytes(HIGH_CKPT.read_bytes())
    if file_sha != EXPECTED_HIGH_SHA or file_sha != arch["high_level"]["file_sha256"]:
        raise RuntimeError("high-level checkpoint SHA mismatch; stopping without searching for another checkpoint")
    env_sha = sha256_bytes(ENV_FILE.read_bytes())
    if env_sha != EXPECTED_ENV_SHA or env_sha != arch["environment"]["env_sha256"]:
        raise RuntimeError("environment SHA mismatch; stopping")
    return {"arch": arch, "sys105": sys105, "high_file_sha": file_sha}


def freeze_untouched_scenarios() -> dict[str, Any]:
    path = OUTPUT / "phase106_untouched_scenario_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scenario_seeds"):
            return payload
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seeds = candidate_untouched_seeds()
    blocked = historical_blocked_seeds()
    overlap = sorted(set(seeds) & blocked)
    if overlap:
        raise RuntimeError(f"untouched seeds overlap history; stopping without replacement: {overlap[:20]}")
    cfg = L3BConfig()
    payload = {
        "label": "FINAL UNTOUCHED TEST",
        "consumed": False,
        "n_scenarios": len(seeds),
        "scenario_seeds": seeds,
        "generation_rule": f"64 consecutive unused-base integers starting at {SCENARIO_BASE}; never reuse 105000 pre-final set; no post-hoc seed swap",
        "hold": HOLD,
        "env_sha256": sha256_bytes(ENV_FILE.read_bytes()),
        "l3b_config": asdict(cfg) | {"step_length_m": cfg.step_length_m},
        "sinr": asdict(cfg),
        "arrival": {"queue_capacity_blocks": cfg.queue_capacity_blocks, "num_hot": cfg.num_hot, "reset_seed_drives_arrivals": True},
        "M": cfg.num_uavs,
        "K": cfg.num_buoys,
        "T": cfg.episode_horizon,
        "not_prefinal": True,
        "not_confirmation": True,
        "one_shot": True,
    }
    dump_frozen(path, payload)
    return payload


def write_preaccess_lock(scen: dict[str, Any], high_file_sha: str, high_param_sha: str) -> str:
    payload = {
        "method_frozen": True,
        "data_frozen": True,
        "metrics_frozen": True,
        "architecture_sha256": sha256_bytes(ARCH_MANIFEST.read_bytes()),
        "checkpoint_sha256": high_file_sha,
        "high_param_sha256": high_param_sha,
        "baseline_definitions_sha256": sha256_bytes(BASELINE_PY.read_bytes()),
        "metric_definitions_sha256": sha256_bytes(EVAL_PY.read_bytes()),
        "statistics_protocol_sha256": sha256_bytes(STATS105_PY.read_bytes()),
        "scenario_manifest_sha256": sha256_bytes((OUTPUT / "phase106_untouched_scenario_manifest.json").read_bytes()),
        "freeze_protocol_sha256": sha256_bytes(FREEZE_FILE.read_bytes()),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "bootstrap": "scenario-paired percentile 2.5/97.5; no invented model-seed CI",
        "final_untouched_consumed": False,
    }
    dump_frozen(OUTPUT / "phase106_preaccess_lock.json", payload)
    return sha256_bytes((OUTPUT / "phase106_preaccess_lock.json").read_bytes())


def _plots(by: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"error": str(error)}
    note = "FINAL UNTOUCHED TEST"
    methods = ["M0", "M1", "M2", "M3"]
    labels = {"M0": "Myopic", "M1": "Exclusive", "M2": "DeepSets-old", "M3": "FINAL-HIER"}
    colors = ["#6c757d", "#84a98c", "#bc4749", "#2a6f97"]

    def bar(key: str, ylabel: str, fname: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([labels[m] for m in methods], [_mean(by[m], key) for m in methods], color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{note}")
        fig.tight_layout()
        path = OUTPUT / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths[key] = str(path)

    bar("payload_bits", "mean payload bits", "phase106_final_payload.png", "Final payload comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    m3p = np.array([float(r["payload_bits"]) for r in sorted(by["M3"], key=lambda x: x["scenario"])])
    for other, color in (("M0", "#6c757d"), ("M1", "#84a98c"), ("M2", "#bc4749")):
        op = np.array([float(r["payload_bits"]) for r in sorted(by[other], key=lambda x: x["scenario"])])
        ax.hist(m3p - op, bins=16, alpha=0.45, label=f"M3-{other}", color=color)
    ax.axvline(0.0, color="black")
    ax.set_xlabel("paired payload delta")
    ax.legend()
    ax.set_title(f"Final paired payload delta\n{note}")
    fig.tight_layout()
    p = OUTPUT / "phase106_final_paired_delta.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["paired"] = str(p)
    bar("completed_blocks", "mean completed blocks", "phase106_final_completed_blocks.png", "Completed blocks comparison")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([labels[m] for m in methods], [_mean(by[m], "residual_queue_blocks") for m in methods], color=colors)
    axes[0].set_title("Residual queue")
    axes[1].bar([labels[m] for m in methods], [_mean(by[m], "dropped_blocks") for m in methods], color=colors)
    axes[1].set_title("Drop")
    fig.suptitle(note)
    fig.tight_layout()
    p = OUTPUT / "phase106_final_queue_drop.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["queue_drop"] = str(p)
    bar("flight_distance_m", "mean flight distance (m)", "phase106_final_flight.png", "Flight distance comparison")
    bar("n_uav_completed", "mean UAVs with a completed block", "phase106_final_service_completion.png", "Service completion comparison")
    return paths


def markdown_tables(main_rows: list[dict[str, Any]], boot: dict[str, Any]) -> str:
    m3p = float(next(r["payload"] for r in main_rows if r["method_id"] == "M3"))
    lines = [
        "| Method | Payload | Relative Gain | Completed Blocks | Residual Queue | Drop | Flight Distance | Legality |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in main_rows:
        if row["method_id"] == "M3":
            gain = "—"
        else:
            gain = f"{(m3p - float(row['payload'])) / max(abs(float(row['payload'])), 1.0):.4f}"
        lines.append(
            f"| {row['method']} | {row['payload']:.0f} | {gain} | {row['completed_blocks']:.3f} | {row['residual_queue']:.3f} | {row['dropped']:.3f} | {row['flight_distance']:.1f} | {row['legality']:.3f} |"
        )
    lines += ["", "| Comparison | Mean Delta | Median Delta | Positive Fraction | 95% CI |", "| --- | --- | --- | --- | --- |"]
    for key, name in (("M3_minus_M0", "M3-M0"), ("M3_minus_M1", "M3-M1"), ("M3_minus_M2", "M3-M2")):
        item = boot[key]
        lines.append(
            f"| {name} | {item['mean_delta']:.0f} | {item['median_delta']:.0f} | {item['positive_scenario_frac']:.3f} | [{item['ci95']['low']:.0f}, {item['ci95']['high']:.0f}] |"
        )
    return "\n".join(lines)


def write_report(payload: dict[str, Any]) -> None:
    m = {row["method_id"]: row for row in payload["main_table"]}
    boot = payload["bootstrap"]
    gate = payload["final_gate"]
    lines = [
        "# HTW Phase-10.6 Final Untouched Evaluation",
        "",
        "FINAL UNTOUCHED TEST. One-shot. No training. No retuning.",
        "",
        f"1. 最终架构：FINAL-HIER（DeepSets-util seed0 + order-free + hold=15 + fly-toward + InRangeZeroMotionGate）。SHA={payload['high_sha'][:16]}…",
        f"2. Final untouched 场景数量：{payload['n_scenarios']}。",
        f"3. 与全部历史 scenario 交集：{payload['overlap_count']}。",
        f"4. final_untouched_consumed={payload['consumed']}。",
        f"5. payload M0/M1/M2/M3 = {m['M0']['payload']} / {m['M1']['payload']} / {m['M2']['payload']} / {m['M3']['payload']}。",
        f"6. M3-M0 Δ={boot['M3_minus_M0']['mean_delta']}，相对提升={boot['M3_minus_M0']['relative_gain']}。",
        f"7. M3-M2 Δ={boot['M3_minus_M2']['mean_delta']}，相对提升={boot['M3_minus_M2']['relative_gain']}。",
        f"8. M3-M1 Δ={boot['M3_minus_M1']['mean_delta']}（M1 仍为 Phase-10.5 冻结的 sequential exclusive，未改定义）。",
        f"9. 95% CI M0/M1/M2 = {boot['M3_minus_M0']['ci95']} / {boot['M3_minus_M1']['ci95']} / {boot['M3_minus_M2']['ci95']}。",
        f"10. completed blocks M0/M1/M2/M3 = {m['M0']['completed_blocks']} / {m['M1']['completed_blocks']} / {m['M2']['completed_blocks']} / {m['M3']['completed_blocks']}。",
        f"11. residual queue = {m['M0']['residual_queue']} / {m['M1']['residual_queue']} / {m['M2']['residual_queue']} / {m['M3']['residual_queue']}。",
        f"12. drop = {m['M0']['dropped']} / {m['M1']['dropped']} / {m['M2']['dropped']} / {m['M3']['dropped']}。",
        f"13. flight distance = {m['M0']['flight_distance']} / {m['M1']['flight_distance']} / {m['M2']['flight_distance']} / {m['M3']['flight_distance']}。",
        f"14. legality = {m['M0']['legality']} / {m['M1']['legality']} / {m['M2']['legality']} / {m['M3']['legality']}。",
        f"15. FINAL Gate={gate['FINAL']}。{gate['text']}",
        f"16. final_system_confirmed={gate['final_system_confirmed']}。",
        f"17. 可写结论：{payload['can_write']}",
        f"18. 不能写：{payload['cannot_write']}",
        "19. 是否重新训练或调过参数？否。",
        "20. 是否允许继续根据 final test 修改方法？否。",
        "",
        payload["markdown"],
        "",
        payload.get("cost_sentence", ""),
        "",
        f"墙钟 {payload['wall_s']:.1f}s。图：{payload['plots']}",
        "",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")
    OUTPUT.joinpath("HTW_Phase10_6_final_untouched_evaluation_report.md").write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    t0 = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    restore = restore_and_check()
    scen = freeze_untouched_scenarios()
    seeds = [int(s) for s in scen["scenario_seeds"]]
    if len(seeds) != N_SCENARIOS:
        raise RuntimeError("frozen scenario count must remain 64")
    overlap = sorted(set(seeds) & historical_blocked_seeds())
    if overlap:
        raise RuntimeError(f"untouched overlap with history; stopping: {overlap[:20]}")
    high, high_file_sha = load_frozen_high(HIGH_CKPT)
    high_param_sha = parameter_sha(high)
    if high_file_sha != EXPECTED_HIGH_SHA:
        raise RuntimeError("SHA mismatch after load")
    if high_param_sha != restore["arch"]["high_level"]["param_sha256"]:
        raise RuntimeError("high-level parameter SHA mismatch; stopping")
    write_preaccess_lock(scen, high_file_sha, high_param_sha)
    dump_frozen(
        OUTPUT / "phase106_protocol.json",
        {
            "phase": "10.6",
            "training": False,
            "one_shot": True,
            "n_scenarios": len(seeds),
            "hold": HOLD,
            "families": ["M0", "M1", "M2", "M3"],
            "label": "FINAL UNTOUCHED TEST",
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "no_further_method_development": True,
        },
    )
    if not OLD_DEEPSETS.is_file():
        raise RuntimeError("M2 DeepSets-old checkpoint missing; cannot invent a substitute")
    old_model = load_deepsets_util(0, OLD_DEEPSETS)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)
    methods = {
        "M0": rule_assigner(legal_myopic_assignment),
        "M1": rule_assigner(exclusive_assignment),
        "M2": learned_assigner(old_model),
        "M3": learned_assigner(high),
    }
    rows: list[dict[str, Any]] = []
    by: dict[str, list[dict[str, Any]]] = {key: [] for key in methods}
    for scenario in seeds:
        print(f"untouched eval {scenario}", flush=True)
        for fam, fn in methods.items():
            met = run_episode(scenario, family=fam, assign_fn=fn)
            rows.append(met)
            by[fam].append(met)
    _write_csv(OUTPUT / "phase106_per_scenario_metrics.csv", rows)

    def pack(fam: str, name: str) -> dict[str, Any]:
        return {
            "method_id": fam,
            "method": name,
            "payload": _mean(by[fam], "payload_bits"),
            "completed_blocks": _mean(by[fam], "completed_blocks"),
            "residual_queue": _mean(by[fam], "residual_queue_blocks"),
            "dropped": _mean(by[fam], "dropped_blocks"),
            "flight_distance": _mean(by[fam], "flight_distance_m"),
            "legality": _mean(by[fam], "exclusive_ok_frac"),
            "in_range": _mean(by[fam], "active_queued_in_range_frac"),
            "idle": _mean(by[fam], "idle_frac"),
            "n_uav_completed": _mean(by[fam], "n_uav_completed"),
            "switch_loss": _mean(by[fam], "target_change_loss_events"),
            "terminal_rho": _mean(by[fam], "terminal_unfinished_rho"),
            "mean_sinr": _mean(by[fam], "mean_sinr"),
            "mean_rate": _mean(by[fam], "mean_active_rate_bps"),
        }

    main_table = [pack("M0", "Myopic"), pack("M1", "Exclusive"), pack("M2", "DeepSets-old"), pack("M3", "FINAL-HIER")]
    _write_csv(OUTPUT / "phase106_final_results.csv", main_table)
    m3p = np.array([float(r["payload_bits"]) for r in sorted(by["M3"], key=lambda x: x["scenario"])])
    boot = {}
    for other, offset in (("M0", 0), ("M1", 1), ("M2", 2)):
        op = np.array([float(r["payload_bits"]) for r in sorted(by[other], key=lambda x: x["scenario"])])
        d = m3p - op
        item = paired_bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED + offset)
        item["relative_gain"] = float(d.mean() / max(float(np.mean(op)), 1.0e-9))
        boot[f"M3_minus_{other}"] = item
    dump_frozen(OUTPUT / "phase106_paired_bootstrap.json", boot)
    gate = classify_final(
        vs_m0=boot["M3_minus_M0"],
        vs_m1=boot["M3_minus_M1"],
        vs_m2=boot["M3_minus_M2"],
        blocks_m3=_mean(by["M3"], "completed_blocks"),
        blocks_m0=_mean(by["M0"], "completed_blocks"),
        blocks_m2=_mean(by["M2"], "completed_blocks"),
        legal_m3=_mean(by["M3"], "exclusive_ok_frac"),
        legal_m0=_mean(by["M0"], "exclusive_ok_frac"),
        legal_m2=_mean(by["M2"], "exclusive_ok_frac"),
        drop_m3=_mean(by["M3"], "dropped_blocks"),
        drop_m0=_mean(by["M0"], "dropped_blocks"),
        drop_m2=_mean(by["M2"], "dropped_blocks"),
        idle_m3=_mean(by["M3"], "idle_frac"),
        idle_m0=_mean(by["M0"], "idle_frac"),
        idle_m2=_mean(by["M2"], "idle_frac"),
    )
    dump_frozen(OUTPUT / "phase106_final_gate.json", gate | {"not_for_further_tuning": True, "trained": False})
    flight_m3 = _mean(by["M3"], "flight_distance_m")
    flight_m0 = _mean(by["M0"], "flight_distance_m")
    if _mean(by["M3"], "payload_bits") >= max(_mean(by["M0"], "payload_bits"), _mean(by["M2"], "payload_bits")) and flight_m3 > flight_m0:
        cost = "FINAL-HIER improves communication payload and service completion, at the cost of increased UAV travel distance. FINAL-HIER提高了累计数据采集量与任务完成量，但需要更大的无人机运动距离。Flight distance is not an energy model."
    else:
        cost = "See payload and flight columns; do not claim uniform dominance."
    if gate["FINAL"] == "A":
        can_write = "Utility-Aligned high-level assignment is effective on final L3-B; learned assignment beats legal myopic/exclusive on payload; order-free decoder, Task Option hold=15, and InRangeZeroMotionGate remain as the frozen stack."
        cannot_write = "HGNN优于DeepSets；PPO低层优于规则低层；Transformer贡献显著；全面优于所有baseline（忽略飞行距离）。"
    elif gate["FINAL"] == "B":
        can_write = "Untouched shows an advantage trend; statistical confirmation is incomplete."
        cannot_write = "最终确认的系统优势；HGNN/PPO/Transformer 贡献。"
    else:
        can_write = "Pre-final SYS-A did not reproduce as FINAL-A."
        cannot_write = "把 Pre-Final 优势写成最终 untouched 确认；HGNN/PPO/Transformer 贡献。"
    status = {
        "architecture_finalized": True,
        "training_finished": True,
        "final_untouched_consumed": True,
        "final_gate": gate["FINAL"],
        "final_system_confirmed": gate["final_system_confirmed"],
        "HGNN_selected": False,
        "low_level_PPO_selected": False,
        "DeepSets_selected": True,
        "utility_aligned_selected": True,
        "order_free_decoder_selected": True,
        "task_option_selected": True,
        "hold": 15,
        "executor": "fly_toward_with_in_range_zero_motion_gate",
        "method_development_stopped": True,
    }
    dump_frozen(OUTPUT / "phase106_final_scientific_status.json", status)
    lock = json.loads((OUTPUT / "phase106_preaccess_lock.json").read_text(encoding="utf-8"))
    lock["final_untouched_consumed"] = True
    dump_frozen(OUTPUT / "phase106_preaccess_lock.json", lock)
    consumed = dict(scen)
    consumed["consumed"] = True
    dump_frozen(OUTPUT / "phase106_untouched_scenario_manifest.json", consumed)
    plots = _plots(by)
    md = markdown_tables(main_table, boot)
    summary = {
        "high_sha": high_file_sha,
        "n_scenarios": len(seeds),
        "overlap_count": 0,
        "consumed": True,
        "main_table": main_table,
        "bootstrap": boot,
        "final_gate": gate,
        "can_write": can_write,
        "cannot_write": cannot_write,
        "cost_sentence": cost,
        "markdown": md,
        "plots": plots,
        "wall_s": perf_counter() - t0,
        "trained": False,
        "high_sha_unchanged": sha256_bytes(HIGH_CKPT.read_bytes()) == high_file_sha,
    }
    dump_frozen(OUTPUT / "phase106_execution_summary.json", summary)
    write_report(summary)
    print(json.dumps({"FINAL": gate["FINAL"], "M3": main_table[3]["payload"], "trained": False, "consumed": True}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
