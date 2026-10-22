"""LL-V2.2: freeze validation, train seeds 1-4, evaluate five seeds vs shared L0."""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.protocol import EXPECTED_HIGH_SHA

from .eval import evaluate_validation
from .protocol import (
    BOOT_SEED,
    CURVE_TRAIN_N,
    D_SAFE_M,
    LOW_SEEDS,
    N_BOOT,
    NEW_SEEDS,
    OUTPUT,
    PHASE1B_CKPT0,
    PHASE2,
    PPO,
    ROOT,
    env_sha,
    high_sha,
    load_p1b_prefreeze,
    load_pools,
)
from .scenarios import freeze_validation_manifest
from .stats import bootstrap_deltas
from .train import train_seed


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_actor(path: Path):
    high, digest = load_frozen_high()
    actor, critic = make_networks()
    payload = torch.load(path, map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    worker = ResidualWorker(high, actor, critic, service_lock=True)
    return worker, digest


def plot_all(per_seed: list[dict], l0_payload: float, curves: list[dict], rel_rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    seeds = [r["model_seed"] for r in per_seed]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(s) for s in seeds], [r["payload"] for r in per_seed], color="#2a6f97")
    ax.axhline(l0_payload, color="#6c757d", ls="--", label="L0")
    ax.set_ylabel("payload")
    ax.set_title("5-seed P1B payload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "five_seed_payload.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(s) for s in seeds], [r["delta"] for r in per_seed], color="#2a6f97")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("delta vs L0")
    ax.set_title("per-seed delta")
    fig.tight_layout()
    fig.savefig(out / "per_seed_delta.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    by = {}
    for row in curves:
        by.setdefault(row["model_seed"], []).append(row)
    for seed, rows in by.items():
        xs = [r["update"] for r in rows]
        ax.plot(xs, [r["payload"] for r in rows], marker="o", label=f"seed{seed}")
    ax.set_xlabel("update")
    ax.set_ylabel("payload")
    ax.set_title("learning curve by seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curve_by_seed.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(s) for s in seeds], [r["projector_intervention_rate"] for r in per_seed])
    ax.set_title("projector intervention by seed")
    fig.tight_layout()
    fig.savefig(out / "projector_by_seed.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(s) for s in seeds], [r["min_sep_worst"] for r in per_seed])
    ax.axhline(1.0, color="r", ls="--")
    ax.set_title("minimum separation worst")
    fig.tight_layout()
    fig.savefig(out / "min_sep_by_seed.png", dpi=140)
    plt.close(fig)
    if rel_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = sorted({r["variant"] for r in rel_rows if r["regime"] in {"near", "high", "all"}})
        x = np.arange(len(LOW_SEEDS))
        width = 0.12
        near_shuffle = [next((r["mean_l2"] for r in rel_rows if r["model_seed"] == s and r["variant"] == "neighbor_shuffle" and r["regime"] == "near"), 0.0) for s in LOW_SEEDS]
        ax.bar(x, near_shuffle, width, label="neighbor shuffle near")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in LOW_SEEDS])
        ax.set_title("relation feature sensitivity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "relation_sensitivity.png", dpi=140)
        plt.close(fig)
    _ = labels if rel_rows else None


def main() -> dict:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefreeze = load_p1b_prefreeze()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch")
    if not PHASE1B_CKPT0.is_file():
        raise RuntimeError("missing Phase-1B seed0 update100 checkpoint")
    train_seeds, _design = load_pools()
    curve_seeds = list(train_seeds[:CURVE_TRAIN_N])
    val_path = PHASE2 / "LLV2_PHASE2_VALIDATION_MANIFEST.json"
    val_manifest = freeze_validation_manifest(val_path)
    val_seeds = [int(s) for s in val_manifest["scenario_seeds"]]
    train_meta = []
    history_rows = []
    curve_rows = []
    ckpt_rows = [
        {
            "model_seed": 0,
            "update": 100,
            "path": str(PHASE1B_CKPT0),
            "primary": True,
            "retrained": False,
        }
    ]
    seed0_dst = OUTPUT / "ckpt_seed0_update100.pt"
    shutil.copyfile(PHASE1B_CKPT0, seed0_dst)
    for seed in NEW_SEEDS:
        meta = train_seed(int(seed), train_seeds, curve_seeds)
        train_meta.append(meta)
        history_rows.extend(meta["history"])
        curve_rows.extend(meta["curve"])
        ckpt_rows.extend(meta["ckpts"])
    p1b_eval = list(csv.DictReader((PHASE2.parent / "phase1b" / "LLV2_PHASE1B_EVAL_CURVE.csv").open(encoding="utf-8")))
    for rec in p1b_eval:
        curve_rows.append(
            {
                "model_seed": 0,
                "update": int(float(rec["update"])),
                "payload": float(rec["payload"]),
                "completed_blocks": float(rec["completed_blocks"]),
                "mean_residual_norm": float(rec["mean_residual_norm"]),
                "projector_intervention_rate": float(rec["projector_intervention_rate"]),
                "source": "phase1b_design_historical",
            }
        )
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    l0_worker = ResidualWorker(high, actor, critic, service_lock=True)
    t_val = time.time()
    l0 = evaluate_validation(l0_worker, val_seeds, zero_residual=True, diagnose=False)
    l0_payloads = {int(r["scenario"]): r["payload"] for r in l0["scenarios"]}
    ckpt_map = {0: PHASE1B_CKPT0}
    for seed in NEW_SEEDS:
        ckpt_map[int(seed)] = OUTPUT / f"ckpt_seed{seed}_update100.pt"
    per_seed = []
    per_scen_rows = []
    safety_rows = []
    proj_rows = []
    rel_csv = []
    sep_csv = []
    interf_csv = []
    delta_mat = np.zeros((len(LOW_SEEDS), len(val_seeds)), dtype=np.float64)
    for si, seed in enumerate(LOW_SEEDS):
        worker, digest = _load_actor(ckpt_map[int(seed)])
        if digest != EXPECTED_HIGH_SHA:
            raise RuntimeError("high SHA changed at eval")
        summary = evaluate_validation(worker, val_seeds, zero_residual=False, diagnose=True)
        payloads = {int(r["scenario"]): r["payload"] for r in summary["scenarios"]}
        deltas = []
        for j, scen in enumerate(val_seeds):
            dlt = payloads[int(scen)] - l0_payloads[int(scen)]
            deltas.append(dlt)
            delta_mat[si, j] = dlt
            rec = [r for r in summary["scenarios"] if int(r["scenario"]) == int(scen)][0]
            per_scen_rows.append({"model_seed": int(seed), "method": "P1B", **rec, "l0_payload": l0_payloads[int(scen)], "delta": dlt})
        mean_p = summary["payload"]
        mean_d = float(np.mean(deltas))
        per_seed.append(
            {
                "model_seed": int(seed),
                "payload": mean_p,
                "delta": mean_d,
                "relative_gain": mean_d / max(l0["payload"], 1.0),
                "median_delta": float(np.median(deltas)),
                "positive_scenario_fraction": float(np.mean(np.asarray(deltas) > 0)),
                "completed_blocks": summary["completed_blocks"],
                "residual_queue": summary["residual_queue"],
                "drop": summary["drop"],
                "flight_distance": summary["flight_distance"],
                "mean_sinr": summary["mean_sinr"],
                "mean_active_rate": summary["mean_active_rate"],
                "unfinished_progress": summary["unfinished_progress"],
                "min_sep": summary["min_sep"],
                "min_sep_worst": summary["min_sep_worst"],
                "collision_count": summary["collision_count"],
                "near_events": summary["near_events"],
                "projector_intervention_rate": summary["projector_intervention_rate"],
                "projector_failure_count": summary["projector_failure_count"],
                "service_lock_violation_count": summary["service_lock_violation_count"],
                "mean_residual_norm": summary["mean_residual_norm"],
                "smoothness": summary["smoothness"],
                "SERVICE_SAFE_ENVELOPE_among_intervened": summary["SERVICE_SAFE_ENVELOPE_among_intervened"],
                "PAIRWISE_COLLISION_among_intervened": summary["PAIRWISE_COLLISION_among_intervened"],
            }
        )
        safety_rows.append(
            {
                "model_seed": int(seed),
                "collision_count": summary["collision_count"],
                "min_sep_worst": summary["min_sep_worst"],
                "projector_failure_count": summary["projector_failure_count"],
                "service_lock_violation_count": summary["service_lock_violation_count"],
                "safety_ok": bool(
                    summary["collision_count"] == 0
                    and summary["projector_failure_count"] == 0
                    and summary["service_lock_violation_count"] == 0
                    and summary["min_sep_worst"] + 1.0e-6 >= D_SAFE_M
                ),
            }
        )
        proj_rows.append(
            {
                "model_seed": int(seed),
                "projector_intervention_rate": summary["projector_intervention_rate"],
                "STEP_LIMIT_among_intervened": summary["STEP_LIMIT_among_intervened"],
                "PAIRWISE_COLLISION_among_intervened": summary["PAIRWISE_COLLISION_among_intervened"],
                "SERVICE_SAFE_ENVELOPE_among_intervened": summary["SERVICE_SAFE_ENVELOPE_among_intervened"],
                "MULTI_CONSTRAINT_among_intervened": summary["MULTI_CONSTRAINT_among_intervened"],
                "OTHER_among_intervened": summary["OTHER_among_intervened"],
            }
        )
        for row in summary.get("relation", []):
            rel_csv.append({"model_seed": int(seed), **row})
        sep = summary.get("separation", {})
        sep_csv.append({"model_seed": int(seed), **sep})
        interf_csv.append({"model_seed": int(seed), **summary.get("interference_response", {})})
    for rec in l0["scenarios"]:
        per_scen_rows.append({"model_seed": "L0", "method": "L0", **rec, "l0_payload": rec["payload"], "delta": 0.0})
    boot = bootstrap_deltas(delta_mat, n_boot=N_BOOT, seed=BOOT_SEED)
    all_safe = all(r["safety_ok"] for r in safety_rows)
    lock_ok = all(r["service_lock_violation_count"] == 0 for r in safety_rows)
    env_low = all(r["SERVICE_SAFE_ENVELOPE_among_intervened"] < 0.25 for r in proj_rows)
    blocks_ok = float(np.mean([r["completed_blocks"] for r in per_seed])) + 1e-9 >= l0["completed_blocks"]
    mean_delta = boot["overall_mean_delta"]
    hier_lo = boot["hierarchical_ci95"]["low"]
    seed_lo = boot["seed_cluster_ci95"]["low"]
    n_pos = boot["n_positive_seeds"]
    med = boot["median_delta_seed_means"]
    if (
        mean_delta > 0
        and hier_lo > 0
        and seed_lo > 0
        and n_pos >= 4
        and med >= 0
        and all_safe
        and lock_ok
        and blocks_ok
        and env_low
    ):
        ms = "A"
        stable = True
        text = "stable_residual_advantage across independent validation."
    elif mean_delta > 0 and (n_pos >= 3) and all_safe:
        ms = "B"
        stable = False
        text = "advantage trend, not stable."
    else:
        ms = "C"
        stable = False
        text = "no stable residual value or safety failure."
        if not all_safe:
            text = "safety failure on at least one seed."
    near_l2 = []
    high_if = []
    high_sinr = []
    for seed in LOW_SEEDS:
        near_l2.append(
            next((r["mean_l2"] for r in rel_csv if r["model_seed"] == seed and r["variant"] == "neighbor_shuffle" and r["regime"] == "near"), 0.0)
        )
        high_if.append(
            next((r["mean_l2"] for r in rel_csv if r["model_seed"] == seed and r["variant"] == "interf_shuffle" and r["regime"] == "high"), 0.0)
        )
        high_sinr.append(
            next((r["mean_l2"] for r in rel_csv if r["model_seed"] == seed and r["variant"] == "sinr_shuffle" and r["regime"] == "high"), 0.0)
        )
    n_unused = sum(1 for x in near_l2 if x < 1.0e-4)
    n_rel = sum(1 for a, b, c in zip(near_l2, high_if, high_sinr) if max(a, b, c) > 0.01)
    if n_unused >= 3 and n_rel <= 1:
        rel_gate = 0
    elif n_rel >= 3:
        rel_gate = 2
    else:
        rel_gate = 1
    if ms == "A" and rel_gate in {0, 1}:
        nxt = "next may target relation-feature utilization only; do not start it automatically."
    elif ms == "A" and rel_gate == 2:
        nxt = "do not add relation modules; prefer more formal validation later."
    elif ms == "B":
        nxt = "analyze seed instability; do not add relation modules yet."
    else:
        nxt = "stop stacking complex RL; keep L0."
    val_wall = time.time() - t_val
    gate = {
        "LLV2_MS": ms,
        "stable_residual_advantage": stable,
        "REL": rel_gate,
        "text": text,
        "next": nxt,
        "all_seeds_safe": all_safe,
        "lock_ok": lock_ok,
        "envelope_low": env_low,
        "blocks_ok": blocks_ok,
        "recommend_next_train": False,
    }
    val_manifest["consumed"] = True
    _dump(val_path, val_manifest)
    protocol = {
        "name": "LL-V2.2",
        "method": "frozen LL-V2.1B ServiceResidualLock residual MAPPO",
        "p1b_prefreeze": prefreeze,
        "ppo": PPO.to_dict(),
        "high_sha": high_sha(),
        "env_sha": env_sha(),
        "retrained_seed0": False,
        "new_seeds": list(NEW_SEEDS),
        "validation": "LLV2_PHASE2_VALIDATION_V1",
        "n_validation": len(val_seeds),
        "visited_phase106": False,
        "primary_checkpoint": "update100",
        "wall_s": time.time() - started,
        "validation_wall_s": val_wall,
        "train_walls": {str(m["model_seed"]): m["wall_s"] for m in train_meta},
    }
    _dump(OUTPUT / "LLV2_PHASE2_PROTOCOL.json", protocol)
    _dump(OUTPUT / "LLV2_PHASE2_TRAINING_SEEDS.json", {"seed0": "reused phase1b u100", "new": list(NEW_SEEDS), "train_pool": "LLV2_train"})
    _dump(OUTPUT / "LLV2_PHASE2_CHECKPOINT_MANIFEST.json", {"primary": "update100", "checkpoints": ckpt_rows, "best_of_curve_forbidden": True})
    _csv(OUTPUT / "LLV2_PHASE2_TRAINING_HISTORY.csv", history_rows)
    _csv(OUTPUT / "LLV2_PHASE2_PER_SEED_METRICS.csv", per_seed)
    _csv(OUTPUT / "LLV2_PHASE2_PER_SCENARIO_METRICS.csv", per_scen_rows)
    _csv(OUTPUT / "LLV2_PHASE2_SAFETY_METRICS.csv", safety_rows)
    _csv(OUTPUT / "LLV2_PHASE2_PROJECTOR_BREAKDOWN.csv", proj_rows)
    _csv(OUTPUT / "LLV2_PHASE2_RELATION_UTILIZATION.csv", rel_csv)
    _csv(OUTPUT / "LLV2_PHASE2_SEPARATION_RESPONSE.csv", sep_csv)
    _csv(OUTPUT / "LLV2_PHASE2_INTERFERENCE_RESPONSE.csv", interf_csv)
    _csv(OUTPUT / "LLV2_PHASE2_LEARNING_CURVES.csv", curve_rows)
    _dump(OUTPUT / "LLV2_PHASE2_BOOTSTRAP.json", boot)
    _dump(OUTPUT / "LLV2_PHASE2_GATE.json", gate | {"bootstrap": boot, "l0_payload": l0["payload"]})
    plot_all(per_seed, l0["payload"], curve_rows, rel_csv, OUTPUT / "figures")
    report = _report(per_seed, l0, boot, gate, val_seeds, train_meta, safety_rows, proj_rows, rel_gate, near_l2, high_if, high_sinr, sep_csv, interf_csv)
    _dump(OUTPUT / "HTW_LowLevelV2_Phase2_Multiseed_Validation_CN.md", report)
    names = [
        "LLV2_PHASE2_PROTOCOL.json",
        "LLV2_PHASE2_TRAINING_SEEDS.json",
        "LLV2_PHASE2_CHECKPOINT_MANIFEST.json",
        "LLV2_PHASE2_TRAINING_HISTORY.csv",
        "LLV2_PHASE2_PER_SEED_METRICS.csv",
        "LLV2_PHASE2_PER_SCENARIO_METRICS.csv",
        "LLV2_PHASE2_SAFETY_METRICS.csv",
        "LLV2_PHASE2_PROJECTOR_BREAKDOWN.csv",
        "LLV2_PHASE2_RELATION_UTILIZATION.csv",
        "LLV2_PHASE2_SEPARATION_RESPONSE.csv",
        "LLV2_PHASE2_INTERFERENCE_RESPONSE.csv",
        "LLV2_PHASE2_BOOTSTRAP.json",
        "LLV2_PHASE2_GATE.json",
        "HTW_LowLevelV2_Phase2_Multiseed_Validation_CN.md",
    ]
    (PHASE2 / "LLV2_PHASE2_VALIDATION_MANIFEST.json").write_text(val_path.read_text(encoding="utf-8"), encoding="utf-8")
    for name in names:
        src = OUTPUT / name
        (PHASE2 / name).write_bytes(src.read_bytes())
        (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).write_bytes(src.read_bytes())
    (ROOT / "results/root_exports/LLV2_PHASE2_VALIDATION_MANIFEST.json").write_bytes(val_path.read_bytes())
    (ROOT / "docs/archive/phase_reports/HTW_LowLevelV2_Phase2_Multiseed_Validation_CN.md").write_text(report, encoding="utf-8")
    return {"gate": gate, "l0": l0["payload"], "mean_delta": mean_delta, "n_val": len(val_seeds)}


def _report(per_seed, l0, boot, gate, val_seeds, train_meta, safety_rows, proj_rows, rel_gate, near_l2, high_if, high_sinr, sep_csv, interf_csv) -> str:
    walls = {m["model_seed"]: round(m["wall_s"], 1) for m in train_meta}
    lines = [
        "# HTW Low-Level V2 Phase-2 Multi-Seed Independent Validation",
        "",
        "方法冻结自 LL-V2.1B。未改 lock/reward/网络/超参。seed0 不重训。非 confirmation / untouched。",
        "",
        f"1. 新增训练 seed：1,2,3,4；seed0 复用 Phase-1B u100。训练 wall-s：{walls}",
        "2. 每新 seed 预算均为 100×48×4=19200 env transitions。",
        f"3. 新 validation：{len(val_seeds)} scenarios（LLV2_PHASE2_VALIDATION_V1）。",
        "4. 与历史 train/design/confirmation/untouched 无交集：是（分配前全量扫描）。",
        f"5. L0 mean payload：{l0['payload']}",
        f"6. 五 seed payload：{[r['payload'] for r in per_seed]}",
        f"7. Delta_s：{[r['delta'] for r in per_seed]}",
        f"8. 正向 seed 数：{boot['n_positive_seeds']}/5",
        f"9. overall mean delta：{boot['overall_mean_delta']}",
        f"10. hierarchical 95% CI：{boot['hierarchical_ci95']}",
        f"11. seed-cluster 95% CI：{boot['seed_cluster_ci95']}",
        f"12. completed blocks P1B mean / L0：{float(np.mean([r['completed_blocks'] for r in per_seed]))} / {l0['completed_blocks']}",
        f"13. collision per seed：{[r['collision_count'] for r in safety_rows]}",
        f"14. min_sep_worst per seed：{[r['min_sep_worst'] for r in safety_rows]}",
        f"15. projector_failure per seed：{[r['projector_failure_count'] for r in safety_rows]}",
        f"16. projector intervention per seed：{[r['projector_intervention_rate'] for r in proj_rows]}",
        f"17. ENVELOPE among intervened：{[r['SERVICE_SAFE_ENVELOPE_among_intervened'] for r in proj_rows]}",
        f"18. neighbor shuffle near L2：{near_l2}",
        f"19. interf/sinr shuffle high L2：{high_if} / {high_sinr}",
        f"20. separation response：{sep_csv}",
        f"21. high-interf ΔSINR/Δrate：{interf_csv}",
        f"22. LLV2-MS Gate：{gate['LLV2_MS']}",
        f"23. stable_residual_advantage：{gate['stable_residual_advantage']}",
        f"24. REL Gate：{rel_gate}",
        f"25. 下一阶段建议：{gate['next']} 本阶段停止，不自动开训。",
        "",
        f"scenario-paired CI：{boot['scenario_paired_ci95']}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
