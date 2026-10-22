"""HTW Integration Phase-3. Executor-conditioned utility. Five seeds. No low-level training."""
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
from wt.lowlevel_v2.integration_phase2.train_hl import split_metrics
from wt.lowlevel_v2.integration_phase2.utility_gen import load_ll_worker, replay_to_hold
from wt.lowlevel_v2.integration_phase2c.search_core import greedy_trace
from wt.lowlevel_v2.phase1b.controller import make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design

from .dataset import load_exec_split, materialize_executor_dataset
from .protocol import (
    BATCH,
    CHECKPOINT_RULE,
    EPOCHS,
    EXPECTED_HIGH_SHA,
    EXPECTED_UTIL_SHA,
    HOLD,
    INTEG3,
    LAMBDA_RANK,
    LAMBDA_REF,
    LOW_SEEDS,
    LR,
    MODEL_SEEDS,
    OLD_HL_CKPT,
    OUTPUT,
    PHASE2B_OUTPUT,
    PHASE2D_OUTPUT,
    PHASE2_OUTPUT,
    POOL_SIZE,
    ROOT,
    TRAIN_NAME,
    TRAIN_WORKERS,
    UTILITY_EPS,
    VAL_NAME,
    abs_ckpt,
    ckpt_path,
    exec_ckpt,
    high_sha,
    load_compat_gate,
    load_phase2_gate,
    load_phase2a_gate,
    load_phase2d_gate,
    prefix_ckpt,
)
from .tests import run_pretests
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
    for seed in (PHASE2D_OUTPUT / "utility_cache.json", PHASE2B_OUTPUT / "utility_cache.json"):
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


def load_loop_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rec = dict(row)
            for key, val in list(rec.items()):
                if key in {"high", "high_seed", "family"}:
                    continue
                try:
                    rec[key] = float(val)
                except (TypeError, ValueError):
                    pass
            rows.append(rec)
    return rows


def eval_model(spec: dict, val: dict, rows: dict) -> None:
    print(f"eval {spec['family']} {spec['seed']}", flush=True)
    model = load_high(spec["path"])
    within = split_metrics(model, val, mechanism=False)
    rows["within"].append({"family": spec["family"], "high_seed": spec["seed"], **within})
    for index in range(int(val["assignment"].shape[0])):
        encoded = _encode_one(model, val, index, "cpu")
        snap = make_snap(val["uav_positions"][index], val["buoy_positions"][index])
        pool = np.asarray(val["candidates"][index], dtype=np.int64)
        u_pool = np.asarray(val["utility"][index], dtype=np.float64)
        hashes = list(val["hashes"][index])
        best_u = float(np.max(u_pool))
        with torch.no_grad():
            greedy = greedy_trace(model, encoded, snap, val["buoy_mask"][index])
            lp = candidate_logprobs(model, encoded, snap, pool, val["buoy_mask"][index]).detach().cpu().numpy()
        g_assn = greedy["assignment"]
        u_g = u_new(val, index, g_assn)
        nr = decode_nr(best_u, u_g)
        rows["decode"].append(
            {
                "family": spec["family"],
                "high_seed": spec["seed"],
                "state_id": index,
                "U_decode": u_g,
                "U_best_pool": best_u,
                "DecodeNR": nr,
                "in_pool": assignment_hash(g_assn) in set(hashes),
                "PoolMass": float(np.exp(float(np.logaddexp.reduce(lp)))),
                "idle_fraction": float(np.mean(g_assn[:6] < 0)),
            }
        )
        print(f"  {spec['family']} {spec['seed']} {index+1}/{val['assignment'].shape[0]} NR={nr:.3f}", flush=True)
        save_u_cache()
    del model


def prefix_baselines(p2d: dict) -> dict:
    nr = [float(p2d["PREFIX_DecodeNR"][str(s)]["median"]) for s in (0, 1)]
    sp = [float(p2d["within_PREFIX"][str(s)]["spearman"]) for s in (0, 1)]
    pay = [float(p2d["PREFIX_closed_loop_seed"][str(s)]) for s in (0, 1)]
    return {
        "DecodeNR_seed": {str(s): float(p2d["PREFIX_DecodeNR"][str(s)]["median"]) for s in (0, 1)},
        "DecodeNR_aggregate": float(np.median(nr)),
        "spearman_seed": {str(s): float(p2d["within_PREFIX"][str(s)]["spearman"]) for s in (0, 1)},
        "spearman_mean": float(np.mean(sp)),
        "closed_loop": float(p2d["PREFIX_closed_loop"]),
        "closed_loop_seed": {str(s): float(p2d["PREFIX_closed_loop_seed"][str(s)]) for s in (0, 1)},
        "closed_loop_std": float(np.std(pay, ddof=1)),
        "OLD_closed_loop": float(p2d["OLD_closed_loop"]),
        "ABS_closed_loop": float(p2d["ABS_closed_loop"]),
    }


def classify(rows: dict, loop_rows: list[dict], pre: dict, dataset: dict, p2d: dict, ckpt_rows: list[dict]) -> dict:
    base = prefix_baselines(p2d)
    exec_nr = {int(s): summarize([r["DecodeNR"] for r in rows["decode"] if r["family"] == "EXEC" and int(r["high_seed"]) == int(s)]) for s in MODEL_SEEDS}
    exec_nr_medians = [exec_nr[int(s)]["median"] for s in MODEL_SEEDS]
    exec_agg = float(np.median(exec_nr_medians))
    within_exec = {int(s): next(r for r in rows["within"] if r["family"] == "EXEC" and int(r["high_seed"]) == int(s)) for s in MODEL_SEEDS}
    exec_sp = [float(within_exec[int(s)]["spearman"]) for s in MODEL_SEEDS]
    exec_sp_mean = float(np.mean(exec_sp))
    exec_pay_seed = {
        int(s): float(np.mean([float(r["payload"]) for r in loop_rows if (r.get("family") == "EXEC" or r.get("high") == "EXEC-HL") and int(r["high_seed"]) == int(s)]))
        for s in MODEL_SEEDS
    }
    exec_pay = float(np.mean(list(exec_pay_seed.values())))
    exec_std = float(np.std(list(exec_pay_seed.values()), ddof=1))
    old_pay = base["OLD_closed_loop"]
    loop_ok = exec_pay + 1.0e-9 >= old_pay
    nr_ok = exec_agg <= base["DecodeNR_aggregate"] + 1.0e-12
    var_ok = exec_std < base["closed_loop_std"] - 1.0e-12
    sp_ok = exec_sp_mean > base["spearman_mean"] + 1.0e-12
    idle = [r["idle_fraction"] for r in rows["decode"] if r["family"] == "EXEC"]
    idle_ok = all(0.15 < float(x) < 0.90 for x in idle)
    legal_ok = all(float(r.get("conflict_rate", 0.0)) <= 1.0e-8 for r in rows["within"] if r["family"] == "EXEC")
    passed = bool(loop_ok and nr_ok and var_ok and sp_ok)
    fails = []
    if not loop_ok:
        fails.append(f"closed-loop {exec_pay:.3e} < OLD {old_pay:.3e}")
    if not nr_ok:
        fails.append(f"DecodeNR {exec_agg:.4f} > PREFIX {base['DecodeNR_aggregate']:.4f}")
    if not var_ok:
        fails.append(f"seed std {exec_std:.3e} not < PREFIX {base['closed_loop_std']:.3e}")
    if not sp_ok:
        fails.append(f"Spearman {exec_sp_mean:.4f} not > PREFIX {base['spearman_mean']:.4f}")
    text = "Executor-conditioned supervision reduces HL-LL utility mismatch." if passed else "Executor-conditioned supervision did not pass. " + "; ".join(fails)
    return {
        "EXEC": "A" if passed else "C",
        "executor_conditioned_supervision_confirmed": passed,
        "training_low_level": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "COLREL": "C",
        "COMM": "B",
        "text": text,
        "next_step": "Stop. Do not retune q, pool, decoder, beam, or checkpoint rule. Do not start the next phase.",
        "fail_reasons": fails,
        "pretests": {"q": pre["q"]["pass"], "identity": pre["identity"]["pass"], "grad": pre["grad"]["pass"]},
        "U_LL": dataset["U_LL"],
        "OLD_closed_loop": old_pay,
        "ABS_closed_loop": base["ABS_closed_loop"],
        "PREFIX_closed_loop": base["closed_loop"],
        "EXEC_closed_loop": exec_pay,
        "EXEC_closed_loop_seed": {str(s): exec_pay_seed[int(s)] for s in MODEL_SEEDS},
        "PREFIX_closed_loop_std": base["closed_loop_std"],
        "EXEC_closed_loop_std": exec_std,
        "PREFIX_DecodeNR_aggregate": base["DecodeNR_aggregate"],
        "PREFIX_DecodeNR": base["DecodeNR_seed"],
        "EXEC_DecodeNR": {str(s): exec_nr[int(s)] for s in MODEL_SEEDS},
        "EXEC_DecodeNR_aggregate": exec_agg,
        "PREFIX_spearman_mean": base["spearman_mean"],
        "EXEC_spearman": {str(s): float(within_exec[int(s)]["spearman"]) for s in MODEL_SEEDS},
        "EXEC_spearman_mean": exec_sp_mean,
        "gate_loop_ok": loop_ok,
        "gate_nr_ok": nr_ok,
        "gate_var_ok": var_ok,
        "gate_spearman_ok": sp_ok,
        "idle_ok": idle_ok,
        "legal_ok": legal_ok,
        "best_epochs": ckpt_rows,
        "in_pool_EXEC": {str(s): float(np.mean([r["in_pool"] for r in rows["decode"] if r["family"] == "EXEC" and int(r["high_seed"]) == int(s)])) for s in MODEL_SEEDS},
        "within_EXEC": {str(s): within_exec[int(s)] for s in MODEL_SEEDS},
    }


def plot_figures(rows: dict, loop_rows: list[dict], gate: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    labs = [f"E{s}" for s in MODEL_SEEDS]
    nrs = [gate["EXEC_DecodeNR"][str(s)]["median"] for s in MODEL_SEEDS]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labs, nrs)
    ax.axhline(gate["PREFIX_DecodeNR_aggregate"], color="C1", linestyle="--", label="PREFIX")
    ax.set_title("DecodeNR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exec_decode_nr.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(gate["OLD_closed_loop"] / 1e6, color="k", linestyle="--", label="OLD")
    ax.axhline(gate["ABS_closed_loop"] / 1e6, color="C1", linestyle=":", label="ABS")
    ax.axhline(gate["PREFIX_closed_loop"] / 1e6, color="C2", linestyle="-.", label="PREFIX")
    ax.bar(labs, [gate["EXEC_closed_loop_seed"][str(s)] / 1e6 for s in MODEL_SEEDS], label="EXEC")
    ax.set_ylabel("payload Mbit")
    ax.set_title("closed-loop payload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exec_closed_loop.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labs, [gate["EXEC_spearman"][str(s)] for s in MODEL_SEEDS])
    ax.axhline(gate["PREFIX_spearman_mean"], color="C1", linestyle="--", label="PREFIX mean")
    ax.set_title("P vs U_LL Spearman")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exec_spearman.png", dpi=130)
    plt.close(fig)


def write_report(gate: dict, protocol: dict, dataset: dict) -> str:
    epochs = "; ".join(f"seed{r.get('high_seed')} epoch={r.get('best_epoch')}" for r in gate.get("best_epochs") or [])
    nr = ", ".join(f"seed{s} median={gate['EXEC_DecodeNR'][str(s)]['median']:.4f}" for s in MODEL_SEEDS)
    sp = ", ".join(f"seed{s}={gate['EXEC_spearman'][str(s)]:.4f}" for s in MODEL_SEEDS)
    pay = ", ".join(f"seed{s}={gate['EXEC_closed_loop_seed'][str(s)]:.6e}" for s in MODEL_SEEDS)
    fails = "; ".join(gate.get("fail_reasons") or []) or "none"
    u = gate["U_LL"]
    return f"""# HTW Integration Phase-3：Executor-Conditioned Utility Supervision

本阶段 **只改标签**：用冻结 Low-Level executor 的 \(U_{{\\mathrm{{LL}}}}\) 线性归一化得到 \(q\)，损失为 \(L_{{\\mathrm{{exec}}}}=-\\sum_i q_i\\log P_\\theta(\\Gamma_i\\mid s)\)。未改低层 MAPPO、SafetyProjector、collision/interference、reward、actor/critic、decoder、DeepSets、ranking/reference、hold、Horizon、candidate pool。正式推理仍是 joint greedy。PREFIX 作为 baseline，未覆盖旧实验。

---

## 审计

1. **executor utility label？** mean={u['mean']:.6g}；std={u['std']:.6g}；min={u['min']:.6g}；max={u['max']:.6g}；n={u['n']}
2. **HL 概率 vs \(U_{{\\mathrm{{LL}}}}\) Spearman？** PREFIX mean={gate['PREFIX_spearman_mean']:.4f}；EXEC mean={gate['EXEC_spearman_mean']:.4f}（{sp}）
3. **DecodeNR？** PREFIX aggregate={gate['PREFIX_DecodeNR_aggregate']:.4f}；EXEC aggregate={gate['EXEC_DecodeNR_aggregate']:.4f}（{nr}）
4. **normalized regret？** 即上列 DecodeNR（\((U_{{\\mathrm{{best}}}}-U_{{\\mathrm{{decode}}}})/(|U_{{\\mathrm{{best}}}}|+\\epsilon)\)）
5. **closed-loop payload？** OLD={gate['OLD_closed_loop']:.6e}；ABS={gate['ABS_closed_loop']:.6e}；PREFIX={gate['PREFIX_closed_loop']:.6e}；EXEC={gate['EXEC_closed_loop']:.6e}（{pay}）
6. **seed 稳定性？** PREFIX std={gate['PREFIX_closed_loop_std']:.6e}；EXEC std={gate['EXEC_closed_loop_std']:.6e}

## 其余记录

- **selected epochs？** {epochs}
- **q 构造？** {dataset['q_method']}；artificial_q=false；prefix_label=false
- **数据集 SHA？** {dataset['source_utility_sha']}
- **Gate loop/NR/var/Spearman？** {gate['gate_loop_ok']}/{gate['gate_nr_ok']}/{gate['gate_var_ok']}/{gate['gate_spearman_ok']}
- **EXEC Gate？** {gate['EXEC']}
- **confirmed？** {str(gate['executor_conditioned_supervision_confirmed']).lower()}
- **失败原因？** {fails}
- **本阶段是否修改低层？否。**
- **是否重新碰 collision/interference？否。**

{gate['text']}

{gate['next_step']}

本阶段停止。不要自动进入下一阶段。
"""


def publish(report: str) -> None:
    names = [
        "HL_EXEC_PROTOCOL.json",
        "HL_EXEC_DATASET.json",
        "HL_EXEC_LABEL_STATS.json",
        "HL_EXEC_PRETESTS.json",
        "HL_EXEC_TRAINING_HISTORY.csv",
        "HL_EXEC_HIGHLEVEL_CHECKPOINTS.json",
        "HL_EXEC_WITHIN_POOL.csv",
        "HL_EXEC_DECODE_UTILITY.csv",
        "HL_EXEC_CLOSED_LOOP_DESIGN.csv",
        "HL_EXEC_GATE.json",
        "HTW_HighLevel_ExecutorConditioned_Supervision_CN.md",
    ]
    for name in names:
        src = OUTPUT / name
        if src.is_file():
            shutil.copy2(src, INTEG3 / name)
            shutil.copy2(src, ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name)
    fig_src = OUTPUT / "figures"
    fig_dst = INTEG3 / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)
    (ROOT / "docs/archive/phase_reports/HTW_HighLevel_ExecutorConditioned_Supervision_CN.md").write_text(report, encoding="utf-8")
    (INTEG3 / "HTW_HighLevel_ExecutorConditioned_Supervision_CN.md").write_text(report, encoding="utf-8")


def run() -> dict:
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("OLD-HL SHA mismatch")
    p2 = load_phase2_gate()
    p2a = load_phase2a_gate()
    p2d = load_phase2d_gate()
    compat = load_compat_gate()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = materialize_executor_dataset()
    load_u_cache()
    val = load_exec_split(VAL_NAME)
    util_sha = sha256_bytes((OUTPUT / f"{TRAIN_NAME}.npz").read_bytes() + (OUTPUT / f"{VAL_NAME}.npz").read_bytes())
    protocol = {
        "name": "HL-EXEC-UTIL",
        "training_high_level": True,
        "training_low_level": False,
        "seeds": list(MODEL_SEEDS),
        "loss": "L_exec + 0.25 L_rank + 0.10 L_ref",
        "L_exec": "-sum_i q_i log P_theta(Gamma_i|s)",
        "q": dataset["q_method"],
        "artificial_q": False,
        "prefix_label": False,
        "candidate_conditional_probability": False,
        "official_decode": "joint greedy argmax",
        "lambda_rank": LAMBDA_RANK,
        "lambda_ref": LAMBDA_REF,
        "epochs": EPOCHS,
        "batch": int(BATCH),
        "lr": LR,
        "checkpoint_rule": CHECKPOINT_RULE,
        "utility_dataset_sha": util_sha,
        "source_utility_sha": dataset["source_utility_sha"],
        "expected_utility_sha": EXPECTED_UTIL_SHA,
        "hold": HOLD,
        "pool_size": POOL_SIZE,
        "PREFIX": p2d.get("PREFIX"),
        "HLALIGN": p2.get("HLALIGN"),
        "HLSUP": p2a.get("gate_name"),
        "COMPAT": compat.get("COMPAT"),
        "COLREL": "C",
        "COMM": "B",
        "from_prefix_checkpoint": False,
        "from_abs_checkpoint": False,
    }
    _dump(OUTPUT / "HL_EXEC_PROTOCOL.json", protocol)
    pre = run_pretests()
    _dump(OUTPUT / "HL_EXEC_PRETESTS.json", pre)
    missing = [int(s) for s in MODEL_SEEDS if not exec_ckpt(int(s)).is_file()]
    infos = {}
    if missing:
        print(f"train EXEC seeds from scratch: {missing}", flush=True)
        n_workers = max(1, min(int(TRAIN_WORKERS), len(missing)))
        if n_workers == 1:
            for seed in missing:
                infos[int(seed)] = train_seed_process(int(seed))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for seed, info in zip(missing, pool.map(train_seed_process, missing)):
                    infos[int(seed)] = info
    hist, ckpt_rows = [], []
    existing = OUTPUT / "HL_EXEC_HIGHLEVEL_CHECKPOINTS.json"
    if infos or not existing.is_file():
        for seed in MODEL_SEEDS:
            if int(seed) in infos:
                hist.extend(infos[int(seed)]["history"])
                ckpt_rows.append({"high_seed": int(seed), "best_epoch": infos[int(seed)]["best_epoch"], "best_val_regret": infos[int(seed)]["best_val_regret"], "checkpoint": infos[int(seed)]["checkpoint"]})
            elif exec_ckpt(int(seed)).is_file():
                ckpt_rows.append({"high_seed": int(seed), "best_epoch": "reused", "checkpoint": str(exec_ckpt(int(seed)))})
        if hist:
            _csv(OUTPUT / "HL_EXEC_TRAINING_HISTORY.csv", hist)
        _dump(OUTPUT / "HL_EXEC_HIGHLEVEL_CHECKPOINTS.json", {"rule": CHECKPOINT_RULE, "items": ckpt_rows})
    else:
        ckpt_rows = json.loads(existing.read_text(encoding="utf-8"))["items"]
    rows = {"within": [], "decode": []}
    decode_path = OUTPUT / "HL_EXEC_DECODE_UTILITY.csv"
    within_path = OUTPUT / "HL_EXEC_WITHIN_POOL.csv"
    if decode_path.is_file() and within_path.is_file():
        rows["decode"] = load_loop_csv(decode_path)
        rows["within"] = load_loop_csv(within_path)
    else:
        for seed in MODEL_SEEDS:
            eval_model({"family": "EXEC", "seed": int(seed), "path": exec_ckpt(int(seed))}, val, rows)
        _csv(within_path, rows["within"])
        _csv(decode_path, rows["decode"])
    loop_path = OUTPUT / "HL_EXEC_CLOSED_LOOP_DESIGN.csv"
    if loop_path.is_file():
        loop_rows = load_loop_csv(loop_path)
    else:
        loop_rows = []
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
                    loop_rows.append(rec)
        with (PHASE2D_OUTPUT / "HL_PREFIX_CLOSED_LOOP_DESIGN.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("high") == "PREFIX-HL" or row.get("family") == "PREFIX":
                    rec = dict(row)
                    rec["family"] = "PREFIX"
                    rec["high"] = "PREFIX-HL"
                    for k, v in list(rec.items()):
                        if k in {"high", "high_seed", "family"}:
                            continue
                        try:
                            rec[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                    loop_rows.append(rec)
        _, design = load_pools()
        print("closed-loop EXEC 5 x 5", flush=True)
        for hs in MODEL_SEEDS:
            high = load_high(exec_ckpt(int(hs)))
            for ll in LOW_SEEDS:
                metrics = evaluate_design(make_stack(high, int(ll)), design, zero_residual=False)
                loop_rows.append({"family": "EXEC", "high": "EXEC-HL", "high_seed": int(hs), "low_seed": int(ll), **{k: v for k, v in metrics.items() if k != "scenarios"}})
        _csv(loop_path, loop_rows)
    gate = classify(rows, loop_rows, pre, dataset, p2d, ckpt_rows)
    _dump(OUTPUT / "HL_EXEC_GATE.json", gate)
    plot_figures(rows, loop_rows, gate, OUTPUT / "figures")
    report = write_report(gate, protocol, dataset)
    (OUTPUT / "HTW_HighLevel_ExecutorConditioned_Supervision_CN.md").write_text(report, encoding="utf-8")
    publish(report)
    return gate


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    gate = run()
    print(json.dumps({"EXEC": gate["EXEC"], "confirmed": gate["executor_conditioned_supervision_confirmed"], "fail_reasons": gate["fail_reasons"]}, indent=2), flush=True)
