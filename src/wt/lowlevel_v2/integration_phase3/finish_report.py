"""Write the official Phase-3 Chinese audit from frozen artifacts. No training, no method change."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman

from .dataset import load_exec_split
from .protocol import (
    INTEG3,
    MODEL_SEEDS,
    OLD_HL_CKPT,
    OUTPUT,
    PHASE2B_OUTPUT,
    PHASE2D_OUTPUT,
    ROOT,
    VAL_NAME,
    abs_ckpt,
    exec_ckpt,
    prefix_ckpt,
)


def _load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rec = dict(row)
            for key, val in list(rec.items()):
                if key in {"family", "high", "high_seed", "model"}:
                    continue
                try:
                    rec[key] = float(val)
                except (TypeError, ValueError):
                    pass
            rows.append(rec)
    return rows


def _mean(xs) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(xs) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _recover_best_epochs(log_path: Path) -> dict[int, dict]:
    pattern = re.compile(r"exec seed=(\d+) epoch (\d+)/80 .*val_regret=([0-9.]+)")
    best: dict[int, dict] = {}
    if not log_path.is_file():
        return best
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        seed = int(match.group(1))
        epoch = int(match.group(2))
        regret = float(match.group(3))
        prev = best.get(seed)
        if prev is None or regret < float(prev["best_val_regret"]) - 1.0e-9:
            best[seed] = {"high_seed": seed, "best_epoch": epoch, "best_val_regret": regret}
    return best


def _load_frozen_high(path: Path) -> OrderFreeAssignmentModel:
    model = OrderFreeAssignmentModel("deepsets")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _rank_corr_for_ckpt(path: Path, val: dict) -> dict[str, float]:
    model = _load_frozen_high(path)
    sps = []
    kens = []
    with torch.no_grad():
        for index in range(int(val["assignment"].shape[0])):
            encoded = _encode_one(model, val, index, "cpu")
            snap = make_snap(val["uav_positions"][index], val["buoy_positions"][index])
            lp = candidate_logprobs(model, encoded, snap, val["candidates"][index], val["buoy_mask"][index]).detach().cpu().numpy()
            u = np.asarray(val["utility"][index], dtype=np.float64)
            sps.append(spearman(u, lp))
            kens.append(kendall_tau(u, lp))
    del model
    return {"spearman": _mean(sps), "kendall": _mean(kens), "n": float(len(sps))}


def _family_rank_corr(val: dict) -> dict:
    payload = {
        "aggregation": "mean of per-state rank correlations on the frozen L=24 val pool",
        "OLD": _rank_corr_for_ckpt(Path(OLD_HL_CKPT), val),
        "ABS": {str(s): _rank_corr_for_ckpt(abs_ckpt(int(s)), val) for s in MODEL_SEEDS},
        "PREFIX": {str(s): _rank_corr_for_ckpt(prefix_ckpt(int(s)), val) for s in (0, 1)},
        "EXEC": {str(s): _rank_corr_for_ckpt(exec_ckpt(int(s)), val) for s in MODEL_SEEDS},
    }
    payload["ABS_mean"] = {
        "spearman": _mean([payload["ABS"][str(s)]["spearman"] for s in MODEL_SEEDS]),
        "kendall": _mean([payload["ABS"][str(s)]["kendall"] for s in MODEL_SEEDS]),
    }
    payload["PREFIX_mean"] = {
        "spearman": _mean([payload["PREFIX"][str(s)]["spearman"] for s in (0, 1)]),
        "kendall": _mean([payload["PREFIX"][str(s)]["kendall"] for s in (0, 1)]),
    }
    payload["EXEC_mean"] = {
        "spearman": _mean([payload["EXEC"][str(s)]["spearman"] for s in MODEL_SEEDS]),
        "kendall": _mean([payload["EXEC"][str(s)]["kendall"] for s in MODEL_SEEDS]),
    }
    return payload


def artifacts_ready() -> bool:
    need = [
        OUTPUT / "HL_EXEC_GATE.json",
        OUTPUT / "HL_EXEC_WITHIN_POOL.csv",
        OUTPUT / "HL_EXEC_DECODE_UTILITY.csv",
        OUTPUT / "HL_EXEC_CLOSED_LOOP_DESIGN.csv",
        OUTPUT / "HL_EXEC_HIGHLEVEL_CHECKPOINTS.json",
        OUTPUT / "HL_EXEC_DATASET.json",
    ]
    if any(not p.is_file() for p in need):
        return False
    return all(exec_ckpt(int(s)).is_file() for s in MODEL_SEEDS)


def write_official_report() -> Path:
    if not artifacts_ready():
        raise RuntimeError("Phase-3 artifacts are not complete; refuse to write the official report")
    gate = json.loads((OUTPUT / "HL_EXEC_GATE.json").read_text(encoding="utf-8"))
    dataset = json.loads((OUTPUT / "HL_EXEC_DATASET.json").read_text(encoding="utf-8"))
    ckpt = json.loads((OUTPUT / "HL_EXEC_HIGHLEVEL_CHECKPOINTS.json").read_text(encoding="utf-8"))
    p2d = json.loads((PHASE2D_OUTPUT / "HL_PREFIX_GATE.json").read_text(encoding="utf-8"))
    p2b = json.loads((PHASE2B_OUTPUT / "HL_ABS_GATE.json").read_text(encoding="utf-8"))
    within_exec = _load_csv(OUTPUT / "HL_EXEC_WITHIN_POOL.csv")
    within_abs = _load_csv(PHASE2B_OUTPUT / "HL_ABS_WITHIN_POOL_METRICS.csv")
    within_pref = _load_csv(PHASE2D_OUTPUT / "HL_PREFIX_WITHIN_POOL.csv")
    loop = _load_csv(OUTPUT / "HL_EXEC_CLOSED_LOOP_DESIGN.csv")

    def fam_metric(rows, family, key, seed=None):
        out = [r for r in rows if r.get("family") == family]
        if seed is not None:
            out = [r for r in out if str(r.get("high_seed")) == str(seed)]
        return [float(r[key]) for r in out if r.get(key) not in {"", None}]

    old_sp = _mean(fam_metric(within_abs, "OLD", "spearman"))
    old_nom = _mean(fam_metric(within_abs, "OLD", "nom5"))
    old_reg = _mean(fam_metric(within_abs, "OLD", "top1_regret"))
    abs_sp = [_mean(fam_metric(within_abs, "ABS", "spearman", s)) for s in MODEL_SEEDS]
    abs_nom = [_mean(fam_metric(within_abs, "ABS", "nom5", s)) for s in MODEL_SEEDS]
    abs_reg = [_mean(fam_metric(within_abs, "ABS", "top1_regret", s)) for s in MODEL_SEEDS]
    pref_sp = [float(p2d["within_PREFIX"][str(s)]["spearman"]) for s in (0, 1)]
    pref_nom = [float(p2d["within_PREFIX"][str(s)]["nom5"]) for s in (0, 1)]
    pref_reg = [float(p2d["within_PREFIX"][str(s)]["top1_regret"]) for s in (0, 1)]
    exec_sp = [float(gate["EXEC_spearman"][str(s)]) for s in MODEL_SEEDS]
    exec_nom = [_mean(fam_metric(within_exec, "EXEC", "nom5", s)) for s in MODEL_SEEDS]
    exec_reg = [_mean(fam_metric(within_exec, "EXEC", "top1_regret", s)) for s in MODEL_SEEDS]
    exec_nr = [float(gate["EXEC_DecodeNR"][str(s)]["median"]) for s in MODEL_SEEDS]
    abs_nr = [float(p2b["ABS_DecodeNR"][str(s)]["median"]) for s in MODEL_SEEDS]
    pref_nr = [float(p2d["PREFIX_DecodeNR"][str(s)]["median"]) for s in (0, 1)]
    old_nr = float(p2b["OLD_DecodeNR"]["median"])
    exec_pay = [float(gate["EXEC_closed_loop_seed"][str(s)]) for s in MODEL_SEEDS]
    old_pay = float(gate["OLD_closed_loop"])
    abs_pay = float(gate["ABS_closed_loop"])
    pref_pay = float(gate["PREFIX_closed_loop"])
    u = dataset["U_LL"]
    recovered = _recover_best_epochs(OUTPUT / "run_phase3.log")
    epoch_bits = []
    for row in ckpt.get("items") or []:
        seed = int(row.get("high_seed"))
        if str(row.get("best_epoch")) == "reused" and seed in recovered:
            epoch_bits.append(
                f"seed{seed} epoch={recovered[seed]['best_epoch']} regret={recovered[seed]['best_val_regret']}"
            )
        else:
            epoch_bits.append(f"seed{row.get('high_seed')} epoch={row.get('best_epoch')} regret={row.get('best_val_regret')}")
    epochs = "; ".join(epoch_bits)
    val = load_exec_split(VAL_NAME)
    ranks = _family_rank_corr(val)
    (OUTPUT / "HL_EXEC_KENDALL.json").write_text(json.dumps(ranks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path = OUTPUT / "HL_EXEC_ROLLOUT_CACHE_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    old_ken = float(ranks["OLD"]["kendall"])
    abs_ken = [float(ranks["ABS"][str(s)]["kendall"]) for s in MODEL_SEEDS]
    pref_ken = [float(ranks["PREFIX"][str(s)]["kendall"]) for s in (0, 1)]
    exec_ken = [float(ranks["EXEC"][str(s)]["kendall"]) for s in MODEL_SEEDS]
    mismatch_solved = bool(gate.get("executor_conditioned_supervision_confirmed"))
    verdict = (
        "是：闭环不低于 OLD，DecodeNR 不差于 PREFIX，seed 方差下降，且 \(P\) 与 \(U_{\\mathrm{LL}}\) 的 Spearman 上升。"
        if mismatch_solved
        else "否：executor-conditioned supervision 没有同时满足四项门控，HL-LL utility mismatch 仍未解决。"
    )
    fail = "; ".join(gate.get("fail_reasons") or []) or "无"
    report = f"""# HTW Phase-3：Executor-Conditioned Utility Supervision 正式报告

本报告只审计已冻结的 Phase-3 产物。未改 Low-Level MAPPO、SafetyProjector、collision/interference、reward、actor/critic、decoder、DeepSets、candidate pool、ranking/reference。未新增 interference auxiliary、collision 模块、trajectory head，也未做 end-to-end validation。PREFIX 作为 baseline。正式推理仍是 joint greedy。

训练循环读取预计算 `utility`，不在 epoch 内调用 `shadow_h15`。因此 **没有向训练加入 utility cache**；训练 cache hit rate 为 N/A，相对训练加速比为 0。评估阶段的 out-of-pool \(U_{{\\mathrm{{LL}}}}\) 仍使用既有 `(scenario, t, assignment_hash)` cache，不改变 \(q_i\)、\(L_{{\\mathrm{{exec}}}}\)、checkpoint 与随机性。

## 结论

**Executor-conditioned utility supervision 是否解决 High-Level planner 与 Low-Level RL executor 的 utility mismatch？{verdict}**

Gate={gate.get('EXEC')}；confirmed={str(mismatch_solved).lower()}。失败原因：{fail}。

下一步：停止。不自动进入下一阶段。不重调 q / pool / decoder / beam / checkpoint。

## 1. OLD / ABS / PREFIX / EXEC 比较

| 方法 | DecodeNR（聚合） | Spearman vs \(U_{{LL}}\) | Kendall vs \(U_{{LL}}\) | NOM5 | closed-loop payload |
|---|---:|---:|---:|---:|---:|
| OLD-HL + RL | {old_nr:.4f} | {old_sp:.4f} | {old_ken:.4f} | {old_nom:.4f} | {old_pay:.6e} |
| ABS-HL + RL | {float(np.median(abs_nr)):.4f} | {_mean(abs_sp):.4f} | {_mean(abs_ken):.4f} | {_mean(abs_nom):.4f} | {abs_pay:.6e} |
| PREFIX-HL + RL | {float(np.median(pref_nr)):.4f} | {_mean(pref_sp):.4f} | {_mean(pref_ken):.4f} | {_mean(pref_nom):.4f} | {pref_pay:.6e} |
| EXEC-HL + RL | {float(gate['EXEC_DecodeNR_aggregate']):.4f} | {float(gate['EXEC_spearman_mean']):.4f} | {_mean(exec_ken):.4f} | {_mean(exec_nom):.4f} | {float(gate['EXEC_closed_loop']):.6e} |

来源：OLD/ABS 用 Phase-2B 正式 CSV/Gate；PREFIX 用 Phase-2D 正式 Gate/Within-pool；EXEC 用本阶段 DecodeNR / Within-pool / 5×5 闭环。Kendall 是冻结 checkpoint 在同一 val pool 上、按 state 平均的 rank correlation，与 Spearman 聚合方式一致。ABS DecodeNR 为五 seed median 的中位数；PREFIX 为两 pilot seed median 的中位数。

## 2. DecodeNR

- OLD median={old_nr:.4f}
- ABS：{', '.join(f'seed{s}={abs_nr[i]:.4f}' for i, s in enumerate(MODEL_SEEDS))}；五 seed median 的中位数={float(np.median(abs_nr)):.4f}
- PREFIX：seed0={pref_nr[0]:.4f}，seed1={pref_nr[1]:.4f}；聚合={float(np.median(pref_nr)):.4f}
- EXEC：{', '.join(f'seed{s}={exec_nr[i]:.4f}' for i, s in enumerate(MODEL_SEEDS))}；聚合={float(gate['EXEC_DecodeNR_aggregate']):.4f}
- EXEC vs PREFIX = {float(gate['EXEC_DecodeNR_aggregate'])-float(np.median(pref_nr)):+.4f}
- 门控 DecodeNR <= PREFIX：{gate.get('gate_nr_ok')}

## 3. Spearman / Kendall

聚合方式：每个 val state 在 L=24 pool 上计算 \(P_\\theta(\\Gamma\\mid s)\) 与 \(U_{{\\mathrm{{LL}}}}\) 的 rank correlation，再对 state 取平均。

- OLD Spearman={old_sp:.4f}；Kendall={old_ken:.4f}
- ABS Spearman={_mean(abs_sp):.4f}（std={_std(abs_sp):.4f}）；Kendall={_mean(abs_ken):.4f}（std={_std(abs_ken):.4f}）
- PREFIX Spearman={_mean(pref_sp):.4f}；Kendall={_mean(pref_ken):.4f}（seed0 Kendall={pref_ken[0]:.4f}，seed1={pref_ken[1]:.4f}）
- EXEC Spearman={_mean(exec_sp):.4f}（std={_std(exec_sp):.4f}）；Kendall={_mean(exec_ken):.4f}（std={_std(exec_ken):.4f}）
- EXEC Spearman 分 seed：{', '.join(f'seed{s}={exec_sp[i]:.4f}' for i, s in enumerate(MODEL_SEEDS))}
- EXEC Kendall 分 seed：{', '.join(f'seed{s}={exec_ken[i]:.4f}' for i, s in enumerate(MODEL_SEEDS))}
- EXEC Spearman mean vs PREFIX={float(gate['EXEC_spearman_mean'])-_mean(pref_sp):+.4f}
- EXEC Kendall mean vs PREFIX={_mean(exec_ken)-_mean(pref_ken):+.4f}

## 4. NOM5

NOM5 = 概率质量落在 \(U \\ge 0.95 U_{{\\mathrm{{best}}}}\) 的候选上。

- OLD={old_nom:.4f}
- ABS={_mean(abs_nom):.4f}（{', '.join(f'seed{s}={abs_nom[i]:.4f}' for i, s in enumerate(MODEL_SEEDS))}）
- PREFIX={_mean(pref_nom):.4f}（seed0={pref_nom[0]:.4f}，seed1={pref_nom[1]:.4f}）
- EXEC={_mean(exec_nom):.4f}（std={_std(exec_nom):.4f}；{', '.join(f'seed{s}={exec_nom[i]:.4f}' for i, s in enumerate(MODEL_SEEDS))}）
- EXEC val regret mean={_mean(exec_reg):.1f}（std={_std(exec_reg):.1f}）；OLD={old_reg:.1f}；ABS={_mean(abs_reg):.1f}；PREFIX={_mean(pref_reg):.1f}

## 5. closed-loop payload

- OLD={old_pay:.6e}
- ABS={abs_pay:.6e}
- PREFIX={pref_pay:.6e}
- EXEC={float(gate['EXEC_closed_loop']):.6e}
- EXEC vs OLD={float(gate['EXEC_closed_loop'])-old_pay:.6e}
- EXEC vs ABS={float(gate['EXEC_closed_loop'])-abs_pay:.6e}
- EXEC vs PREFIX={float(gate['EXEC_closed_loop'])-pref_pay:.6e}
- 门控 closed-loop >= OLD：{gate.get('gate_loop_ok')}

## 6. 五 seed 均值方差

| seed | closed-loop payload | DecodeNR median | Spearman | Kendall | NOM5 |
|---|---:|---:|---:|---:|---:|
| 0 | {exec_pay[0]:.6e} | {exec_nr[0]:.4f} | {exec_sp[0]:.4f} | {exec_ken[0]:.4f} | {exec_nom[0]:.4f} |
| 1 | {exec_pay[1]:.6e} | {exec_nr[1]:.4f} | {exec_sp[1]:.4f} | {exec_ken[1]:.4f} | {exec_nom[1]:.4f} |
| 2 | {exec_pay[2]:.6e} | {exec_nr[2]:.4f} | {exec_sp[2]:.4f} | {exec_ken[2]:.4f} | {exec_nom[2]:.4f} |
| 3 | {exec_pay[3]:.6e} | {exec_nr[3]:.4f} | {exec_sp[3]:.4f} | {exec_ken[3]:.4f} | {exec_nom[3]:.4f} |
| 4 | {exec_pay[4]:.6e} | {exec_nr[4]:.4f} | {exec_sp[4]:.4f} | {exec_ken[4]:.4f} | {exec_nom[4]:.4f} |

- payload mean={_mean(exec_pay):.6e}；std={_std(exec_pay):.6e}；min={min(exec_pay):.6e}；max={max(exec_pay):.6e}；max−min={max(exec_pay)-min(exec_pay):.6e}
- PREFIX two-seed payload std={float(gate['PREFIX_closed_loop_std']):.6e}
- 方差是否下降：{str(bool(gate.get('gate_var_ok'))).lower()}
- selected epochs / val candidate regret：{epochs}
- \(U_{{LL}}\) label：mean={u['mean']:.6g}；std={u['std']:.6g}；min={u['min']:.6g}；max={u['max']:.6g}；n={u['n']}；非正个数={u.get('n_nonpositive')}
- \(q\)：{dataset['q_method']}；artificial_q=false；prefix_label=false

## 7. 是否解决 High-Level 与 Low-Level utility mismatch

四项门控必须同时成立：

- closed-loop >= OLD：{gate.get('gate_loop_ok')}（EXEC {float(gate['EXEC_closed_loop']):.6e} vs OLD {old_pay:.6e}）
- DecodeNR <= PREFIX：{gate.get('gate_nr_ok')}（EXEC {float(gate['EXEC_DecodeNR_aggregate']):.4f} vs PREFIX {float(gate['PREFIX_DecodeNR_aggregate']):.4f}）
- seed variance 降低：{gate.get('gate_var_ok')}
- Spearman 提升：{gate.get('gate_spearman_ok')}（EXEC {float(gate['EXEC_spearman_mean']):.4f} vs PREFIX {float(gate['PREFIX_spearman_mean']):.4f}）

**结论：{verdict}**

{gate.get('text')}

训练 cache 审计：repeated_low_level_rollout_in_training={audit.get('repeated_low_level_rollout_in_training')}；code_changed={audit.get('code_changed')}；cache_hit_rate_training={audit.get('cache_hit_rate_training')}；speedup_if_cache_added_to_training={audit.get('speedup_if_cache_added_to_training')}。{audit.get('action', '')}

冻结：Low-Level MAPPO、SafetyProjector、collision/interference、reward、Actor/Critic、decoder、DeepSets、candidate pool、ranking/reference loss、hold=15、L=24。

未做：interference auxiliary、collision 模块、trajectory head、end-to-end independent validation、beam / Pool-MAP 正式化、已完成 seed 的重训。

本阶段停止。不要自动进入下一阶段。
"""
    names = [
        ROOT / "docs/archive/phase_reports/HTW_ExecutorConditionedUtility_Phase3_Report.md",
        INTEG3 / "HTW_ExecutorConditionedUtility_Phase3_Report.md",
        OUTPUT / "HTW_ExecutorConditionedUtility_Phase3_Report.md",
    ]
    for path in names:
        path.write_text(report, encoding="utf-8")
    return names[0]


if __name__ == "__main__":
    if artifacts_ready():
        path = write_official_report()
        print(f"wrote {path}", flush=True)
    else:
        print("WAITING_FOR_TRAINING", flush=True)
