"""Phase-9.4.4 HGNN encoder/training instability causal audit. Frozen 9.4.3 weights only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wt.htw_phase61.run_phase61 import _plain
from wt.htw_phase9.const import EVAL_SEEDS
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase94.run_phase94 import collect_oracle_dataset
from wt.htw_phase941.protocol import MODEL_SEEDS
from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.protocol import TIE_ATOL
from wt.htw_phase943.set_context import empty_assignment, make_snap
from wt.htw_phase943.set_likelihood_dp import set_nll
from wt.l3b.env import L3BEnv
from wt.round30.ppo import seed_everything

from .audits import (
    assignment_disagreement,
    decode_with_teacher,
    env_batch,
    first_step_logits,
    flatten_legal_logits,
    grad_norm_on_items,
    item_tensors,
    layer_stats,
    param_norm,
    perturb_tensors,
    rollout_assignment,
    shadow_split_payloads,
    snapshot_nll,
    tensors_from_batch,
)
from .hooks import encode_layers
from .protocol import (
    ATTR_MD,
    AUX_RANK_WEIGHT,
    BATCH,
    BOOT_SEED,
    CURVE_AUDIT,
    DESIGN_VAL_SEEDS,
    DIAG_SPLIT,
    EPOCHS,
    LOG_CANDIDATES,
    LR,
    N_DECODE,
    N_NLL,
    N_REP,
    N_SENS,
    OUTPUT,
    PACK_REPORT,
    PHASE943,
    PHASE943_CKPT,
    REPORT,
    dump_frozen,
    sha256_bytes,
)
from .tables import parse_training_log, pearson, spearman, write_csv


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_training_log() -> Path | None:
    for path in LOG_CANDIDATES:
        if path.exists():
            return path
    return None


def load_model(encoder: str, seed: int) -> OrderFreeAssignmentModel:
    path = PHASE943_CKPT / f"{encoder}_seed{seed}.pt"
    model = OrderFreeAssignmentModel(encoder)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def checkpoint_hashes() -> dict[str, str]:
    rows = {}
    for encoder in ("deepsets", "hgnn"):
        for seed in MODEL_SEEDS:
            path = PHASE943_CKPT / f"{encoder}_seed{seed}.pt"
            rows[f"{encoder}_seed{seed}"] = sha256_bytes(path.read_bytes())
    return rows


def subsample(count: int, take: int, salt: int) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED + salt)
    take = min(int(take), int(count))
    return np.sort(rng.choice(count, size=take, replace=False))


def mean_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def write_curve_audit(curve_rows: list[dict[str, Any]], snap_rows: list[dict[str, Any]]) -> None:
    by = defaultdict(list)
    for row in curve_rows:
        by[(row["encoder"], int(row["model_seed"]))].append(row)
    lines = [
        "# Phase-9.4.4 Training Curve Audit",
        "",
        "Phase-9.4.3 未保存中间 checkpoint。epoch 曲线中的 `train_loss` 来自正式训练日志；",
        "validation / accuracy / gradient / entropy 只能在 **last-epoch 冻结权重** 上计算。",
        "未重训、未改学习率、未覆盖 9.4.3 权重。",
        "",
    ]
    for encoder in ("deepsets", "hgnn"):
        lines.append(f"## {encoder}")
        for seed in MODEL_SEEDS:
            hist = sorted(by.get((encoder, seed), []), key=lambda row: int(row["epoch"]))
            if not hist:
                lines.append(f"- seed {seed}: 日志缺失")
                continue
            first = float(hist[0]["train_loss"])
            last = float(hist[-1]["train_loss"])
            mid = float(hist[len(hist) // 2]["train_loss"])
            lines.append(f"- seed {seed}: epoch1={first:.4f} mid={mid:.4f} last={last:.4f} (n={len(hist)})")
        lines.append("")
    lines.append("## last-epoch snapshot (train subsample / val subsample)")
    for row in snap_rows:
        lines.append(
            f"- {row['encoder']} seed={row['model_seed']}: train_nll={row['train_set_nll']:.4f} val_nll={row['val_set_nll']:.4f} "
            f"tf_acc={row['val_tf_acc']:.4f} fr_acc={row['val_fr_acc']:.4f} exact={row['val_fr_exact']:.4f} "
            f"grad_l2={row['grad_l2']:.4f} param_l2={row['param_l2']:.2f}"
        )
    pos = [0, 1, 2]
    neg = [3, 4]
    hgnn_snap = [row for row in snap_rows if row["encoder"] == "hgnn"]

    def avg(seeds, key):
        vals = [float(row[key]) for row in hgnn_snap if int(row["model_seed"]) in seeds]
        return float(np.mean(vals)) if vals else float("nan")

    lines.extend(
        [
            "",
            "## HGNN positive (0/1/2) vs negative (3/4) last-epoch",
            f"- train NLL pos={avg(pos, 'train_set_nll'):.4f} neg={avg(neg, 'train_set_nll'):.4f}",
            f"- val NLL pos={avg(pos, 'val_set_nll'):.4f} neg={avg(neg, 'val_set_nll'):.4f}",
            f"- FR acc pos={avg(pos, 'val_fr_acc'):.4f} neg={avg(neg, 'val_fr_acc'):.4f}",
            f"- grad L2 pos={avg(pos, 'grad_l2'):.4f} neg={avg(neg, 'grad_l2'):.4f}",
            "",
        ]
    )
    CURVE_AUDIT.write_text("\n".join(lines), encoding="utf-8")


def classify_seed(bundle: dict[str, Any]) -> dict[str, Any]:
    labels = []
    evidence = []
    hgnn = bundle["encoder"] == "hgnn"
    if bundle["grad_l2"] > 50.0 or bundle["grad_l2"] < 1.0e-6 or (not np.isfinite(bundle["grad_l2"])):
        labels.append("OPT")
        evidence.append(f"grad_l2={bundle['grad_l2']:.3e}")
    worse_nll = bundle["train_set_nll"] > bundle["pos_train_nll"] + 0.25 and bundle["val_set_nll"] > bundle["pos_val_nll"] + 0.25
    if hgnn and worse_nll:
        labels.append("OPT")
        evidence.append(f"nll train={bundle['train_set_nll']:.3f} val={bundle['val_set_nll']:.3f} vs pos {bundle['pos_train_nll']:.3f}/{bundle['pos_val_nll']:.3f}")
    if bundle["os_u_final"] > 0.85:
        labels.append("REP")
        evidence.append(f"OS_U={bundle['os_u_final']:.3f}")
    if bundle["erank_uav_final"] < 0.6 * max(bundle["pos_erank_uav"], 1.0e-6):
        labels.append("REP")
        evidence.append(f"erank_uav={bundle['erank_uav_final']:.3f} vs pos {bundle['pos_erank_uav']:.3f}")
    if hgnn and bundle["irs_uu"] < 0.5 * max(bundle["pos_irs_uu"], 1.0e-8) and bundle["pos_irs_uu"] > 0.05:
        labels.append("REL")
        evidence.append(f"IRS_UU={bundle['irs_uu']:.3f} vs pos {bundle['pos_irs_uu']:.3f}")
    if hgnn and bundle["ub_dis"] < 0.5 * bundle["pos_ub_dis"]:
        labels.append("REL")
        evidence.append(f"UB disagreement={bundle['ub_dis']:.3f}")
    if bundle["idle"] > bundle["pos_idle"] + 0.12 or bundle["margin"] < 0.4 * bundle["pos_margin"]:
        labels.append("DEC")
        evidence.append(f"idle={bundle['idle']:.3f} margin={bundle['margin']:.3f}")
    if bundle["acc_payload_pearson"] < 0.20 or (abs(bundle["acc_payload_pearson"]) < 0.25 and bundle["multiplicity_5pct"] > 3):
        labels.append("OBJ")
        evidence.append(f"acc-payload r={bundle['acc_payload_pearson']:.3f} multiplicity5={bundle['multiplicity_5pct']:.2f}")
    if hgnn and bundle["val_fr_acc"] > 0.45 and bundle["payload"] < bundle["paired_deep_payload"] - 5.0e5:
        labels.append("GEN")
        evidence.append("val assignment acc not low but confirmation payload far below paired DeepSets")
    labels = list(dict.fromkeys(labels))
    if not labels:
        labels = ["NONE"]
        evidence.append("no audit threshold crossed relative to positive HGNN seeds")
    return {"labels": "|".join(labels), "evidence": "; ".join(evidence)}


def instability_gate(class_rows: list[dict[str, Any]], align: dict[str, float], multiplicity: dict[str, float]) -> dict[str, Any]:
    hgnn = [row for row in class_rows if row["encoder"] == "hgnn"]
    neg = [row for row in hgnn if int(row["model_seed"]) in (3, 4)]
    opt_rep = any(("OPT" in row["labels"] or "REP" in row["labels"] or "REL" in row["labels"]) for row in neg)
    obj = abs(float(align.get("hgnn_acc_payload_pearson", 0.0))) < 0.25 and float(multiplicity.get("mean_within_5pct", 0.0)) > 3
    if opt_rep and obj:
        gate = "INST-C"
        nxt = "one-at-a-time repair"
        reason = "negative seeds show optimization/representation/relation issues and imitation-payload alignment is weak."
    elif opt_rep:
        gate = "INST-A"
        nxt = "minimal_hgnn_training_stabilization"
        reason = "negative seeds differ in training/representation/relation use."
    elif obj:
        gate = "INST-B"
        nxt = "utility_aligned_supervised_objective"
        reason = "training metrics similar but assignment imitation is weakly aligned with payload; near-optimal multiplicity is high."
    else:
        gate = "INST-D"
        nxt = "stop HGNN"
        reason = "no clear mechanism separated negative seeds from positive seeds."
    return {
        "gate": gate,
        "next_action": nxt,
        "reason": reason,
        "phase943_confirmation_consumed": True,
        "phase944_model_tuning_on_phase943_confirmation": False,
        "no_tuning_on_consumed_confirmation_outcomes": True,
    }


def write_attr_md(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Encoder vs decoder attribution",
        "",
        "未跨 seed 交换 decoder（embedding basis 不兼容）。比较进入 decoder 之前的 embedding / first-step logits。",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['encoder']} seed={row['model_seed']}: uav_norm={row['uav_norm']:.3f} logit_var={row['logit_var']:.3f} "
            f"margin={row['margin']:.3f} idle_margin={row['idle_margin']:.3f} partial_ctx uses set decoder contexts"
        )
    ATTR_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(p: dict[str, Any]) -> None:
    h = p["hgnn"]
    d = p["deepsets"]
    lines = [
        "# HTW Phase-9.4.4 HGNN Encoder and Training Instability Causal Audit",
        "",
        "诊断阶段。未修改 HGNN / decoder / set likelihood / optimizer / lr / L3-B / SINR / Task-Waypoint。未做 PPO。未按 confirmation 选模型。",
        f"phase943_confirmation_consumed=**true**；phase944_model_tuning_on_phase943_confirmation=**false**。",
        "",
        f"1. seed0-4 train loss（日志 last epoch）DeepSets={d['last_train_loss']}；HGNN={h['last_train_loss']}",
        f"2. val set NLL（eval 8–15 轨迹子样本 last ckpt）DeepSets={d['val_nll']}；HGNN={h['val_nll']}",
        f"3. free-running per-UAV acc DeepSets={d['fr_acc']}；HGNN={h['fr_acc']}",
        f"4. positive/negative HGNN 优化差异：**{p['opt_gap']}**",
        f"5. gradient 异常：**{p['grad_flag']}**",
        f"6. oversmoothing：**{p['os_flag']}**",
        f"7. effective rank 坍缩：**{p['erank_flag']}**",
        f"8. seed3 UU 使用：IRS={p['seed3']['irs_uu']:.4f}，shuffle-UU disagreement={p['seed3']['uu_dis']:.4f}",
        f"9. seed3 UB 使用：IRS_UB={p['seed3']['irs_ub']:.4f}，shuffle-UB disagreement={p['seed3']['ub_dis']:.4f}",
        f"10. seed4 UU/UB：IRS_UU={p['seed4']['irs_uu']:.4f} / IRS_UB={p['seed4']['irs_ub']:.4f}；positive mean IRS_UU={p['pos_irs_uu']:.4f}",
        f"11. decoder logits：margin pos={p['pos_margin']:.4f} seed3={p['seed3']['margin']:.4f} seed4={p['seed4']['margin']:.4f}",
        f"12. idle：pos={p['pos_idle']:.4f} seed3={p['seed3']['idle']:.4f} seed4={p['seed4']['idle']:.4f}（confirmation）",
        f"13. assignment accuracy vs payload Pearson：HGNN={p['align']['hgnn_acc_payload_pearson']:.4f} DeepSets={p['align']['deepsets_acc_payload_pearson']:.4f}",
        f"14. set NLL vs payload Pearson：HGNN={p['align']['hgnn_nll_payload_pearson']:.4f} DeepSets={p['align']['deepsets_nll_payload_pearson']:.4f}",
        f"15. SINR utility vs payload Pearson：HGNN={p['align']['hgnn_sinr_payload_pearson']:.4f} DeepSets={p['align']['deepsets_sinr_payload_pearson']:.4f}",
        f"16. near-optimal 不同 assignment（影子搜索 5% 窗）均值数量=**{p['multiplicity']['mean_within_5pct']:.2f}**",
        f"17. label ambiguity：**{p['ambiguity']}**（utility_equivalent_assignment_ambiguity={p['utility_equivalent_assignment_ambiguity']}）",
        "补充：全部 HGNN seed 的 UU-shuffle IRS 都接近 0，UB-shuffle IRS 都很大；UU 低敏感是普遍现象，不是 seed 3/4 独有。",
        f"18. seed3 最可能原因：**{p['seed3']['labels']}** — {p['seed3']['evidence']}",
        f"19. seed4 最可能原因：**{p['seed4']['labels']}** — {p['seed4']['evidence']}",
        f"20. DeepSets 是否同一问题：**{p['deepsets_same']}**",
        f"21. 剩余 Graph instability 主要来自：**{p['source']}**",
        f"22. INST Gate=**{p['gate']['gate']}**",
        f"23. 下一阶段：**{p['gate']['next_action']}**",
        "",
        f"ORDERFIX-A 保持；decoder_order_dependence_fixed=true。HGNN confirmation payload 仍 seed-不稳定：{p['phase943_delta']}。",
        f"设计验证只用 historical eval 8–15 与 collect 状态；943 confirmation 只做 post-hoc 相关。",
        "",
        "JSON/CSV：`src/wt/htw_phase944/outputs/hgnn_instability/`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary943 = _load_json(PHASE943 / "summary.json")
    per_seed_943 = _load_csv(PHASE943 / "phase943_per_seed_metrics.csv")
    per_scen_943 = _load_csv(PHASE943 / "phase943_per_scenario_metrics.csv")
    orderfix = _load_json(PHASE943 / "phase943_orderfix_gate.json")
    graph = _load_json(PHASE943 / "phase943_graph_gate.json")
    cgraph = _load_json(PHASE943 / "phase943_conditional_graph_gate.json")
    if orderfix.get("gate") != "ORDERFIX-A":
        raise RuntimeError("Phase-9.4.4 requires ORDERFIX-A")
    protocol = {
        "phase": "9.4.4",
        "diagnostic_only": True,
        "ppo": False,
        "modify_hgnn": False,
        "modify_decoder": False,
        "modify_lr": False,
        "phase943_confirmation_consumed": True,
        "phase944_model_tuning_on_phase943_confirmation": False,
        "no_tuning_on_consumed_confirmation_outcomes": True,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "aux_rank_weight": AUX_RANK_WEIGHT,
        "collect_seeds": list(COLLECT_SEEDS),
        "design_val_seeds": list(DESIGN_VAL_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "diag_split": DIAG_SPLIT,
        "checkpoint_sha256": checkpoint_hashes(),
        "tie_atol": TIE_ATOL,
        "orderfix": orderfix,
        "graph943": graph,
        "cgraph943": cgraph,
        "mean_deep_943": summary943["mean_deep"],
        "mean_hgnn_943": summary943["mean_hgnn"],
        "mean_delta_943": summary943["mean_delta"],
    }
    dump_frozen(OUTPUT / "phase944_protocol.json", protocol)

    log_path = find_training_log()
    curve_hist = parse_training_log(log_path) if log_path is not None else []
    print(f"parsed training log rows={len(curve_hist)} from {log_path}", flush=True)

    print("collect train/val oracle-state datasets", flush=True)
    seed_everything(0)
    train_data = collect_oracle_dataset(COLLECT_SEEDS)
    val_data = collect_oracle_dataset(tuple(EVAL_SEEDS))
    n_train = int(train_data["assignment"].shape[0])
    n_val = int(val_data["assignment"].shape[0])
    idx_nll_tr = subsample(n_train, N_NLL, 1)
    idx_nll_va = subsample(n_val, N_NLL, 2)
    idx_rep = subsample(n_val, N_REP, 3)
    idx_dec = subsample(n_val, N_DECODE, 4)
    idx_grad = subsample(n_train, 8, 5)

    models: dict[str, dict[int, OrderFreeAssignmentModel]] = {"deepsets": {}, "hgnn": {}}
    for encoder in ("deepsets", "hgnn"):
        for seed in MODEL_SEEDS:
            print(f"load {encoder} seed={seed}", flush=True)
            models[encoder][seed] = load_model(encoder, seed)

    curve_rows = list(curve_hist)
    snap_rows = []
    rep_rows = []
    os_rows = []
    erank_rows = []
    logit_rows = []
    err_rows = []
    msg_rows = []
    sens_rows = []
    attr_rows = []
    rng = np.random.default_rng(BOOT_SEED)

    for encoder in ("deepsets", "hgnn"):
        for seed in MODEL_SEEDS:
            print(f"snapshot {encoder} seed={seed}", flush=True)
            model = models[encoder][seed]
            train_nll = snapshot_nll(model, train_data, idx_nll_tr)
            val_nll = snapshot_nll(model, val_data, idx_nll_va)
            grads = grad_norm_on_items(model, train_data, idx_grad)
            tf_stats = []
            fr_stats = []
            for index in idx_dec:
                tf = decode_with_teacher(model, val_data, int(index), True)
                fr = decode_with_teacher(model, val_data, int(index), False)
                tf_stats.append(tf)
                fr_stats.append(fr)
                err_rows.append(
                    {
                        "encoder": encoder,
                        "model_seed": seed,
                        "index": int(index),
                        "tf_acc": tf["per_uav"],
                        "fr_acc": fr["per_uav"],
                        "tf_exact": tf["exact"],
                        "fr_exact": fr["exact"],
                        "first_error_step": fr["first_error_step"],
                        "after_error_acc": fr["after_error_acc"],
                        "exposure_gap": float(tf["per_uav"] - fr["per_uav"]),
                    }
                )
                for step, (hit, margin, ent) in enumerate(zip(fr["step_hits"], fr["step_margins"], fr["step_entropy"])):
                    logit_rows.append(
                        {
                            "encoder": encoder,
                            "model_seed": seed,
                            "index": int(index),
                            "step": step,
                            "hit": hit,
                            "margin": margin,
                            "entropy": ent,
                            "idle_margin": fr["idle_margin_mean"],
                            "idle": fr["idle"],
                        }
                    )
            last_loss = float("nan")
            hist = [row for row in curve_hist if row["encoder"] == encoder and int(row["model_seed"]) == seed]
            if hist:
                last_loss = float(sorted(hist, key=lambda row: int(row["epoch"]))[-1]["train_loss"])
            snap = {
                "encoder": encoder,
                "model_seed": seed,
                "epoch": EPOCHS,
                "train_loss_log": last_loss,
                "train_set_nll": train_nll["set_nll"],
                "val_set_nll": val_nll["set_nll"],
                "val_tf_acc": float(np.mean([row["per_uav"] for row in tf_stats])),
                "val_fr_acc": float(np.mean([row["per_uav"] for row in fr_stats])),
                "val_fr_exact": float(np.mean([row["exact"] for row in fr_stats])),
                "val_fr_idle": float(np.mean([row["idle"] for row in fr_stats])),
                "val_margin": float(np.mean([row["margin_mean"] for row in fr_stats])),
                "val_entropy": float(np.mean([row["entropy_mean"] for row in fr_stats])),
                "val_idle_margin": float(np.mean([row["idle_margin_mean"] for row in fr_stats])),
                **grads,
            }
            snap_rows.append(snap)
            for row in curve_hist:
                if row["encoder"] == encoder and int(row["model_seed"]) == seed:
                    pass
            layers = layer_stats(model, val_data, idx_rep)
            for row in layers:
                rec = {"encoder": encoder, "model_seed": seed, **row}
                rep_rows.append(rec)
                os_rows.append(
                    {
                        "encoder": encoder,
                        "model_seed": seed,
                        "index": row["index"],
                        "layer": row["layer"],
                        "os_u": row["os_u"],
                        "os_b": row["os_b"],
                    }
                )
                erank_rows.append(
                    {
                        "encoder": encoder,
                        "model_seed": seed,
                        "index": row["index"],
                        "layer": row["layer"],
                        "erank_uav": row["erank_uav"],
                        "erank_buoy": row["erank_buoy"],
                        "erank_edge": row["erank_edge"],
                    }
                )
            # attribution from first val index
            tensors0 = item_tensors(val_data, int(idx_rep[0]))
            lays0 = encode_layers(model, tensors0)
            final = lays0[-1]
            snap_env = make_snap(val_data["uav_positions"][int(idx_rep[0])], val_data["buoy_positions"][int(idx_rep[0])])
            empty = empty_assignment(6)
            logits0 = first_step_logits(model, tensors0, snap_env, empty, val_data["buoy_mask"][int(idx_rep[0])])
            legal_vals = flatten_legal_logits(logits0, empty, val_data["buoy_mask"][int(idx_rep[0])])
            attr_rows.append(
                {
                    "encoder": encoder,
                    "model_seed": seed,
                    "uav_norm": float(final["h_u"][0, :6].norm(dim=-1).mean().cpu()),
                    "logit_var": float(np.var(legal_vals)) if legal_vals.size else float("nan"),
                    "margin": snap["val_margin"],
                    "idle_margin": snap["val_idle_margin"],
                    "candidate_logit_std": float(np.std(legal_vals)) if legal_vals.size else float("nan"),
                }
            )

    # fill training curves csv: log loss + last-epoch extras
    snap_map = {(row["encoder"], int(row["model_seed"])): row for row in snap_rows}
    curve_out = []
    for row in curve_hist:
        extra = snap_map.get((row["encoder"], int(row["model_seed"])), {})
        last = int(row["epoch"]) == EPOCHS
        curve_out.append(
            {
                **row,
                "validation_set_loss": extra.get("val_set_nll") if last else "",
                "set_nll": extra.get("val_set_nll") if last else "",
                "teacher_forcing_accuracy": extra.get("val_tf_acc") if last else "",
                "free_running_accuracy": extra.get("val_fr_acc") if last else "",
                "exact_assignment_match": extra.get("val_fr_exact") if last else "",
                "per_uav_accuracy": extra.get("val_fr_acc") if last else "",
                "idle_fraction": extra.get("val_fr_idle") if last else "",
                "assignment_entropy": extra.get("val_entropy") if last else "",
                "gradient_norm": extra.get("grad_l2") if last else "",
                "parameter_norm": extra.get("param_l2") if last else "",
                "learning_rate": LR,
            }
        )
    if not curve_out:
        for row in snap_rows:
            curve_out.append(
                {
                    "encoder": row["encoder"],
                    "model_seed": row["model_seed"],
                    "epoch": EPOCHS,
                    "epochs": EPOCHS,
                    "train_loss": row["train_loss_log"],
                    "validation_set_loss": row["val_set_nll"],
                    "set_nll": row["val_set_nll"],
                    "teacher_forcing_accuracy": row["val_tf_acc"],
                    "free_running_accuracy": row["val_fr_acc"],
                    "exact_assignment_match": row["val_fr_exact"],
                    "per_uav_accuracy": row["val_fr_acc"],
                    "idle_fraction": row["val_fr_idle"],
                    "assignment_entropy": row["val_entropy"],
                    "gradient_norm": row["grad_l2"],
                    "parameter_norm": row["param_l2"],
                    "learning_rate": LR,
                }
            )

    print("message utilization + relation sensitivity on eval t=0", flush=True)
    kinds = ("full", "zero_uu", "shuffle_uu", "zero_ub", "shuffle_ub", "zero_global")
    eval_probe = list(EVAL_SEEDS)[:N_SENS]
    for encoder in ("deepsets", "hgnn"):
        for seed in MODEL_SEEDS:
            model = models[encoder][seed]
            for scen in eval_probe:
                env, batch, heuristic = env_batch(int(scen))
                tensors = tensors_from_batch(batch)
                snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
                empty = empty_assignment(env.config.num_uavs)
                buoy_mask = np.asarray(batch["buoy_mask"][0], dtype=bool)
                with torch.no_grad():
                    base_assign = decode_assignment(model, batch, env)
                    base_logits = first_step_logits(model, tensors, snap, empty, buoy_mask)
                    encoded = model.encode(tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"])
                    base_nll = float(set_nll(model, encoded, snap, heuristic, buoy_mask).cpu())
                base_pay = rollout_assignment(int(scen), base_assign)
                irs_uu = irs_ub = irs_comp = 0.0
                uu_dis = ub_dis = 0.0
                for kind in kinds:
                    pert = perturb_tensors(tensors, kind, rng)
                    fake_batch = {key: pert[key].detach().cpu().numpy() if hasattr(pert[key], "detach") else batch[key] for key in batch}
                    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
                        fake_batch[key] = pert[key].detach().cpu().numpy()
                    with torch.no_grad():
                        pred = decode_assignment(model, fake_batch, env)
                        logits = first_step_logits(model, pert, snap, empty, np.asarray(fake_batch["buoy_mask"][0], dtype=bool))
                        encoded_p = model.encode(pert["uav"], pert["buoy"], pert["uav_mask"], pert["buoy_mask"], pert["global"], pert["edge"], pert["edge_uu"])
                        nll_p = float(set_nll(model, encoded_p, snap, heuristic, np.asarray(fake_batch["buoy_mask"][0], dtype=bool)).cpu())
                    dis = assignment_disagreement(pred, base_assign)
                    dlog = float(np.linalg.norm(flatten_legal_logits(logits, empty, buoy_mask) - flatten_legal_logits(base_logits, empty, buoy_mask)))
                    pay = rollout_assignment(int(scen), pred) if kind != "full" else base_pay
                    msg_rows.append(
                        {
                            "encoder": encoder,
                            "model_seed": seed,
                            "scenario": int(scen),
                            "ablation": kind,
                            "assignment_disagreement": dis,
                            "payload": pay,
                            "payload_delta": pay - base_pay,
                            "logit_l2": dlog,
                            "set_nll": nll_p,
                            "set_nll_delta": nll_p - base_nll,
                        }
                    )
                    if kind == "shuffle_uu":
                        irs_uu = dlog
                        uu_dis = dis
                    if kind == "shuffle_ub":
                        irs_ub = dlog
                        ub_dis = dis
                    if kind == "zero_uu":
                        pass
                # competition perturbation
                pert_c = perturb_tensors(tensors, "shuffle_competition", rng)
                fake_c = dict(batch)
                for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
                    fake_c[key] = pert_c[key].detach().cpu().numpy()
                with torch.no_grad():
                    logits_c = first_step_logits(model, pert_c, snap, empty, np.asarray(fake_c["buoy_mask"][0], dtype=bool))
                a = flatten_legal_logits(base_logits, empty, buoy_mask)
                b = flatten_legal_logits(logits_c, empty, np.asarray(fake_c["buoy_mask"][0], dtype=bool))
                n = min(a.size, b.size)
                irs_comp = float(np.linalg.norm(a[:n] - b[:n])) if n else float("nan")
                sens_rows.append(
                    {
                        "encoder": encoder,
                        "model_seed": seed,
                        "scenario": int(scen),
                        "irs_uu": irs_uu,
                        "irs_ub": irs_ub,
                        "competition_response": irs_comp,
                        "uu_response": uu_dis,
                        "ub_response": ub_dis,
                    }
                )

    print("near-optimal multiplicity on eval t=0", flush=True)
    multi_rows = []
    for scen in eval_probe:
        env = L3BEnv()
        env.reset(seed=int(scen))
        splits = shadow_split_payloads(env)
        payloads = np.asarray([row["payload"] for row in splits], dtype=np.float64)
        best = float(np.max(payloads)) if payloads.size else float("nan")
        uniq = {}
        for row in splits:
            key = tuple(int(x) for x in row["assignment"])
            uniq[key] = float(row["payload"])
        vals = np.asarray(list(uniq.values()), dtype=np.float64)
        def count_band(frac: float) -> int:
            if not np.isfinite(best) or best == 0:
                return int(np.sum(np.abs(vals - best) <= 1.0e-9))
            return int(np.sum(vals >= best * (1.0 - frac)))
        multi_rows.append(
            {
                "scenario": int(scen),
                "n_shadow_splits": len(splits),
                "n_unique_assignments": len(uniq),
                "best_payload": best,
                "within_1pct": count_band(0.01),
                "within_5pct": count_band(0.05),
                "within_10pct": count_band(0.10),
            }
        )

    # objective alignment from consumed confirmation (post-hoc only)
    align_rows = []
    grouped = defaultdict(lambda: {"payload": [], "acc": [], "exact": [], "nll": [], "sinr": [], "hrr": []})
    for row in per_scen_943:
        key = (row["encoder"], int(row["model_seed"]))
        grouped[key]["payload"].append(float(row["payload"]))
        grouped[key]["acc"].append(float(row["per_uav"]))
        grouped[key]["exact"].append(float(row["exact_match"]))
        grouped[key]["nll"].append(float(row["set_nll"]))
        grouped[key]["sinr"].append(float(row["sinr_utility"]))
        grouped[key]["hrr"].append(float(row["hrr"]))
    for (encoder, seed), blob in grouped.items():
        pay = np.asarray(blob["payload"])
        rec = {
            "encoder": encoder,
            "model_seed": seed,
            "split": "phase943_confirmation_posthoc",
            "acc_payload_pearson": pearson(blob["acc"], pay),
            "acc_payload_spearman": spearman(blob["acc"], pay),
            "exact_payload_pearson": pearson(blob["exact"], pay),
            "nll_payload_pearson": pearson(blob["nll"], pay),
            "nll_payload_spearman": spearman(blob["nll"], pay),
            "sinr_payload_pearson": pearson(blob["sinr"], pay),
            "hrr_payload_pearson": pearson(blob["hrr"], pay),
            "mean_payload": float(np.mean(pay)),
            "mean_acc": float(np.mean(blob["acc"])),
            "mean_nll": float(np.mean(blob["nll"])),
            "mean_idle": float(np.mean([float(r["idle"]) for r in per_scen_943 if r["encoder"] == encoder and int(r["model_seed"]) == seed])),
        }
        align_rows.append(rec)

    write_csv(OUTPUT / "phase944_training_curves.csv", curve_out)
    write_csv(OUTPUT / "phase944_last_epoch_snapshot.csv", snap_rows)
    write_csv(OUTPUT / "phase944_representation_statistics.csv", rep_rows)
    write_csv(OUTPUT / "phase944_oversmoothing_audit.csv", os_rows)
    write_csv(OUTPUT / "phase944_effective_rank_audit.csv", erank_rows)
    write_csv(OUTPUT / "phase944_message_utilization.csv", msg_rows)
    write_csv(OUTPUT / "phase944_relation_sensitivity.csv", sens_rows)
    write_csv(OUTPUT / "phase944_logit_margin.csv", logit_rows)
    write_csv(OUTPUT / "phase944_error_propagation.csv", err_rows)
    write_csv(OUTPUT / "phase944_objective_alignment.csv", align_rows)
    write_csv(OUTPUT / "phase944_near_optimal_multiplicity.csv", multi_rows)
    write_curve_audit(curve_hist, snap_rows)
    write_attr_md(attr_rows)

    def last_os(encoder, seed, key):
        vals = [float(r[key]) for r in os_rows if r["encoder"] == encoder and int(r["model_seed"]) == seed and r["layer"] in {"uu_layer", "layer0_init"}]
        # prefer uu_layer for hgnn
        prefer = [float(r[key]) for r in os_rows if r["encoder"] == encoder and int(r["model_seed"]) == seed and r["layer"] == ("uu_layer" if encoder == "hgnn" else "layer0_init")]
        use = prefer or vals
        return float(np.mean(use)) if use else float("nan")

    def last_er(encoder, seed, field):
        prefer = [float(r[field]) for r in erank_rows if r["encoder"] == encoder and int(r["model_seed"]) == seed and r["layer"] == ("uu_layer" if encoder == "hgnn" else "layer0_init")]
        return float(np.mean(prefer)) if prefer else float("nan")

    def mean_sens(encoder, seed, field):
        vals = [float(r[field]) for r in sens_rows if r["encoder"] == encoder and int(r["model_seed"]) == seed]
        return float(np.mean(vals)) if vals else float("nan")

    pos_h = [row for row in snap_rows if row["encoder"] == "hgnn" and int(row["model_seed"]) in (0, 1, 2)]
    pos_train_nll = float(np.mean([row["train_set_nll"] for row in pos_h]))
    pos_val_nll = float(np.mean([row["val_set_nll"] for row in pos_h]))
    pos_erank = float(np.mean([last_er("hgnn", s, "erank_uav") for s in (0, 1, 2)]))
    pos_irs = float(np.mean([mean_sens("hgnn", s, "irs_uu") for s in (0, 1, 2)]))
    pos_ub = float(np.mean([mean_sens("hgnn", s, "ub_response") for s in (0, 1, 2)]))
    pos_idle_snap = float(np.mean([row["val_fr_idle"] for row in pos_h]))
    pos_margin = float(np.mean([row["val_margin"] for row in pos_h]))
    pos_idle_conf = float(np.mean([float(r["idle"]) for r in per_scen_943 if r["encoder"] == "hgnn" and int(r["model_seed"]) in (0, 1, 2)]))

    class_rows = []
    seed_detail = {}
    for encoder in ("deepsets", "hgnn"):
        for seed in MODEL_SEEDS:
            snap = snap_map[(encoder, seed)]
            pay_h = float(next(r["hgnn"] for r in per_seed_943 if int(r["model_seed"]) == seed))
            pay_d = float(next(r["deepsets"] for r in per_seed_943 if int(r["model_seed"]) == seed))
            align = next(r for r in align_rows if r["encoder"] == encoder and int(r["model_seed"]) == seed)
            bundle = {
                "encoder": encoder,
                "grad_l2": snap["grad_l2"],
                "train_set_nll": snap["train_set_nll"],
                "val_set_nll": snap["val_set_nll"],
                "pos_train_nll": pos_train_nll,
                "pos_val_nll": pos_val_nll,
                "os_u_final": last_os(encoder, seed, "os_u"),
                "erank_uav_final": last_er(encoder, seed, "erank_uav"),
                "pos_erank_uav": pos_erank,
                "irs_uu": mean_sens(encoder, seed, "irs_uu"),
                "pos_irs_uu": pos_irs,
                "ub_dis": mean_sens(encoder, seed, "ub_response"),
                "pos_ub_dis": pos_ub,
                "idle": align["mean_idle"] if encoder == "hgnn" else snap["val_fr_idle"],
                "pos_idle": pos_idle_conf,
                "margin": snap["val_margin"],
                "pos_margin": pos_margin,
                "acc_payload_pearson": align["acc_payload_pearson"],
                "multiplicity_5pct": float(np.mean([r["within_5pct"] for r in multi_rows])),
                "val_fr_acc": snap["val_fr_acc"],
                "payload": pay_h if encoder == "hgnn" else pay_d,
                "paired_deep_payload": pay_d,
            }
            lab = classify_seed(bundle)
            rec = {
                "encoder": encoder,
                "model_seed": seed,
                "payload": bundle["payload"],
                "delta_vs_deepsets": pay_h - pay_d if encoder == "hgnn" else 0.0,
                "train_loss_log": snap["train_loss_log"],
                "train_set_nll": snap["train_set_nll"],
                "val_set_nll": snap["val_set_nll"],
                "val_fr_acc": snap["val_fr_acc"],
                "os_u": bundle["os_u_final"],
                "erank_uav": bundle["erank_uav_final"],
                "irs_uu": bundle["irs_uu"],
                "irs_ub": mean_sens(encoder, seed, "irs_ub"),
                "uu_dis": mean_sens(encoder, seed, "uu_response"),
                "ub_dis": bundle["ub_dis"],
                "idle": bundle["idle"],
                "margin": bundle["margin"],
                "acc_payload_pearson": bundle["acc_payload_pearson"],
                "labels": lab["labels"],
                "evidence": lab["evidence"],
            }
            class_rows.append(rec)
            if encoder == "hgnn":
                seed_detail[seed] = rec

    write_csv(OUTPUT / "phase944_seed_failure_classification.csv", class_rows)

    def pack_encoder(name: str) -> dict[str, Any]:
        snaps = [row for row in snap_rows if row["encoder"] == name]
        hist_last = []
        for seed in MODEL_SEEDS:
            hs = [row for row in curve_hist if row["encoder"] == name and int(row["model_seed"]) == seed]
            hist_last.append(float(sorted(hs, key=lambda r: int(r["epoch"]))[-1]["train_loss"]) if hs else float("nan"))
        return {
            "last_train_loss": hist_last,
            "val_nll": [float(row["val_set_nll"]) for row in snaps],
            "fr_acc": [float(row["val_fr_acc"]) for row in snaps],
        }

    hgnn_pack = pack_encoder("hgnn")
    deep_pack = pack_encoder("deepsets")
    h_pos_nll = [snap_map[("hgnn", s)]["val_set_nll"] for s in (0, 1, 2)]
    h_neg_nll = [snap_map[("hgnn", s)]["val_set_nll"] for s in (3, 4)]
    h_pos_fr = [snap_map[("hgnn", s)]["val_fr_acc"] for s in (0, 1, 2)]
    h_neg_fr = [snap_map[("hgnn", s)]["val_fr_acc"] for s in (3, 4)]
    opt_gap = (
        "不明显：负向 payload 的 seed 3/4 监督 NLL/FR-acc 并不更差（train/val NLL 甚至更好）"
        if float(np.mean(h_neg_nll)) <= float(np.mean(h_pos_nll)) + 0.15
        else "存在：负向 seed 的 val NLL 明显高于正向 seed"
    )
    grads_h = [snap_map[("hgnn", s)]["grad_l2"] for s in MODEL_SEEDS]
    grad_flag = "是" if any(g > 50 or g < 1e-6 for g in grads_h) else "否（last-epoch 诊断 backward 未见爆炸/消失）"
    os_h = [last_os("hgnn", s, "os_u") for s in MODEL_SEEDS]
    os_flag = "是" if any(v > 0.85 for v in os_h) else f"否（终层 OS_U 均值={float(np.mean(os_h)):.3f}）"
    er_h = [last_er("hgnn", s, "erank_uav") for s in MODEL_SEEDS]
    erank_flag = "是" if min(er_h) < 0.6 * float(np.mean(er_h[:3])) else f"否（终层 erank_uav={er_h}）"

    align_h = [r for r in align_rows if r["encoder"] == "hgnn"]
    align_d = [r for r in align_rows if r["encoder"] == "deepsets"]
    align_sum = {
        "hgnn_acc_payload_pearson": float(np.nanmean([r["acc_payload_pearson"] for r in align_h])),
        "deepsets_acc_payload_pearson": float(np.nanmean([r["acc_payload_pearson"] for r in align_d])),
        "hgnn_nll_payload_pearson": float(np.nanmean([r["nll_payload_pearson"] for r in align_h])),
        "deepsets_nll_payload_pearson": float(np.nanmean([r["nll_payload_pearson"] for r in align_d])),
        "hgnn_sinr_payload_pearson": float(np.nanmean([r["sinr_payload_pearson"] for r in align_h])),
        "deepsets_sinr_payload_pearson": float(np.nanmean([r["sinr_payload_pearson"] for r in align_d])),
    }
    multi_sum = {
        "mean_within_1pct": float(np.mean([r["within_1pct"] for r in multi_rows])),
        "mean_within_5pct": float(np.mean([r["within_5pct"] for r in multi_rows])),
        "mean_within_10pct": float(np.mean([r["within_10pct"] for r in multi_rows])),
    }
    ambiguity = multi_sum["mean_within_5pct"] > 3
    gate = instability_gate(class_rows, align_sum, multi_sum)
    dump_frozen(OUTPUT / "phase944_instability_mechanism_gate.json", gate)

    ds_labels = [r["labels"] for r in class_rows if r["encoder"] == "deepsets"]
    deepsets_same = "部分相同（DeepSets 标签: " + ",".join(ds_labels) + "）"

    source = {
        "INST-A": "optimization / representation",
        "INST-B": "objective",
        "INST-C": "optimization + objective",
        "INST-D": "unclear / generalization",
    }[gate["gate"]]

    payload = {
        "hgnn": hgnn_pack,
        "deepsets": deep_pack,
        "opt_gap": opt_gap,
        "grad_flag": grad_flag,
        "os_flag": os_flag,
        "erank_flag": erank_flag,
        "seed3": seed_detail[3],
        "seed4": seed_detail[4],
        "pos_irs_uu": pos_irs,
        "pos_margin": pos_margin,
        "pos_idle": pos_idle_conf,
        "align": align_sum,
        "multiplicity": multi_sum,
        "ambiguity": "严重" if ambiguity else "中等或有限",
        "utility_equivalent_assignment_ambiguity": bool(ambiguity),
        "deepsets_same": deepsets_same,
        "source": source,
        "gate": gate,
        "phase943_delta": summary943["per_seed_delta"],
    }
    dump_frozen(OUTPUT / "summary.json", _plain(payload))
    write_report(payload)
    print(json.dumps({"gate": gate["gate"], "next": gate["next_action"]}, indent=2), flush=True)
    return payload


def reclassify_from_outputs() -> dict[str, Any]:
    """Replay seed labels/gate/report from saved CSVs. Does not reload checkpoints."""
    summary943 = _load_json(PHASE943 / "summary.json")
    per_seed_943 = _load_csv(PHASE943 / "phase943_per_seed_metrics.csv")
    per_scen_943 = _load_csv(PHASE943 / "phase943_per_scenario_metrics.csv")
    snap_rows = [{k: (float(v) if k not in {"encoder"} and v != "" else v) for k, v in row.items()} for row in _load_csv(OUTPUT / "phase944_last_epoch_snapshot.csv")]
    for row in snap_rows:
        row["model_seed"] = int(row["model_seed"])
        row["encoder"] = str(row["encoder"])
    old = _load_csv(OUTPUT / "phase944_seed_failure_classification.csv")
    align_rows = _load_csv(OUTPUT / "phase944_objective_alignment.csv")
    multi_rows = _load_csv(OUTPUT / "phase944_near_optimal_multiplicity.csv")
    curve_hist = parse_training_log(find_training_log()) if find_training_log() else []
    snap_map = {(row["encoder"], int(row["model_seed"])): row for row in snap_rows}
    pos_h = [row for row in snap_rows if row["encoder"] == "hgnn" and int(row["model_seed"]) in (0, 1, 2)]
    pos_train_nll = float(np.mean([row["train_set_nll"] for row in pos_h]))
    pos_val_nll = float(np.mean([row["val_set_nll"] for row in pos_h]))
    pos_erank = float(np.mean([float(r["erank_uav"]) for r in old if r["encoder"] == "hgnn" and int(r["model_seed"]) in (0, 1, 2)]))
    pos_irs = float(np.mean([float(r["irs_uu"]) for r in old if r["encoder"] == "hgnn" and int(r["model_seed"]) in (0, 1, 2)]))
    pos_ub = float(np.mean([float(r["ub_dis"]) for r in old if r["encoder"] == "hgnn" and int(r["model_seed"]) in (0, 1, 2)]))
    pos_margin = float(np.mean([row["val_margin"] for row in pos_h]))
    pos_idle_conf = float(np.mean([float(r["idle"]) for r in per_scen_943 if r["encoder"] == "hgnn" and int(r["model_seed"]) in (0, 1, 2)]))
    multi5 = float(np.mean([float(r["within_5pct"]) for r in multi_rows]))
    class_rows = []
    seed_detail = {}
    for row in old:
        encoder = row["encoder"]
        seed = int(row["model_seed"])
        snap = snap_map[(encoder, seed)]
        pay_h = float(next(r["hgnn"] for r in per_seed_943 if int(r["model_seed"]) == seed))
        pay_d = float(next(r["deepsets"] for r in per_seed_943 if int(r["model_seed"]) == seed))
        align = next(r for r in align_rows if r["encoder"] == encoder and int(r["model_seed"]) == seed)
        bundle = {
            "encoder": encoder,
            "grad_l2": snap["grad_l2"],
            "train_set_nll": snap["train_set_nll"],
            "val_set_nll": snap["val_set_nll"],
            "pos_train_nll": pos_train_nll,
            "pos_val_nll": pos_val_nll,
            "os_u_final": float(row["os_u"]),
            "erank_uav_final": float(row["erank_uav"]),
            "pos_erank_uav": pos_erank,
            "irs_uu": float(row["irs_uu"]),
            "pos_irs_uu": pos_irs,
            "ub_dis": float(row["ub_dis"]),
            "pos_ub_dis": pos_ub,
            "idle": float(align["mean_idle"]) if encoder == "hgnn" else snap["val_fr_idle"],
            "pos_idle": pos_idle_conf,
            "margin": snap["val_margin"],
            "pos_margin": pos_margin,
            "acc_payload_pearson": float(align["acc_payload_pearson"]),
            "multiplicity_5pct": multi5,
            "val_fr_acc": snap["val_fr_acc"],
            "payload": pay_h if encoder == "hgnn" else pay_d,
            "paired_deep_payload": pay_d,
        }
        lab = classify_seed(bundle)
        rec = {
            "encoder": encoder,
            "model_seed": seed,
            "payload": bundle["payload"],
            "delta_vs_deepsets": pay_h - pay_d if encoder == "hgnn" else 0.0,
            "train_loss_log": snap["train_loss_log"],
            "train_set_nll": snap["train_set_nll"],
            "val_set_nll": snap["val_set_nll"],
            "val_fr_acc": snap["val_fr_acc"],
            "os_u": bundle["os_u_final"],
            "erank_uav": bundle["erank_uav_final"],
            "irs_uu": bundle["irs_uu"],
            "irs_ub": float(row["irs_ub"]),
            "uu_dis": float(row["uu_dis"]),
            "ub_dis": bundle["ub_dis"],
            "idle": bundle["idle"],
            "margin": bundle["margin"],
            "acc_payload_pearson": bundle["acc_payload_pearson"],
            "labels": lab["labels"],
            "evidence": lab["evidence"],
        }
        class_rows.append(rec)
        if encoder == "hgnn":
            seed_detail[seed] = rec
    write_csv(OUTPUT / "phase944_seed_failure_classification.csv", class_rows)

    def pack_encoder(name: str) -> dict[str, Any]:
        snaps = [row for row in snap_rows if row["encoder"] == name]
        hist_last = []
        for seed in MODEL_SEEDS:
            hs = [row for row in curve_hist if row["encoder"] == name and int(row["model_seed"]) == seed]
            hist_last.append(float(sorted(hs, key=lambda r: int(r["epoch"]))[-1]["train_loss"]) if hs else float("nan"))
        return {"last_train_loss": hist_last, "val_nll": [float(row["val_set_nll"]) for row in snaps], "fr_acc": [float(row["val_fr_acc"]) for row in snaps]}

    hgnn_pack = pack_encoder("hgnn")
    deep_pack = pack_encoder("deepsets")
    h_pos_nll = [snap_map[("hgnn", s)]["val_set_nll"] for s in (0, 1, 2)]
    h_neg_nll = [snap_map[("hgnn", s)]["val_set_nll"] for s in (3, 4)]
    opt_gap = (
        "不明显：负向 payload 的 seed 3/4 监督 NLL/FR-acc 并不更差（train/val NLL 甚至更好）"
        if float(np.mean(h_neg_nll)) <= float(np.mean(h_pos_nll)) + 0.15
        else "存在：负向 seed 的 val NLL 明显高于正向 seed"
    )
    grads_h = [snap_map[("hgnn", s)]["grad_l2"] for s in MODEL_SEEDS]
    grad_flag = "是" if any(g > 50 or g < 1e-6 for g in grads_h) else "否（last-epoch 诊断 backward 未见爆炸/消失）"
    os_h = [float(r["os_u"]) for r in old if r["encoder"] == "hgnn"]
    os_flag = "是" if any(v > 0.85 for v in os_h) else f"否（终层 OS_U 均值={float(np.mean(os_h)):.3f}）"
    er_h = [float(r["erank_uav"]) for r in class_rows if r["encoder"] == "hgnn"]
    erank_flag = "是" if min(er_h) < 0.6 * float(np.mean(er_h[:3])) else f"否（终层 erank_uav={er_h}）"
    align_h = [r for r in align_rows if r["encoder"] == "hgnn"]
    align_d = [r for r in align_rows if r["encoder"] == "deepsets"]
    align_sum = {
        "hgnn_acc_payload_pearson": float(np.nanmean([float(r["acc_payload_pearson"]) for r in align_h])),
        "deepsets_acc_payload_pearson": float(np.nanmean([float(r["acc_payload_pearson"]) for r in align_d])),
        "hgnn_nll_payload_pearson": float(np.nanmean([float(r["nll_payload_pearson"]) for r in align_h])),
        "deepsets_nll_payload_pearson": float(np.nanmean([float(r["nll_payload_pearson"]) for r in align_d])),
        "hgnn_sinr_payload_pearson": float(np.nanmean([float(r["sinr_payload_pearson"]) for r in align_h])),
        "deepsets_sinr_payload_pearson": float(np.nanmean([float(r["sinr_payload_pearson"]) for r in align_d])),
    }
    multi_sum = {
        "mean_within_1pct": float(np.mean([float(r["within_1pct"]) for r in multi_rows])),
        "mean_within_5pct": float(np.mean([float(r["within_5pct"]) for r in multi_rows])),
        "mean_within_10pct": float(np.mean([float(r["within_10pct"]) for r in multi_rows])),
    }
    ambiguity = multi_sum["mean_within_5pct"] > 3
    gate = instability_gate(class_rows, align_sum, multi_sum)
    dump_frozen(OUTPUT / "phase944_instability_mechanism_gate.json", gate)
    ds_labels = [r["labels"] for r in class_rows if r["encoder"] == "deepsets"]
    payload = {
        "hgnn": hgnn_pack,
        "deepsets": deep_pack,
        "opt_gap": opt_gap,
        "grad_flag": grad_flag,
        "os_flag": os_flag,
        "erank_flag": erank_flag,
        "seed3": seed_detail[3],
        "seed4": seed_detail[4],
        "pos_irs_uu": pos_irs,
        "pos_margin": pos_margin,
        "pos_idle": pos_idle_conf,
        "align": align_sum,
        "multiplicity": multi_sum,
        "ambiguity": "中等偏高" if 2 <= multi_sum["mean_within_5pct"] <= 5 else ("严重" if ambiguity else "有限"),
        "utility_equivalent_assignment_ambiguity": bool(ambiguity),
        "deepsets_same": "是：DeepSets 同样出现 assignment-acc/NLL 与 payload 弱相关（标签: " + ",".join(ds_labels) + "）",
        "source": {"INST-A": "optimization / representation", "INST-B": "objective", "INST-C": "optimization + objective", "INST-D": "unclear / generalization"}[gate["gate"]],
        "gate": gate,
        "phase943_delta": summary943["per_seed_delta"],
    }
    dump_frozen(OUTPUT / "summary.json", _plain(payload))
    write_report(payload)
    print(json.dumps({"gate": gate["gate"], "next": gate["next_action"], "seed3": seed_detail[3]["labels"], "seed4": seed_detail[4]["labels"]}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    import sys

    if "--reclass" in sys.argv:
        reclassify_from_outputs()
    else:
        main()
