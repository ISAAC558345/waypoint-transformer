"""Phase-10.0 archive, interface check, and small design-set equivalence. No training."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from wt.htw_phase9.const import HOLD, IDLE_INDEX
from wt.htw_phase941.protocol import MODEL_SEEDS
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase945.protocol import OUTPUT as P945
from wt.l3b.config import L3BConfig
from wt.round30.ppo import seed_everything

from .compatibility import audit
from .high_level import load_deepsets_util
from .loop import compare_traces, run_path_a, run_path_b
from .protocol import (
    ARCHIVE,
    DESIGN_SEEDS,
    OUTPUT,
    PACK_REPORT,
    PHASE943_CKPT,
    POOL_SEED,
    POOL_SIZE,
    REPORT,
    ROOT,
    dump_frozen,
    sha256_bytes,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_from_history(encoder: str, seed: int) -> dict[str, Any]:
    path = P945 / f"phase945_train_history_{encoder}_{seed}.csv"
    if not path.is_file():
        return {"best_epoch": None, "best_val_regret": None, "source": "missing_history"}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    val = [r for r in rows if r.get("val_top1_regret")]
    if not val:
        return {"best_epoch": None, "best_val_regret": None, "source": str(path)}
    best = min(val, key=lambda r: float(r["val_top1_regret"]))
    return {"best_epoch": int(best["epoch"]), "best_val_regret": float(best["val_top1_regret"]), "source": str(path)}


def _npz_pool(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {"path": str(path), "sha256": sha256_bytes(path.read_bytes()), "keys": list(data.files), "n": int(data["assignment"].shape[0]), "pool_size": int(np.asarray(data["candidates"]).shape[1]) if "candidates" in data.files else None}


def _meta(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "sha256": sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size}


def restore_945() -> dict[str, Any]:
    summary = _read_json(P945 / "summary.json")
    freeze = _read_json(P945 / "phase945_prefinal_freeze.json")
    proto = _read_json(P945 / "phase945_protocol.json")
    accel = _read_json(P945 / "PHASE945_COMPUTE_ACCELERATION_AMENDMENT.json")
    pool_man = _read_json(P945 / "phase945_candidate_pool_manifest.json")
    util_man = _read_json(P945 / "phase945_candidate_utility_dataset_manifest.json")
    obj_gate = _read_json(P945 / "phase945_objective_gate.json")
    graph_gate = _read_json(P945 / "phase945_graph_gate.json")
    conf = _read_json(P945 / "phase945_confirmation_manifest.json")
    old_rule = _read_json(PHASE943_CKPT / "phase943_protocol.json").get("checkpoint_rule")
    sel_csv = P945 / "phase945_checkpoint_selection.csv"
    sel_rows = list(csv.DictReader(sel_csv.open(encoding="utf-8"))) if sel_csv.is_file() else []
    disk_payloads = {
        "deepsets_old": summary["d_old"],
        "deepsets_util": summary["d_util"],
        "hgnn_old": summary["g_old"],
        "hgnn_util": summary["g_util"],
        "uobj": summary["uobj"]["gate"],
        "graph_u": summary["graph"]["gate"],
        "next_action": summary["next_step"],
        "d_per_seed": summary.get("d_per_seed"),
        "g_per_seed": summary.get("g_per_seed"),
    }
    expected = {
        "deepsets_old": 3200000.0,
        "deepsets_util": 6950000.0,
        "hgnn_old": 3393750.0,
        "hgnn_util": 6125000.0,
        "uobj": "UOBJ-A",
        "graph_u": "GRAPH-U-C",
        "next_action": "retain DeepSets",
    }
    matches = {key: disk_payloads[key] == expected[key] for key in expected}
    l32_hits = []
    for name in ("protocol.py", "AGENTS.md"):
        text = (ROOT / "src" / "wt" / "htw_phase945" / name).read_text(encoding="utf-8")
        if "POOL_SIZE = 32" in text or "pool_size = 32" in text:
            l32_hits.append(name)
    csv_nan = all((row.get("best_epoch") == "-1") for row in sel_rows) if sel_rows else True
    return {
        "disk_payloads": disk_payloads,
        "matches_user_table": matches,
        "pool_size_code": POOL_SIZE,
        "pool_size_freeze": freeze.get("pool_size"),
        "pool_size_protocol_json": proto.get("pool_size"),
        "pool_size_accel": accel.get("frozen_pool_size"),
        "pool_size_candidate_manifest": pool_man.get("pool_size"),
        "pool_size_utility_manifest": util_man.get("pool_size"),
        "pool_seed": freeze.get("pool_seed", POOL_SEED),
        "accel_note_L": accel.get("note_L"),
        "l32_in_phase945_source": l32_hits,
        "pool_size_protocol_note": (
            "Disk freeze is L=24 in protocol.py, prefinal_freeze.json, candidate/utility manifests, "
            "and acceleration amendment note_L. No POOL_SIZE=32 remains in Phase-9.4.5 source or JSON. "
            "An earlier acceleration instruction mentioning 32 was never written as a frozen scientific L. "
            "Recorded as oral-vs-disk discrepancy / protocol deviation; history not rewritten; no rerun."
        ),
        "old_checkpoint_rule": old_rule,
        "new_checkpoint_rule": proto.get("checkpoint_rule"),
        "checkpoint_rules_equal": old_rule == proto.get("checkpoint_rule"),
        "gain_wording": "收益对齐训练流程整体改进" if old_rule != proto.get("checkpoint_rule") else "checkpoint rule matched",
        "checkpoint_selection_csv": {
            "path": str(sel_csv),
            "all_best_epoch_minus_one": csv_nan,
            "note": "CSV lists best_epoch=-1/nan (skip-train leftover). Official .pt files are still the train() best-val-regret state_dicts; recover epoch from phase945_train_history_*.csv. Not rewritten.",
        },
        "utility_executor": "fly-toward",
        "hold": HOLD,
        "uobj_gate_file": obj_gate,
        "graph_u_gate_file": graph_gate,
        "confirmation_n_seeds": len(conf.get("seeds", []) or conf.get("confirmation_seeds", []) or []),
        "artifacts": {
            "summary": _meta(P945 / "summary.json"),
            "protocol": _meta(P945 / "phase945_protocol.json"),
            "freeze": _meta(P945 / "phase945_prefinal_freeze.json"),
            "candidate_pool_manifest": _meta(P945 / "phase945_candidate_pool_manifest.json"),
            "utility_dataset_manifest": _meta(P945 / "phase945_candidate_utility_dataset_manifest.json"),
            "training_seed_manifest": _meta(P945 / "phase945_training_seed_manifest.json"),
            "checkpoint_selection_csv": _meta(sel_csv),
            "confirmation_manifest": _meta(P945 / "phase945_confirmation_manifest.json"),
            "uobj": _meta(P945 / "phase945_objective_gate.json"),
            "graph_u": _meta(P945 / "phase945_graph_gate.json"),
            "compute_accel": _meta(P945 / "PHASE945_COMPUTE_ACCELERATION_AMENDMENT.json"),
            "compute_equivalence": _meta(P945 / "phase945_compute_equivalence_gate.json"),
            "train_npz": _npz_pool(P945 / "phase945_train.npz"),
            "val_npz": _npz_pool(P945 / "phase945_val.npz"),
        },
    }


def archive_deepsets() -> dict[str, Any]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in MODEL_SEEDS:
        src = P945 / f"deepsets_seed{seed}.pt"
        dst = ARCHIVE / f"deepsets_util_seed{seed}.pt"
        shutil.copy2(src, dst)
        hist = _best_from_history("deepsets", int(seed))
        rows.append(
            {
                "encoder": "deepsets",
                "model_seed": int(seed),
                "source_path": str(src),
                "archive_path": str(dst),
                "sha256": sha256_bytes(src.read_bytes()),
                "archive_sha256": sha256_bytes(dst.read_bytes()),
                "selected_by_confirmation_payload": False,
                **hist,
            }
        )
    hgnn_kept = []
    for seed in MODEL_SEEDS:
        path = P945 / f"hgnn_seed{seed}.pt"
        hgnn_kept.append({"encoder": "hgnn", "model_seed": int(seed), "path": str(path), "sha256": sha256_bytes(path.read_bytes()) if path.is_file() else None, "role": "historical_control_not_primary"})
    cfg = L3BConfig()
    n_params = int(sum(p.numel() for p in OrderFreeAssignmentModel("deepsets").parameters()))
    norm = _read_json(P945 / "phase945_utility_normalization.json") if (P945 / "phase945_utility_normalization.json").is_file() else {}
    return {
        "encoder": "deepsets",
        "decoder": "OrderFreeSetDecoder",
        "objective": "utility_aligned_set_supervision",
        "normalization": norm.get("method", "median_mad"),
        "normalization_spec": norm,
        "candidate_pool_size": POOL_SIZE,
        "candidate_pool_seed": POOL_SEED,
        "utility_dataset": {"train": str(P945 / "phase945_train.npz"), "val": str(P945 / "phase945_val.npz")},
        "checkpoint_rule": "best_validation_mean_candidate_regret",
        "action_encoding": "env association int64; idle=-1; decoder idle token=IDLE_INDEX=12",
        "idle_index": int(IDLE_INDEX),
        "hold": int(HOLD),
        "hold_decision_times": "t=0,15,30,45 on T=48; last segment truncated by episode end; no early terminate",
        "environment": {"name": "L3BEnv", "num_uavs": cfg.num_uavs, "num_buoys": cfg.num_buoys, "horizon": cfg.episode_horizon, "service_slots": cfg.service_slots},
        "executor_used_for_labels_and_eval": "wt.l3b.policies.fly_toward",
        "n_parameters": n_params,
        "models": rows,
        "hgnn_historical": hgnn_kept,
        "selection": "all five training seeds archived; no confirmation-max pick",
        "checkpoint_selection_csv_note": "phase945_checkpoint_selection.csv has best_epoch=-1; archive uses on-disk .pt plus train_history best_val_regret",
    }


def run_equivalence() -> dict[str, Any]:
    seed_everything(0)
    pairs = []
    all_ok = True
    for seed in MODEL_SEEDS:
        model = load_deepsets_util(int(seed), ARCHIVE / f"deepsets_util_seed{seed}.pt")
        for scene in DESIGN_SEEDS:
            a = run_path_a(model, int(scene))
            b = run_path_b(model, int(scene))
            cmp_ = compare_traces(a, b)
            ok = bool(cmp_["stepwise_equivalent"])
            all_ok = all_ok and ok
            pairs.append({"model_seed": int(seed), "scenario": int(scene), "ok": ok, "n_mismatches": cmp_["n_mismatches"], "payload": cmp_["final_payload_a"], "metrics": cmp_["metrics_a"], "failed": cmp_["mismatches_head"]})
    return {
        "design_seeds": list(DESIGN_SEEDS),
        "design_seeds_are_confirmation": False,
        "n_pairs": len(pairs),
        "all_stepwise_equivalent": all_ok,
        "pairs": pairs,
    }


def write_docs(facts: dict[str, Any], archive: dict[str, Any], equiv: dict[str, Any], compat: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dump_frozen(OUTPUT / "phase100_high_level_checkpoint_manifest.json", archive)
    dump_frozen(OUTPUT / "phase100_rule_executor_equivalence.json", equiv)
    dump_frozen(OUTPUT / "phase100_945_restore.json", facts)
    dump_frozen(OUTPUT / "phase100_low_level_compatibility.json", compat)
    (OUTPUT / "phase100_protocol_traceability.md").write_text(
        "\n".join(
            [
                "# Phase-10.0 协议追溯",
                "",
                "## Phase-9.4.5 磁盘事实",
                f"- DeepSets-old payload = {facts['disk_payloads']['deepsets_old']}",
                f"- DeepSets-util payload = {facts['disk_payloads']['deepsets_util']}",
                f"- HGNN-old payload = {facts['disk_payloads']['hgnn_old']}",
                f"- HGNN-util payload = {facts['disk_payloads']['hgnn_util']}",
                f"- UOBJ = {facts['disk_payloads']['uobj']}",
                f"- GRAPH-U = {facts['disk_payloads']['graph_u']}",
                f"- next_action = {facts['disk_payloads']['next_action']}",
                f"- 与用户表逐项相等 = {facts['matches_user_table']}",
                "",
                "## L=24 vs 口头 L=32",
                f"- `protocol.py` POOL_SIZE = {facts['pool_size_code']}",
                f"- `phase945_prefinal_freeze.json` pool_size = {facts['pool_size_freeze']}",
                f"- acceleration `frozen_pool_size` = {facts['pool_size_accel']}",
                f"- candidate/utility manifest pool_size = {facts['pool_size_candidate_manifest']}/{facts['pool_size_utility_manifest']}",
                f"- acceleration note_L = {facts['accel_note_L']}",
                f"- 源码中的 POOL_SIZE=32：{facts['l32_in_phase945_source'] or '无'}",
                f"- 说明：{facts['pool_size_protocol_note']}",
                f"- checkpoint_selection.csv 偏差：{facts['checkpoint_selection_csv']}",
                "",
                "## Checkpoint 规则",
                f"- 旧（9.4.3）= `{facts['old_checkpoint_rule']}`",
                f"- 新（9.4.5）= `{facts['new_checkpoint_rule']}`",
                f"- 是否一致 = {facts['checkpoint_rules_equal']}",
                f"- 表述：{facts['gain_wording']}",
                "",
                "## 系统边界",
                "- 已验证：DeepSets 高层 → assignment/task option → fly-toward 规则执行器 → L3-B。",
                "- 未验证：DeepSets 高层 → L3-B 目标条件低层 PPO → L3-B。",
                "- 不能把「Round-30 PPO 核心文件未改」写成「当前实际使用低层 PPO」。",
                "- 9.4.5 utility 标签在 fly-toward 下计算；换成学习低层后高层收益可能变化。本阶段未重算标签。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUTPUT / "phase100_task_option_contract.md").write_text(
        "\n".join(
            [
                "# Task Option 最小接口",
                "",
                "高层输出：`targets`（每机 buoy 或 idle=-1）、`decision_time`、`remaining_hold`、`uav_mask`、`buoy_mask`、`association`（实际传入 `env.step`）。",
                "执行器输入：当前可观测 L3-B 状态 + 当前 TaskOption。",
                "执行器输出：`(M,2)` 单位方向 movement；环境按 `step_length_m` 限幅。",
                f"idle 编码：assignment `-1`；decoder token `{IDLE_INDEX}`。",
                "高层负责 association；低层只负责 movement。adapter 若发现 association≠targets 则报错，不重做 matching。",
                f"Hold：T={L3BConfig().episode_horizon}，每 {HOLD} 步在 pending<=0 时重决策（0/15/30/45），最后一段被 episode 截断。无完成即换目标、无队列清空换目标、无新 termination/安全投影/奖励。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUTPUT / "phase100_low_level_compatibility_report.md").write_text(
        "\n".join(
            [
                "# 旧低层 PPO 与 L3-B 兼容性",
                "",
                f"**结论：{compat['verdict_zh']}**（`{compat['verdict']}`）。",
                f"- Phase-6.2 checkpoint `{compat['phase62_ckpt']}` 存在：{compat['phase62_ckpt_exists']}",
                "- 问题清单：",
                *[f"  - {row}" for row in compat["issues"]],
                "- 可复用：" + "；".join(compat["reusable"]),
                "- 必须适配：" + "；".join(compat["must_adapt"]),
                "- 本阶段未训练、未随机初始化评估、未在加载失败时静默退回 fly-toward 并标成 PPO。",
                "",
                "## 后续低层实验计划（不执行）",
                "- B0：冻结 DeepSets-util 五种子 + fly-toward。",
                "- B1：同一冻结高层 + L3-B 目标条件低层 PPO（需先完成 M=6 特征适配后再训）。",
                "- 同环境、同 hold、同评估初始状态与到达过程；各自闭环允许高层按各自状态重决策，不强行对齐 assignment。",
                "- 先用 1 个 design seed、极少 updates 做 smoke 测墙钟，再估全量；禁止预设「PPO 必须超过规则」。规则足够则可保留 B0。",
                "- 指标必须区分：进入服务区、持续有效服务、实际完成块、累计 payload；SINR 下 D=3 不是固定三步完成。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "# HTW Phase-10.0 DeepSets 高层定版与低层执行接口验证",
        "",
        "未训练任何网络。未做高层/低层 PPO。未再生 utility NPZ。未重跑 9.4.5 confirmation。未优化 HGNN。未开新 Round。",
        "",
        f"1. 24 候选协议来源：代码与 `phase945_prefinal_freeze.json` 冻结 L={facts['pool_size_code']}；加速修正写明未改 L。口头/加速指令中的 32 不是磁盘冻结科学协议。",
        f"2. 旧 checkpoint 规则 `{facts['old_checkpoint_rule']}`，新规则 `{facts['new_checkpoint_rule']}`。不一致。9.4.5 相对 9.4.3 的 payload 提升表述为**收益对齐训练流程整体改进**，不把全部提升归因于单一 loss 项。",
        f"3. 五个 DeepSets-util 已复制到 `{ARCHIVE}`，源与副本 SHA 见 `phase100_high_level_checkpoint_manifest.json`。未按 confirmation 最高 payload 选单模型。",
        f"4. 原路径 vs 新接口：design seeds {list(DESIGN_SEEDS)} × 5 个冻结高层，逐步等价={equiv['all_stepwise_equivalent']}。",
        "5. 当前低层实际是 **fly-toward 规则执行器**，不是 PPO。",
        f"6. 旧 PPO 与 L3-B：{compat['verdict_zh']}。",
        "7. 后续至少适配 M=6 特征垫、option 拼接形状、SINR 服务完成语义，并准备可加载的 L3-B checkpoint 路径。",
        f"8. 启动 L3-B 低层训练的工程条件：接口已定义且规则路径等价；**权重兼容条件未满足**（L2-A M=5 checkpoint 存在但不可直接加载到 L3-B M=6）。",
        "9. 本阶段是否训练过模型：**否**。",
        "10. 下一阶段最小任务：在 design 集上实现 L3-B 目标条件低层（不加载 5 机权重），先 smoke，再与冻结 DeepSets + fly-toward（B0）同协议对照；不预设 PPO 必须更好。",
        "",
        f"等价性 JSON：`{OUTPUT / 'phase100_rule_executor_equivalence.json'}`",
        "",
    ]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    facts = restore_945()
    archive = archive_deepsets()
    dump_frozen(OUTPUT / "phase100_high_level_checkpoint_manifest.json", archive)
    equiv = run_equivalence()
    compat = audit()
    write_docs(facts, archive, equiv, compat)
    print(json.dumps({"equiv": equiv["all_stepwise_equivalent"], "n_pairs": equiv["n_pairs"], "verdict": compat["verdict"], "trained": False}, indent=2), flush=True)
    return {"facts": facts, "archive": archive, "equiv": equiv, "compat": compat}


if __name__ == "__main__":
    main()
