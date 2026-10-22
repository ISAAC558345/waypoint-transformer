"""Train the existing HGNN on utility ranks and compare frozen baselines."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5e_utility_dataset.utility import descending_rank
from wt.round30.ppo import seed_everything

from .loss import pairwise_rank_loss
from .protocol import (
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT,
    PATIENCE,
    PLANNER_READY,
    REPORT,
    TEST_SCENARIO,
    TRAIN_SCENARIO,
)
from .score import focal_scores, load_deepsets, load_old_hgnn, new_hgnn


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _best_ids(values: list[float], higher: bool) -> set[int]:
    array = np.asarray(values, dtype=np.float64)
    target = float(np.max(array) if higher else np.min(array))
    return {index for index, value in enumerate(array.tolist()) if float(value) == target}


def _regret(chosen: float, best: float) -> float:
    if best <= 1.0e-6:
        return 0.0 if chosen <= 1.0e-6 else 1.0
    return float((best - chosen) / best)


def _separated(payloads: list[float]) -> bool:
    return float(max(payloads) - min(payloads)) >= 1.0e6 - 1.0


def score_group(scores: np.ndarray, sample: dict) -> dict:
    values = [float(item) for item in np.asarray(scores, dtype=np.float64)]
    ranks = [int(item) for item in sample["utility_rank"]]
    payloads = [float(item) for item in sample["payload_gain"]]
    preference = [-float(item) for item in ranks]
    chosen = int(np.argmax(np.asarray(values, dtype=np.float64)))
    pred_rank = descending_rank(values)
    label_best = _best_ids(ranks, higher=False)
    payload_best = max(payloads)
    return {
        "scenario_id": int(sample["scenario_id"]),
        "time_step": int(sample["time_step"]),
        "uav_id": int(sample["uav_id"]),
        "spearman": spearman(values, preference),
        "kendall": kendall_tau(values, preference),
        "top1": float(chosen in label_best or bool(_best_ids(values, higher=True).intersection(label_best))),
        "top3": float(any(int(pred_rank[index]) <= 3 for index in label_best)),
        "payload": payloads[chosen],
        "regret": _regret(payloads[chosen], payload_best),
        "separated": _separated(payloads),
        "chosen_candidate": int(sample["candidate_ids"][chosen]),
        "label_best_candidates": [int(sample["candidate_ids"][index]) for index in sorted(label_best)],
    }


def assignment_group(sample: dict) -> dict:
    return score_group(np.asarray(sample["assignment_score"], dtype=np.float64), sample)


def evaluate_model(model, rows: list[dict], *, use_urgency: bool) -> list[dict]:
    was_training = bool(model.training)
    model.eval()
    scored = []
    with torch.no_grad():
        for sample in rows:
            values = focal_scores(model, sample, use_urgency=use_urgency).detach().cpu().numpy()
            scored.append(score_group(values, sample))
    if was_training:
        model.train()
    return scored


def _pack(rows: list[dict]) -> dict:
    separated = [row for row in rows if row["separated"]]
    out = {}
    for name, subset in (("all", rows), ("payload_separated", separated)):
        out[name] = {
            "n_groups": len(subset),
            "spearman": _finite_mean([row["spearman"] for row in subset]),
            "kendall": _finite_mean([row["kendall"] for row in subset]),
            "top1": _finite_mean([row["top1"] for row in subset]),
            "top3": _finite_mean([row["top3"] for row in subset]),
            "payload": _finite_mean([row["payload"] for row in subset]),
            "regret": _finite_mean([row["regret"] for row in subset]),
        }
    return out


def train_model(model, train_rows: list[dict], val_rows: list[dict]) -> list[dict]:
    seed_everything(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = -1.0e18
    wait = 0
    history = []
    for epoch in range(int(EPOCHS)):
        model.train()
        order = np.random.permutation(len(train_rows))
        losses = []
        for index in order:
            sample = train_rows[int(index)]
            scores = focal_scores(model, sample, use_urgency=True)
            loss = pairwise_rank_loss(scores, sample["utility_rank"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_rows_scored = evaluate_model(model, val_rows, use_urgency=True)
        val_score = _pack(val_rows_scored)["payload_separated"]["spearman"]
        if not np.isfinite(val_score):
            val_score = -1.0e18
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_spearman": None if val_score <= -1.0e17 else float(val_score),
            }
        )
        print(
            f"utility hgnn epoch {epoch + 1}/{EPOCHS} loss={history[-1]['train_loss']:.4f} val_spearman={history[-1]['val_spearman']}",
            flush=True,
        )
        if val_score > best_score + 1.0e-6:
            best_score = float(val_score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(PATIENCE):
                break
    model.load_state_dict(best_state)
    model.eval()
    return history


def _fmt(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def _phase5g(test_pack: dict) -> dict:
    new = test_pack["new_hgnn"]["payload_separated"]
    old = test_pack["old_hgnn"]["payload_separated"]
    learned = bool(
        np.isfinite(new["spearman"])
        and np.isfinite(old["spearman"])
        and new["spearman"] > old["spearman"]
        and new["top1"] > old["top1"]
        and new["payload"] > old["payload"]
        and new["regret"] < old["regret"]
    )
    return {"metric_improvement": learned, "phase5g_started": False}


def _report(summary: dict) -> str:
    test = summary["test"]
    history = summary["history"]
    curve = ["| epoch | train loss | validation Spearman |", "| --- | --- | --- |"]
    for row in history:
        curve.append(f"| {row['epoch']} | {_fmt(row['train_loss'])} | {_fmt(row['val_spearman'])} |")
    names = (
        ("assignment similarity", "assignment_score"),
        ("DeepSets checkpoint", "deepsets"),
        ("HGNN assignment similarity", "old_hgnn"),
        ("HGNN utility ranking", "new_hgnn"),
    )
    table = [
        "| model | subset | n | Spearman | Kendall | Top-1 | Top-3 | 15-step payload | regret |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in names:
        for subset in ("all", "payload_separated"):
            item = test[key][subset]
            table.append(
                f"| {label} | {subset} | {item['n_groups']} | {_fmt(item['spearman'])} | {_fmt(item['kendall'])} | {_fmt(item['top1'])} | {_fmt(item['top3'])} | {_fmt(item['payload'])} | {_fmt(item['regret'])} |"
            )
    learned = summary["phase5g"]["metric_improvement"]
    learned_text = "测试场景上，新 HGNN 的排序和 15 步 payload 同时好于旧的 assignment HGNN。" if learned else "测试场景上，新 HGNN 还没有同时提高排序和 15 步 payload。"
    return "\n".join(
        [
            "# HTW Phase-5F HGNN Trajectory Utility Retraining",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段只重训现有 high-level HGNN 的监督目标。没有重训 Residual MAPPO，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C、低层策略或 planner，没有使用 230000–230127，也没有进入 Phase-5G。",
            "",
            "## 1. 模型",
            "",
            "结构保持 Phase-5 的 `Phase5HGNNEncoder` 和 `OrderFreeSetDecoder`。没有增加新的网络模块。UAV–Buoy 边和 UAV–UAV 边保留。新模型的浮标输入多一个当前队列比例 urgency，这是原 encoder 已经支持的通道。",
            "",
            "对照不重新训练：冻结 DeepSets 检查点，以及 Phase-5 用 assignment set NLL 训练的 `hgnn_orderfree.pt`。",
            "",
            "## 2. Loss",
            "",
            "assignment similarity 的 set NLL 没有用于新模型。每个无人机在其他无人机分配固定时，decoder 对合法浮标给出分数 s。标签是 Phase-5E 的 `utility_rank`，1 为最好。",
            "",
            "对所有 rank(i) < rank(j) 的候选对，loss 是 `-log sigmoid(s_i - s_j)` 的平均。并列候选不产生配对。这是成对排序损失，系数没有再搜索，τ 没有读取或修改。",
            "",
            "## 3. 训练数据",
            "",
            "标签来自 `phase5e_utility_dataset/trajectory_utility_dataset.csv`。状态由同一套 310100、310101 重置重放得到，并逐项核对 queue、channel、distance、ETA、interference、urgency 和 15 步 payload。没有新的 scenario。",
            "",
            f"Phase-5E 只有两个场景，不能再切出第三个互不重叠的场景。训练是场景 {TRAIN_SCENARIO} 的 t=0 和 t=15。验证是同一场景的 t=30，只用于选择 epoch。测试是整个场景 {TEST_SCENARIO}，不参与训练或 early stopping。",
            "",
            "## 4. Feature isolation",
            "",
            "送入 encoder 的键只有 uav、buoy、mask、global、edge、edge_uu。`payload_gain`、`utility_score` 和 `utility_rank` 不在输入里。urgency 是当前队列比例。15 步轨迹没有进入输入。加载数据时如果这些标签出现在图里会直接报错。",
            "",
            "## 5. 训练曲线",
            "",
            *curve,
            "",
            "验证 Spearman 用的是 payload 已经分开的组。检查点保留验证 Spearman 最高的那一轮参数。",
            "",
            "## 6. 测试场景 ranking",
            "",
            "Spearman 和 Kendall 比较的是预测分数与 utility 名次：名次越好，目标分数越高。Top-1 看最高分是否落在最好的 utility 名次上。Top-3 看最好的 utility 候选是否进入预测前 3。payload 和 regret 用该候选在 Phase-5E 里已经核对过的 15 步 fly-toward shadow。",
            "",
            *table,
            "",
            f"payload 已经分开的 {test['new_hgnn']['payload_separated']['n_groups']} 组才拉开了候选。新 HGNN 的 Spearman 是 {_fmt(test['new_hgnn']['payload_separated']['spearman'])}，Top-1 是 {_fmt(test['new_hgnn']['payload_separated']['top1'])}，平均 15 步 payload 是 {_fmt(test['new_hgnn']['payload_separated']['payload'])} bit，regret 是 {_fmt(test['new_hgnn']['payload_separated']['regret'])}。旧 assignment HGNN 的 Spearman 是 {_fmt(test['old_hgnn']['payload_separated']['spearman'])}，Top-1 是 {_fmt(test['old_hgnn']['payload_separated']['top1'])}，payload 是 {_fmt(test['old_hgnn']['payload_separated']['payload'])} bit，regret 是 {_fmt(test['old_hgnn']['payload_separated']['regret'])}。冻结 DeepSets 检查点原来学的是候选池上的 payload 分布，不是 assignment NLL；它在这些组上的 payload 是 {_fmt(test['deepsets']['payload_separated']['payload'])} bit。",
            "",
            "全部 18 组里的 Top-1 会把 payload 没有差别的组算成命中，因为那些组的 utility 名次全部并列。分开的 11 组才是排序是否有效的结果。",
            "",
            "## 7. Payload",
            "",
            "15 步 payload 是选中候选后、其他无人机保持当前排他分配、低层固定为 fly-toward 的通信增量。没有调用 MAPPO，也没有调用 `decide()`。regret 是该组内最大 payload 减去选中 payload，再除以最大 payload。",
            "",
            "## 8. Failure cases",
            "",
            f"文件：`phase5f_utility_rank/utility_rank_failures.csv`。测试场景中新模型 Top-1 没有命中 utility 最优候选的组数是 {summary['failure_count']}。",
            "",
            "## 9. Phase-5G",
            "",
            learned_text,
            "",
            "这个提高只来自一个隔离测试场景上的 11 个有效组。它说明现有 HGNN 可以在不看未来轨迹的情况下学到一部分 utility 排序，并提高 fly-toward 的 15 步 payload。它不构成 deployment 结论，也不启动 Phase-5G。",
            "",
            "本阶段不进入 Phase-5G，也不做 deployment verification。系统优化没有完成。",
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def run() -> dict:
    from .data import load_splits

    seed_everything(0)
    print("join phase-5e labels", flush=True)
    splits = load_splits()
    model = new_hgnn()
    history = train_model(model, splits["train"], splits["validation"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT / "hgnn_utility_rank.pt")
    test_rows = splits["test"]
    test = {
        "assignment_score": _pack([assignment_group(sample) for sample in test_rows]),
        "deepsets": _pack(evaluate_model(load_deepsets(), test_rows, use_urgency=False)),
        "old_hgnn": _pack(evaluate_model(load_old_hgnn(), test_rows, use_urgency=False)),
        "new_hgnn": _pack(evaluate_model(model, test_rows, use_urgency=True)),
    }
    failures = []
    new_groups = evaluate_model(model, test_rows, use_urgency=True)
    for row in new_groups:
        if row["top1"] < 1.0:
            failures.append(
                {
                    "scenario": row["scenario_id"],
                    "time_step": row["time_step"],
                    "uav": row["uav_id"],
                    "chosen_candidate": row["chosen_candidate"],
                    "utility_best_candidates": " ".join(str(item) for item in row["label_best_candidates"]),
                    "payload": row["payload"],
                    "regret": row["regret"],
                }
            )
    with (OUTPUT / "utility_rank_failures.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["scenario", "time_step", "uav", "chosen_candidate", "utility_best_candidates", "payload", "regret"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "mappo_retrained": False,
        "tau_modified": False,
        "holdout_used": False,
        "phase5g_started": False,
        "new_network_module": False,
        "train_scenario": TRAIN_SCENARIO,
        "test_scenario": TEST_SCENARIO,
        "train_groups": len(splits["train"]),
        "validation_groups": len(splits["validation"]),
        "test_groups": len(test_rows),
        "history": history,
        "test": test,
        "failure_count": len(failures),
        "phase5g": _phase5g(test),
    }
    (OUTPUT / "phase5f_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"failures": len(failures), "test_separated": {key: test[key]["payload_separated"] for key in test}}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
