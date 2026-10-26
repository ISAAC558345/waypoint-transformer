"""Run smoke, then training and paired evaluation. Merging is a separate decision."""
from __future__ import annotations

import csv
import json

from .criteria import merge_decision
from .evaluate import evaluate_all
from .protocol import (
    GATE_NOT_MERGED,
    GATE_SMOKE_BLOCKED,
    GATE_SMOKE_PASS,
    GATE_TRAINING_BLOCKED,
    LAMBDA_U,
    OUTPUT,
    REPORT,
    RHO,
    TRAIN_SEEDS,
)
from .report import render_report
from .smoke import run_smoke, watched_hashes
from .train import train_variant


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _schema(summary: dict) -> dict:
    return {
        "schema_version": "synthetic_priority_reward_experiment_v1",
        "synthetic_tag": True,
        "lambda_u": LAMBDA_U,
        "rho": RHO,
        "l1_lambda_u": 0.0,
        "train_seeds": list(TRAIN_SEEDS),
        "formal_actor_sha256": summary.get("formal_actor_sha256"),
        "thresholds_changed_after_results": False,
        "trained_l0": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "merged": bool(summary.get("merged")),
    }


def _publish(summary: dict, rows: list[dict] | None = None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        _write_csv(OUTPUT / "scenario_rows.csv", rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "schema.json").write_text(json.dumps(_schema(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")


def main() -> None:
    smoke = run_smoke()
    summary = {
        "gate": smoke["gate"],
        "merged": False,
        "smoke": smoke,
        "problems": list(smoke["problems"]),
        "watched_sha256": smoke["watched_sha256"],
        "formal_actor_sha256": smoke["formal_actor_sha256"],
        "training_text": "训练尚未运行。",
        "decision_text": "Smoke 未通过，不训练，不合并。",
        "manifest": [],
    }
    if smoke["gate"] != GATE_SMOKE_PASS:
        summary["gate"] = GATE_SMOKE_BLOCKED
        _publish(summary)
        print(summary["gate"], len(summary["problems"]))
        return
    summary["decision_text"] = "Smoke 已通过。训练或评估尚未完成，不合并。"
    manifests = []
    try:
        for variant in ("L1", "L2"):
            for seed in TRAIN_SEEDS:
                manifests.append(train_variant(variant, int(seed)))
    except RuntimeError as exc:
        summary["gate"] = GATE_TRAINING_BLOCKED
        summary["problems"] = [str(exc)]
        summary["manifest"] = manifests
        summary["training_text"] = "训练在非有限损失或梯度处停止，超参数和预算没有改动。"
        summary["decision_text"] = "训练失败，不合并。"
        summary["watched_sha256_after"] = watched_hashes()
        _publish(summary)
        print(summary["gate"], exc)
        return
    rows = evaluate_all()
    decision = merge_decision(rows)
    after = watched_hashes()
    if after != smoke["watched_sha256"]:
        summary["problems"] = ["watched formal files changed during training or evaluation"]
        decision["merge_l2"] = False
    summary["manifest"] = manifests
    summary["decision"] = decision
    summary["watched_sha256_after"] = after
    summary["training_text"] = "；".join(
        f"{item['variant']} seed {item['model_seed']} sha256 {item['sha256']} 用时 {item['elapsed_seconds']:.1f} 秒"
        for item in manifests
    )
    if decision["merge_l2"] and not summary["problems"]:
        summary["gate"] = "PRIORITY_TAG_EXPERIMENT_MERGE_READY"
        summary["decision_text"] = "L2 达到预注册标准。可以创建 PR 并合并到 feature/maritime-wt-stage1a。本程序不自动合并。"
    else:
        summary["gate"] = GATE_NOT_MERGED
        summary["merged"] = False
        if decision["l1_meets_same_criteria"] and not decision["merge_l2"]:
            summary["decision_text"] = "L2 未达到预注册标准，不合并。L1 单独达到同一 payload 与安全标准，只在报告中分开记录，不把含 L2 奖励的实验分支合并。"
        else:
            summary["decision_text"] = "结果未达到预注册标准或统计上不充分，保留实验分支，不合并。"
    _publish(summary, rows)
    print(summary["gate"], decision["l2_passing_seeds"], decision["l1_passing_seeds"])


if __name__ == "__main__":
    main()
