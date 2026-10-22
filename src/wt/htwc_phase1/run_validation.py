"""Run HTW Phase-1 validation stages 0--3 on a mini budget. No Round, no fresh/untouched."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .validation import (
    MINI_UPDATES,
    SMOKE_CONFIG,
    build_specs,
    eval_slice,
    evaluate_baseline,
    interface_shapes,
    train_baseline,
    train_slice,
    warmup_new_predictor,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "validation_mini"
REPORT = Path(__file__).resolve().parent / "HTW_Phase1_validation_report.md"
REPO_REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase1_validation_report.md"


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if key != "agent" and key != "rows"}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def write_report(payload: Mapping[str, Any]) -> None:
    evals = payload["evaluation"]
    rho_old = evals["tw_old"]["action_relevance"].get("mean_rho")
    rho_new = evals["htw_new"]["action_relevance"].get("mean_rho")
    payload_old = evals["tw_old"]["mean_payload_bits"]
    payload_new = evals["htw_new"]["mean_payload_bits"]
    payload_state = evals["state_only"]["mean_payload_bits"]
    new_better_payload = payload_new > payload_old and payload_new > payload_state
    new_better_rho = np.isfinite(rho_new) and np.isfinite(rho_old) and float(rho_new) > float(rho_old)
    effective = bool(new_better_payload or new_better_rho)
    reasons = []
    if not new_better_payload:
        reasons.append("mini-budget PPO 下 New TW 的 cumulative payload 未同时超过 State-only 与 Old TW。")
    if not new_better_rho:
        reasons.append("mean rho_new 未稳定大于 rho_old，低层未必在跟踪 waypoint。")
    reasons.append("TW_new 预测器仅做了 planner 窗口上的短 warmup，不是正式监督训练。")
    reasons.append("planner 为 independent_queue_rate，不是最优轨迹。")
    reasons.append("PPO 更新次数远小于 Round-30 正式预算，结果只作接口级验证。")
    lines = [
        "# HTW Phase-1 Validation Report",
        "",
        "本报告对应 **mini-budget 验证**，不是 Round-30 规模正式训练。未访问 fresh/untouched/protected JSONL，未修改 HGNN、环境、reward、PPO 核心公式。",
        "",
        "## 1. 三个 baseline 结构",
        "",
        "| Baseline | 名称 | UAV 输入维 | TW 来源 |",
        "|---|---|---:|---|",
        f"| 1 | State-only PPO | {payload['interfaces']['state_only']} | 无 |",
        f"| 2 | Original Task-Waypoint PPO | {payload['interfaces']['tw_old']} | 冻结 Round-21 DeepSets `phi=[service,progress,return]`，不重训预测器 |",
        f"| 3 | New HTW Task-Waypoint PPO | {payload['interfaces']['htw_new']} | warmup 后的 TW_new 预测器 `goal,waypoint,horizon` |",
        "",
        "## 2. 输入差异",
        "",
        "- Baseline 1：`feature_dict` 原始 8 维 UAV 特征。",
        "- Baseline 2：拼接 11 维 TW_old（每机 service5 + progress5 + return1）。",
        "- Baseline 3：拼接 8 维 TW_new（goal5 + waypoint2 + horizon1）。",
        "- 环境、reward、PPO clip/GAE/optimizer、seed、episode 评估集相同。",
        "",
        "## 3. 训练配置",
        "",
        f"- PPO：`rollout_length={SMOKE_CONFIG.rollout_length}`, `num_envs={SMOKE_CONFIG.num_envs}`, `update_epochs={SMOKE_CONFIG.update_epochs}`, `clip_eps={SMOKE_CONFIG.clip_eps}`。",
        f"- Mini-formal 更新次数：{MINI_UPDATES} / baseline（环境步约 {payload['training']['state_only']['environment_steps']}）。",
        f"- Seed：{payload['seed']}。训练场景：`training_scenarios()[:16]`；评估：随后 4 个训练场景（非 fresh、非 untouched）。",
        f"- TW_new predictor warmup loss 末值：{payload['predictor_warmup']['loss_last']:.6f}。",
        "",
        "## 4. Payload 结果",
        "",
        f"- State-only：{payload_state:.3f} bits",
        f"- Old TW：{payload_old:.3f} bits",
        f"- New TW：{payload_new:.3f} bits",
        "",
        "## 5. Reward 结果",
        "",
        f"- State-only return：{evals['state_only']['mean_return']:.6f}",
        f"- Old TW return：{evals['tw_old']['mean_return']:.6f}",
        f"- New TW return：{evals['htw_new']['mean_return']:.6f}",
        "",
        "## 6. Queue 结果",
        "",
        f"- State-only residual queue：{evals['state_only']['mean_residual_queue_bits']:.3f}",
        f"- Old TW residual queue：{evals['tw_old']['mean_residual_queue_bits']:.3f}",
        f"- New TW residual queue：{evals['htw_new']['mean_residual_queue_bits']:.3f}",
        "",
        "## 7. Action relevance",
        "",
        f"- Old TW mean rho：{rho_old}",
        f"- New TW mean rho：{rho_new}",
        f"- rho_new > rho_old：{str(new_better_rho).lower()}",
        "",
        "Fairness / energy：",
        "",
        f"- Jain fairness New/Old/State：{evals['htw_new']['mean_jain_fairness']:.4f} / {evals['tw_old']['mean_jain_fairness']:.4f} / {evals['state_only']['mean_jain_fairness']:.4f}",
        f"- Energy efficiency New/Old/State：{evals['htw_new']['mean_energy_efficiency']:.4f} / {evals['tw_old']['mean_energy_efficiency']:.4f} / {evals['state_only']['mean_energy_efficiency']:.4f}",
        "",
        "## 8. Waypoint 有效性",
        "",
        f"- New TW mean ||W-p||（归一化坐标）：{evals['htw_new']['mean_waypoint_distance']}",
        f"- New TW 动作与 waypoint 方向夹角均值（度）：{evals['htw_new']['mean_waypoint_angle_deg']}",
        "",
        "## 9. 结论",
        "",
        f"**New Task-Waypoint 在本次 mini-budget 下是否被证明优于 Old Task-Waypoint：{'有限迹象，未构成正式有效性证明' if effective else '否'}。**",
        "",
        "说明：",
        "",
        *[f"- {item}" for item in reasons],
        "",
        "下一阶段若要进入 Hierarchical Controller，需要先用更大、仍非 Round 级的授权预算，或改用 planner 教师 TW_new 做上界诊断。当前不进入 HGNN。",
        "",
        f"原始 JSON：`{OUTPUT.as_posix()}/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    REPO_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shapes = interface_shapes()
    if shapes["state_only"] != 8 or shapes["tw_old"] != 19 or shapes["htw_new"] != 16:
        raise RuntimeError(f"Stage 0 interface mismatch: {shapes}")
    train, evaluation = train_slice(), eval_slice()
    predictor, warmup = warmup_new_predictor(train)
    specs = build_specs(predictor)
    trained = {}
    for key in ("state_only", "tw_old", "htw_new"):
        trained[key] = train_baseline(specs[key], train, MINI_UPDATES)
    evaluations = {}
    for key in ("state_only", "tw_old", "htw_new"):
        evaluations[key] = evaluate_baseline(specs[key], trained[key]["agent"], evaluation)
        trained[key] = {name: value for name, value in trained[key].items() if name != "agent"}
    payload = {
        "seed": 0,
        "stage0_interfaces": True,
        "interfaces": shapes,
        "predictor_warmup": {"loss_curve": warmup, "loss_last": warmup[-1]},
        "training": trained,
        "evaluation": evaluations,
        "fresh_access": 0,
        "untouched_access": 0,
        "protected_jsonl_access": 0,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(_plain(main()), indent=2, sort_keys=True)[:2000])
