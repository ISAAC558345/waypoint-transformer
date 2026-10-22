"""Non-learning Level-3-A competition probes. No PPO, no HGNN, no training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import L3AConfig
from .env import L3AEnv, legal_assignment_count
from .policies import (
    exclusive_myopic_assignment,
    hold_fixed,
    myopic_assignment,
    oracle_option_assignment,
    rollout,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "causal"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase8_1_relational_causal_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase8_1_relational_causal_report.md"
SEEDS = tuple(range(8))
HOLD = 15


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def assignment_contrast(seed: int) -> dict[str, float]:
    """Same start: all UAVs pile on the hottest buoy vs exclusive oracle init."""
    env = L3AEnv(L3AConfig(seed=seed))
    env.reset(seed=seed)
    exclusive = oracle_option_assignment(env)
    piled = np.full(env.config.num_uavs, int(np.argmax(env.queue)), dtype=np.int64)
    piled_roll = rollout(L3AEnv(L3AConfig(seed=seed)), hold_fixed(piled), seed, hold=HOLD)
    excl_roll = rollout(L3AEnv(L3AConfig(seed=seed)), hold_fixed(exclusive), seed, hold=HOLD)
    return {
        "payload_all_on_hot": piled_roll["payload_bits"],
        "payload_exclusive_hold": excl_roll["payload_bits"],
        "delta_payload": float(excl_roll["payload_bits"] - piled_roll["payload_bits"]),
        "piled_conflict_rate": piled_roll["mean_conflict_rate"],
        "exclusive_conflict_rate": excl_roll["mean_conflict_rate"],
        "assignments_differ": float(not np.array_equal(piled, exclusive)),
    }


def pairwise_swap_delta(seed: int) -> dict[str, float]:
    """Same state: UAV 0+1 both on hot 0 vs only UAV 0 on hot 0 (UAV 1 idle)."""
    base_env = L3AEnv(L3AConfig(seed=seed))
    base_env.reset(seed=seed)
    z_both = np.full(base_env.config.num_uavs, -1, dtype=np.int64)
    z_one = z_both.copy()
    z_both[0] = 0
    z_both[1] = 0
    z_one[0] = 0
    both = rollout(L3AEnv(L3AConfig(seed=seed)), hold_fixed(z_both), seed, hold=HOLD)
    one = rollout(L3AEnv(L3AConfig(seed=seed)), hold_fixed(z_one), seed, hold=HOLD)
    return {
        "payload_two_on_hot": both["payload_bits"],
        "payload_one_on_hot": one["payload_bits"],
        "delta_payload": float(one["payload_bits"] - both["payload_bits"]),
        "two_conflict_rate": both["mean_conflict_rate"],
        "one_conflict_rate": one["mean_conflict_rate"],
    }


def compare_policies() -> dict[str, Any]:
    myopic_rows = [rollout(L3AEnv(), myopic_assignment, seed, hold=HOLD) for seed in SEEDS]
    exclusive_rows = [rollout(L3AEnv(), exclusive_myopic_assignment, seed, hold=HOLD) for seed in SEEDS]
    oracle_rows = [rollout(L3AEnv(), oracle_option_assignment, seed, hold=HOLD) for seed in SEEDS]
    contrasts = [assignment_contrast(seed) for seed in SEEDS]
    pairwise = [pairwise_swap_delta(seed) for seed in SEEDS]
    myopic_pay = np.array([row["payload_bits"] for row in myopic_rows], dtype=np.float64)
    exclusive_pay = np.array([row["payload_bits"] for row in exclusive_rows], dtype=np.float64)
    oracle_pay = np.array([row["payload_bits"] for row in oracle_rows], dtype=np.float64)
    myopic_conf = np.array([row["mean_conflict_rate"] for row in myopic_rows], dtype=np.float64)
    exclusive_conf = np.array([row["mean_conflict_rate"] for row in exclusive_rows], dtype=np.float64)
    oracle_conf = np.array([row["mean_conflict_rate"] for row in oracle_rows], dtype=np.float64)
    agree = np.array([row["oracle_agreement"] for row in myopic_rows], dtype=np.float64)
    contrast_delta = np.array([row["delta_payload"] for row in contrasts], dtype=np.float64)
    pair_delta = np.array([row["delta_payload"] for row in pairwise], dtype=np.float64)
    return {
        "seeds": list(SEEDS),
        "hold": HOLD,
        "myopic": myopic_rows,
        "exclusive_myopic": exclusive_rows,
        "oracle_option": oracle_rows,
        "assignment_contrasts": contrasts,
        "pairwise": pairwise,
        "mean_myopic_payload": float(np.mean(myopic_pay)),
        "mean_exclusive_myopic_payload": float(np.mean(exclusive_pay)),
        "mean_oracle_payload": float(np.mean(oracle_pay)),
        "oracle_minus_myopic": float(np.mean(oracle_pay) - np.mean(myopic_pay)),
        "oracle_beats_myopic": bool(float(np.mean(oracle_pay)) > float(np.mean(myopic_pay)) + 1.0),
        "mean_myopic_conflict": float(np.mean(myopic_conf)),
        "mean_exclusive_conflict": float(np.mean(exclusive_conf)),
        "mean_oracle_conflict": float(np.mean(oracle_conf)),
        "mean_myopic_drop": float(np.mean([row["dropped_blocks"] for row in myopic_rows])),
        "mean_oracle_drop": float(np.mean([row["dropped_blocks"] for row in oracle_rows])),
        "mean_myopic_oracle_agreement": float(np.mean(agree)),
        "mean_contrast_delta": float(np.mean(contrast_delta)),
        "contrast_changes_payload": bool(float(np.mean(np.abs(contrast_delta))) > 1.0),
        "mean_pairwise_delta": float(np.mean(pair_delta)),
        "pairwise_nonadditive": bool(float(np.mean(pair_delta)) > 1.0),
        "legal_conflict_free_assignments": legal_assignment_count(6, 12),
    }


def write_report(payload: Mapping[str, Any]) -> None:
    cfg = L3AConfig()
    option_ok = payload["oracle_beats_myopic"]
    lines = [
        "# HTW Phase-8.1 Relational Causal Validation Report",
        "",
        "独立 **Level-3-A** 关系因果床：**M=6，K=12，hold=15**。只加入 **UAV–UAV 竞争**。",
        "未训练、未改 HGNN、未改低层 PPO、未加 SINR、未加协同服务、未加能量、未开 Round、未改 L2-A。",
        "",
        "运动固定为飞向当前 option 目标（因果运动脚本）。低层 PPO 权重未加载、未更新。",
        "",
        "## 1. Level-3 环境定义",
        "",
        "代码：`src/wt/l3a/`。与锁定 L2-A / Round-19/30/35 隔离。",
        "",
        f"- 规模：M={cfg.num_uavs}，K={cfg.num_buoys}，T={cfg.episode_horizon}，option hold={payload['hold']}。",
        f"- 无冲突合法部分单射数 **{payload['legal_conflict_free_assignments']}**。",
        f"- 几何：3 个紧密热门 buoy（x≈28–32 m，初始 Q=4），3 个近距冷 buoy（初始 Q=1），6 个远簇 buoy（t=4…36 偶数步到达）。",
        f"- 6 架 UAV 从热门簇附近出发（y 有种子抖动）。簇间约 100 m，步长 {cfg.step_length_m:.1f} m。",
        f"- 服务：D_k={cfg.service_slots}；关联且进入半径 {cfg.service_radius_m:.0f} m 才累积 ρ；切换则 ρ←0。",
        "- **未启用** 各向同性 1/n 干扰、SINR、协同、能量。",
        "",
        "HRL 骨架：高层每 15 步给出离散 buoy option，低层因果执行为飞向该 buoy。本阶段 option 由脚本给出，不是神经网络。",
        "",
        "## 2. 新增关系变量",
        "",
        "相对 L2-A 的关键变化：请求关联 **不再** `project_exclusive`（更近者留下）。",
        "",
        "约束（合法服务）：",
        "",
        "`sum_m x_mk <= 1` 对每个 buoy k。",
        "",
        "若多个 UAV 同时选择同一 buoy：",
        "",
        "- `assignment` 保留冲突请求（竞争边可见）；",
        "- 这些 UAV **本步服务进度为 0**（冲突全灭，不是留下最近机）。",
        "",
        "| 变量 | 含义 |",
        "| --- | --- |",
        "| `assignment` | 请求的 UAV→buoy，允许重叠 |",
        "| `conflict_uav_mask` | 是否处于竞争冲突 |",
        "| `conflict_rate` | 冲突 UAV 比例 |",
        "| `competition_pairs` | (i,j,k) 共争 buoy k |",
        "| `hot_buoy_mask` | 高价值竞争 buoy |",
        "| `exclusive_ok` | 当前请求是否满足排他 |",
        "",
        "观测仍不含 `remaining_arrivals()`；该函数只给 Oracle 影子搜索。",
        "",
        "## 3. 为什么 L2-A 不足",
        "",
        "Phase-7：MLP payload 1.575e7 > DeepSets 1.25e7 > HGNN 4.50e6。",
        "",
        "L2-A 用 `project_exclusive` 把第二近的机改成 idle，策略看不到竞争损失；独立 greedy 配上环境消歧后接近 matching 分类。",
        "L3-A 让重叠选择直接损失 payload，关系决策进入收益。",
        "",
        "## 4. 验证实验（非学习）",
        "",
        "1. **Myopic**：每机独立 `argmax_k Q_k/(dist+1)`，允许碰撞；option hold=15。",
        "2. **Oracle option**：t=0 影子搜索簇间机数 n_far，簇内排他最近队列；hold=15。",
        "3. **Assignment 对比**：同初态，全员 hold 最热 buoy vs hold 排他 Oracle 初值。",
        "4. **成对关系**：仅 UAV0+UAV1 同选热门 vs 仅 UAV0 选热门（UAV1 idle）。",
        "",
        "诊断项 Exclusive myopic：顺序排他 matching（L2 风格），用于判断「只要禁止碰撞是否足够」。",
        "",
        "| 方法 | mean payload bits | mean conflict rate | mean dropped |",
        "| --- | ---: | ---: | ---: |",
        f"| Myopic independent（关系盲） | {payload['mean_myopic_payload']:.0f} | {payload['mean_myopic_conflict']:.3f} | {payload['mean_myopic_drop']:.1f} |",
        f"| Exclusive myopic（诊断） | {payload['mean_exclusive_myopic_payload']:.0f} | {payload['mean_exclusive_conflict']:.3f} | — |",
        f"| Oracle option | {payload['mean_oracle_payload']:.0f} | {payload['mean_oracle_conflict']:.3f} | {payload['mean_oracle_drop']:.1f} |",
        f"| 差（Oracle − Myopic） | {payload['oracle_minus_myopic']:.0f} | — | {payload['mean_oracle_drop'] - payload['mean_myopic_drop']:.1f} |",
        "",
        f"**Oracle > Myopic：{'成立' if option_ok else '不成立'}**",
        "",
        f"- 同状态全员挤热门 vs 排他 hold：平均 payload 差（排他 − 全挤）= {payload['mean_contrast_delta']:.0f} bits。",
        f"  不同关系决策改变 payload：**{'是' if payload['contrast_changes_payload'] else '否'}**。",
        f"- 两机同抢热门 vs 一机独占：平均 payload 差（独占 − 同抢）= {payload['mean_pairwise_delta']:.0f} bits。",
        f"  非加性竞争：**{'是' if payload['pairwise_nonadditive'] else '否'}**。",
        f"- Myopic 与 Oracle 逐步 option 一致率 = {payload['mean_myopic_oracle_agreement']:.3f}。",
        "",
        "种子 0–7。未训练任何模型。",
        "",
        "## 5. 成功标准与下一步",
        "",
        f"- 关系决策空间：{'已出现（Oracle > Myopic）' if option_ok else '未出现，需加强竞争几何'}。",
        "- 本阶段 **不训练** MLP / DeepSets / HGNN。",
        "- 若本报告 Oracle > Myopic，下一阶段才在 **同一 L3-A 环境、冻结低层、hold=15** 下比较三个高层编码器。",
        f"- Exclusive myopic payload = {payload['mean_exclusive_myopic_payload']:.0f}，相对 Oracle 差 {payload['mean_oracle_payload'] - payload['mean_exclusive_myopic_payload']:.0f}。",
        "  独立 Myopic 的失败主要来自碰撞全灭；一旦改成排他 matching，收益已接近 Oracle。",
        "  因此：Level-3-A **已经产生关系决策空间**（独立选择会互相伤害），但 **HGNN 仍未证明必要**——简单排他投影可能再次让 MLP/DeepSets 足够。",
        "  比较三个编码器时，应让策略输出可冲突的 z，由环境执行冲突全灭，而不是在环境里默默消歧。",
        "",
        "数值：`src/wt/l3a/outputs/causal/summary.json`",
        "",
    ]
    text = "\n".join(lines)
    REPORT.write_text(text, encoding="utf-8")
    PACK_REPORT.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = compare_policies()
    (OUTPUT / "summary.json").write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    keys = (
        "mean_myopic_payload",
        "mean_exclusive_myopic_payload",
        "mean_oracle_payload",
        "oracle_minus_myopic",
        "oracle_beats_myopic",
        "mean_myopic_conflict",
        "mean_contrast_delta",
        "mean_pairwise_delta",
        "pairwise_nonadditive",
    )
    print(json.dumps({key: result[key] for key in keys}, indent=2))
