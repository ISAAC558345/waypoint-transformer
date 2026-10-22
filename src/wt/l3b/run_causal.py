"""Non-learning Level-3-B SINR probes. No PPO, no HGNN, no training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import L3BConfig
from .env import L3BEnv, legal_assignment_count
from .policies import exclusive_matching, myopic_assignment, oracle_option_assignment, rollout

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "causal"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase8_2_SINR_relational_validation_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase8_2_SINR_relational_validation_report.md"
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


def compare_policies() -> dict[str, Any]:
    myopic_rows = [rollout(L3BEnv(), myopic_assignment, seed, hold=HOLD) for seed in SEEDS]
    exclusive_rows = [rollout(L3BEnv(), exclusive_matching, seed, hold=HOLD) for seed in SEEDS]
    oracle_rows = [rollout(L3BEnv(), oracle_option_assignment, seed, hold=HOLD) for seed in SEEDS]
    myopic_pay = np.array([row["payload_bits"] for row in myopic_rows], dtype=np.float64)
    exclusive_pay = np.array([row["payload_bits"] for row in exclusive_rows], dtype=np.float64)
    oracle_pay = np.array([row["payload_bits"] for row in oracle_rows], dtype=np.float64)
    exclusive_rate = np.array([row["mean_rate_bps"] for row in exclusive_rows], dtype=np.float64)
    oracle_rate = np.array([row["mean_rate_bps"] for row in oracle_rows], dtype=np.float64)
    gap = float(np.mean(oracle_pay) - np.mean(exclusive_pay))
    rel = gap / max(float(np.mean(exclusive_pay)), 1.0)
    return {
        "seeds": list(SEEDS),
        "hold": HOLD,
        "myopic": myopic_rows,
        "exclusive_matching": exclusive_rows,
        "oracle_option": oracle_rows,
        "mean_myopic_payload": float(np.mean(myopic_pay)),
        "mean_exclusive_payload": float(np.mean(exclusive_pay)),
        "mean_oracle_payload": float(np.mean(oracle_pay)),
        "oracle_minus_myopic": float(np.mean(oracle_pay) - np.mean(myopic_pay)),
        "oracle_minus_exclusive": gap,
        "oracle_vs_exclusive_rel": float(rel),
        "oracle_beats_exclusive": bool(gap > 5.0e6 or rel > 0.20),
        "oracle_beats_myopic": bool(float(np.mean(oracle_pay)) > float(np.mean(myopic_pay)) + 1.0),
        "mean_myopic_conflict": float(np.mean([row["mean_conflict_rate"] for row in myopic_rows])),
        "mean_exclusive_conflict": float(np.mean([row["mean_conflict_rate"] for row in exclusive_rows])),
        "mean_oracle_conflict": float(np.mean([row["mean_conflict_rate"] for row in oracle_rows])),
        "mean_myopic_drop": float(np.mean([row["dropped_blocks"] for row in myopic_rows])),
        "mean_exclusive_drop": float(np.mean([row["dropped_blocks"] for row in exclusive_rows])),
        "mean_oracle_drop": float(np.mean([row["dropped_blocks"] for row in oracle_rows])),
        "mean_exclusive_rate": float(np.mean(exclusive_rate)),
        "mean_oracle_rate": float(np.mean(oracle_rate)),
        "mean_oracle_n_near": float(np.mean([row["oracle_n_near"] for row in oracle_rows])),
        "mean_oracle_n_far": float(np.mean([row["oracle_n_far"] for row in oracle_rows])),
        "mean_exclusive_active": float(np.mean([row["mean_active_frac"] for row in exclusive_rows])),
        "mean_oracle_active": float(np.mean([row["mean_active_frac"] for row in oracle_rows])),
        "mean_myopic_oracle_agreement": float(np.mean([row["oracle_agreement"] for row in myopic_rows])),
        "legal_conflict_free_assignments": legal_assignment_count(6, 12),
    }


def write_report(payload: Mapping[str, Any]) -> None:
    cfg = L3BConfig()
    ok = payload["oracle_beats_exclusive"]
    lines = [
        "# HTW Phase-8.2 SINR Relational Validation Report",
        "",
        "独立 **Level-3-B**：M=6，K=12，hold=15。在锁定的 L3-A 竞争机制上 **只加 SINR 耦合与动态关系图**。",
        "未训练、未改 HGNN、未改低层 PPO、未加 cooperative service、未开 Round、未改 `src/wt/l3a/`。",
        "",
        "运动：飞向当前 option。低层 PPO 未加载、未更新。",
        "",
        "## 1. 环境定义",
        "",
        "代码：`src/wt/l3b/`。",
        "",
        f"- 规模：M={cfg.num_uavs}，K={cfg.num_buoys}，T={cfg.episode_horizon}，hold={payload['hold']}。",
        "- 近簇 / 远簇各 6 个 buoy，簇内间距约 2–6 m，簇间约 100 m。",
        "- 仍保留 UAV–UAV 竞争：`sum_m x_mk > 1` 则冲突机本步不发射、不服务。",
        "- **未加入** 协同服务、能量。",
        "",
        "速率：",
        "",
        "`R_mk = B log2(1 + SINR_mk)`，",
        "`SINR_mk = G_mk P / (σ² + Σ_{j≠m} G_jk P κ_{ℓk})`。",
        "",
        f"B={cfg.bandwidth_hz:.1e} Hz，P={cfg.tx_power}，σ²={cfg.noise_power}，α={cfg.path_loss_exp}，",
        f"同簇 κ={cfg.kappa_same_cluster}，跨簇 κ={cfg.kappa_cross_cluster}。",
        "",
        "服务进度 `Δρ = min(1, (R / R_ref) / D)`，`R_ref` 为 `min_link` 处的孤立链路速率。同簇密铺会显著压低 SINR。",
        "Oracle 允许 **idle**（不发射），用来减干扰。",
        "",
        "## 2. 动态关系图",
        "",
        "每步重算，写入 `info()`：",
        "",
        "| 边 / 特征 | 含义 |",
        "| --- | --- |",
        "| `distance_uav_buoy` | UAV–Buoy 距离 |",
        "| `distance_uav_uav` | UAV–UAV 距离 |",
        "| `rate_bps` | 当前并发 SINR 下的速率 |",
        "| `rate_if_assigned` | 固定其他链路时，自己改选 k 的假设速率 |",
        "| `interference_uav_uav` | 当前链路上 j 对 i 接收 buoy 的干扰功率 |",
        "| `queue_pressure` | Q_k / Q_max |",
        "| `competition_pairs` | 同 buoy 冲突边 |",
        "",
        "观测不含未来到达；`remaining_arrivals()` 只给 Oracle 影子。",
        "",
        "## 3. 为何 Phase-8.1 不够",
        "",
        "L3-A：Myopic 2.625e6（conflict 0.913），Exclusive myopic 4.338e7，Oracle 4.763e7。",
        "排他规则已接近 Oracle，问题仍可被排序 matching 解掉。",
        "L3-B 让「合法排他」仍可能很差：同簇塞满会互相压速率。",
        "",
        "## 4. 非学习结果",
        "",
        "1. Myopic：独立 Q/距离，允许碰撞。",
        "2. Exclusive matching：顺序排他 Q/距离，**忽略 SINR 与 idle**。",
        "3. Oracle：t=0 影子搜索 `(n_near, n_far)`，其余 idle；簇内再排他。",
        "",
        "| 方法 | mean payload | conflict | mean rate (bps) | dropped | active frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Myopic | {payload['mean_myopic_payload']:.0f} | {payload['mean_myopic_conflict']:.3f} | — | {payload['mean_myopic_drop']:.1f} | — |",
        f"| Exclusive matching | {payload['mean_exclusive_payload']:.0f} | {payload['mean_exclusive_conflict']:.3f} | {payload['mean_exclusive_rate']:.3e} | {payload['mean_exclusive_drop']:.1f} | {payload['mean_exclusive_active']:.3f} |",
        f"| Oracle option | {payload['mean_oracle_payload']:.0f} | {payload['mean_oracle_conflict']:.3f} | {payload['mean_oracle_rate']:.3e} | {payload['mean_oracle_drop']:.1f} | {payload['mean_oracle_active']:.3f} |",
        f"| Oracle − Exclusive | {payload['oracle_minus_exclusive']:.0f} | — | — | — | — |",
        "",
        f"**Oracle 明显高于 Exclusive matching：{'成立' if ok else '不成立'}**",
        f"（相对差 {payload['oracle_vs_exclusive_rel']:.2%}；判据：绝对差 > 5e6 或相对差 > 20%）",
        "",
        f"- Oracle 平均选用 n_near={payload['mean_oracle_n_near']:.2f}，n_far={payload['mean_oracle_n_far']:.2f}（其余 idle）。",
        f"- Myopic 与 Oracle option 一致率 = {payload['mean_myopic_oracle_agreement']:.3f}。",
        "",
        "种子 0–7。未训练任何神经网络。",
        "",
        "## 5. 成功标准与下一步",
        "",
        f"- 关系推理空间（相对简单规则）：{'已增强' if ok else '仍不足，排他 matching 仍接近 Oracle'}。",
        "- 本阶段不训练 MLP / DeepSets / HGNN。",
        "- 若成立，下一阶段才在 L3-B 上比较三个高层编码器；策略应输出可冲突 z，环境用冲突全灭 + SINR，不要再做「更近者留下」。",
        "- 仍未加入 cooperative service。",
        "",
        "数值：`src/wt/l3b/outputs/causal/summary.json`",
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
        "mean_exclusive_payload",
        "mean_oracle_payload",
        "oracle_minus_exclusive",
        "oracle_vs_exclusive_rel",
        "oracle_beats_exclusive",
        "mean_oracle_n_near",
        "mean_oracle_n_far",
        "mean_exclusive_rate",
        "mean_oracle_rate",
    )
    print(json.dumps({key: result[key] for key in keys}, indent=2))
