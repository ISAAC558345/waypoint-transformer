"""Non-learning causal probes for Level-2-A. No PPO, no HGNN, no Round."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import L2AConfig
from .env import L2AEnv, legal_assignment_count
from .policies import fly_toward, hold_fixed, myopic_assignment, oracle_option_assignment, rollout

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "causal"
REPORT = ROOT / "docs" / "archive" / "phase_reports" / "level2_causal_validation_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "level2_causal_validation_report.md"
SEEDS = tuple(range(8))


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


def association_flip_delta(seed: int) -> dict[str, float]:
    """Same start; hold two different exclusive assignments for the whole episode."""
    env = L2AEnv(L2AConfig(seed=seed))
    env.reset(seed=seed)
    base = oracle_option_assignment(env)
    alt = base.copy()
    if int(alt[0]) >= 0:
        alt[0] = 6 if int(alt[0]) < 6 else 0
    alt = alt.copy()
    from .env import project_exclusive

    alt = project_exclusive(alt, env.uav_positions, env.buoy_positions)
    left = rollout(L2AEnv(L2AConfig(seed=seed)), hold_fixed(base), seed)
    right = rollout(L2AEnv(L2AConfig(seed=seed)), hold_fixed(alt), seed)
    return {
        "payload_hold_oracle_init": left["payload_bits"],
        "payload_hold_flipped": right["payload_bits"],
        "delta_payload": float(right["payload_bits"] - left["payload_bits"]),
        "assignments_differ": float(not np.array_equal(base, alt)),
    }


def compare_policies() -> dict[str, Any]:
    myopic_rows = [rollout(L2AEnv(), myopic_assignment, seed) for seed in SEEDS]
    oracle_rows = [rollout(L2AEnv(), oracle_option_assignment, seed) for seed in SEEDS]
    flips = [association_flip_delta(seed) for seed in SEEDS]
    myopic_pay = np.array([row["payload_bits"] for row in myopic_rows], dtype=np.float64)
    oracle_pay = np.array([row["payload_bits"] for row in oracle_rows], dtype=np.float64)
    agree = np.array([row["oracle_agreement"] for row in myopic_rows], dtype=np.float64)
    myopic_drop = np.array([row["dropped_blocks"] for row in myopic_rows], dtype=np.float64)
    oracle_drop = np.array([row["dropped_blocks"] for row in oracle_rows], dtype=np.float64)
    abs_flip = np.array([abs(row["delta_payload"]) for row in flips], dtype=np.float64)
    return {
        "seeds": list(SEEDS),
        "myopic": myopic_rows,
        "oracle_option": oracle_rows,
        "flips": flips,
        "mean_myopic_payload": float(np.mean(myopic_pay)),
        "mean_oracle_payload": float(np.mean(oracle_pay)),
        "oracle_minus_myopic": float(np.mean(oracle_pay) - np.mean(myopic_pay)),
        "oracle_beats_myopic": bool(float(np.mean(oracle_pay)) > float(np.mean(myopic_pay)) + 1.0),
        "mean_myopic_drop": float(np.mean(myopic_drop)),
        "mean_oracle_drop": float(np.mean(oracle_drop)),
        "mean_myopic_oracle_agreement": float(np.mean(agree)),
        "assignment_not_solved": bool(float(np.mean(agree)) < 0.80),
        "mean_abs_flip_payload": float(np.mean(abs_flip)),
        "flip_changes_payload": bool(float(np.mean(abs_flip)) > 1.0),
        "legal_assignments": legal_assignment_count(5, 12),
    }


def write_report(payload: Mapping[str, Any]) -> None:
    cfg = L2AConfig()
    option_ok = payload["oracle_beats_myopic"]
    lines = [
        "# Level-2-A Causal Validation Report",
        "",
        "独立 L2-A 物理验证床：**M=5，K=12**。未训练、未改 PPO 核心、未加 HGNN、未训练 Task-Waypoint、未开 Round、未改锁定 Level-1 / Round-19/30/35 环境。",
        "",
        "本阶段目标不是提高学习性能，而是证明该 benchmark **开始需要** 长期任务规划 / HRL（以及后续 HTW/HGNN 的结构空间）。",
        "",
        "「State-only」在此定义为非学习代理：**当前队列+距离上的最近正向队列 matching**，再沿该关联飞向 buoy。它对应 L1 中「状态已足够解 assignment」的机制，而不是 PPO 训练曲线。",
        "",
        "## 1. Level-2 环境定义",
        "",
        "代码：`src/wt/l2a/`。与锁定海事 L1 隔离。",
        "",
        f"- 规模：M={cfg.num_uavs}，K={cfg.num_buoys}，T={cfg.episode_horizon}。合法部分单射关联数 **{payload['legal_assignments']}**（L1 的 M=2,K=5 仅为 31）。",
        "- 几何：近簇 6 buoy（x≈18–42 m），远簇 6 buoy（x≈118–142 m）；UAV 从近簇出发。",
        f"- 运动：v_max Δt = {cfg.step_length_m:.1f} m，簇间约 100 m ⇒ **飞行 ≥4 slot**（空间延迟）。",
        f"- 服务：D_k={cfg.service_slots}；仅关联且进入服务半径 {cfg.service_radius_m:.0f} m 才累积进度 ρ；**切换则 ρ←0**。",
        f"- 队列：Q_k(t+1)=clip(max(Q_k(t)-S_k(t),0)+A_k(t), 0, {cfg.queue_capacity_blocks})。近簇仅有初始 Q=3；远簇在 t=4…36 的偶数步 A_k=1。溢出计入 dropped。",
        "- 完成一块（ρ=1）发放 1e6 bits payload。附近并发服务有干扰缩放 1/n。",
        "- 关联：控制器给出 x_mk 请求，环境投影为排他匹配（每 buoy 至多一机，每机至多一 buoy）。",
        "",
        "## 2. 新增状态变量",
        "",
        "写入 `info()`、**不含未来到达轨迹**：",
        "",
        "| 变量 | 形状 | 含义 |",
        "| --- | --- | --- |",
        "| `uav_positions_m` | (5,2) | UAV 位置 |",
        "| `buoy_positions_m` | (12,2) | buoy 位置 |",
        "| `queue_blocks` | (12,) | Q_k(t) |",
        "| `service_progress` | (5,) | 当前占用进度 ρ |",
        "| `assignment` | (5,) | 当前 x 的 UAV→buoy，idle=-1 |",
        "| `eta_slots` | (5,12) | 以当前步长计的飞行时间 |",
        "| `cumulative_payload_bits` | scalar | 累计 payload |",
        "",
        "`remaining_arrivals()` 只给 Oracle 脚本，不进入观测。",
        "",
        "## 3. 新增动作变量",
        "",
        "| 动作 | 形状 | 含义 |",
        "| --- | --- | --- |",
        "| movement | (5,2) | 单位方向位移，范数截断到 1 |",
        "| association | (5,) | 请求的 buoy index 或 idle；经 `project_exclusive` |",
        "",
        "因果实验中运动固定为「飞向当前关联目标」，只对照 **关联/option**。",
        "",
        "## 4. 长期耦合证明",
        "",
        f"- 同一初态、整段 hold 两种关联：平均 \|Δpayload\| = {payload['mean_abs_flip_payload']:.0f} bits。不同 buoy 选择 **改变未来 payload：{'是' if payload['flip_changes_payload'] else '否'}**。",
        f"- Myopic 与 Oracle option 的逐步 assignment 一致率 = {payload['mean_myopic_oracle_agreement']:.3f}（L1 上 PPO 曾达到约 0.95 的 option 完成率）。",
        f"- 当前贪心是否仍「接近 100% 解 assignment」：**{'否' if payload['assignment_not_solved'] else '仍过高'}**。",
        "- 机制来源：远簇到达 + 多 slot 飞行 + D≥3 占用 + 排他 x_mk。贪心先清近簇，远簇队列溢出。",
        "",
        "## 5. Oracle Option vs State-only",
        "",
        "Oracle：在 t=0 对簇间机数 n_far∈{0..5} 做 **影子全轨迹** 搜索，然后 **hold 该簇 option**，簇内再最近队列排他。影子可用已知 A_k 过程；观测向量仍无未来 A。",
        "",
        "| 方法 | mean payload bits | mean dropped blocks |",
        "| --- | ---: | ---: |",
        f"| Myopic nearest-queue（State-only 代理） | {payload['mean_myopic_payload']:.0f} | {payload['mean_myopic_drop']:.1f} |",
        f"| Oracle cluster option | {payload['mean_oracle_payload']:.0f} | {payload['mean_oracle_drop']:.1f} |",
        f"| 差（Oracle − Myopic） | {payload['oracle_minus_myopic']:.0f} | {payload['mean_oracle_drop'] - payload['mean_myopic_drop']:.1f} |",
        "",
        f"**Oracle Option > State-only：{'成立' if option_ok else '不成立'}**",
        "",
        "种子 0–7：近簇出发位置有小抖动。未训练 PPO。",
        "",
        "## 6. 对 HTW / HRL / HGNN 的含义",
        "",
        "- **HRL**：存在可 hold 的 buoy/簇 option，且优于当前状态贪心 ⇒ 需要高层任务分配。",
        "- **HTW**：L1 失败是因为无任务分解间隙；L2-A 已出现该间隙，后续才值得把 option 做成可学习 TW。",
        "- **HGNN**：本阶段未加入。图关联在 M=5,K=12（约 1.7e5 合法匹配）上有结构空间，但 **尚未证明必须用 HGNN**；先锁定因果床。",
        "",
        f"数值：`src/wt/l2a/outputs/causal/summary.json`",
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
    print(json.dumps({key: result[key] for key in ("mean_myopic_payload", "mean_oracle_payload", "oracle_minus_myopic", "oracle_beats_myopic", "mean_myopic_oracle_agreement", "assignment_not_solved", "mean_abs_flip_payload", "legal_assignments")}, indent=2))
