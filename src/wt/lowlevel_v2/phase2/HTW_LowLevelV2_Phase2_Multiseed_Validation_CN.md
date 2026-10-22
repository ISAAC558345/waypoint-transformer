# HTW Low-Level V2 Phase-2 Multi-Seed Independent Validation

方法完全冻结自 LL-V2.1B（ServiceResidualLock + 同 MAPPO / reward / d_safe / alpha）。seed0 **未重训**，复用 `phase1b/outputs/ckpt_update100.pt`。新独立集 `LLV2_PHASE2_VALIDATION_V1`：222000–222063，64 场景。environment SHA=`de22347ea625a330f61e8f62b2a973066f4669c7498fd58bac2fd8bf706632ba`。高层 SHA 训练前后一致。未访问 106000–106063。primary 仅 u100。

## 训练

| seed | 是否新训 | env transitions | wall-s |
|---|---|---|---|
| 0 | 否（P1B u100） | 19200（历史） | — |
| 1 | 是 | 19200 | 96.1 |
| 2 | 是 | 19200 | 102.2 |
| 3 | 是 | 19200 | 104.4 |
| 4 | 是 | 19200 | 105.6 |

曲线中 seed0 来自历史 design eval；seed1–4 来自 LLV2_train 前 16 条（仅监控崩溃，不选点）。

## Independent validation vs 共享 L0

L0 mean payload = **10.234e6**（64 场景各跑一次）。

| seed | payload | Delta_s | relative | +scen frac | blocks | proj rate | ENVELOPE among int. |
|---|---|---|---|---|---|---|---|
| 0 | 11.328e6 | +1.094e6 | +10.7% | 0.625 | 11.328 | 9.2% | 1.91% |
| 1 | 8.578e6 | **−1.656e6** | −16.2% | 0.109 | 8.578 | 4.8% | 0.68% |
| 2 | 13.359e6 | +3.125e6 | +30.5% | 0.859 | 13.359 | 59.1% | 0% |
| 3 | 10.469e6 | +0.234e6 | +2.3% | 0.453 | 10.469 | 50.5% | 0% |
| 4 | 10.328e6 | +0.094e6 | +0.9% | 0.484 | 10.328 | 58.1% | 0% |

正向 seed：**4/5**。overall mean Δ=**+0.578e6**。median of seed means=+0.234e6。

- hierarchical 95% CI：**[−0.800e6, +1.997e6]** 跨 0
- seed-cluster 95% CI：**[−0.756e6, +1.941e6]** 跨 0
- scenario-paired 辅助 CI：[+0.138e6, +1.006e6]（不单独作为主门）

P1B mean completed blocks 10.813 ≥ L0 10.234。

## 安全（每 seed 分开）

min_sep_worst 全部 ∈ [0.99999999872, 0.99999999908]。  
1e-9 计数：seed0=2，seed4=8，其余 0。  
**已冻结 1e-6 容差下 collision=0，全体 projector_failure=0，service_lock_violation=0。**  
无 SAFETY-FAIL。不得用 payload 抵消；此处也无需抵消。

## Projector / SERVICE lock

ENVELOPE 占介入：0–1.91%，跨 seed 保持低。 pairwise 主导。seed2/3/4 总体介入偏高（50–59%），但是 pairwise 微校正，不是服务区逃逸。

## 关系 / 避碰 / 通信

- neighbor shuffle **near L2 ≈ 1e-9**，五 seed 均未利用 neighbor。
- interference shuffle high L2 ≤ 3.3e-4，基本未用。
- SINR/rate shuffle high L2 0.041–0.153，有响应。
- separation_response mean ≈ 0，median=0：不能声称主动避碰。
- 高干扰下 ΔSINR/Δrate：seed2 略正，其余≈0 或微负。不是稳定通信优化。

## Gate

**LLV2-MS-B**。`stable_residual_advantage=false`。  
**REL-1**（neighbor/interference 未用；SINR 在部分状态有响应）。

下一阶段唯一建议：先分析 **seed1 不稳定**（同方法、同预算、同 train pool 下明显差于 L0）。**不要**加 attention / HGNN / interference penalty，不要加长预算，不要改失败 seed。本阶段停止，不自动开下一训练阶段。
