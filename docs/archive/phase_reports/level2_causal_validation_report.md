# Level-2-A Causal Validation Report

独立 L2-A 物理验证床：**M=5，K=12**。未训练、未改 PPO 核心、未加 HGNN、未训练 Task-Waypoint、未开 Round、未改锁定 Level-1 / Round-19/30/35 环境。

本阶段目标不是提高学习性能，而是证明该 benchmark **开始需要** 长期任务规划 / HRL（以及后续 HTW/HGNN 的结构空间）。

「State-only」在此定义为非学习代理：**当前队列+距离上的最近正向队列 matching**，再沿该关联飞向 buoy。它对应 L1 中「状态已足够解 assignment」的机制，而不是 PPO 训练曲线。

## 1. Level-2 环境定义

代码：`src/wt/l2a/`。与锁定海事 L1 隔离。

- 规模：M=5，K=12，T=48。合法部分单射关联数 **169021**（L1 的 M=2,K=5 仅为 31）。
- 几何：近簇 6 buoy（x≈18–42 m），远簇 6 buoy（x≈118–142 m）；UAV 从近簇出发。
- 运动：v_max Δt = 22.0 m，簇间约 100 m ⇒ **飞行 ≥4 slot**（空间延迟）。
- 服务：D_k=3；仅关联且进入服务半径 16 m 才累积进度 ρ；**切换则 ρ←0**。
- 队列：Q_k(t+1)=clip(max(Q_k(t)-S_k(t),0)+A_k(t), 0, 4)。近簇仅有初始 Q=3；远簇在 t=4…36 的偶数步 A_k=1。溢出计入 dropped。
- 完成一块（ρ=1）发放 1e6 bits payload。附近并发服务有干扰缩放 1/n。
- 关联：控制器给出 x_mk 请求，环境投影为排他匹配（每 buoy 至多一机，每机至多一 buoy）。

## 2. 新增状态变量

写入 `info()`、**不含未来到达轨迹**：

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `uav_positions_m` | (5,2) | UAV 位置 |
| `buoy_positions_m` | (12,2) | buoy 位置 |
| `queue_blocks` | (12,) | Q_k(t) |
| `service_progress` | (5,) | 当前占用进度 ρ |
| `assignment` | (5,) | 当前 x 的 UAV→buoy，idle=-1 |
| `eta_slots` | (5,12) | 以当前步长计的飞行时间 |
| `cumulative_payload_bits` | scalar | 累计 payload |

`remaining_arrivals()` 只给 Oracle 脚本，不进入观测。

## 3. 新增动作变量

| 动作 | 形状 | 含义 |
| --- | --- | --- |
| movement | (5,2) | 单位方向位移，范数截断到 1 |
| association | (5,) | 请求的 buoy index 或 idle；经 `project_exclusive` |

因果实验中运动固定为「飞向当前关联目标」，只对照 **关联/option**。

## 4. 长期耦合证明

- 同一初态、整段 hold 两种关联：平均 \|Δpayload\| = 1875000 bits。不同 buoy 选择 **改变未来 payload：是**。
- Myopic 与 Oracle option 的逐步 assignment 一致率 = 0.648（L1 上 PPO 曾达到约 0.95 的 option 完成率）。
- 当前贪心是否仍「接近 100% 解 assignment」：**否**。
- 机制来源：远簇到达 + 多 slot 飞行 + D≥3 占用 + 排他 x_mk。贪心先清近簇，远簇队列溢出。

## 5. Oracle Option vs State-only

Oracle：在 t=0 对簇间机数 n_far∈{0..5} 做 **影子全轨迹** 搜索，然后 **hold 该簇 option**，簇内再最近队列排他。影子可用已知 A_k 过程；观测向量仍无未来 A。

| 方法 | mean payload bits | mean dropped blocks |
| --- | ---: | ---: |
| Myopic nearest-queue（State-only 代理） | 17000000 | 77.6 |
| Oracle cluster option | 28750000 | 67.5 |
| 差（Oracle − Myopic） | 11750000 | -10.1 |

**Oracle Option > State-only：成立**

种子 0–7：近簇出发位置有小抖动。未训练 PPO。

## 6. 对 HTW / HRL / HGNN 的含义

- **HRL**：存在可 hold 的 buoy/簇 option，且优于当前状态贪心 ⇒ 需要高层任务分配。
- **HTW**：L1 失败是因为无任务分解间隙；L2-A 已出现该间隙，后续才值得把 option 做成可学习 TW。
- **HGNN**：本阶段未加入。图关联在 M=5,K=12（约 1.7e5 合法匹配）上有结构空间，但 **尚未证明必须用 HGNN**；先锁定因果床。

数值：`src/wt/l2a/outputs/causal/summary.json`
