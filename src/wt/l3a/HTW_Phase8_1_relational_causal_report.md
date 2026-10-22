# HTW Phase-8.1 Relational Causal Validation Report

独立 **Level-3-A** 关系因果床：**M=6，K=12，hold=15**。只加入 **UAV–UAV 竞争**。
未训练、未改 HGNN、未改低层 PPO、未加 SINR、未加协同服务、未加能量、未开 Round、未改 L2-A。

运动固定为飞向当前 option 目标（因果运动脚本）。低层 PPO 权重未加载、未更新。

## 1. Level-3 环境定义

代码：`src/wt/l3a/`。与锁定 L2-A / Round-19/30/35 隔离。

- 规模：M=6，K=12，T=48，option hold=15。
- 无冲突合法部分单射数 **1442173**。
- 几何：3 个紧密热门 buoy（x≈28–32 m，初始 Q=4），3 个近距冷 buoy（初始 Q=1），6 个远簇 buoy（t=4…36 偶数步到达）。
- 6 架 UAV 从热门簇附近出发（y 有种子抖动）。簇间约 100 m，步长 22.0 m。
- 服务：D_k=3；关联且进入半径 16 m 才累积 ρ；切换则 ρ←0。
- **未启用** 各向同性 1/n 干扰、SINR、协同、能量。

HRL 骨架：高层每 15 步给出离散 buoy option，低层因果执行为飞向该 buoy。本阶段 option 由脚本给出，不是神经网络。

## 2. 新增关系变量

相对 L2-A 的关键变化：请求关联 **不再** `project_exclusive`（更近者留下）。

约束（合法服务）：

`sum_m x_mk <= 1` 对每个 buoy k。

若多个 UAV 同时选择同一 buoy：

- `assignment` 保留冲突请求（竞争边可见）；
- 这些 UAV **本步服务进度为 0**（冲突全灭，不是留下最近机）。

| 变量 | 含义 |
| --- | --- |
| `assignment` | 请求的 UAV→buoy，允许重叠 |
| `conflict_uav_mask` | 是否处于竞争冲突 |
| `conflict_rate` | 冲突 UAV 比例 |
| `competition_pairs` | (i,j,k) 共争 buoy k |
| `hot_buoy_mask` | 高价值竞争 buoy |
| `exclusive_ok` | 当前请求是否满足排他 |

观测仍不含 `remaining_arrivals()`；该函数只给 Oracle 影子搜索。

## 3. 为什么 L2-A 不足

Phase-7：MLP payload 1.575e7 > DeepSets 1.25e7 > HGNN 4.50e6。

L2-A 用 `project_exclusive` 把第二近的机改成 idle，策略看不到竞争损失；独立 greedy 配上环境消歧后接近 matching 分类。
L3-A 让重叠选择直接损失 payload，关系决策进入收益。

## 4. 验证实验（非学习）

1. **Myopic**：每机独立 `argmax_k Q_k/(dist+1)`，允许碰撞；option hold=15。
2. **Oracle option**：t=0 影子搜索簇间机数 n_far，簇内排他最近队列；hold=15。
3. **Assignment 对比**：同初态，全员 hold 最热 buoy vs hold 排他 Oracle 初值。
4. **成对关系**：仅 UAV0+UAV1 同选热门 vs 仅 UAV0 选热门（UAV1 idle）。

诊断项 Exclusive myopic：顺序排他 matching（L2 风格），用于判断「只要禁止碰撞是否足够」。

| 方法 | mean payload bits | mean conflict rate | mean dropped |
| --- | ---: | ---: | ---: |
| Myopic independent（关系盲） | 2625000 | 0.913 | 78.0 |
| Exclusive myopic（诊断） | 43375000 | 0.000 | — |
| Oracle option | 47625000 | 0.000 | 50.4 |
| 差（Oracle − Myopic） | 45000000 | — | -27.6 |

**Oracle > Myopic：成立**

- 同状态全员挤热门 vs 排他 hold：平均 payload 差（排他 − 全挤）= 53875000 bits。
  不同关系决策改变 payload：**是**。
- 两机同抢热门 vs 一机独占：平均 payload 差（独占 − 同抢）= 4000000 bits。
  非加性竞争：**是**。
- Myopic 与 Oracle 逐步 option 一致率 = 0.123。

种子 0–7。未训练任何模型。

## 5. 成功标准与下一步

- 关系决策空间：已出现（Oracle > Myopic）。
- 本阶段 **不训练** MLP / DeepSets / HGNN。
- 若本报告 Oracle > Myopic，下一阶段才在 **同一 L3-A 环境、冻结低层、hold=15** 下比较三个高层编码器。
- Exclusive myopic payload = 43375000，相对 Oracle 差 4250000。
  独立 Myopic 的失败主要来自碰撞全灭；一旦改成排他 matching，收益已接近 Oracle。
  因此：Level-3-A **已经产生关系决策空间**（独立选择会互相伤害），但 **HGNN 仍未证明必要**——简单排他投影可能再次让 MLP/DeepSets 足够。
  比较三个编码器时，应让策略输出可冲突的 z，由环境执行冲突全灭，而不是在环境里默默消歧。

数值：`src/wt/l3a/outputs/causal/summary.json`
