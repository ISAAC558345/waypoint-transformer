# HTW Phase-9.4.1 Graph Value Stability and Independent Confirmation Audit

本阶段不修改模型。Phase-9.4 Masked Autoregressive Decoder、HGNN/DeepSets 编码器、loss、optimizer、epochs、L3-B、SINR、hold=15、低层 fly-toward 全部冻结。未做 PPO，未改 Task-Waypoint。

历史 Phase-9.4 报告中的 “Oracle” 在本报告统一改称为 **shadow-search / heuristic reference**（`(n_near, n_far, idle)` 有限阴影搜索）。负 NHRR 只表示优于该启发式，不表示超过真正全局最优。未篡改 Phase-9.4 原文。

## 数据边界

- 已用于 collect/train 的 L3-B scenario seeds：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- 历史 eval（8–15）保持只读，不回训。
- Confirmation seeds（预注册）：[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
- 与历史占用集合相交：[]
- Confirmation SHA-256：`be3c8123ef0ac7368dbceed960961fb81efec2b9bd1cd9cc8034cff6d1867773`
- Checkpoint 规则（训练前冻结）：`last_epoch_fixed_phase94`（最后一轮，与 Phase-9.4 相同；不用 confirmation payload 选点）

## 训练实例

1. DeepSets+AR 独立训练 seed 数：**5**
2. HGNN+AR 独立训练 seed 数：**5**（一一配对）
model_seed=0 复用 Phase-9.4 正式 checkpoint；1–4 按冻结协议重训。

## Confirmation 结果

3. confirmation scenario 数：**16**
4. 是否全部为此前未使用普通场景：**True**
5. DeepSets+AR mean payload = **3687500**
6. HGNN+AR mean payload = **3762500**
7. mean paired delta (HGNN−DeepSets) = **75000**
8. median paired delta = **0**
9. hierarchical bootstrap 95% CI = **[-1012500, 1112500]**
10. seed-cluster bootstrap 95% CI = **[-862500, 937500]**
11. 各 training seed mean delta：[875000.0, 1125000.0, 687500.0, -1437500.0, -875000.0]
12. 正向 training seed 比例 = **0.60**（3/5）
13. conflict：DeepSets=0.0000，HGNN=0.0000
14. idle：DeepSets=0.4083，HGNN=0.4339
15. SINR utility：DeepSets=8.1216e+05，HGNN=8.4756e+05
16. high-interference subset delta = **220000**
17. low-interference subset delta = **-650000**
18. high-competition subset delta = **600000**
19. low-competition subset delta = **0**
20. graph advantage 是否集中在关系复杂状态：**True**（high/medium interference 与 high competition 上 delta 更大，但 high-competition 仅 2 个 scenario、medium-interference 仅 2 个，不能当作稳定机制结论）
21. permutation audit 是否通过：**False**（UAV DeepSets=False HGNN=False；Buoy DeepSets=False HGNN=False）。固定 UAV 解码顺序的 AR decoder 本身就不是 UAV 置换等变的。
22. 原所谓 Oracle 是否已改称 heuristic reference：**是**
23. 是否仍存在负 NHRR：**True**。解释：若出现负 NHRR，表示该策略在对应 scenario 上超过 shadow-search heuristic，不是全局 Oracle。
24. GRAPH Gate = **GRAPH-B**
25. stable_graph_advantage = **False**
26. 是否允许进入 high-level PPO fine-tune：**False**

graph_value_suggestive=True
Gate 原因：均值 HGNN>DeepSets，但 CI 跨 0 或 training-seed 方向不稳定。

Sanity：DeepSets+AR vs Hungarian 4.00e6 → mean 3687500；HGNN+AR vs Hungarian 2.625e6 → mean 3762500。

下一阶段：仅 GRAPH-A 才允许 Phase-9.5 supervised-preserving RL。GRAPH-B 不 PPO、不改架构。GRAPH-C 停止 HGNN 主线，保留 DeepSets+AR。

JSON/CSV：`src/wt/htw_phase941/outputs/graph_stability/`
