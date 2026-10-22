# HTW Low-Level V2 Phase-0 报告

本阶段只建设接口。未训练。未修改 FINAL-HIER。未访问 Phase-10.6 untouched。

1. 新低层完整结构：冻结高层 Task Option → L0 base (fly-toward+gate) → residual desired → SafetyProjector / ServiceSafeEnvelope → L3-B movement。未来算法名 ICAR-MAPPO。
2. 来自现有 final 系统：DeepSets-util checkpoint、Order-Free 高层接口、Task Option、hold=15、fly-toward、InRangeZeroMotionGate、L3-B SINR/rate/interference 公式。
3. 新加入：neighbor set encoder schema、residual action 合同、水平安全盘 placeholder、SafetyProjector、ServiceSafeEnvelope、critic schema、新 seed registry、MAPPO skeleton（零残差初始化）。
4. Actor 看：self、task、communication（现有 SINR/rate/interference）、neighbor_set、d_nearest。无未来到达。
5. Critic 看：全体 UAV self/task、association、queue/progress、SINR/rate、两两距离、干扰矩阵、剩余 hold 与剩余时间。
6. Residual：a_des = clip(a_base + alpha * tanh(delta_rl))。Zero residual 时 a_des == a_base：True。
7. 碰撞体积：二维水平安全盘。L3-B 无原生 UAV 半径。历史 maritime `multi_uav_min_separation_m=15` 不能移植（起飞间距约 1.2 m）。placeholder d_safe=1.0 m，requires_pretraining_freeze=true。
8. SafetyProjector：迭代推开预测位置，再还原为步长受限动作；不可行则 failure，不称为安全。
9. 通信干扰：经现有 SINR→rate→payload 进入未来主奖励；Actor/Critic 观察现有干扰特征。
10. 第一版不加额外 interference penalty：避免第二套人工通信损失；真实 payload 已含干扰。
11. ServiceSafeEnvelope：在有效服务区内时，预测下一位置须留在 service_radius - safety_margin 内。旧 hard-zero gate 仍是 L0。
12. 旧 final untouched 是否访问？**否。**
13. 本阶段是否训练？**否。**
14. 是否修改旧 FINAL-HIER？**否。**
15. 新 train/design 是否与历史隔离？train/design 新池与扫描到的历史种子及 106000–106063 无交集。confirmation/final untouched 未创建。
16. 是否具备进入 LL-V2.1 单 seed Residual MAPPO pilot 的工程条件？接口与测试具备后可以进入 **pilot 工程准备**，但必须先由用户冻结 d_safe 与 alpha_residual；本阶段 **不自动训练**。

checkpoint SHA: `dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`
neighbor perm error: 1.1920928955078125e-07
actor init |mu|_max: 0.0
critic dim: 188 (expected 188)
通信位置扰动响应: True
