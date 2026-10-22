# HTW Phase-9.4.2 Conditional Graph Advantage and Decoder-Order Sensitivity Audit

本阶段为冻结模型的 **inference-only** 机制验证。未训练、未 PPO、未改 HGNN/DeepSets/AR decoder/L3-B/SINR/低层/Task-Waypoint。
未根据模型 payload 挑选 scenario。I(s)/C(s) 在 confirmation 之前冻结。Primary 只使用 canonical decode order **O0**。
heuristic reference = shadow-search `(n_near, n_far, idle)`。负 NHRR 不表示超过全局最优。

1. 新独立 scenario 数量：**32**（seeds=[52, 59, 63, 71, 73, 79, 84, 89, 44, 46, 64, 66, 67, 74, 75, 77, 115, 202, 218, 258, 263, 355, 378, 598, 41, 47, 48, 49, 51, 56, 58, 60]）
2. R1–R4 数量：R1=8，R2=8，R3=8，R4=8
3. interference 阈值：q33=0.0558593，q66=0.0649852（复用 Phase-9.4.1）
4. competition 阈值：q33=0.5，q67=0.5（COLLECT_SEEDS tertiles）
5. R1 delta mean/CI = **-250000** / [-1000000, 575000]
6. R2 delta mean/CI = **-75000** / [-1325000, 1375000]
7. R3 delta mean/CI = **-225000** / [-1125625, 675000]
8. R4 delta mean/CI = **100000** / [-750000, 950000]
9. R4 正向 training seed 比例 = **0.40**（2/5）
10. Graph×Interference interaction (highI−lowI) = **250000** CI=[-812500, 1475000]
11. Graph×Competition interaction (highC−lowC) = **100000** CI=[-975000, 1087812]
12. relation-complexity interaction (R4−R1) = **350000** CI=[-675000, 1300000]
13. HGNN 优势是否随关系复杂度增强：**True**
14. low-complexity (R1) 下 HGNN 是否反而较差：**True**
15. DeepSets OSR (mean) = **0.3906**
16. HGNN OSR (mean) = **0.5545**
17. 模型优劣是否随 order 反转：**True**（反转率=0.498）
18. ORDER Gate = **ORDER-C**
19. 负 training seed 失败原因：**负 seed [2, 3, 4] 在 R1 与 R4 均失败，偏全局训练失败而非单一 regime。**
20. CGRAPH Gate = **CGRAPH-B**
21. conditional_graph_advantage_confirmed = **False**
22. 高关系复杂度稳定优势？**suggestive：R4 均值正向但不稳定。**
23. HGNN 普遍优于 DeepSets？**否。总体 paired delta 未形成稳定普遍优势。**
24. 下一阶段选择：**fix decoder order dependence**

CGRAPH 原因：R4 均值 HGNN>DeepSets，但 CI 跨 0 或 training-seed 不稳定。
ORDER 原因：HGNN vs DeepSets 优劣随 UAV decode order 频繁反转。

JSON/CSV：`src/wt/htw_phase942/outputs/conditional_graph/`
