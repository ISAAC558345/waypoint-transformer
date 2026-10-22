# HTW Phase-9.4.3 Permutation-Equivariant Order-Free Global Assignment Decoder

未改 HGNN/DeepSets encoder、L3-B、SINR、低层、Task-Waypoint。未做 PPO。只替换高层 assignment decoder 与 set-level supervised loss。

1. 旧 decoder 问题：固定 UAV 顺序 AR 造成 ORDER-C（OSR≈0.39/0.55，优劣反转≈0.50）。
2. 新 decoder 每步在 remaining UAV×合法 buoy∪idle 上共享打分，选择 (m,k)；谁先分配由模型决定，不是外部 index 循环。
3. 是否 combination-aware：是，每步条件于 partial assignment set。
4. 是否保留 partial SINR context：是，且只依赖已选 assignment **集合**。
5. DP vs 6!：**True**（diff=3.815e-06）
6. UAV permutation equivariance = **1.0000** / HGNN **1.0000**
7. Buoy = **1.0000** / **1.0000**
8. Joint = **1.0000** / **1.0000**
9. candidate enumeration invariance = **1.0000**
10. PPV (DeepSets/HGNN) = **0.0000** / **0.0000**
11. ORDERFIX Gate = **ORDERFIX-A**
12. decoder_order_dependence_fixed = **True**
13. DeepSets mean payload = **3146875**
14. HGNN mean payload = **3590625**
15. mean paired delta = **443750**
16. median delta = **1000000**
17. hierarchical 95% CI = **{'ci_hi': 2121953.1249999995, 'ci_lo': -1415937.5, 'mean': 443750.0, 'n_boot': 2000}**
18. 5 model-seed delta = **[515625.0, 2515625.0, 2421875.0, -203125.0, -3031250.0]**
19. 正向 seed 比例 = **0.6**
20. R1 = **{'ci_hi': 2162812.499999998, 'ci_lo': -1450000.0, 'mean': 375000.0, 'median': 1000000.0, 'n': 16, 'n_pos_seeds': 3, 'name': 'R1', 'per_seed': [-1125000.0, 2750000.0, 2062500.0, 875000.0, -2687500.0], 'positive_seed_frac': 0.6}**
21. R2 = **{'ci_hi': 2425000.0, 'ci_lo': -1212812.5, 'mean': 675000.0, 'median': 1000000.0, 'n': 16, 'n_pos_seeds': 3, 'name': 'R2', 'per_seed': [1187500.0, 2687500.0, 2562500.0, -500000.0, -2562500.0], 'positive_seed_frac': 0.6}**
22. R3 = **{'ci_hi': 2125000.0, 'ci_lo': -2337812.5, 'mean': -62500.0, 'median': 1000000.0, 'n': 16, 'n_pos_seeds': 3, 'name': 'R3', 'per_seed': [625000.0, 2000000.0, 2875000.0, -1500000.0, -4312500.0], 'positive_seed_frac': 0.6}**
23. R4 = **{'ci_hi': 2275000.0, 'ci_lo': -1162812.5, 'mean': 787500.0, 'median': 1000000.0, 'n': 16, 'n_pos_seeds': 4, 'name': 'R4', 'per_seed': [1375000.0, 2625000.0, 2187500.0, 312500.0, -2562500.0], 'positive_seed_frac': 0.8}**
24. CGRAPH Gate = **CGRAPH-B**
25. GRAPH Gate = **GRAPH-B**
26. seed 2/3/4 是否仍失败：**[3, 4]**
27. 先前 graph instability 可归因于 decoder order 的部分：**order 已修，剩余不稳定更可能来自 encoder/训练。**
28. HGNN 普遍优于 DeepSets？**否**
29. 仅高关系复杂度条件优势？**见 CGRAPH**
30. 下一阶段：**investigate remaining encoder instability**

ORDERFIX 原因：结构审计通过；PPV deep=0.0000 hgnn=0.0000。
训练=已完成。confirmation seeds=[607, 610, 611, 621, 622, 632, 633, 637, 638, 640, 641, 665, 679, 682, 686, 707, 609, 613, 618, 627, 628, 629, 631, 635, 645, 647, 651, 652, 658, 662, 664, 676, 649, 661, 673, 674, 738, 764, 766, 798, 878, 904, 947, 1015, 1120, 1215, 1254, 1320, 600, 601, 604, 616, 617, 620, 624, 630, 634, 636, 639, 643, 653, 654, 656, 657]。

JSON：`src/wt/htw_phase943/outputs/order_free_decoder/`
