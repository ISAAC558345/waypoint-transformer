# HTW Phase-9.4.5 Utility-Aligned Set Supervision for Global Assignment

只改 supervised objective。未改 HGNN/DeepSets encoder、Order-Free Decoder、L3-B、SINR、hold=15、低层、Task-Waypoint。未做 PPO。
phase943/941/942 confirmation 未进入 candidate/utility/tau/lambda/checkpoint。

1. candidate pool size = **24**
2. 生成：model-independent 合法池（heuristic / exclusive / 1-2 edge / competition / interference / idle / seeded diverse）seed=94501
3. utility = 冻结 L3-B + fly-toward 下 **H=15 payload increment**
4. 是否 actual H15 payload：**是**
5. 旧 reference 仍是 heuristic/shadow-search：**是**（L_ref 权重=0.1）
6. train near-optimal multiplicity (within 5% best-observed) mean n=**6.69**
7. utility-logP Spearman old/new DeepSets=**-0.0260/0.2852** HGNN=**-0.0047/0.3125**
8. candidate regret old/new DeepSets=**643750.0/368750.0** HGNN=**637500.0/368750.0**
9. NOM_5 old/new DeepSets=**0.4704/0.5960** HGNN=**0.4722/0.6009**
10. DeepSets old payload = **3200000**
11. DeepSets util payload = **6950000**
12. HGNN old payload = **3393750**
13. HGNN util payload = **6125000**
14. DeepSets objective delta = **3750000** CI={'mean': 3750000.0, 'ci_lo': 2478125.0, 'ci_hi': 4878203.125, 'n_boot': 2000}
15. HGNN objective delta = **2731250** CI={'mean': 2731250.0, 'ci_lo': 834375.0, 'ci_hi': 4609453.125, 'n_boot': 2000}
16. 对应 95% CI 见上。
17. seed0-4 DeepSets delta=[4578125.0, 4109375.0, 3500000.0, 5296875.0, 1265625.0]；HGNN delta=[281250.0, 953125.0, 1781250.0, 5062500.0, 5578125.0]
18. seed3/4 HGNN-DeepSets util delta=[-953125.0, 1171875.0]（old [-718750.0, -3140625.0]）
19. conflict util DeepSets/HGNN=**0.0000/0.0000**
20. idle util DeepSets/HGNN=**0.5440/0.5538**
21. SINR utility util DeepSets/HGNN=**1033681.2374/925790.5864**
22. UOBJ Gate=**UOBJ-A**
23. objective_alignment_confirmed=**True**
24. HGNN-util vs DeepSets-util delta=**-825000**
25. GRAPH-U Gate=**GRAPH-U-C**
26. 稳定 Graph Value=**False**
27. R4={'name': 'R4', 'n': 16, 'mean': -700000.0, 'median': -1000000.0, 'ci_lo': -2862812.5, 'ci_hi': 987500.0, 'n_pos_seeds': 3, 'positive_seed_frac': 0.6, 'per_seed': [-4125000.0, -2125000.0, 1000000.0, 375000.0, 1375000.0]}
28. 本阶段是否支持进入 PPO：**否。本阶段仍不 PPO。**
29. 下一阶段：**retain DeepSets**

OBJ (validation)=OBJ-A。checkpoint=best validation candidate regret，每 10 epoch。
confirmation seeds=[1321, 1322, 1323, 1328, 1333, 1335, 1347, 1353, 1355, 1358, 1375, 1377, 1382, 1386, 1388, 1392, 1331, 1334, 1336, 1339, 1341, 1343, 1345, 1348, 1357, 1362, 1366, 1368, 1374, 1380, 1381, 1383, 1384, 1396, 1466, 1815, 1855, 1887, 1949, 1967, 2073, 2100, 2181, 2223, 2263, 2265, 2369, 2371, 1329, 1337, 1338, 1342, 1356, 1359, 1360, 1365, 1369, 1370, 1372, 1373, 1387, 1405, 1406, 1414]。

JSON：`src/wt/htw_phase945/outputs/utility_aligned/`
