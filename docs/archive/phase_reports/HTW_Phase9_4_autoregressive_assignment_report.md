# HTW Phase-9.4 Masked Autoregressive Global Assignment Decoder

本阶段冻结 Level-3-B（M=6，K=12，hold=15）、SINR、低层 fly-toward、Oracle 定义与 train/eval split。
未修改 HGNN 层数/宽度，未重调 Phase-9.3 ranking，未做 PPO，未改 Task-Waypoint。
只替换高层 assignment decoder：固定 UAV 顺序的 Masked Autoregressive Decoder。

## 0. 磁盘冻结事实

- Phase-9.2 DeepSets+Hungarian payload = 4000000（conflict=0.000）
- Phase-9.3 HGNN+Hungarian payload = 2625000（ranking acc≈0.992）
- Phase-9.2 MLP option classification payload = 4375000（conflict=0.099）
与用户给出的 2.625e6 / 4.00e6 / 0.992 一致（以磁盘 JSON 为准）。
监督数据 collect seeds [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]，epochs=80，batch=32，Adam lr=0.001，aux ranking λ=0.05。
Eval seeds [8, 9, 10, 11, 12, 13, 14, 15]。Decoder 参数量（DeepSets/HGNN 共用）= 135297。

## 1. AR Decoder 结构是什么？

固定 UAV 解码顺序 m=0..M-1。每步在 mask 后的 {0..K-1, idle} 上做 softmax。
条件输入：`[uav_m, buoy_k, UB_edge_mk, graph, global, partial_assignment_embedding, candidate SINR/rate physics]`。
partial assignment 与 SINR context 随已选链路更新；非法 buoy 在 softmax 前 logits=-1e9。
DeepSets 与 HGNN 共用同一 `MaskedARDecoder`；唯一差异是 encoder。

## 2. 为什么比 Hungarian 更适合 SINR 组合收益？

Hungarian 假设 `Ĵ(X)=Σ_i Σ_k x_ik S_ik`。SINR 下 UAV1→buoy1 会改变 UAV2→buoy2 的干扰与速率，
因此需要 `P(z_m | G, X_<m)` 的条件组合分配，而不是一次性加性边分。

## 3–20. 数值与门控

| 方法 | payload | Oracle regret | NRegret | exact match | per-UAV acc | conflict | idle | SINR utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Exclusive | 1375000 | 3250000 | 0.703 | — | 0.166 | 0.000 | 0.000 | 4.450e+05 |
| B1 MLP (9.2) | 4375000 | 250000 | 0.054 | — | 0.418 | 0.099 | — | 9.433e+05 |
| B2 DeepSets+Hung | 4000000 | 625000 | 0.135 | — | 0.304 | 0.000 | — | 5.449e+05 |
| B3 HGNN+Hung | 2625000 | 2000000 | 0.432 | — | 0.408 | 0.000 | — | 5.594e+05 |
| B4 DeepSets+AR | 4250000 | 375000 | 0.081 | 0.125 | 0.443 | 0.000 | 0.234 | 9.870e+05 |
| B5 HGNN+AR | 5625000 | -1000000 | -0.216 | 0.344 | 0.719 | 0.000 | 0.354 | 1.015e+06 |
| Oracle | 4625000 | 0 | 0 | 1 | 1 | 0.000 | 0.417 | 9.581e+05 |

3. DeepSets+AR payload = **4250000**
4. HGNN+AR payload = **5625000**
5. DeepSets+AR − Hungarian = 250000；HGNN+AR − Hungarian = 3000000
6. Oracle payload (eval 8–15, hold=15, fly-toward) = **4625000**
7. DeepSets+AR normalized Oracle regret = **0.0811**
8. HGNN+AR normalized Oracle regret = **-0.2162**
9. whole-assignment exact match：DeepSets=0.1250，HGNN=0.3438
10. per-UAV accuracy（free-running）：DeepSets=0.4427，HGNN=0.7188
11. conflict rate：DeepSets=0.0000，HGNN=0.0000
12. idle rate：DeepSets=0.2344，HGNN=0.3542
13. teacher-forcing accuracy：DeepSets=0.5104，HGNN=0.7656
14. free-running accuracy：DeepSets=0.4427，HGNN=0.7188
15. exposure gap (TF−FR)：DeepSets=0.0677，HGNN=0.0469
16. SINR utility (mean rate)：DeepSets=9.8704e+05，HGNN=1.0153e+06
17. HGNN−DeepSets 逐 seed payload 差：[-1000000.0, 5000000.0, 4000000.0, 0.0, 0.0, 0.0, 3000000.0, 0.0]
    均值=1375000，bootstrap 95% CI=[0, 2875000]
18. DEC Gate = **DEC-A**
19. GRAPH Gate = **GRAPH-C**
20. 当前证据是否支持「HGNN 优于 DeepSets」：**否（仅 suggestive / unstable）**

补充：eval seeds 8–15 上 HGNN+AR 闭环 payload 高于 Oracle split heuristic。
这不表示 Oracle 标签错误；该 Oracle 是 (n_near, n_far, idle) 阴影搜索，不是穷尽联合分配。
HGNN idle=0.354，介于 DeepSets idle=0.234 与 Oracle idle=0.417 之间。

## 门控解释

- DEC：至少一个 AR decoder 相对其 Hungarian 对照 payload 上升、Oracle regret 下降、conflict=0。
- GRAPH：均值上 HGNN+AR payload 更高、NRegret 更低、conflict=0；paired bootstrap 95% CI=[0, 2875000]，下界未严格大于 0，只能写 suggestive / unstable。
- 情形（A/B/C）：**A（均值次序）/ GRAPH-C（CI 不稳定）**

下一阶段规则（第 28 条）：GRAPH-A 未过则不进入 Phase-9.5 PPO。
DEC-A 已成立时应带着最佳 AR decoder 前进，不要继续加 HGNN 层。
若要 RL，必须先得到稳定 GRAPH-A，且必须保护 supervised policy。

本阶段未做 PPO、未改 Task-Waypoint、未改低层。

JSON：`src/wt/htw_phase94/outputs/autoregressive_assignment/summary.json`
