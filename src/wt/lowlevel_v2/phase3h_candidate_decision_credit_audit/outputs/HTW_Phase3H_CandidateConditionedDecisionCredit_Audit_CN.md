# HTW Phase-3H：Candidate-Conditioned Decision Credit Audit

本阶段只做离线诊断：Phase-3F 的 candidate-geometry signal（OOF≈0.546）究竟来自连续距离，还是 first-hold 已存在的 **离散决策 / feasible-set 结构**。未训练 Low-Level / High-Level，未改 Actor/Critic、reward、decoder、pool、ranking、checkpoint、正式 greedy。未把未来 hold / state / action / trajectory / `F` / `U_traj` 当作输入。不重复 Phase-3G 的 leftover×access 连续公式作为新发现。

指定离散 proxy（先看原始计数，再组内 z-score；不发明复杂公式）：

- `degree` / `N_feasible(Γ)`：选择 Γ 后，t=0 未占用 buoy 上仍 `in_range` 的 (UAV, buoy) 边数。
- `weighted_degree`：同上，边权为该 leftover buoy 的当前 queue。
- `Flex`：exclusive occupation 下，每个 UAV 的合法单边改写数（改 idle，或改到未占用且 in-range 的 buoy）。

`S_disc = z(U_single) + α z(Flex)`，指定 α=0.50；0.25 / 0.75 只报敏感性。STRUCT_OC 与 FHAT 沿用 Phase-3G / 3E。只改 first-hold 一次；hold 2/3/4 的 `U_traj` 来自已锁定的 Phase-3C 表，不重新 rollout。主判断以 **EXEC** 为准。

Sanity：LOCAL=9.000e6，TRAJ=11.88e6。复现通过。

**经验判断：当前可观测的 leftover feasible-set / Flex 不能解释 candidate-geometry 的 future-value signal。最大化 degree 或 Flex 会把闭环打到约 4.5e6。证据支持 C：连续几何公式与离散 feasible-set 都不足，first-hold observable information 本身不够。**

---

## 0. 已锁定前序事实

| 阶段 | 锁定数字 |
|---|---|
| 3C | LOCAL=9.000e6，TRAJ=11.88e6 |
| 3E | U_single OOF=0.193，state+cand=0.527，full=0.553；预测恢复≈0.135 |
| 3F | candidate geometry OOF≈0.546≈full 0.553；U_single OOF≈0.295 |
| 3G | within-state：U=0.259，O_traj=0.002，STRUCT_OC=0.187，FHAT=0.353；闭环 STRUCT_OC=9.250e6，FHAT=9.375e6 |

Phase-3G 已否定 leftover_queue×access/competition 的加性结构。本阶段换成 **离散边计数 / 合法改写数**，不再重做那条连续公式。

---

## 1. 原始分布（EXEC，未归一化）

| proxy | min | max | mean | std | n_unique |
|---|---:|---:|---:|---:|---:|
| degree | 0 | 36 | 21.66 | 8.35 | 7 |
| weighted_degree | 0 | 90 | 39.89 | 24.88 | 13 |
| Flex | 6 | 39 | 25.10 | 7.26 | 22 |
| leftover_coverable | 0 | 6 | 3.61 | 1.39 | 7 |
| conflict | 0 | 0 | 0 | 0 | 1 |
| pool_degree | 0 | 11 | 1.38 | 1.99 | 8 |

degree / Flex 不是常数：每个 state 内 degree 有 6–7 个不同值，Flex 有 11–14 个。先看分布再归一化是成立的。

`conflict` 全为 0：正式 24-pool 已经 exclusive legalize，组内没有 assignment conflict 可用来排序。不能把“更少 conflict”写成高 F 的可观测标记。

`degree` 与 `Flex` 几乎共线（pooled Spearman=0.949），二者主要是 `n_occupied` 的离散反面（与占用数 Spearman 约 −0.72 / −0.58）。这是 leftover 任务集合的边计数，不是一条新的物理量。

额外记录 `pool_degree`：当前 24-pool 里 Hamming 距离=1 的邻居数。用户指定的三个最小 proxy 仍是 degree / weighted_degree / Flex；`S_disc` 只用 Flex。

---

## 2. Within-state ranking vs F（EXEC）

| score | Spearman med [p25,p75] | Kendall med | top-1 | top-3 | NDCG@3 |
|---|---:|---:|---:|---:|---:|
| U_single | 0.259 [0.115, 0.377] | 0.167 | 0.062 | 0.104 | 0.473 |
| degree | 0.036 [−0.085, 0.132] | 0.007 | 0.000 | 0.104 | 0.358 |
| weighted_degree | 0.018 [−0.165, 0.177] | −0.002 | 0.000 | 0.083 | 0.346 |
| Flex | 0.041 [−0.079, 0.084] | 0.005 | 0.000 | 0.083 | 0.363 |
| STRUCT_OC | 0.187 [−0.003, 0.334] | 0.143 | 0.062 | 0.125 | 0.507 |
| S_disc | 0.187 [0.022, 0.326] | 0.129 | 0.062 | 0.104 | 0.482 |
| FHAT | **0.353** [0.176, 0.524] | 0.236 | 0.062 | 0.125 | 0.528 |
| pool_degree（额外） | 0.175 [−0.012, 0.350] | 0.145 | 0.188 | 0.250 | 0.590 |

degree / Flex 的组内相关接近 0，区间跨过 0，top-1=0。它们比 Phase-3G 的 `O_traj`（0.002）只高一点，远低于 Phase-3F geometry OOF 0.546，也低于 `U_single` 0.259。

`pool_degree` 是本阶段最强的纯离散项（0.175，NDCG@3=0.590），仍解释不了 0.546，也不能写成未来价值的原因。

vs `U_traj`：LOCAL 0.581，S_disc 0.497，Flex 0.098，degree 0.121。离散项单独几乎不排序轨迹效用。

四方法对照（within-state median Spearman vs F）：

| family | U_single | degree | Flex | S_disc | FHAT |
|---|---:|---:|---:|---:|---:|
| OLD | 0.343 | 0.120 | 0.147 | 0.332 | 0.483 |
| ABS | 0.097 | 0.162 | 0.114 | 0.117 | 0.197 |
| PREFIX | −0.064 | **−0.257** | **−0.307** | −0.130 | 0.096 |
| EXEC | 0.259 | 0.036 | 0.041 | 0.187 | 0.353 |

PREFIX 上 leftover degree / Flex 与 F **负相关**。离散 feasible-set 不是可移植的正 credit。

---

## 3. First-hold 闭环（后续仍正式 greedy）

| method | U_traj | F percentile | R | top-1 vs TRAJ |
|---|---:|---:|---:|---:|
| LOCAL | 9.000e6 | 0.565 | 0.000 | 0.125 |
| STRUCT_OC | 9.250e6 | 0.615 | 0.135 | 0.125 |
| DEGREE | **4.500e6** | 0.487 | −1.995 | 0.000 |
| WDEGREE | **4.500e6** | 0.487 | −1.995 | 0.000 |
| FLEX | **4.688e6** | 0.529 | −1.687 | 0.000 |
| S_disc | 9.250e6 | 0.607 | 0.077 | 0.125 |
| FHAT | **9.375e6** | 0.617 | 0.135 | 0.125 |
| TRAJ | 11.88e6 | 0.987 | 1.000 | 1.000 |

单独 `argmax degree / Flex` 把 EXEC 闭环打到约 4.5e6，低于 NATURAL 量级的另一侧崩溃。最大化“留下更多后续可行边”在 first-hold 上是有害选择，不是弱选择。

`S_disc`（α=0.25/0.50/0.75）三个指定值的 `U_traj` 都是 9.250e6，R=0.077。不是搜参赢家。

相对 LOCAL 的 +0.250e6 **只来自 1/16 个 state**（221003：6.00e6 → 10.00e6）。其余 15 个选点与 LOCAL 相同。top-1 vs TRAJ 仍是 0.125。

---

## 4. Replacement / structure consistency（不是因果）

对每个 state，把 first-hold 的 LOCAL candidate 在纸面上换成另一个合法 candidate，只比较 t=0 可观测结构，不 rollout。

EXEC：

- top-F 比 LOCAL 有更高 Flex 的比例 = **0**
- 更高 degree 的比例 = **0**
- 更低 conflict 的比例 = **0**（conflict 全 0）
- mean ΔFlex(topF−LOCAL) = **−6.19**
- mean Δdegree = **−7.13**

高 F candidate 在 first-hold 上已经表现出 **更小** 的 leftover feasible-set，不是更大。这与“留下更多后续选择 → 更高 F”的假设方向相反。

高/低 F 四分位的均值几乎重叠（degree 22.4 vs 21.2，Flex 25.8 vs 24.6）。组内排序没有被 leftover 边数拉开。

只能写：高 F 与更小 leftover degree **一致地同时出现**；不能写 Flex 是 F 的原因。

---

## 5. 四方法指定 S_disc

| method | LOCAL | STRUCT_OC | S_disc | FHAT | TRAJ | R_disc |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 1.169e7 | 1.194e7 | 1.194e7 | **1.219e7** | 1.381e7 | 0.091 |
| ABS | 1.112e7 | 1.181e7 | 1.156e7 | **1.188e7** | 1.312e7 | 0.111 |
| PREFIX | 1.031e7 | **1.088e7** | 1.069e7 | 1.075e7 | 1.438e7 | 0.042 |
| EXEC | 9.000e6 | 9.250e6 | 9.250e6 | **9.375e6** | 1.188e7 | 0.077 |

四家都是 FHAT 不差于 S_disc。S_disc 没有更稳的 first-hold 选择。

---

## 6. 对 Phase-3F geometry signal 的含义

Phase-3F：17 维 candidate geometry 联合 OOF≈0.546。Phase-3G：连续 leftover×access 压不成那条信号。Phase-3H：把同一几何收成 **离散边 / 合法改写 / leftover 可覆盖任务**，组内 Spearman 仍约 0.04。

因此：

1. geometry signal 不是“选 Γ 后还剩多少 in-range 后续边”。
2. 也不是 assignment flexibility（合法单边改写数）。
3. 正式 pool 内没有 conflict 可分。
4. pool 内 Hamming-1 邻居数只有弱相关（0.175），仍远低于 0.546。
5. ExtraTrees 用的是多维联合（SINR / 干扰 / 距离 / assignment token），不是一张 leftover 匹配图。
6. F 含 later-greedy；t=0 leftover 匹配图描述不了 hold 2/3/4。

candidate geometry 有 signal，但 **不能进一步解释为当前可观测 feasible-set structure**。

---

## 7. 六个问题

**Q1. candidate geometry 的 signal 能否由当前 feasible-set structure 解释？**  
**不能。** EXEC 最强指定离散项 Flex=0.041，degree=0.036。对照 `O_traj`=0.002、geometry OOF=0.546。额外的 `pool_degree`=0.175 仍不够。PREFIX 上 degree/Flex 甚至为负。

**Q2. degree / weighted-degree / Flex 与 F 的 within-state ranking？**  
Spearman median：degree **0.036**，weighted_degree **0.018**，Flex **0.041**。Kendall 接近 0。top-1 全是 0。NDCG@3 约 0.35–0.36，低于 LOCAL 的 0.473。

**Q3. S_disc 是否比 U_single 有稳定改善？**  
**否。** 组内 Spearman 0.187 < U_single 0.259。闭环 9.250e6 > 9.000e6，但只改 1 个 state，且排序变差。不是稳定改善。

**Q4. S_disc 能否提高实际 first-hold trajectory recovery？**  
**相对 LOCAL：能抬一点，但是脆弱的。** 9.250e6 vs 9.000e6，R=0.077，F percentile 0.565→0.607。15/16 选点不变。单独用 degree/Flex 会把 `U_traj` 打到 4.5e6。不能写成有效算法。

**Q5. S_disc 是否超过 Phase-3E FHAT？**  
**否。** 9.250e6 < 9.375e6；within Spearman 0.187 < 0.353；R 0.077 < 0.135。四方法里也没有一家 S_disc 稳定压过 FHAT。

**Q6. 当前证据支持哪一种解释？**  
**C. 两者都不足，observable information 本身不足。**

- 不是 A：Phase-3G 已否定连续 leftover/access；本阶段不再把它救回来。
- 不是 B：离散 leftover feasible-set / Flex 组内相关≈0，高 F 的 leftover degree 更小，最大化它有害。
- 是 C：first-hold 能写成的连续公式和离散匹配图，都吸收不了 Phase-3F 的 0.546 OOF；预测恢复仍停在约 0.13。剩余 gap 仍是 later-greedy / 未观测未来，不是“缺一张更漂亮的可行集计数”。

Flex 即使有弱相关，也只是与 F 一致的可观测结构相关，**不是** future value 的原因。本阶段不把任一 α 写成最终参数。

---

## 8. 阶段纪律与产物

未训练。未改正式方法。未进入 End-to-End Independent Validation。未进入 Phase-4。

产物：`src/wt/lowlevel_v2/phase3h_candidate_decision_credit_audit/outputs/`

- `candidate_structure.csv`
- `within_state_metrics.csv`
- `replacement_audit.csv`
- `selection_metrics.csv`
- `recovery_metrics.csv`
- `method_comparison.csv`
- `figures/candidate_f_vs_degree.png`
- `figures/candidate_f_vs_flex.png`
- `figures/within_state_ranking.png`
- `figures/first_hold_selection.png`
- `figures/recovery_comparison.png`

本阶段停止。
