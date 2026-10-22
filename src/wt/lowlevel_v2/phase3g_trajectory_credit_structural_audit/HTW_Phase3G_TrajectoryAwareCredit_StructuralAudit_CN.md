# HTW Phase-3G：Trajectory-Aware Credit Structural Approximation Audit

本阶段只做离线诊断：检验 `F(Γ)` 能否被分解为若干 **first-hold 可观测** 的 trajectory-aware credit components，而不是再学 `x → F`。未训练 Low-Level / High-Level，未改 Actor/Critic、reward、SafetyProjector、decoder、candidate pool、ranking、checkpoint、正式 evaluation、`L_exec`。未把 `F` / `U_traj` / 后续 hold / 未来 state-action-trajectory / future decoder 当作输入。未做 Transformer / MLP / XGBoost 大规模训练或超参搜索。

指定结构（预先写死，不是搜参赢家）：

`O_traj = leftover_queue · (1 + idle_frac) · leftover_access / (1 + competition)`

`C_transition = in_range_frac / (1 + competition)`

`leftover_access = 1 / (1 + d_left / service_radius)`，`d_left` 为当前未分配且 queue>0 的 buoy 到最近 UAV 的距离均值。

组内 z-score 后：`S = U_single + α · std(U) · z(O) + δ · std(U) · z(C)`。

指定对照：`STRUCT_O = (α=0.5, δ=0)`，`STRUCT_C = (α=0, δ=0.25)`，`STRUCT_OC = (0.5, 0.25)`。Phase-3E 对照：`FHAT = U_single + 0.25 F_hat`（OOF ExtraTrees-C，scenario GroupKFold）。α 网格 `{0.25, 0.5, 0.75, 1.0}` 只报敏感性，不取最优当结论。

协议与 Phase-3C/3D/3E/3F 相同：只改 first-hold 一次；hold 2/3/4 继续正式 greedy。设计集 16 scenario × 24 candidate。主判断以 **EXEC** 为准。

Sanity（必须对齐 Phase-3C）：NATURAL=5.812e6，**LOCAL=9.000e6**，**TRAJ=11.88e6**。本阶段复现通过。

**经验判断：指定 leftover / access / transition 公式不能构成稳定的 F 分解。STRUCT 相对 LOCAL 只有 2/16 state 的脆弱抬升，且不优于 Phase-3E 的 generic F_hat。证据不足以支持“应用结构化近似替代直接预测 F”。**

---

## 0. 与前序阶段的关系

| 阶段 | 已锁定事实 |
|---|---|
| 3C | first-hold `U_single` 与最终 `U_traj` mismatch：LOCAL 9.00e6 vs TRAJ 11.88e6 |
| 3D | oracle `S_β = U_single + β F` 有价值；β=0.50 时 EXEC `U_traj`=11.44e6，Spearman=0.946 |
| 3E | OOF：U_single only=0.193，state+cand=0.527，full=0.553；预测选择最佳 `U_traj`=9.375e6，`R_pred`≈0.135 |
| 3F | candidate geometry OOF=0.546≈full 0.553；within-state Spearman(`U_single`,F)≈0.259，(`F_hat`,F)≈0.577；top-1≈0.062；判断 information-limited |

Phase-3F 的建议是停止堆模型、改看 F 的结构。本阶段就是那一次结构假设检验，不是算法上线。

---

## 1. Observable proxy inventory

见 `outputs/observable_proxy_inventory.csv`。每一项都经过 `available_at_first_hold = true` 才允许进入 score。`F`、`U_traj`、future trajectory 标记为 forbidden，未使用。

| 组 | proxy | 含义 | 入模 |
|---|---|---|---|
| A | `U_single` | first-hold shadow increment | 是 |
| B | `mean/max_uav_target_dist`, `cand_mean_sinr` | candidate–state / UAV / 通信几何 | 是 |
| C | 同上 + assignment 放到 t=0 的距离 | candidate–UAV | 是 |
| D | `n_unique_buoys`, leftover buoy 集合 | 与其它任务 / buoy 的关系 | 是 |
| E | `n_in_range`, `leftover_min_dist`, `leftover_access` | 可达 / 距离 | 是 |
| F | `leftover_queue`, `leftover_hot`, `idle_frac`, `O_traj` | neighborhood / remaining-task opportunity | 是 |
| G | `competition_count` | assignment 竞争 | 是 |
| H | `assigned_queue_mean`, `queue_hot_mass` | t=0 已有 queue / progress | 是 |

`queue_hot_mass` 在 t=0 设计集上几乎跨 candidate 常数，只能作 state 背景，不能作 within-state 排序。`remaining_arrivals` 与后续 hold 一律不用。

---

## 2. Component 与 F（EXEC within-state）

只报 **state 内** 相关，不把跨 scenario 的 pooled 相关当成排序证据。

| component | Spearman vs F median | p25 | p75 |
|---|---:|---:|---:|
| U_single | 0.259 | 0.115 | 0.377 |
| -competition | 0.183 | 0.041 | 0.250 |
| leftover_access | 0.081 | -0.007 | 0.277 |
| leftover_queue | 0.018 | -0.165 | 0.177 |
| C_transition | 0.018 | -0.120 | 0.177 |
| O_traj | 0.002 | -0.109 | 0.112 |
| n_in_range | -0.010 | -0.106 | 0.068 |
| -mean_dist | -0.088 | -0.272 | 0.006 |

最相关的 **可观测量仍是 `U_single`（0.259）**。在非 `U_single` 的结构项里，只有 **-competition（0.183）** 有稳定弱正号；leftover-queue / leftover-access / `O_traj` / `C_transition` 都接近 0，区间跨过 0。

Pooled scatter（`component_vs_F.png`）与组内结论一致：`O_traj`、`C_transition`、`leftover_queue` 对 F 几乎是云团，Spearman≈0。

四方法对照（within-state median Spearman vs F）：

| family | U_single | O_traj | leftover_queue | C_transition | -competition | -distance |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 0.343 | 0.080 | 0.071 | -0.095 | 0.093 | -0.293 |
| ABS | 0.097 | 0.155 | 0.140 | -0.053 | 0.010 | 0.078 |
| PREFIX | -0.064 | **-0.293** | **-0.306** | 0.313 | -0.027 | **0.319** |
| EXEC | 0.259 | 0.002 | 0.018 | 0.018 | 0.183 | -0.088 |

`O_traj` / leftover-queue 的符号 **跨方法翻转**。PREFIX 上“留下更多 queue”反而对应更低 F。因此 leftover-opportunity 不是一个可移植的 F 分量。

---

## 3. Within-state ranking（必须看，不能只看全局相关）

对每个 EXEC state 的 24 个 candidate，比较 score 与 F、`U_traj`。

### 3.1 vs F

| score | Spearman med [p25,p75] | Kendall med | top-1 mean | top-3 mean | NDCG@3 mean |
|---|---:|---:|---:|---:|---:|
| LOCAL (`U_single`) | 0.259 [0.115, 0.377] | 0.167 | 0.062 | 0.104 | 0.473 |
| O_traj | 0.002 [-0.109, 0.112] | -0.011 | **0.000** | 0.146 | 0.378 |
| C_transition | 0.018 [-0.120, 0.177] | -0.007 | 0.000 | 0.083 | 0.316 |
| STRUCT_OC | 0.187 [-0.003, 0.334] | 0.143 | 0.062 | 0.125 | 0.507 |
| FHAT | **0.353** [0.176, 0.524] | 0.236 | 0.062 | 0.125 | **0.528** |

`O_traj` 单独排序 **从未** 命中组内 F 第一名。STRUCT_OC 的组内 Spearman（0.187）**低于** LOCAL（0.259），也低于 FHAT（0.353）。NDCG@3 只有 FHAT 略高。top-1 全部停在 0.062，与 Phase-3F 一致。

### 3.2 vs `U_traj`

| score | Spearman med [p25,p75] | top-1 mean |
|---|---:|---:|
| LOCAL | 0.581 [0.420, 0.646] | 0.125 |
| O_traj | 0.111 [0.037, 0.238] | — |
| STRUCT_OC | 0.476 [0.402, 0.584] | 0.125 |
| FHAT | **0.627** [0.552, 0.724] | 0.125 |

对最终轨迹效用，FHAT 仍高于 LOCAL；STRUCT_OC 反而略低于 LOCAL。结构化加项没有把 first-hold 选择推得更接近 `argmax U_traj`。

---

## 4. First-hold 实际闭环（后续仍正式 greedy）

| method | U_traj | remaining F | F percentile | R vs LOCAL | DecodeNR | top-1 vs TRAJ | ranking-regret median |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOCAL | 9.000e6 | 4.388e6 | 0.565 | 0.000 | 0.000 | 0.125 | 1.000 |
| FHAT | **9.375e6** | 4.800e6 | 0.617 | 0.135 | 0.010 | 0.125 | 1.000 |
| STRUCT_O | 9.188e6 | 4.588e6 | 0.602 | 0.058 | 0.003 | 0.125 | 1.000 |
| STRUCT_C | **8.500e6** | 3.913e6 | 0.529 | 0.077* | 0.006 | 0.062 | 1.000 |
| STRUCT_OC | 9.250e6 | 4.675e6 | 0.615 | 0.135 | 0.009 | 0.125 | 1.000 |
| TRAJ-ORACLE | 11.88e6 | 9.625e6 | 0.987 | 1.000 | 0.516 | 1.000 | 0.000 |

`payload` 与 `U_traj` 相同（轨迹终点 payload 增量）。`R = (U_method - U_local) / (U_traj_oracle - U_local)`，分母为 0 的 scenario 记 NaN，不进均值。

\*STRUCT_C 的 **mean R=0.077 与 mean `U_traj`=8.50e6 方向不一致**：这是「比率均值 ≠ 均值比率」。有的 scenario 分母很小、轻微正恢复会抬高 mean R。闭环判断以 **mean `U_traj`** 为准：STRUCT_C **低于** LOCAL，单独加 transition proxy 有害。

α 敏感性（只加 `O_traj`，不取最优）：

| alpha | U_traj | R | F percentile |
|---:|---:|---:|---:|
| 0.25 | 9.188e6 | 0.058 | 0.602 |
| 0.50 | 9.188e6 | 0.058 | 0.602 |
| 0.75 | 9.250e6 | 0.135 | 0.615 |
| 1.00 | 9.250e6 | 0.135 | 0.615 |

更大 α 只是把那 2 个改选 scenario 再确认一遍，不是新的最优算法。

---

## 5. Counterfactual consistency：STRUCT 选中的 candidate 是否更接近 TRAJ-ORACLE

EXEC 16 个 state，STRUCT_OC 与 LOCAL **14/16 选同一个 candidate**。仅两处改选：

| scenario | LOCAL `U_traj` / F | STRUCT_OC | FHAT |
|---|---|---|---|
| 221003 | 6.00e6 / 1.80e6 | **9.00e6 / 5.00e6** | 9.00e6 / 5.00e6 |
| 221006 | 8.00e6 / 3.60e6 | **9.00e6 / 5.00e6** | 8.00e6 / 3.60e6（未改） |
| 221000 | 7.00e6 / 3.60e6（未改） | 同 LOCAL | **10.00e6 / 7.00e6** |

STRUCT_OC 的整段 +0.250e6 均值，**全部来自这 2 个 state**。FHAT 的 +0.375e6 来自 221000 与 221003。

选中 candidate 的真实 F percentile：LOCAL 0.565 → STRUCT_OC 0.615 → FHAT 0.617。有抬升，但仍远低于 TRAJ 的 0.987。top-1 vs `argmax U_traj` **三者都是 0.125**，STRUCT **并没有更经常贴近轨迹最优**。

ranking-regret 中位仍是 1.0：对绝大多数 state，STRUCT 选中的 F 等于 LOCAL 选中的 F。相关略变、选点几乎不变——这正是本阶段禁止「相关升高就宣称算法有效」的原因。

---

## 6. Component ablation

| score | EXEC `U_traj` | F pct | R | top-1 vs TRAJ | within Spearman vs F |
|---|---:|---:|---:|---:|---:|
| U_single | 9.000e6 | 0.565 | 0.000 | 0.125 | 0.259 |
| U_single + O_traj | 9.188e6 | 0.602 | 0.058 | 0.125 | — |
| U_single + C_transition | 8.500e6 | 0.529 | 0.077* | 0.062 | — |
| U_single + O_traj + C_transition | 9.250e6 | 0.615 | 0.135 | 0.125 | 0.187 |
| U_single + 0.25 F_hat | 9.375e6 | 0.617 | 0.135 | 0.125 | 0.353 |
| TRAJ-ORACLE | 11.88e6 | 0.987 | 1.000 | 1.000 | — |

O_traj 单独有很小正贡献；C_transition 单独损害 mean `U_traj`；二者相加的闭环 R 追上 FHAT，但绝对 `U_traj` 仍低于 FHAT，组内排序还更差。没有一项 ablation 把 top-1 从 0.125 抬上去。

---

## 7. 四方法指定 STRUCT_OC

| method | LOCAL | FHAT | STRUCT_O | STRUCT_OC | TRAJ | R_OC | R_FHAT |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD | 1.169e7 | **1.219e7** | 1.206e7 | 1.194e7 | 1.381e7 | 0.145 | 0.182 |
| ABS | 1.112e7 | **1.188e7** | 1.156e7 | 1.181e7 | 1.312e7 | 0.222 | 0.333 |
| PREFIX | 1.031e7 | 1.075e7 | 1.050e7 | **1.088e7** | 1.438e7 | 0.132 | 0.069 |
| EXEC | 9.000e6 | **9.375e6** | 9.188e6 | 9.250e6 | 1.188e7 | 0.135 | 0.135 |

只有 PREFIX 上 STRUCT_OC 的闭环略高于 FHAT；OLD / ABS / EXEC 都是 FHAT 更好或持平。STRUCT 不是跨方法更稳的选择器。

---

## 8. 为什么 candidate geometry 有 signal，但直接预测 F 的 top-1 很差

这是本阶段对 Phase-3F 现象的结构解释，不是新模型。

1. **Phase-3F 的 geometry signal 是多特征联合、且含跨 state 变异。** ExtraTrees 在 17 维 candidate geometry 上 OOF Spearman=0.546，within-state Spearman(`F_hat`,F)≈0.577。那不是 leftover-queue 这一维。
2. **把 geometry 压成一条物理公式后，组内排序消失。** `O_traj` within-state Spearman≈0.002，top-1=0。leftover-queue、access、in-range 单独都不能复现 ExtraTrees 抓住的交互。
3. **F 含 later-greedy。** t=0 leftover 只描述“现在没吃掉的 queue 质量/距离”，不描述 hold 2/3/4 的正式 greedy 会如何走。PREFIX 上 leftover 甚至与 F 负相关：留下更多任务并不等于留下更好的后续机会。
4. **组内 F 的第一名对局部扰动极敏感。** 即使 ExtraTrees 相关 0.55、FHAT 组内 0.35，top-1 仍是 0.062。STRUCT 相关更低，更不可能稳定打中 `argmax F` / `argmax U_traj`。
5. **因此：geometry 有粗粒度 signal ≠ 存在可加的、可解释的 first-hold credit 分解。** 当前失败的是「leftover × access / competition」这条假设，不是 Phase-3D「oracle F 有用」被推翻。

---

## 9. 五个问题

**Q1. first-hold 可观测信息中，哪些结构 component 与 F 最相关？**  
EXEC 上仍是 **`U_single`（median Spearman=0.259）**。非 local-value 项里只有 **-competition（0.183）** 有弱正号；`leftover_access`=0.081；`O_traj` / leftover-queue / `C_transition` ≈ 0。PREFIX 上最相关的反而是 **-distance（0.319）** 与 `C_transition`（0.313），leftover-queue 为 **-0.306**。没有一个跨方法稳定的“未来机会”分量。

**Q2. `O_traj` 是否比 `U_single` 更能解释 future trajectory value？**  
**否。** EXEC：`O_traj` vs F = 0.002，`U_single` vs F = 0.259；`O_traj` vs `U_traj` = 0.111，`U_single` vs `U_traj` = 0.581。`O_traj` 组内 top-1 vs F = 0。

**Q3. 结构化 score 是否比 Phase-3E generic predictor 更稳定？**  
**否。** 组内 Spearman：STRUCT_OC 0.187 < FHAT 0.353 <（Phase-3F 的 `F_hat` 0.577）。闭环：STRUCT_OC 9.250e6 < FHAT 9.375e6；R 同为 0.135。跨方法 4 家里 3 家 FHAT 不差于 STRUCT。STRUCT 没有表现出更好的稳定性。

**Q4. 结构化 score 能否提高实际 first-hold trajectory recovery？**  
**相对 LOCAL：能抬一点，但是脆弱的。** STRUCT_OC=9.250e6 vs LOCAL=9.000e6，R=0.135；选中 F percentile 0.565→0.615。14/16 state 选点不变，增益只来自 221003 与 221006。top-1 vs TRAJ 仍是 0.125。STRUCT_C 把 mean `U_traj` 降到 8.500e6。相对 FHAT：**没有提高**。不能把相关或某一个 α 写成有效算法。

**Q5. 当前证据是否足以支持“trajectory-aware credit 应采用可观测结构化近似，而不是直接预测完整 F”？**  
**不足，不能这么写。**  
支持结构路线的部分：leftover/competition 等字段确实可观测；竞争项有弱信号；STRUCT_OC 相对 LOCAL 有 2-state 闭环正贡献。  
不支持“用结构化近似替代预测 F”的部分：指定 `O_traj` 几乎不解释 F；STRUCT 组内排序与闭环都不优于 Phase-3E `F_hat`；公式符号跨方法翻转；top-1 / ranking-regret 几乎不动。  
更准确的阶段结论是：

> first-hold 可写成 leftover / access / transition 的简单加性 credit，**当前证据不支持**；Phase-3F 的 geometry signal 也 **不能** 被这条公式吸收。oracle F 仍然有价值（Phase-3D），但可观测结构近似尚未成立。剩余 gap 仍主要是 later-greedy / 未观测未来，而不是“缺一个更漂亮的 α”。

本阶段不把任一 α、δ 写成最终参数。相关升高 ≠ 算法有效。

---

## 10. 阶段纪律与产物

未训练、未改正式方法、未进入 Phase-4。

产物目录：`src/wt/lowlevel_v2/phase3g_trajectory_credit_structural_audit/outputs/`

- `observable_proxy_inventory.csv`
- `component_metrics.csv`
- `within_state_metrics.csv`
- `selection_metrics.csv`
- `ablation_metrics.csv`
- `ranking_metrics.csv`
- `recovery_metrics.csv`
- `figures/component_vs_F.png`
- `figures/within_state_ranking.png`
- `figures/first_hold_selection.png`
- `figures/recovery_comparison.png`
- `figures/ranking_regret.png`

本阶段停止。
