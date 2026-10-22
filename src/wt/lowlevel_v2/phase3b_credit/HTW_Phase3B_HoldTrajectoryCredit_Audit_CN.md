# HTW Phase-3B：Hold-to-Trajectory Credit Audit

本阶段只做离线诊断。未训练、未改模型、未改 checkpoint、未改 q / L_exec / ranking / reference / DeepSets / decoder / candidate pool / beam / Pool-MAP / Low-Level MAPPO / SafetyProjector / collision / interference / reward / tau / lambda / 正式 evaluation pipeline。

全部数字复用已冻结的 Phase-3 Trajectory Utility Audit 产物：

- 设计集 221000–221015（16 scenario）
- OLD / ABS / PREFIX / EXEC 各 **seed0**
- 闭环 Low-Level seed=0
- 自然 4-hold 轨迹 64 条；第一 hold 24-pool counterfactual 1536 条

Sanity（不重新证明）：沿已实现轨迹 `U_traj ≈ Σ_t U_single_t`。四方法 Spearman 约 0.98–0.997，interaction gap 中位数 0。

**总判定：C — 两者同时存在：EXEC seed0 第一 hold 局部选点差，且四条方法的 local ranking 与 trajectory ranking 都不一致。** 结论跟数据，不预设 long-horizon credit assignment 一定成立。

## 1. 自然轨迹 hold 贡献

每条已实现 4-hold 轨迹记录 `state` 只通过已有 hold 序列表达：`Γ_t`、`U_single_t`、`U_traj`。

定义（只用于诊断，不是新训练 target）：

- `G_t = Σ_{r=t}^{4} U_single_r` remaining return
- `F_t = Σ_{r=t+1}^{4} U_single_r` future return

下面的 Pearson / Spearman / Kendall / top-1 / top-3 / normalized regret 是 **16 条已实现轨迹跨 scenario 的排序**，不是 hold 内 24-pool 排序。t=4 时 `F=0`，相关为 NA。

| method | hold | U_single mean | G mean | F mean | Spearman(U,G) | Spearman(U,F) | top-1(U,G) | top-1(U,F) | nregret(U→G) | nregret(U→F) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OLD | 1 | 3.675e+06 | 1.084e+07 | 7.162e+06 | 0.344 | 0.185 | 0.000 | 0.000 | 0.400 | 0.600 |
| OLD | 2 | 3.062e+06 | 7.162e+06 | 4.100e+06 | 0.391 | -0.197 | 1.000 | 0.000 | 0.000 | 0.286 |
| OLD | 3 | 3.288e+06 | 4.100e+06 | 8.125e+05 | 0.950 | 0.135 | 1.000 | 0.000 | 0.000 | 0.500 |
| OLD | 4 | 8.125e+05 | 8.125e+05 | 0.000e+00 | 1.000 | NA | 1.000 | NA | 0.000 | 0.000 |
| ABS | 1 | 3.775e+06 | 9.838e+06 | 6.062e+06 | 0.709 | 0.279 | 0.000 | 0.000 | 0.357 | 0.556 |
| ABS | 2 | 3.125e+06 | 6.062e+06 | 2.938e+06 | 0.656 | 0.235 | 1.000 | 0.000 | 0.000 | 0.200 |
| ABS | 3 | 2.438e+06 | 2.938e+06 | 5.000e+05 | 0.974 | 0.662 | 1.000 | 0.000 | 0.000 | 0.000 |
| ABS | 4 | 5.000e+05 | 5.000e+05 | 0.000e+00 | 1.000 | NA | 1.000 | NA | 0.000 | 0.000 |
| PREFIX | 1 | 4.275e+06 | 1.142e+07 | 7.150e+06 | 0.291 | -0.641 | 0.000 | 0.000 | 0.077 | 0.222 |
| PREFIX | 2 | 3.838e+06 | 7.150e+06 | 3.312e+06 | 0.468 | -0.156 | 1.000 | 0.000 | 0.000 | 0.200 |
| PREFIX | 3 | 2.875e+06 | 3.312e+06 | 4.375e+05 | 0.676 | -0.512 | 0.000 | 0.000 | 0.000 | 1.000 |
| PREFIX | 4 | 4.375e+05 | 4.375e+05 | 0.000e+00 | 1.000 | NA | 1.000 | NA | 0.000 | 0.000 |
| EXEC | 1 | 6.875e+05 | 5.775e+06 | 5.088e+06 | 0.891 | 0.841 | 1.000 | 0.000 | 0.000 | 0.222 |
| EXEC | 2 | 2.488e+06 | 5.088e+06 | 2.600e+06 | 0.900 | 0.776 | 1.000 | 0.000 | 0.000 | 0.429 |
| EXEC | 3 | 2.162e+06 | 2.600e+06 | 4.375e+05 | 0.968 | 0.374 | 1.000 | 1.000 | 0.000 | 0.000 |
| EXEC | 4 | 4.375e+05 | 4.375e+05 | 0.000e+00 | 1.000 | NA | 1.000 | NA | 0.000 | 0.000 |

第一 hold 交叉象限（相对中位数，16 scenario）：EXEC 高U低F=1，低U高F=1；PREFIX 高U低F=5，低U高F=1。

EXEC hold1：U_single mean=6.875e+05，F mean=5.088e+06。PREFIX hold1：U_single=4.275e+06，F=7.150e+06。

沿自然轨迹，EXEC 第一 hold 局部低，后续 F 也低于 PREFIX。**没有**“局部差、未来补回来”的已实现补偿。

## 2. 第一 hold counterfactual credit（本阶段主实验）

同一初始 state：强制第一 hold 为官方 24-pool 之一，hold 2/3/4 继续正式 greedy decoder。得到 `Γ → U_single(Γ|s) → U_traj(Γ)`。

此处 `G_1 = U_traj`（整条 remaining），`F_1 = U_traj - U_single`。

| method | Spearman | Pearson | Kendall | top-1 | top-3 | local-argmax 的 traj regret | U_traj gap | argmax 不一致率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OLD | 0.619 | 0.620 | 0.450 | 0.188 | 0.438 | 0.150 | 2.125e+06 | 0.812 |
| ABS | 0.511 | 0.554 | 0.372 | 0.250 | 0.437 | 0.153 | 2.000e+06 | 0.750 |
| PREFIX | 0.391 | 0.359 | 0.271 | 0.000 | 0.167 | 0.278 | 4.062e+06 | 1.000 |
| EXEC | 0.539 | 0.522 | 0.387 | 0.125 | 0.333 | 0.242 | 2.875e+06 | 0.875 |

第一 hold 的 `U_single` 对 **future F**（不是含自身的 G）的 pool 内排序：

| method | Spearman(U,F) | top-1(U,F) | nregret(U→F) |
|---|---:|---:|---:|
| OLD | 0.327 | 0.000 | 0.356 |
| ABS | 0.065 | 0.062 | 0.400 |
| PREFIX | -0.043 | 0.000 | 0.547 |
| EXEC | 0.235 | 0.062 | 0.545 |

## 3. 局部目标错位 vs 未来状态效应

对每个第一 hold 候选：

- `Δ_local = U_single(Γ) − U_single(Γ_actual)`
- `A_traj = U_traj(Γ) − U_traj(Γ_actual)`
- `Δ_future = F(Γ) − F(Γ_actual)`

`A_traj = Δ_local + Δ_future`。

| method | Spearman(Δ_local, A_traj) | Spearman(Δ_future, A_traj) | Spearman(Δ_local, Δ_future) | local-best 的 A_traj | local-best 的 Δ_future |
|---|---:|---:|---:|---:|---:|
| OLD | 0.539 | 0.914 | 0.208 | 8.125e+05 | -1.250e+05 |
| ABS | 0.419 | 0.850 | -0.059 | 1.312e+06 | 4.750e+05 |
| PREFIX | 0.292 | 0.886 | -0.132 | -1.250e+06 | -1.588e+06 |
| EXEC | 0.653 | 0.944 | 0.419 | 3.188e+06 | -7.375e+05 |

EXEC：A_traj 更跟 Δ_future（Spearman=0.944）而不是 Δ_local（0.653）。均值份额 A_traj_share_future=0.716，share_local=0.284。
`Δ_local` 与 `Δ_future` 的 Spearman=0.419（EXEC 为正；PREFIX=-0.132）。

同一方法下，把第一 hold 换成 pool 内两种 oracle 后的闭环均值：

| method | actual U_traj | pool argmax U_single 的 U_traj | pool argmax U_traj |
|---|---:|---:|---:|
| OLD | 1.088e+07 | 1.169e+07 | 1.381e+07 |
| ABS | 9.812e+06 | 1.112e+07 | 1.312e+07 |
| PREFIX | 1.156e+07 | 1.031e+07 | 1.438e+07 |
| EXEC | 5.812e+06 | 9.000e+06 | 1.188e+07 |

PREFIX 的 pool `argmax U_single` 反而低于其 OOD greedy（local-best A_traj 为负）：它已经选到了高即时效用。EXEC 换成 pool `argmax U_single` 可抬升约 3.2e6，仍低于 PREFIX 自然轨迹；换成 `argmax U_traj` 才接近 PREFIX。
注意：自然第一 hold 大多不在 24-pool 内，因此 pool oracle **不是** OOD greedy 的上界。221006 的 traj regret 为负，即 OOD 实际轨迹高于该 scenario 的 pool-best U_traj。

## 4. 四方法统一表

| method | hold | local U_single | remaining G | local-vs-G Spearman | top-1(U,G) | first-hold cf Spearman | first-hold cf top-1 | cf traj regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OLD | 1 | 3.675e+06 | 1.084e+07 | 0.344 | 0.000 | 0.619 | 0.188 | 0.150 |
| OLD | 2 | 3.062e+06 | 7.162e+06 | 0.391 | 1.000 | NA | NA | NA |
| OLD | 3 | 3.288e+06 | 4.100e+06 | 0.950 | 1.000 | NA | NA | NA |
| OLD | 4 | 8.125e+05 | 8.125e+05 | 1.000 | 1.000 | NA | NA | NA |
| ABS | 1 | 3.775e+06 | 9.838e+06 | 0.709 | 0.000 | 0.511 | 0.250 | 0.153 |
| ABS | 2 | 3.125e+06 | 6.062e+06 | 0.656 | 1.000 | NA | NA | NA |
| ABS | 3 | 2.438e+06 | 2.938e+06 | 0.974 | 1.000 | NA | NA | NA |
| ABS | 4 | 5.000e+05 | 5.000e+05 | 1.000 | 1.000 | NA | NA | NA |
| PREFIX | 1 | 4.275e+06 | 1.142e+07 | 0.291 | 0.000 | 0.391 | 0.000 | 0.278 |
| PREFIX | 2 | 3.838e+06 | 7.150e+06 | 0.468 | 1.000 | NA | NA | NA |
| PREFIX | 3 | 2.875e+06 | 3.312e+06 | 0.676 | 0.000 | NA | NA | NA |
| PREFIX | 4 | 4.375e+05 | 4.375e+05 | 1.000 | 1.000 | NA | NA | NA |
| EXEC | 1 | 6.875e+05 | 5.775e+06 | 0.891 | 1.000 | 0.539 | 0.125 | 0.242 |
| EXEC | 2 | 2.488e+06 | 5.088e+06 | 0.900 | 1.000 | NA | NA | NA |
| EXEC | 3 | 2.162e+06 | 2.600e+06 | 0.968 | 1.000 | NA | NA | NA |
| EXEC | 4 | 4.375e+05 | 4.375e+05 | 1.000 | 1.000 | NA | NA | NA |

## 5. EXEC 专项

- 自然第一 hold `U_single`：mean=6.875e+05，median=5.000e+05。PREFIX：4.275e+06。此前约 0.69e6 vs 4.28e6 得到确认。
- 自然 `U_traj`：mean=5.812e+06（PREFIX 1.156e+07）。正式五 seed EXEC 仍是 8.6925e6；本条是 seed0。
- 自然 future `F_1`：EXEC 5.125e+06 vs PREFIX 7.288e+06。后续没有补偿。
- 自然第一 hold `U_single` 在 24-pool 中的 percentile：mean=0.440，median=0.396。
- 自然 `U_traj` 在 24 条 counterfactual 中的 percentile：mean=0.508，median=0.583。
- oracle-local first hold 与 oracle-trajectory first hold 不一致率=0.875。
- local regret：mean=0.844，median=0.900。
- trajectory regret：mean=0.496，median=0.571。

EXEC 逐 scenario（seed0）：

| scenario | U_single | U_traj | F | local pct | traj pct | oracle-local U | oracle-local traj | oracle-traj U | oracle-traj traj | disagree | local regret | traj regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 221000 | 1.000e+06 | 5.000e+06 | 4.000e+06 | 0.792 | 0.625 | 3.400e+06 | 7.000e+06 | 1.000e+06 | 1.000e+07 | 1 | 0.706 | 0.500 |
| 221001 | 1.000e+06 | 1.000e+07 | 9.000e+06 | 0.750 | 0.875 | 4.000e+06 | 1.100e+07 | 1.000e+06 | 1.200e+07 | 1 | 0.750 | 0.167 |
| 221002 | 0.000e+00 | 3.000e+06 | 3.000e+06 | 0.125 | 0.167 | 5.000e+06 | 1.000e+07 | 4.000e+06 | 1.400e+07 | 1 | 1.000 | 0.786 |
| 221003 | 0.000e+00 | 2.000e+06 | 2.000e+06 | 0.208 | 0.208 | 4.200e+06 | 6.000e+06 | 3.000e+06 | 1.000e+07 | 1 | 1.000 | 0.800 |
| 221004 | 0.000e+00 | 3.000e+06 | 3.000e+06 | 0.250 | 0.167 | 4.400e+06 | 6.000e+06 | 1.000e+06 | 1.000e+07 | 1 | 1.000 | 0.700 |
| 221005 | 1.000e+06 | 8.000e+06 | 7.000e+06 | 0.792 | 0.625 | 5.000e+06 | 8.000e+06 | 1.000e+06 | 1.300e+07 | 1 | 0.800 | 0.385 |
| 221006 | 4.000e+06 | 1.100e+07 | 7.000e+06 | 0.958 | 1.000 | 4.400e+06 | 8.000e+06 | 2.000e+06 | 9.000e+06 | 1 | 0.091 | -0.222 |
| 221007 | 1.000e+06 | 8.000e+06 | 7.000e+06 | 0.583 | 0.667 | 5.000e+06 | 1.100e+07 | 4.000e+06 | 1.100e+07 | 1 | 0.800 | 0.273 |
| 221008 | 1.000e+06 | 1.000e+07 | 9.000e+06 | 0.667 | 0.833 | 5.000e+06 | 1.000e+07 | 1.000e+06 | 1.200e+07 | 1 | 0.800 | 0.167 |
| 221009 | 0.000e+00 | 2.000e+06 | 2.000e+06 | 0.208 | 0.208 | 5.000e+06 | 8.000e+06 | 4.000e+06 | 1.400e+07 | 1 | 1.000 | 0.857 |
| 221010 | 1.000e+06 | 1.000e+07 | 9.000e+06 | 0.500 | 0.875 | 4.000e+06 | 1.400e+07 | 4.000e+06 | 1.400e+07 | 0 | 0.750 | 0.286 |
| 221011 | 0.000e+00 | 2.000e+06 | 2.000e+06 | 0.167 | 0.083 | 5.000e+06 | 8.000e+06 | 1.000e+06 | 1.300e+07 | 1 | 1.000 | 0.846 |
| 221012 | 0.000e+00 | 3.000e+06 | 3.000e+06 | 0.083 | 0.250 | 5.000e+06 | 8.000e+06 | 2.000e+06 | 1.400e+07 | 1 | 1.000 | 0.786 |
| 221013 | 0.000e+00 | 2.000e+06 | 2.000e+06 | 0.292 | 0.208 | 4.400e+06 | 7.000e+06 | 0.000e+00 | 9.000e+06 | 1 | 1.000 | 0.778 |
| 221014 | 1.000e+06 | 9.000e+06 | 8.000e+06 | 0.500 | 0.792 | 5.000e+06 | 8.000e+06 | 2.000e+06 | 1.100e+07 | 1 | 0.800 | 0.182 |
| 221015 | 0.000e+00 | 5.000e+06 | 5.000e+06 | 0.167 | 0.542 | 5.000e+06 | 1.400e+07 | 5.000e+06 | 1.400e+07 | 0 | 1.000 | 0.643 |

若 EXEC 存在 long-horizon compensation，应看到：自然 `U_single` 低、但 `U_traj` 在 24 条 cf 中偏高，或 `F` 高于 PREFIX。当前两者都不成立。

## 6. 五个问题

**Q1. U_single 是否能预测当前 hold 开始后的 remaining return？**  
只能部分预测 **G（含自身）**，不能当作 **F** 或轨迹选点分数。
沿 16 条自然轨迹：PREFIX hold1 Spearman(U,G)=0.291，Spearman(U,F)=-0.641（高即时、低未来）。EXEC hold1 Spearman(U,G)=0.891，Spearman(U,F)=0.841——这是 seed0 崩塌簇同时压低 U 与 F 的跨 scenario 相关，不是“局部效用能预测未来”。
正确检验是 24-pool counterfactual：U_single vs U_traj Spearman EXEC=0.539、PREFIX=0.391；U_single vs F 仅 0.235 / -0.043。U_single 仍是正确的即时增量，但不能单独当 long-horizon 排序量。

**Q2. first-hold local utility ranking 与 first-hold trajectory ranking 是否一致？**  
不一致。EXEC top-1=0.125，Spearman=0.539，不一致率=0.875。PREFIX top-1=0.000，Spearman=0.391。argmax U_single 与 argmax U_traj 在多数 scenario 上不是同一个 Γ。

**Q3. counterfactual 后，后续状态是否产生显著 compensation / degradation？**  
会。A_traj 与 Δ_future 的 Spearman=0.944，与 Δ_local 为 0.653。U_single 对 F 的 pool 内 top-1=0.062。
但这是 **候选之间** 的未来状态效应，不是 EXEC seed0 已实现 greedy 的补偿：其 F 低于 PREFIX（5.125e+06 vs 7.288e+06），U_traj 也更低。

**Q4. EXEC 的主要问题？**  
**C**：两者同时存在：EXEC seed0 第一 hold 局部选点差，且四条方法的 local ranking 与 trajectory ranking 都不一致。
- A（local selection）：EXEC seed0 第一 hold U_single 约 6.875e+05，pool percentile 中位 0.396；PREFIX 约 4.275e+06。后续未补偿。
- B（credit）：四条方法共享 first-hold local≠traj ranking（top-1 约 0–0.25）。PREFIX 尽管 credit 也不对齐，但 greedy 选了高即时 U_LL，闭环仍最高。
数据能同时看见 A 和 B，不能把 EXEC seed0 的失败只写成 B。

**Q5. 下一阶段最小且唯一值得验证的假设（本阶段不实现、不训练）：**  
离线表已经表明：同一后续 greedy 下，`U_traj(argmax U_traj) > U_traj(argmax U_single)`（EXEC 均值 gap 约 2.9e6）。因此不必再验证“oracle 查表谁更大”。
还没被策略实现、因而唯一值得验证的假设是：**只改第一 hold、冻结后续 greedy 时，EXEC 若能选到接近 `argmax U_traj` 的 Γ_1，闭环能否追上 PREFIX 自然轨迹（本表 pool traj-best EXEC=1.188e+07 vs PREFIX actual=1.156e+07）。** 若能，则 EXEC seed0 的主缺口是第一 hold credit / 选点；若不能，则后续 hold 的 EXEC greedy 仍是独立问题。本阶段不训练、不改 decoder。

完整 Pearson / Kendall / top-3 见 `src/wt/lowlevel_v2/phase3b_credit/outputs/method_hold_summary.csv` 与 `PHASE3B_CREDIT_SUMMARY.json`。自然 hold 行的 `F` 来自 `Σ_{r>t} U_single`；EXEC 专项里的 `F` 来自 `U_traj − U_single`，二者相差已知的 interaction gap。

## 7. 数据边界

- 只使用 seed0 × Low-Level seed=0。正式 EXEC 五 seed 均值 8.6925e6 不被本条自然轨迹代表。
- 后续 hold 没有 24-pool counterfactual `U_traj`，因此 t=2/3/4 的 credit 只能做跨 scenario 相关，不能做候选排序。
- 自然第一 hold 几乎都不在官方 24-pool 内；pool 内 oracle 只回答“池内换 Γ_1 会怎样”。
- 没有重训、没有新 rollout。

本阶段停止。不要自动进入下一阶段。不要训练。不要改方法或 checkpoint。
