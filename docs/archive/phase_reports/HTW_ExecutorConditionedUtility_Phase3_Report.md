# HTW Phase-3：Executor-Conditioned Utility Supervision 正式报告

本报告只审计已冻结的 Phase-3 产物。未改 Low-Level MAPPO、SafetyProjector、collision/interference、reward、actor/critic、decoder、DeepSets、candidate pool、ranking/reference。未新增 interference auxiliary、collision 模块、trajectory head，也未做 end-to-end validation。PREFIX 作为 baseline。正式推理仍是 joint greedy。

训练循环读取预计算 `utility`，不在 epoch 内调用 `shadow_h15`。因此 **没有向训练加入 utility cache**；训练 cache hit rate 为 N/A，相对训练加速比为 0。评估阶段的 out-of-pool \(U_{\mathrm{LL}}\) 仍使用既有 `(scenario, t, assignment_hash)` cache，不改变 \(q_i\)、\(L_{\mathrm{exec}}\)、checkpoint 与随机性。

## 结论

**Executor-conditioned utility supervision 是否解决 High-Level planner 与 Low-Level RL executor 的 utility mismatch？否：executor-conditioned supervision 没有同时满足四项门控，HL-LL utility mismatch 仍未解决。**

Gate=C；confirmed=false。失败原因：closed-loop 8.692e+06 < OLD 1.091e+07; DecodeNR 0.5000 > PREFIX 0.2375; seed std 2.212e+06 not < PREFIX 1.061e+05; Spearman 0.3976 not > PREFIX 0.4239。

下一步：停止。不自动进入下一阶段。不重调 q / pool / decoder / beam / checkpoint。

## 1. OLD / ABS / PREFIX / EXEC 比较

| 方法 | DecodeNR（聚合） | Spearman vs \(U_{LL}\) | Kendall vs \(U_{LL}\) | NOM5 | closed-loop payload |
|---|---:|---:|---:|---:|---:|
| OLD-HL + RL | 0.2500 | 0.1593 | 0.1853 | 0.2913 | 1.091250e+07 |
| ABS-HL + RL | 0.3258 | 0.3944 | 0.3386 | 0.4808 | 8.440000e+06 |
| PREFIX-HL + RL | 0.2375 | 0.4239 | 0.3640 | 0.4373 | 1.128750e+07 |
| EXEC-HL + RL | 0.5000 | 0.3976 | 0.3385 | 0.4372 | 8.692500e+06 |

来源：OLD/ABS 用 Phase-2B 正式 CSV/Gate；PREFIX 用 Phase-2D 正式 Gate/Within-pool；EXEC 用本阶段 DecodeNR / Within-pool / 5×5 闭环。Kendall 是冻结 checkpoint 在同一 val pool 上、按 state 平均的 rank correlation，与 Spearman 聚合方式一致。ABS DecodeNR 为五 seed median 的中位数；PREFIX 为两 pilot seed median 的中位数。

## 2. DecodeNR

- OLD median=0.2500
- ABS：seed0=0.3258, seed1=0.2500, seed2=0.4000, seed3=0.2250, seed4=0.9250；五 seed median 的中位数=0.3258
- PREFIX：seed0=0.2250，seed1=0.2500；聚合=0.2375
- EXEC：seed0=0.5600, seed1=0.4800, seed2=0.7500, seed3=0.3591, seed4=0.5000；聚合=0.5000
- EXEC vs PREFIX = +0.2625
- 门控 DecodeNR <= PREFIX：False

## 3. Spearman / Kendall

聚合方式：每个 val state 在 L=24 pool 上计算 \(P_\theta(\Gamma\mid s)\) 与 \(U_{\mathrm{LL}}\) 的 rank correlation，再对 state 取平均。

- OLD Spearman=0.1593；Kendall=0.1853
- ABS Spearman=0.3944（std=0.0350）；Kendall=0.3386（std=0.0294）
- PREFIX Spearman=0.4239；Kendall=0.3640（seed0 Kendall=0.3753，seed1=0.3527）
- EXEC Spearman=0.3976（std=0.0223）；Kendall=0.3385（std=0.0229）
- EXEC Spearman 分 seed：seed0=0.3798, seed1=0.4069, seed2=0.3684, seed3=0.4139, seed4=0.4193
- EXEC Kendall 分 seed：seed0=0.3187, seed1=0.3500, seed2=0.3092, seed3=0.3597, seed4=0.3550
- EXEC Spearman mean vs PREFIX=-0.0263
- EXEC Kendall mean vs PREFIX=-0.0255

## 4. NOM5

NOM5 = 概率质量落在 \(U \ge 0.95 U_{\mathrm{best}}\) 的候选上。

- OLD=0.2913
- ABS=0.4808（seed0=0.4906, seed1=0.4092, seed2=0.5280, seed3=0.5654, seed4=0.4106）
- PREFIX=0.4373（seed0=0.4598，seed1=0.4147）
- EXEC=0.4372（std=0.0358；seed0=0.4604, seed1=0.4316, seed2=0.3770, seed3=0.4620, seed4=0.4550）
- EXEC val regret mean=343750.0（std=49804.3）；OLD=987500.0；ABS=363750.0；PREFIX=334375.0

## 5. closed-loop payload

- OLD=1.091250e+07
- ABS=8.440000e+06
- PREFIX=1.128750e+07
- EXEC=8.692500e+06
- EXEC vs OLD=-2.220000e+06
- EXEC vs ABS=2.525000e+05
- EXEC vs PREFIX=-2.595000e+06
- 门控 closed-loop >= OLD：False

## 6. 五 seed 均值方差

| seed | closed-loop payload | DecodeNR median | Spearman | Kendall | NOM5 |
|---|---:|---:|---:|---:|---:|
| 0 | 5.550000e+06 | 0.5600 | 0.3798 | 0.3187 | 0.4604 |
| 1 | 1.056250e+07 | 0.4800 | 0.4069 | 0.3500 | 0.4316 |
| 2 | 7.275000e+06 | 0.7500 | 0.3684 | 0.3092 | 0.3770 |
| 3 | 1.057500e+07 | 0.3591 | 0.4139 | 0.3597 | 0.4620 |
| 4 | 9.500000e+06 | 0.5000 | 0.4193 | 0.3550 | 0.4550 |

- payload mean=8.692500e+06；std=2.212316e+06；min=5.550000e+06；max=1.057500e+07；max−min=5.025000e+06
- PREFIX two-seed payload std=1.060660e+05
- 方差是否下降：false
- selected epochs / val candidate regret：seed0 epoch=reused regret=None; seed1 epoch=reused regret=None; seed2 epoch=reused regret=None; seed3 epoch=reused regret=None; seed4 epoch=70 regret=318750.0
- \(U_{LL}\) label：mean=1.64523e+06；std=1.52169e+06；min=0；max=7e+06；n=2304；非正个数=629
- \(q\)：U/sum(U); if any U<=0 then U-min(U)+eps then normalize；artificial_q=false；prefix_label=false

## 7. 是否解决 High-Level 与 Low-Level utility mismatch

四项门控必须同时成立：

- closed-loop >= OLD：False（EXEC 8.692500e+06 vs OLD 1.091250e+07）
- DecodeNR <= PREFIX：False（EXEC 0.5000 vs PREFIX 0.2375）
- seed variance 降低：False
- Spearman 提升：False（EXEC 0.3976 vs PREFIX 0.4239）

**结论：否：executor-conditioned supervision 没有同时满足四项门控，HL-LL utility mismatch 仍未解决。**

Executor-conditioned supervision did not pass. closed-loop 8.692e+06 < OLD 1.091e+07; DecodeNR 0.5000 > PREFIX 0.2375; seed std 2.212e+06 not < PREFIX 1.061e+05; Spearman 0.3976 not > PREFIX 0.4239

训练 cache 审计：repeated_low_level_rollout_in_training=False；code_changed=False；cache_hit_rate_training=None；speedup_if_cache_added_to_training=0.0。No cache added. Continue current seed2/3/4 training.

冻结：Low-Level MAPPO、SafetyProjector、collision/interference、reward、Actor/Critic、decoder、DeepSets、candidate pool、ranking/reference loss、hold=15、L=24。

未做：interference auxiliary、collision 模块、trajectory head、end-to-end independent validation、beam / Pool-MAP 正式化、已完成 seed 的重训。

本阶段停止。不要自动进入下一阶段。
