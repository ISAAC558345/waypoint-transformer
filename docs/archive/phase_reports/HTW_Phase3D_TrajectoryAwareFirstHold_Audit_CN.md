# HTW Phase-3D：Trajectory-Aware First-Hold Credit Minimal Audit

本阶段只做离线假设检验。未训练、未改模型 / checkpoint / Low-Level / decoder / pool / U_single / 正式 evaluation。

`F` 来自 Phase-3C 已冻结的 first-hold counterfactual remaining return，**不是**在线可观测特征。因此本报告不声称已经解决 Low-Level。

分数（只用于离线选第一 hold）：

`S_β(Γ) = U_single(Γ) + β F(Γ)`，`β ∈ {0, 0.25, 0.5, 0.75, 1}`。

选 `Γ* = argmax S_β` 后，hold 2/3/4 仍是当时正式 greedy。β=0 必须等于 LOCAL-ORACLE；β=1 必须等于 TRAJ-ORACLE。

Sanity：**通过**。β=0 与 LOCAL-ORACLE、β=1 与 TRAJ-ORACLE 的均值差均 ≤ 1.0。EXEC TRAJ-ORACLE=1.188e+07，与 Phase-3C 11.875e6 一致。

**EXEC：NATURAL=5.812e+06，β=0 / LOCAL=9.000e+06，β=1 / TRAJ=1.188e+07。**

## 1. EXEC β 扫描

| beta | final U_traj | F | U_single | DecodeNR | Spearman | Kendall | top-1 | top-3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 9.000e+06 | 4.388e+06 | 4.612e+06 | 0.000 | 0.539 | 0.387 | 0.125 | 0.333 |
| 0.25 | 1.081e+07 | 6.512e+06 | 4.300e+06 | 0.066 | 0.831 | 0.677 | 0.250 | 0.500 |
| 0.5 | 1.144e+07 | 7.600e+06 | 3.838e+06 | 0.159 | 0.946 | 0.820 | 0.312 | 0.771 |
| 0.75 | 1.175e+07 | 8.562e+06 | 3.188e+06 | 0.306 | 0.981 | 0.899 | 0.500 | 0.896 |
| 1.0 | 1.188e+07 | 9.625e+06 | 2.250e+06 | 0.516 | 1.000 | 0.908 | 1.000 | 1.000 |

对照：NATURAL U_traj=5.812e+06，LOCAL-ORACLE=9.000e+06，TRAJ-ORACLE=1.188e+07。
EXEC 均值序列是否随 β 非降：True。β=0.5 相对 LOCAL→TRAJ 剩余增益的回收比例=0.848。

## 2. 四方法 β audit

| method | beta | final U_traj | top-1 | Spearman | Kendall |
|---|---:|---:|---:|---:|---:|
| OLD | 0.0 | 1.169e+07 | 0.188 | 0.619 | 0.450 |
| OLD | 0.25 | 1.306e+07 | 0.500 | 0.880 | 0.735 |
| OLD | 0.5 | 1.362e+07 | 0.688 | 0.943 | 0.831 |
| OLD | 0.75 | 1.381e+07 | 0.812 | 0.978 | 0.901 |
| OLD | 1.0 | 1.381e+07 | 1.000 | 1.000 | 0.912 |
| ABS | 0.0 | 1.112e+07 | 0.250 | 0.511 | 0.372 |
| ABS | 0.25 | 1.256e+07 | 0.500 | 0.822 | 0.654 |
| ABS | 0.5 | 1.306e+07 | 0.562 | 0.923 | 0.787 |
| ABS | 0.75 | 1.312e+07 | 0.625 | 0.974 | 0.876 |
| ABS | 1.0 | 1.312e+07 | 1.000 | 1.000 | 0.888 |
| PREFIX | 0.0 | 1.031e+07 | 0.000 | 0.391 | 0.271 |
| PREFIX | 0.25 | 1.256e+07 | 0.125 | 0.750 | 0.582 |
| PREFIX | 0.5 | 1.394e+07 | 0.500 | 0.900 | 0.749 |
| PREFIX | 0.75 | 1.431e+07 | 0.688 | 0.971 | 0.865 |
| PREFIX | 1.0 | 1.438e+07 | 1.000 | 1.000 | 0.882 |
| EXEC | 0.0 | 9.000e+06 | 0.125 | 0.539 | 0.387 |
| EXEC | 0.25 | 1.081e+07 | 0.250 | 0.831 | 0.677 |
| EXEC | 0.5 | 1.144e+07 | 0.312 | 0.946 | 0.820 |
| EXEC | 0.75 | 1.175e+07 | 0.500 | 0.981 | 0.899 |
| EXEC | 1.0 | 1.188e+07 | 1.000 | 1.000 | 0.908 |

PREFIX / OLD / ABS 同样扫描，用来看 trajectory-aware score 是否系统性改善 first-hold 排序，而不是比较谁最高。

PREFIX：NATURAL=1.156e+07，β=0=1.031e+07，β=1=1.438e+07。

## 3. 四个问题

**Q1. β>0 是否比 β=0 更接近 TRAJ-ORACLE？**  
是。EXEC β=0 → 9.000e+06，β=0.5 → 1.144e+07，β=1 → 1.188e+07。
排序：Spearman β=0=0.539，β=0.5=0.946，β=1=1.000；top-1 0.125 → 0.312 → 1.000。

**Q2. 是否存在明显 sweet spot，还是必须接近 β=1？**  
对 **闭环 U_traj**，最早达到 TRAJ 95% 的网格点是 β=0.5（EXEC β=0.5 为 1.144e+07 vs TRAJ 1.188e+07）。对 **精确 top-1**，仍要到 β=1 才到 1.0（β=0.5 只有 0.312）。因此：不必精确 β=1 才能恢复 payload；但若目标是命中 argmax U_traj，则要完整 F。

**Q3. EXEC 的 first-hold mismatch 能否靠 future-return 明显修正？**  
能。加入 F 后，选点从 LOCAL 9.000e+06 移向 TRAJ 1.188e+07。
这只证明 **存在** 可用的 trajectory-aware credit signal。F 本身由完整 4-hold counterfactual 得到，不能直接喂给在线执行器。

**Q4. 下一阶段最符合数据的最小假设？**  
**C** — 有效信号是每个第一 hold 候选的未来回报 F；β=0.5 已回收 LOCAL→TRAJ 剩余增益的 0.848，因此下一阶段应预测候选级 F（或折扣 F），而不是重训后续 hold / Low-Level。A 只是把同一分数做成 first-hold critic 的实现方式；B 会被误读成 multi-hold executor credit，与 Phase-3C 不符。
不是最终算法。不要改 Low-Level，不要把 β 或 F 当成在线输入。

本阶段停止。不训练。不改方法或 checkpoint。不进入 End-to-End Independent Validation。
