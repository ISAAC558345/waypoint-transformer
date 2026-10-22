# HTW Low-Level V2 Phase-2B：Base-Relative Residual Trust-Region Pilot（LL-V2.2B）

本阶段是 **development pilot**。只训练 **seed0、seed1**，均从头初始化 Actor/Critic。Primary 一律 **update100**。未访问 Phase-2 validation `222000–222063`，未创建新的 independent validation，未训练 seed2/3/4，未网格搜索 beta。

正式结论保持：

- `LLV2-MS-B`
- `stable_residual_advantage=false`
- `REL-1`

Phase-2A 依据：`primary=ACTION`，`secondary=NAV`，`S1-B`。

单元测试 **16/16 PASS** 后才训练。SafetyProjector SHA 未变；高层 checkpoint SHA 未变。

---

## 唯一改动

**Base-Relative Residual Trust Region**（含 base-zero freeze + 二维欧氏范数 cap）。

顺序：`a_base` → Actor `r_raw=0.25*tanh(raw_delta)` → ServiceResidualLock → Trust Region → `a_des` → SafetyProjector → envelope → env。

- `beta_residual = 0.25`
- `epsilon_base_fraction = 0.05`
- `a_max = 1.0`（来自 `ACTION_SCALE` / `clip_unit_step`，未重定义动作单位）
- `epsilon_base = 0.05`
- cap 只改幅度、保持方向；`C_residual` 用 `r_used`；PPO log_prob 仍对 raw sample。
- **未改** reward / lr / PPO clip / entropy / Critic / Actor 隐层 / Neighbor / relation / alpha / d_safe / lock / projector / 高层。

训练预算：每 seed **100 updates × 48 × 4 = 19,200** env transitions，**115,200** agent transitions。评估仅 `LLV2_design`。

---

## Design 对照（u100）

| tag | payload | Δ vs L0 | used residual | arrival | time-in-range | idle | collision |
|---|---|---|---|---|---|---|---|
| L0 | 10.375e6 | 0 | 0 | 6.06 | 104.7 | 0.552 | 0 |
| **new seed0** | **10.3125e6** | **−0.0625e6** | 0.00473 | 5.71 | 106.9 | 0.553 | 0 |
| **new seed1** | **12.1875e6** | **+1.8125e6** | 0.00617 | 5.60 | 101.3 | 0.553 | 0 |
| old seed0 | 11.0625e6 | +0.6875e6 | 0.0133 | 5.90 | 103.3 | 0.552 | 0 |
| old seed1 | 8.4375e6 | −1.9375e6 | 0.0283 | 7.44 | 67.1 | 0.675 | 0 |
| old seed2 参考 | 12.625e6 | +2.250e6 | 0.0157 | 6.13 | 96.1 | 0.598 | 0 |

学习曲线 payload（design）：

- seed0：u0 10.375 / u20 10.312 / u40 10.562 / u60 10.438 / u80 10.562 / **u100 10.312** e6
- seed1：u0 10.375 / u20 10.688 / u40 11.000 / u60 11.438 / u80 12.438 / **u100 12.188** e6

---

## 机制

**ACTION：** old seed1 used residual 0.0283 → new 0.00617（约 22%）。raw 仍约 0.050，但 **base-zero freeze ≈ 90.5%** 把空闲/无飞向修正清零。cap activation = 0（本批 design 上范数 cap 几乎未触发，主要靠 freeze）。原“residual 改 base 过多”被直接抑制。

**NAV：** old seed1 到达 7.44 → new 5.60，**接近并略快于 L0 6.06**。飞行距离 748 → 591。time-in-range 67 → 101（接近 L0 105）。

**SERVICE：** lock violation 全 0。进圈 SINR/rate 正常。未重新破坏 SERVICE。

**高层：** old seed1 idle 0.675 → new 0.553（回到 L0 附近）；assignment-change 2.25 → 2.94。闭环负反馈减轻。

**安全：** collision=0；`min_sep_worst + 1e-6 ≥ 1.0`；projector_failure=0。介入几乎全是 PAIRWISE。

**REL：** neighbor shuffle 仍 ~1e-9。不是成功条件，未加 relation 模块。

---

## Gate

**P2B-B**

- new seed1 **≥ L0**，相对 old seed1 **+3.75e6**，arrival 与 used residual 均改善，安全通过。
- new seed0 **略低于 L0**（−0.0625e6），且低于 old Phase-1B seed0。统一 trust region 对原正向 seed 有轻微代价。
- 因此 **不能** 宣称 action_trust_region_supported=true，**不能** 自动进入 5-seed 独立验证。
- **禁止** 本轮调 beta。停止。

---

## 二十五问

1. 唯一改了什么？**Base-Relative Residual Trust Region**（base-zero freeze + 相对 `||a_base||` 的欧氏 cap）。  
2. `beta_residual`？**0.25**  
3. `epsilon_base_fraction`？**0.05**  
4. `epsilon_base` 实际数值？**0.05**（`a_max=1.0`）  
5. 是否改 reward/lr/network？**否。**  
6. seed0/seed1 预算？各 **100 updates，19200 env / 115200 agent**  
7. 两 seed u0/20/40/60/80/100 payload？seed0：**10.375 / 10.312 / 10.562 / 10.438 / 10.562 / 10.312 e6**；seed1：**10.375 / 10.688 / 11.000 / 11.438 / 12.438 / 12.188 e6**  
8. L0 payload（design）？**10.375e6**  
9. new seed0−L0？**−0.0625e6**  
10. new seed1−L0？**+1.8125e6**  
11. 相对 old seed1？payload **8.4375e6 → 12.1875e6（+3.75e6）**  
12. raw residual norm（new0/new1）？**0.0595 / 0.0502**  
13. used residual norm？**0.00473 / 0.00617**（old seed1 **0.0283**）  
14. cap activation rate？**0**（两 seed）  
15. base-zero freeze rate？**0.925 / 0.905**  
16. old→new seed1 arrival？**7.44 → 5.60**（L0 6.06）  
17. never-reach fraction？new1 **0.558**，L0 **0.555**，old1 **0.401**（old1 进圈更少、停留更短，该比例不可单独解读为更好）  
18. 高层 idle/assignment？idle **0.675 → 0.553**；assignment-change **2.25 → 2.94**  
19. completed blocks？new0 **10.3125**，new1 **12.1875**，L0 **10.375**，old1 **8.4375**  
20. collision？**0 / 0**  
21. minimum separation worst？new0 **0.99999999956**，new1 **0.99999999935**（+1e-6 ≥ 1.0）  
22. projector failure？**0 / 0**  
23. relation 利用？neighbor/interference 仍 ~1e-9，**无实质变化**  
24. P2B Gate？**B**  
25. 是否支持重新进入 5-seed 独立验证？**否。** 停止；不调 beta；不训练另外 3 个 seed。
