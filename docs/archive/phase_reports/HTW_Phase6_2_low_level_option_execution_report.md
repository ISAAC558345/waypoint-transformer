# HTW Phase-6.2 Goal-conditioned Low-level Option Execution

在 Level-2-A 上只训练低层：输入 `[state, z]`（z 为 Oracle buoy one-hot），输出 movement。
未训 π_H、未加 HGNN、未用 Transformer / Task-Waypoint、未改环境、未改 PPO 核心、未开 Round。
与 Phase-6.1 相同：seed=0，80 updates，rollout=48，num_envs=4，训练种子 [0, 1, 2, 3, 4, 5, 6, 7]，评估种子 [8, 9, 10, 11, 12, 13, 14, 15]。

## 1. 为什么 6.1 低层执行不了 option

环境 `r_comm` 只在 **进入服务半径并连续占用 D=3** 后才给 1 块 payload。随机/未成形运动几乎拿不到该信号，PPO 看不到 z 与位移的因果。影子 fly-toward 用同一 z 能到 2.775e7，证明失败在低层执行，不在 option。
距离辅助把 `||p−b_z||` 换成步长单位（不是 waypoint tracking），completion 是 **是否进入目标 buoy 服务区**。

## 2. Reward 网格（仅 controller）

- A: `r_low = r_comm`
- B: `r_low = r_comm + λ_c · 1[in service radius of z]`
- C: `r_low = r_comm + λ_c · completion − λ_d · mean_uav(||p−b_z|| / vΔt)`

λ_c ∈ [0.1, 0.5, 1.0]，λ_d ∈ [0.01, 0.05]。评估 return 仍是环境块数和。

## 3. 结果

| variant | λ_c | λ_d | payload | env return | completion | service | movement m | t_in_range | t_service |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| state_only | 0.00 | 0.00 | 2875000 | 2.88 | 0.108 | 0.012 | 779.0 | 8.1 | 35.8 |
| A_r_comm | 0.00 | 0.00 | 500000 | 0.50 | 0.110 | 0.002 | 1348.1 | 22.6 | 45.2 |
| B_c0.1 | 0.10 | 0.00 | 10125000 | 10.12 | 0.316 | 0.042 | 732.5 | 12.9 | 29.5 |
| C_c0.1_d0.01 | 0.10 | 0.01 | 6375000 | 6.38 | 0.101 | 0.027 | 1037.6 | 20.4 | 31.8 |
| C_c0.1_d0.05 | 0.10 | 0.05 | 9750000 | 9.75 | 0.223 | 0.041 | 1363.6 | 16.8 | 27.1 |
| B_c0.5 | 0.50 | 0.00 | 5375000 | 5.38 | 0.775 | 0.022 | 1614.9 | 6.2 | 40.5 |
| C_c0.5_d0.01 | 0.50 | 0.01 | 5875000 | 5.88 | 0.702 | 0.024 | 1883.7 | 6.8 | 24.8 |
| C_c0.5_d0.05 | 0.50 | 0.05 | 10250000 | 10.25 | 0.586 | 0.043 | 2677.3 | 5.6 | 33.7 |
| B_c1.0 | 1.00 | 0.00 | 4500000 | 4.50 | 0.617 | 0.019 | 2861.8 | 3.8 | 37.4 |
| C_c1.0_d0.01 | 1.00 | 0.01 | 9500000 | 9.50 | 0.686 | 0.040 | 2321.3 | 4.3 | 34.3 |
| C_c1.0_d0.05 | 1.00 | 0.05 | 8000000 | 8.00 | 0.748 | 0.033 | 3036.3 | 4.8 | 34.8 |

Oracle option + fly-toward 影子 payload = 27750000。
**最佳低层**：`C_c0.5_d0.05`（λ_c=0.5, λ_d=0.05），payload 10250000，completion 0.586，service 0.043。
相对 State-only payload 差 = 7375000 bits。

## 4. 三个问题

1. **原低层失败原因：** 稀疏 `r_comm` 与 D=3 占用不同步；没有进入目标区域的中间信号，z 的 one-hot 无法被运动策略利用。
2. **哪种 goal reward 有效：** **B 与 C 有效，A 无效。** 最优 payload 是 C（λ_c=0.5, λ_d=0.05）= 10250000；B（λ_c=0.1）几乎同样高且位移更短（732 m vs 2677 m）。A 仅 `r_comm` 仍约 5e5，复现 6.1 失败。
3. **是否进入 learned high-level policy：可以进入（有保留）。** 成功标准：Option 低层 payload 明显高于 State-only（差 ≥ 1000000 bits）且 service success > 0.002。当前差 7.375e6、service 0.043。低层仍明显低于 fly-toward 影子 2.775e7（约 37%），π_H 可以开始，但低层执行尚未饱和。

未训练 π_H。

JSON：`src/wt/htw_phase62/outputs/goal_execution/summary.json`
