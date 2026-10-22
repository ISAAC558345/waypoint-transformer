# Phase-10.2 指标口径

## Phase-10.1 in_range_active_frac（历史实现，未改）
- 分母：assignment >= 0 的 UAV-时隙（排除 idle）。
- 不排除无效目标（目标越界会在 env 中改成 idle）。
- **不排除空队列**。空队列仍可计入 in-range。
- 距离在 **env.step 之后** 计算。
- r_aux 在 step 之后、针对本步已执行 option 计算。
- progress_reset：任意 rho>0 变为 ~0，**包含正常完成清零**。

## 本阶段补充指标
- raw_in_range_frac：所有 UAV-时隙，含 idle（idle 的 in-range 恒为 False）。
- active_in_range_frac：非 idle。
- active_queued_in_range_frac：非 idle 且 step 开始时目标 queue>0。

## Payload ratio
- mean(B1)/mean(B0) = 0.8529411764705882
- mean((B1-B0))/mean(B0) = -0.14705882352941177
- mean_i(B1_i/B0_i) = 0.8663194444444444（Phase-10.1 `mean_ratio`）
