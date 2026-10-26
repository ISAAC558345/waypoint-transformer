# E1 报告：优先级桶账本原型与奖励尺度

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`445ab1f5999279fba969a968ae115fd44d703f8e`
- 前置 Gate：`URGENCY_PRIORITY_SCHEMA_BLOCKED`
- 账本 Gate：`PRIORITY_LEDGER_PROTOTYPE_PASS`
- 尺度 Gate：`URGENCY_REWARD_SCALE_READY`
- 本阶段 Gate：`URGENCY_REWARD_SCALE_READY`
- `lambda_u` 最终值：未选。`lambda_selected = null`
- 性质：独立原型和 16 步脚本化 smoke。没有训练，没有完整 benchmark，没有修改正式 Planner、奖励、动作、SafetyProjector、checkpoint 或旧数据集。

## 1. 直接结论

合成 tag 能复现。每个正到达块只有一个二元标签。服务、丢弃和剩余队列按 tag 分开，并与正式 `served_bits`、`generated/admitted/dropped` 和 `q_bits` 对齐。最大账本差距是 9.313e-10 bit。打开账本与不打开账本时，正式状态指纹差是 0。

短程 smoke 实际服务了普通 bit 和 tag=1 bit。`lambda_u=0.05` 只作为高风险参照。候选 0、0.0005、0.001、0.002 的 bonus 按这次 tag-aware 服务量核算，并与已保存惩罚探针以及单块解析上界比较。本阶段不选最终权重。

## 2. 实际字段

正式 `L3BBitQueueEnvV2.step` 返回的相关字段是 `served_bits`、`generated_arrival_bits`、`admitted_arrival_bits`、`dropped_bits_step`。返回字典里没有 `float_snap_bits`。snap 使用同一步 `cumulative_snap_bits` 的增量，来源记为 `cumulative_snap_bits_delta`。账本不调用 `arrivals_at`，也不重算转移。

## 3. 合成 tag

- `synthetic_priority_tag` 为 0 或 1。日志和本报告都标明 synthetic。
- `tag_rate=0.25`，`beta_tag=0.25`，种子 `SeedSequence([606, scenario_id, admission_step, buoy])`。
- reset 存量是 tag=0、`age_origin=reset_convention`、`admission_step=0`。
- `u=clip(clip(age/48,0,1)+0.25*tag,0,1)`，在扣减服务之前计算。
- FIFO 键是 `admission_step` 和插入顺序。tag 不参与排序。

Smoke 场景 321000，16 步。UAV 0–2 在服务半径内悬停服务近端浮标，半径外才飞向对应远端浮标。UAV 3–5 飞向远端浮标，进入半径后悬停。正式 `step` 先移动再服务，所以半径内必须给零位移，否则一步 22 m 会在服务前离开 16 m 半径。这不是 MAPPO，也不是正式 Planner。

16 步累计：普通服务 23,218,883.687 bit，tag=1 服务 5,764,427.683 bit，urgency-weighted 服务 3,598,256.779 bit。到达块抽样 36 次，其中 tag=1 为 10 次。接纳普通 21,534,650.261 bit，接纳 tag=1 10,000,000.000 bit。丢弃普通 4,465,349.739 bit，丢弃 tag=1 0 bit。结束时剩余普通 13,315,766.573 bit，剩余 tag=1 4,235,572.317 bit，正式 `q_bits` 总和 17,551,338.890 bit。重复运行的指纹差是 0，tag 序列一致：True。

## 4. 守恒

每步检查 `served_ordinary + served_tagged = served_bits`，`generated = admitted + dropped`，接纳和丢弃也按 tag 相加回到正式向量。桶内 bit 之和与 `q_bits` 的差不超过 1 bit。正 snap 从最老桶扣，不计入 `urgency_served_bits`。本步新到达在服务扣减之后入桶。丢弃只从本步新到达桶扣除，容量不足时可以只丢弃该块的一部分，整块仍只有一个 tag。本段 snap 合计 0 bit。

## 5. 奖励尺度

已保存探针 `results/phase6i_urgency_lowlevel/smoke_summary.json` 的 L1 一次更新：payload 项均值 1.470713，正式奖励均值 1.470543，投影与残差惩罚合计 0.000170。这条探针没有把投影项和残差项拆开，也不是 tag-aware 数据。旧 buoy-level urgency 比例没有进入下表。

短程 smoke 的 payload 项均值 1.811457，urgency 项均值 0.224891。样本只有 16 步，P95/P99 只描述这 16 步，不是训练分布。

| lambda_u | 角色 | bonus 均值 | P50 | P95 | P99 | 最大 | 相对 payload 最大比例 | 相对已记录惩罚最大比例 | 不等 payload 排序 | 减去惩罚后的符号变化步数 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 候选 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 未改变 | 0 |
| 0.0005 | 候选 | 0.000112 | 0.000112 | 0.000231 | 0.000315 | 0.000336 | 0.000118 | 1.978069 | 未改变 | 0 |
| 0.0010 | 候选 | 0.000225 | 0.000224 | 0.000463 | 0.000630 | 0.000671 | 0.000235 | 3.956139 | 未改变 | 0 |
| 0.0020 | 候选 | 0.000450 | 0.000447 | 0.000925 | 0.001259 | 0.001343 | 0.000471 | 7.912277 | 未改变 | 0 |
| 0.0500 | 高风险参照 | 0.011245 | 0.011180 | 0.023130 | 0.031478 | 0.033565 | 0.011768 | 197.806932 | 未改变 | 0 |

相对 payload 的比例是该步 bonus 除以该步 payload 项。相对惩罚的比例用的是上面那条已保存惩罚合计，不是同一次 smoke 的投影代价，因为这条脚本没有 Actor，也没有调用 SafetyProjector。`ceiling_over_recorded_penalty` 在 summary 里，按 `lambda_u * 探针 payload 项 / 探针惩罚` 计算，表示该探针上 urgency=1 时的上界。

`step_order_changed` 记录完整排序是否变化。表中的“不等 payload 排序”只统计 payload 项相差超过 1 bit 的步是否被对调。非零权重只改变 payload 项完全相同的步的先后，用来打破平局，没有改变不等 payload 的顺序。

减去同一条惩罚后，符号是否变化已写入表中。bonus 本身非负，所以不会把正的 payload 项变成负值。

单块解析界限不依赖这条飞行路径。一块 1e6 bit、tag=1、服务时年龄为 0 的信息，`u=0.25`，bonus 等于 `lambda_u * 0.25`。一块年龄达到 48 的信息 `u=1`，bonus 等于 `lambda_u`。bonus 的一般上界是 `lambda_u * served_bits / 1e6`。

| lambda_u | 角色 | 新到达 tag=1 一块的 bonus | 年龄 48 一块的 bonus | 新 tag 块 / 已记录惩罚 | 年龄 48 块 / 已记录惩罚 |
|---|---|---|---|---|---|
| 0 | 候选 | 0 | 0 | 0 | 0 |
| 0.0005 | 候选 | 0.000125 | 0.000500 | 0.736652 | 2.946607 |
| 0.0010 | 候选 | 0.000250 | 0.001000 | 1.473304 | 5.893215 |
| 0.0020 | 候选 | 0.000500 | 0.002000 | 2.946607 | 11.786430 |
| 0.0500 | 高风险参照 | 0.012500 | 0.050000 | 73.665186 | 294.660743 |

## 6. 测试与未改文件

单元测试覆盖 reset、tag 复现、FIFO、同一步到达不进服务、整块丢弃、部分丢弃、snap、capture/restore、空队列标记，以及这次 smoke 与正式转移一致。正式文件哈希写在 `summary.json` 的 `watched_sha256`。本阶段新增的是 `src/wt/benchmark_v2/phase6i_priority_tag/`、`tests/test_priority_tag_prototype.py`、`results/phase6i_priority_tag_prototype/` 和本报告。

## 7. 限制与停止

16 步脚本不能代表开发池的奖励分布，也不能单独决定 `lambda_u`。0.05 仍不是训练候选。下一阶段要先复核本报告，再由负责人确认最终 `lambda_u`，然后才做 observation/reward builder 和训练。现在不训练 L1/L2，不改正式奖励，不跑完整 benchmark。
