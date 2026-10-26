# 实验报告：合成 priority-tag 的有界紧急奖励

- 分支：`experiment/priority-tag-urgency-reward`
- 基线分支：`feature/maritime-wt-stage1a`
- Gate：`PRIORITY_TAG_EXPERIMENT_SMOKE_PASS`
- 是否合并：否
- 标签性质：synthetic priority tag，不是真实海事业务标签。

## 1. 预注册设置

本报告不改验收标准。L1 只增加观测，奖励仍是正式 `r_base`。L2 使用同一观测，并加上预注册候选：

`r = r_base + min(0.0005 * urgency_served_bits/1e6, 0.001 * served_bits/1e6)`

`r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual`。`u = clip(clip(age/48,0,1)+0.25*synthetic_tag,0,1)`，在服务前读取。服务量为 0 时 bonus 为 0。

训练种子是 0、1、2。优化器、rollout、更新次数沿用正式 MAPPO：300 次更新，4 个环境，每次 48 步。初始化是 legacy checkpoint 的显式移植，新输入列权重为 0，不是静默部分加载。L0 使用冻结的 `phase6g_seed0.pt`，不重新训练。

合并需要至少 2 个 L2 种子同时满足：紧急度加权服务和 tagged 服务的配对 95% 下界都大于 0；payload 配对均值不小于 0 且下界不小于 0；安全违规和约束失败不增加；丢弃、SINR、干扰的配对均值不变差；紧急度增益和 payload 结论不被单个场景带动。

## 2. Smoke

Smoke Gate：`PRIORITY_TAG_EXPERIMENT_SMOKE_PASS`。首个场景用时 0.561010 秒，总用时 1.745167 秒。问题：无。

## 3. 正式文件哈希

- `src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt`：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`
- `src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py`：`79fc576924aaabc74df7b5badb4f23cd4fbbba915162ca646e6b819855f425b3`
- `src/wt/lowlevel_v2/observation.py`：`7311f0ff5c6f7ef9367fce7eba011bace123dfd9685f967e6794ad13df3dc679`
- `src/wt/lowlevel_v2/residual_policy.py`：`fabd7fff68d575b74de4e0be67d6a35e60990e10d82f135269a86664dcfe6927`
- `src/wt/lowlevel_v2/safety_projector.py`：`75478f547b5741b85735143cf2a0301bcd9b281d4dcd49c355e615a7e84fa6d7`
- `src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py`：`dc67c0402a273400fc5490f72b0d8fda34713a0baadb1005de06f17e3be8bf9a`
- `results/phase5_hgnn_urgency/checkpoints/mappo_original.pt`：`44d44cf96f666aa7810bcd9b6546e2956d35fbf280e5f312c983444b25dbce14`

正式 L0 checkpoint：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。

## 4. 训练

Smoke 已通过。L1/L2 训练尚未开始，预算仍是 300 次更新、种子 0/1/2。

## 5. 配对评估

评估使用登记的 development pool，确定性动作，同一个场景重置种子。没有读取 reserved 或 test split。

### L2

尚无评估。

### L1

尚无评估。

L2 通过种子数：缺失。L1 通过种子数：缺失。

## 6. 合并决定

评估尚未完成，不合并。

固定轨迹上的奖励上升不能代替这些标准。正式分支、原始 checkpoint 和 E0–E4 结果保持不变。
