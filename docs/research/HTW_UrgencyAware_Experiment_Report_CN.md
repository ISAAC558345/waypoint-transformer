# 实验报告：合成 priority-tag 的有界紧急奖励

- 分支：`experiment/priority-tag-urgency-reward`
- 基线分支：`feature/maritime-wt-stage1a`
- Gate：`PRIORITY_TAG_EXPERIMENT_NOT_MERGED`
- 是否合并：否
- 标签性质：synthetic priority tag，不是真实海事业务标签。

## 1. 预注册设置

本报告不改验收标准。L1 只增加观测，奖励仍是正式 `r_base`。L2 使用同一观测，并加上预注册候选：

`r = r_base + min(0.0005 * urgency_served_bits/1e6, 0.001 * served_bits/1e6)`

`r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual`。`u = clip(clip(age/48,0,1)+0.25*synthetic_tag,0,1)`，在服务前读取。服务量为 0 时 bonus 为 0。

训练种子是 0、1、2。优化器、rollout、更新次数沿用正式 MAPPO：300 次更新，4 个环境，每次 48 步。初始化是 legacy checkpoint 的显式移植，新输入列权重为 0，不是静默部分加载。L0 使用冻结的 `phase6g_seed0.pt`，不重新训练。

合并需要至少 2 个 L2 种子同时满足：紧急度加权服务和 tagged 服务的配对 95% 下界都大于 0；payload 配对均值不小于 0 且下界不小于 0；安全违规和约束失败不增加；丢弃、SINR、干扰的配对均值不变差；紧急度增益和 payload 结论不被单个场景带动。

## 2. Smoke

Smoke Gate：`PRIORITY_TAG_EXPERIMENT_SMOKE_PASS`。首个场景用时 0.390256 秒，总用时 1.518398 秒。问题：无。

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

L1 seed 0 sha256 cedc9bc17b448639793a1d75d1a3c4681b7d060eb2659dfce5a66f2f675ac9c3 用时 776.9 秒；L1 seed 1 sha256 b1574ddaec7ba4006dbcba0999031a60eea8e91323ac4c68f29ae08b8530aee7 用时 749.6 秒；L1 seed 2 sha256 ee13fe4ba0b6fd4a9354b798eb1cd27ff5c4fb4f32459d1e3c5745c4c89f3583 用时 771.6 秒；L2 seed 0 sha256 4a84fdc6db2f14b8dd29c57efcc861194ec3fbbc9e1e52c3047f8e2d77aab41f 用时 803.8 秒；L2 seed 1 sha256 0a8f113de24a76a43bfd29246c39f3c6cd8dff85d9744b28a9d30f0c5c34eb7a 用时 1331.8 秒；L2 seed 2 sha256 55fe2acece8772f1bdbf4e02b228781c91892fc1228c292f7c86e5e78a1bea8c 用时 800.1 秒

## 5. 配对评估

评估使用登记的 development pool，确定性动作，同一个场景重置种子。没有读取 reserved 或 test split。

### L2

- L2 seed 0：通过=否，紧急度差均值 789,703.963938，95% 区间 [-65,513.437760, 1,623,717.556907]，tagged 差均值 -239,099.026146，95% 区间 [-805,992.393062, 327,646.581698]，payload 差均值 712,169.002222，95% 区间 [-794,255.837554, 2,117,444.267246]，payload 变差场景 12，安全差 0，约束失败差 0，丢弃差均值 200,707.914129，SINR 差均值 0.869937，干扰差均值 0.002602。
- L2 seed 1：通过=否，紧急度差均值 757,413.910022，95% 区间 [92,595.796054, 1,387,433.787436]，tagged 差均值 -346,001.878918，95% 区间 [-982,554.265842, 319,650.360926]，payload 差均值 1,411,400.211571，95% 区间 [-126,268.022709, 2,858,177.802901]，payload 变差场景 7，安全差 0，约束失败差 0，丢弃差均值 -537,165.106837，SINR 差均值 0.385900，干扰差均值 0.001426。
- L2 seed 2：通过=否，紧急度差均值 882,556.445550，95% 区间 [-127,194.703980, 1,843,584.827901]，tagged 差均值 -154,433.470984，95% 区间 [-989,581.453090, 628,951.791625]，payload 差均值 565,664.707532，95% 区间 [-1,395,309.894661, 2,498,051.262972]，payload 变差场景 14，安全差 2.000000，约束失败差 0，丢弃差均值 -83,982.222224，SINR 差均值 0.078538，干扰差均值 -0.000666。

### L1

- L1 seed 0：通过=否，紧急度差均值 1,194,793.908404，95% 区间 [517,431.370135, 1,903,423.218486]，tagged 差均值 -8,649.995762，95% 区间 [-719,839.406790, 667,529.733246]，payload 差均值 965,549.291373，95% 区间 [-530,797.517816, 2,547,738.361135]，payload 变差场景 11，安全差 0，约束失败差 0，丢弃差均值 265,385.309218，SINR 差均值 1.391338，干扰差均值 0.004505。
- L1 seed 1：通过=否，紧急度差均值 706,236.659528，95% 区间 [-172,626.283722, 1,529,279.586965]，tagged 差均值 69,363.106146，95% 区间 [-708,260.797682, 786,458.387878]，payload 差均值 764,066.304416，95% 区间 [-773,627.285070, 2,218,590.711507]，payload 变差场景 10，安全差 0，约束失败差 0，丢弃差均值 -252,663.373793，SINR 差均值 0.655716，干扰差均值 0.001054。
- L1 seed 2：通过=否，紧急度差均值 1,317,534.855462，95% 区间 [464,433.631446, 2,202,049.333729]，tagged 差均值 186,135.146737，95% 区间 [-650,014.279784, 1,031,776.472653]，payload 差均值 2,666,318.627394，95% 区间 [550,922.772101, 4,659,131.985751]，payload 变差场景 8，安全差 1.000000，约束失败差 0，丢弃差均值 -784,429.036955，SINR 差均值 0.314915，干扰差均值 0.002881。

L2 通过种子数：0。L1 通过种子数：0。

## 6. 合并决定

结果未达到预注册标准或统计上不充分，保留实验分支，不合并。

固定轨迹上的奖励上升不能代替这些标准。正式分支、原始 checkpoint 和 E0–E4 结果保持不变。
