# Phase-6I-AA：方案 C 正式 Planner 接入

Gate 为 **PLANNER_INTEGRATION_READY**。正式入口已经调用方案 C Epoch 1，接口回归、合法性检查、15 步 smoke 和两段完整 episode 都通过。这只说明决策链路可以运行。

`planner_ready = True`

`deployment_verified = False`

Phase-6I-Z 仍然是 **CLOSED_LOOP_GAIN_INCONCLUSIVE**。本阶段没有改写那份报告，也没有把方案 C 写成已经稳定优于 `G1_RATE k=6`。没有训练，没有读取 reserved test `322064–322079`。

## 1. 修改的文件

| 文件 | 作用 |
| --- | --- |
| `src/wt/benchmark_v2/phase6f_system_bringup/episode.py` | 增加 `run_active_episode`。原来的 `run_episode(..., "j15")` 仍使用 `G1_RATE k=5` 和 J15 |
| `src/wt/benchmark_v2/phase6iaa_formal_planner/protocol.py` | 正式策略、k=6 合同、Gate 和回滚配置 |
| `src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py` | 候选、方案 C 评分、TaskOption 和冻结 MAPPO 执行 |
| `src/wt/benchmark_v2/phase6iaa_formal_planner/audit.py` | 接口、smoke 和有限 episode |
| `tests/test_phase6iaa_formal_planner.py` | 非法分数、同分、保留集和特征维度 |

Phase-6F 的 `CANDIDATE_K` 仍是 5。当前正式循环不读这个常量，候选只来自 Phase-6I-Y 的 `frozen_candidate_keys`。

## 2. 调用链

`run_active_episode` 读取 `phase6iaa_formal_planner/active_policy.json`。

1. 用当前环境调用 `frozen_candidate_keys`，得到 `G1_RATE k=6` 的合法池，再按冻结的字典序绑定评分下标。
2. `high_level_policy = "scheme_c"` 时，`graph_from_env` 用与 `load_decision` 相同的节点、边和全局特征。边宽是 3，队列通道仍由评分函数追加一次。输入里没有 J15、未来 SINR、未来干扰或 rollout 后的交付量。
3. `SchemeCAdapter.infer` 对全部候选评分。同分时稳定排序保留更小的下标。选中下标必须落回同一条 assignment。
4. `make_option` 写成 hold 为 15 的 TaskOption。
5. 冻结 MAPPO-V2 的确定性均值经过原有 `SafetyProjector`，再调用原来的 `env.step`。中途不重新评分，也不更新网络。

评分失败、没有合法候选、出现 NaN/Inf 或 assignment 非法时，这次新决策被中止，不另选一行，也不退回 k=5。若当时没有仍在有效期内的 TaskOption，episode 停止。

`high_level_policy = "g1_rate"` 使用同一候选池上的冻结加和规则，不加载方案 C 做选择。

## 3. 候选池与 checkpoint

启动日志记录的哈希在审计前后没有变化。

| 对象 | SHA-256 |
| --- | --- |
| 方案 C Epoch 1 | `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864` |
| HGNN Epoch 6 | `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde` |
| MAPPO-V2 seed 0 | `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e` |
| Phase-6I-Y 候选配置 | `3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63` |

当前配置是 `high_level_policy = "scheme_c"`，生成器 `G1_RATE`，k = 6，hold = 15。本次每个决策的候选数都是 13327。池生成大约 0.22–0.26 秒，方案 C 评分大约 0.006–0.010 秒。

322000 的 t=0 现场图与 `load_decision` 一致，现场选择与保存图上的方案 C 分数绑定到同一条 assignment。322048 的 t=0 选择是下标 4853，assignment `[0, 2, 1, 3, 5, 4]`，与 Phase-6I-Y 记录相同。

## 4. 接口回归与 smoke

`tests.test_phase6iaa_formal_planner`、`tests.test_phase6f_system_bringup` 和 `tests.test_phase6iy_candidate_adapter` 共 12 项通过。空分数、NaN 和含非法行的分数不会返回下标。同分保留第一行。322064 被拒绝。历史 `CANDIDATE_K` 仍是 5。

Smoke 是 322000 和 322001 从 t=0 执行 15 步。两侧都没有非法 assignment、计账失败、非有限值或异常。payload 与逐步 `served_bits` 之和一致。

| 场景 | assignment | 交付 bit | 完成 block | 剩余队列 bit | 平均 SINR | 平均干扰 | SafetyProjector |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 322000 | `[0, 2, 1, 3, 5, 4]` | 15,000,000 | 15 | 24,000,000 | 0.159 | 0.078 | 5 |
| 322001 | `[2, 0, 1, 3, 5, 4]` | 15,000,000 | 15 | 24,000,000 | 0.350 | 0.044 | 5 |

这两个 15 步窗口的交付量正好等于 15 个 block。完整 episode 的交付量带有小数，用的是环境 `served_bits`，不是 `progress × 1e6`。

## 5. 完整 episode

完整 episode 只跑了训练集 322002 和验证集 322048。每段 48 步、4 次决策，时刻是 0、15、30、45。方案 C 实际评分 10 次：2 次 smoke 加 8 次完整决策。基线对照使用同一初始 reset 和同一个冻结 MAPPO，但 t=0 的 assignment 不同，之后的状态各自演化。这不是 Phase-6I-Z 那个固定 t=15 快照上的 15 步窗口。

方案 C：

| 场景 | 交付 bit | 完成 block | 剩余队列 bit | 平均 SINR | 平均干扰 | 干预步 | 耗时 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 322002 | 69,252,475.119 | 66 | 8,342,069.452 | 0.299 | 0.085 | 25 | 1.826 |
| 322048 | 69,937,810.666 | 67 | 4,700,084.214 | 0.473 | 0.042 | 26 | 1.924 |

`G1_RATE k=6`：

| 场景 | 交付 bit | 完成 block | 剩余队列 bit | 平均 SINR | 平均干扰 | 干预步 | 耗时 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 322002 | 63,363,363.458 | 60 | 7,379,667.280 | 0.213 | 0.064 | 23 | 2.225 |
| 322048 | 74,655,957.749 | 71 | 3,004,598.327 | 0.283 | 0.053 | 22 | 2.196 |

配对差是方案 C 减基线：322002 为 +5,889,111.661 bit，322048 为 -4,718,147.082 bit。一正一负。不能据此声称方案 C 的执行收益已经成立。322048 在 t=45 的基线 assignment 含有一个合法 idle：`[-1, 7, 8, 6, 9, 11]`。

四段完整 episode 和两段 smoke 都没有异常终止。逐步选择、候选数和耗时在 `phase6iaa_formal_planner/summary.json`。

## 6. 基线回滚

把 `phase6iaa_formal_planner/active_policy.json` 里的 `high_level_policy` 改成 `"g1_rate"` 后，再调用 `run_active_episode`。不需要改源码，也不需要重训。该配置仍使用 `G1_RATE k=6`，不会把方案 C 的分数套到 k=5 上。

审计结束时该文件是 `"scheme_c"`。

## 7. Gate 与状态

**PLANNER_INTEGRATION_READY**

`planner_ready = True`

`deployment_verified = False`

独立泛化验证和部署运行验证都还没有做。Phase-6I-Z 的置信区间仍跨过 0。下一步不要把这次接入当成部署通过，也不要读取 reserved test。

本阶段到此停止。
