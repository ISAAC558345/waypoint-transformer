# Phase-6I-Y：正式入口候选池统一与一致性验证

Gate 为 **CANDIDATE_POOL_ALIGNMENT_SUPPORTED**。独立适配层在 51/51 个状态上复现了冻结的 `G1_RATE k=6` 池：候选集合、逐行内容和评分索引都与方案 C 读取的候选文件相同。正式 planner 仍是 `G1_RATE k=5`，没有被修改，也没有被调用。

`planner_ready = False`

`deployment_verified = False`

## 修改文件清单

新增文件：

- `src/wt/benchmark_v2/phase6iy_candidate_adapter/protocol.py`
- `src/wt/benchmark_v2/phase6iy_candidate_adapter/adapter.py`
- `src/wt/benchmark_v2/phase6iy_candidate_adapter/audit.py`
- `tests/test_phase6iy_candidate_adapter.py`
- `phase6iy_candidate_adapter/summary.json`

没有修改 `build_pools`、`enumerate_legal`、`G1_RATE` 的评分公式、idle 语义、合法性规则或 tie-break。没有修改 Phase-6F 的 `CANDIDATE_K = 5`。没有改写任何 HGNN、方案 C 或 MAPPO checkpoint。

## 候选生成调用链

适配入口是 `generate_frozen_pool`。它不导入正式 planner。

1. `phase6ix_candidate_alignment.audit._restore` 把冻结快照中的位置、队列和 association 写回环境对象。不调用 `step`。
2. `phase6e.generators.build_pools(env, (("G1_RATE", 6),))`。k 来自 Phase-6I-B 的 `SELECTED_K = 6`。
3. `jobs._score_current` 里的同一句：`keys = sorted({_key(item) for item in raw})`。`_key` 直接从 `jobs.py` 导入。
4. 每个 key 再经原有 `is_legal` 检查。非法计数不为 0 时按原路径抛出 `ILLEGAL`，不另写过滤器。
5. `CandidateBinding` 把 `score_index` 和完整 assignment 绑在一起。`assignment_at` 与 `score_index_of` 必须互相指回，不能只靠调用方假设行号。

`G1_RATE` 的加和选择仍是排序后的 `np.argmax`，与冻结 manifest 的 `additive_assignment` 使用同一规则。

## k=6 配置来源

配置来自现有代码，不是根据上一份报告重写的：

| 项 | 来源 | 值 |
| --- | --- | --- |
| 生成器 | `phase6ib_highlevel_dataset.protocol.GENERATOR` | `G1_RATE` |
| k | `SELECTED_K` | 6 |
| 边分数 | `proposal_scores` 的归一化瞬时假设速率 | 未改 |
| 名次 | `top_k` 的稳定 mergesort | 速率相同时保留更小浮标编号 |
| idle | `phase6d.pool.enumerate_legal` | 每架 UAV 都可 idle，idle 不占浮标 |
| 合法性 | `is_legal` | 拒绝越界目标和重复浮标 |
| 存储顺序 | `jobs._score_current` | 去重后的字典序 |
| 正式 planner | `phase6f_system_bringup.protocol.CANDIDATE_K` | 仍是 5 |

配置规范化 JSON 的 SHA-256 是 `3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63`。审计前后没有变化。

## 51 个状态的逐项一致性

检查范围与 Phase-6I-X 相同：训练 322000 的 t=0、15、30，以及验证 322048–322063 的 48 个状态。快照重算速率与保存的 `uav_buoy_hypothetical_rate` 最大绝对差是 0。状态哈希与冻结 manifest 的 `state_hash` 在 51/51 个状态上一致。

| 检查 | 通过 |
| --- | --- |
| 候选总数 13327 | 51/51 |
| 集合一致 | 51/51 |
| 顺序一致 | 51/51 |
| 评分索引绑定 | 51/51 |
| 同一输入重复生成 | 51/51 |
| 方案 C 分数落到绑定的 assignment | 51/51 |
| 与 Phase-6I-W 已记录的验证集方案 C 选择一致 | 48/48 |
| `additive_assignment` | 51/51 |

枚举结果在去重前后都是 13327 行，没有被排序额外删掉候选。非法行是 0。每个状态有 12607 行含 idle，其余 720 行是不含 idle 的完整匹配。

集合检查用 assignment 元组的集合比较。顺序检查用原始行逐项比较，没有先把冻结文件排序后再宣布索引一致。修订候选文件、冻结候选文件和 `load_decision` 返回的矩阵三者一致。

方案 C 的选择由 `SchemeCAdapter.infer` 对适配层矩阵打分。选中下标上的绑定 assignment 与分数向量给出的 assignment 相同。例如 322048 t=0 的选中行是索引 4853，assignment 为 `[0, 2, 1, 3, 5, 4]`，与 Phase-6I-W 的记录相同。没有第一个差异位置。

## checkpoint 与配置哈希

审计结束后哈希没有变化：

| 文件 | SHA-256 |
| --- | --- |
| 方案 C Epoch 1 | `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864` |
| HGNN Epoch 6 | `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde` |
| MAPPO-V2 | `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e` |
| 适配配置 | `3535008ed197f28ed3620a25493a9cc17fbb1fca9742d49a3d58957883319a63` |

进程工作集是 354,312,192 字节。环境步数是 0，J15 没有重算。reserved test `322064–322079` 没有读取。

## Gate 及未解决问题

**CANDIDATE_POOL_ALIGNMENT_SUPPORTED**。

尚未解决的问题：

- 正式 planner 的决策逻辑和 `CANDIDATE_K = 5` 都没有切换。适配层还没有接到 planner 上。
- 方案 C 与 `G1_RATE k=6` 基线在冻结 MAPPO 下的执行收益还没有评估。本阶段的一致性不是执行收益。
- `planner_ready = False`，`deployment_verified = False`。

下一阶段使用的接口是 `generate_frozen_pool`。它返回带 `score_index` 的 `FrozenCandidatePool`。方案 C 只能对 `assignment_matrix()` 打分，再用 `assignment_at` 取回 assignment。

## 下一阶段唯一建议

只评估方案 C 与 `G1_RATE k=6` 加和基线在冻结 MAPPO-V2 下的实际执行收益。候选必须来自这个适配层，不能回到 k=5，也不能把适配层接进正式 planner。

本阶段到此停止。不接入正式 planner，不运行闭环 rollout，不宣称部署验证通过。
