# 下一阶段训练方案：将信息紧急程度纳入高层 HGNN 与低层 MAPPO

- 仓库：`ISAAC558345/waypoint-transformer`
- 分支：`feature/maritime-wt-stage1a`
- 目标：训练 urgency-aware 高层与低层候选，同时保留当前正式模型作为冻结基线。
- 本文件是执行计划，不表示代码已经修改或训练已经完成。

## 1. P0：先审计当前队列与标签语义

先检查：
- `src/wt/benchmark_v2/bit_queue_v2.py`
- `src/wt/benchmark_v2/phase6ib_highlevel_dataset/schema.py`
- `src/wt/benchmark_v2/phase6ib_highlevel_dataset/hgnn_revision.py`
- `src/wt/lowlevel_v2/observation.py`
- `src/wt/lowlevel_v2/critic_state.py`
- 高层 J15 标签生成、Scheme C 图特征打包和低层奖励计算。

当前环境用 `q_bits` 表示浮标聚合队列，服务已有队列后再接纳本步新到达；目前不能假定已经有逐包时间戳或 deadline。因此不能直接把 queue length 当成信息紧急程度，也不能凭空添加 deadline/SLA。

P0 必须明确：
1. `arrivals()` 的实际语义和服务是否可合理视为 FIFO；
2. 是否已有真实队列等待年龄、到达时刻或 deadline；
3. 高层标签是否只按 15 步 payload 生成；
4. 低层 Actor/Critic 目前从哪里获得目标浮标、队列与通信状态；
5. 所有需要改动的特征维度、数据集版本、标签和 checkpoint 接口。

交付 `docs/research/HTW_UrgencyAware_Training_Audit_CN.md`。审计完成前不启动长训练。

## 2. 定义紧急程度

### 首选：真实队列等待年龄

如果当前队列语义支持 FIFO，并且能准确记账，就按浮标维护年龄分桶的 bit/block 队列。每步按现有顺序服务旧数据、更新剩余数据年龄、接纳新到达。维护：
- `oldest_wait_age_k`：当前队列最老数据的等待步数；
- `mean_wait_age_k` 或 `age_weighted_backlog_k`；
- `urgency_valid_k`：是否有真实排队数据。

归一化定义为 `urgency_k = clip(oldest_wait_age_k / age_scale_steps, 0, 1)`，其中 `age_scale_steps` 必须在配置中明确并固定。年龄桶总 bit 数必须与 `q_bits` 一致，服务、到达、丢弃均需守恒测试。

如果聚合队列无法可信地表示 FIFO，就不要伪造真实年龄；可先定义清楚标记为 `queue_priority_proxy` 的代理指标，并说明它不等于真实等待时延或 deadline 紧急度。代理不能只重复现有 queue length 特征，必须验证其随所代表的等待程度单调变化。

## 3. 高层：Urgency-Aware HGNN

只增加输入特征不够。如果高层训练标签仍只最大化 payload，模型不一定会学会优先服务紧急信息。

实施要求：
1. 将已核实的 `urgency_k` 加入 buoy node feature；如加入 edge feature，明确其语义，并保证同一状态所有候选使用相同的当前 urgency。
2. 版本化 schema、特征维度、数据集 manifest、state hash 和 checkpoint metadata。不要静默改变旧模型输入维度。
3. 保留现有 payload-only 的 `J15_MAPPO_CURRENT` 标签作为对照，另建 urgency-aware 标签和数据集，不覆盖旧 shard。
4. urgency-aware 窗口效用可定义为 `U = normalized_payload + lambda_urgency * normalized_urgent_service`，其中 `urgent_service = sum_t sum_k urgency_k[t] * served_bits_k[t]`。尺度和 `lambda_urgency` 必须写入配置，只能在训练/开发验证池中选择。
5. 对 listwise/ranking 训练，使用 urgency-aware utility 生成候选排序标签；不能只改输入而继续使用与紧急程度无关的标签。
6. 不使用未来 arrivals、未来 served bits 或 reserved-test 信息构造当前状态特征。

## 4. 低层：Urgency-Conditioned Residual MAPPO

低层不负责重新分配目标，但需要根据当前任务目标的紧急程度调整运动执行，并通过真实服务收益学习如何尽快到达、有效服务。

实施要求：
1. 将目标浮标的 urgency 加入 Actor 的 task observation；idle task 必须通过 mask/validity 字段与真实 urgency=0 区分。
2. 将每架 UAV 当前目标 urgency 和必要的全局紧急度摘要加入 centralized Critic state。显式版本化输入 schema 和 checkpoint。
3. 在低层奖励加入 urgency-weighted served bits：
   `r_t = r_payload + lambda_urgency * sum_k(urgency_k[t] * served_bits_k[t]) / PAYLOAD_SCALE - lambda_project * c_project - lambda_residual * c_residual`
   urgency 必须在本步服务前根据当前可观测状态计算，并与本步 served_bits 的浮标索引对应。禁止使用未来状态回填奖励。
4. 先核对奖励各项的数值尺度，再确定固定的 `lambda_urgency`；至少保留 `lambda_urgency=0` 的原奖励对照。不能盲目加入大权重。
5. 保持动作结构 `fly_toward + residual -> clip -> SafetyProjector` 不变；除非后续独立证据说明动作结构本身有缺陷。
6. 新 checkpoint 写入独立目录。输入维度改变时必须显式记录 warm-start/权重迁移策略，禁止静默忽略不匹配参数。
7. 训练任务分布覆盖合法的 Scheme C、G1_RATE 及其他合法 assignment，记录分布和随机种子。

## 5. 训练顺序

### Stage A：环境状态和守恒测试
先实现真实 urgency 或明确标注的 proxy。测试年龄更新、服务先于到达、空队列无效标记、年龄桶与 q_bits 守恒，以及原有 payload/queue accounting。原有回归测试必须继续通过。

### Stage B：特征和标签
建立新版本的高层/低层 observation schema。验证 urgency 变化符合定义、无未来泄漏、输入维度一致。生成 payload-only 与 urgency-aware 两套高层标签，并报告标签排序差异。

### Stage C：Smoke test
高层、低层分别短程运行，检查 finite loss/gradient/action、合法 assignment、队列年龄守恒、payload 计账、checkpoint 保存和重载。失败则停止，不启动长训练。

### Stage D：分开训练
1. 先训练 urgency-aware 低层候选，冻结原 MAPPO-V2 作为对照。
2. 再训练 urgency-aware 高层候选，使用新特征和新标签；训练时低层执行器固定。
3. 两层分别通过评估后，才运行组合测试；不要一开始就联合训练。
4. 只有单层和组合收益稳定后，才考虑交替训练或联合微调。

## 6. 必须进行的完整闭环 Benchmark

所有组使用相同的场景、reset seeds 和环境配置：

| 组别 | 高层 | 低层 | 目的 |
|---|---|---|---|
| A | 原始 Scheme C | 原始 MAPPO-V2 | 冻结正式基线 |
| B | 原始 Scheme C | urgency-aware MAPPO | 隔离低层紧急度因素 |
| C | urgency-aware HGNN | 原始 MAPPO-V2 | 隔离高层紧急度因素 |
| D | urgency-aware HGNN | urgency-aware MAPPO | 测量组合及交互作用 |
| E | G1_RATE k=6 | 原始 MAPPO-V2 | 原有高层基线 |
| F | G1_RATE k=6 | urgency-aware MAPPO | 检查低层跨高层泛化 |

主指标是完整 48 步 episode 的实际累计 `payload_bits`。紧急度指标包括按 urgency 分组的 served bits、最老/平均等待年龄、高紧急度队列首次服务时间、剩余队列和 dropped bits。只有环境有真实 deadline 时才可报告 deadline miss rate。

辅助指标包括 SINR、干扰、服务步数、路径长度、SafetyProjector 触发率、安全违规率和不同训练 seed 的稳定性。以场景为配对单位，报告均值差、95% bootstrap CI、正负收益场景数；不把同一 episode 的时间步视为独立样本。reserved test 只用于最终一次确认，不得调参。

## 7. 验收门槛

- `URGENCY_STATE_AUDIT_COMPLETE`：真实等待年龄/代理语义与数据来源明确。
- `URGENCY_ACCOUNTING_TESTS_PASS`：年龄桶、队列、服务、到达和丢弃守恒。
- `URGENCY_FEATURE_LABEL_SMOKE_PASS`：高低层特征/标签版本化、无未来泄漏、短程训练通过。
- `URGENCY_LOWLEVEL_EVAL_COMPLETE`：低层新旧策略完整闭环对比完成。
- `URGENCY_HIGHLEVEL_EVAL_COMPLETE`：高层新旧策略在固定低层下对比完成。
- `URGENCY_CLOSED_LOOP_EVAL_COMPLETE`：六组完整闭环结果齐全。
- `URGENCY_GAIN_SUPPORTED` 或 `URGENCY_GAIN_INCONCLUSIVE`：依据预注册统计结果决定是否保留紧急度机制。

不得只凭训练 reward 上升就宣称改进。若 urgency-aware 策略提高紧急队列服务但降低总 payload，应如实报告这一权衡，不得只挑选有利指标。

## 8. 保护边界与交付

- 不覆盖当前正式 Scheme C、`phase6g_seed0.pt`、历史结果或旧数据集。
- 所有新数据集、模型、配置、日志放入独立版本目录。
- 先执行 P0 审计并报告结论，再进入 Stage A；不得未经审计直接大规模改环境和训练。
- 每阶段完成后汇报 Gate、diff、测试、模型/配置哈希和下一步建议，随后停止等待确认。
