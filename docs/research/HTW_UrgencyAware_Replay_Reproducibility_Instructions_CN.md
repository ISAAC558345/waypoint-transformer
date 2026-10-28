# Stage E7：冻结策略评估复现误差定位指令

- 仓库：ISAAC558345/waypoint-transformer
- 正式基线分支：feature/maritime-wt-stage1a
- 来源实验分支：experiment/priority-tag-urgency-reward
- 来源诊断分支：experiment/priority-tag-reward-attribution
- 当前诊断分支：experiment/priority-tag-replay-repro
- 来源诊断提交：51adade38d8f3da5402e67506cf875d80224e82d
- 前置 Gate：URGENCY_REWARD_ATTRIBUTION_BLOCKED
- 本阶段 Gate：REPLAY_DIVERGENCE_LOCALIZED / REPLAY_DIVERGENCE_INCONCLUSIVE / REPLAY_DIVERGENCE_BLOCKED
- 本阶段仅定位复现差异；不训练、不调奖励、不扩大评估、不合并任何分支。

## 1. 已知问题

E6 复核了 224 条已保存场景记录，summary 统计对账一致，age+tag 增量与 combined urgency 的分解误差约为 0。但在冻结 L0 seed 0 的场景 321101 重放时：
- payload 差 1.553167 bit，超过固定 1 bit 容差；
- urgency_served_bits 差 0.135255 bit；
- served_tagged_bits 差 0.262044 bit；
- served_ordinary_bits 差 1.291123 bit；
- dropped_ordinary_bits 差 1.494478 bit；
- path_length 差约 5.745e-6 m。
绕过归因包装、直接走原评估路径，也出现相同差异。因此尚不能归因于新增 instrumentation。

不放宽 1 bit 容差，不把现有部分重放用于年龄/tag 归因。

## 2. 目标

定位已保存评估和当前重放之间的首次差异发生在哪个步骤、哪个状态字段，以及差异来源属于：
A. 代码/配置/依赖版本不一致；
B. reset、外生 arrivals、随机数状态或 seed 不一致；
C. 动作张量、动作变换或 SafetyProjector 的数值差异；
D. 环境队列更新顺序、float snap、served/dropped 计算或累计求和差异；
E. 仅输出汇总精度/序列化造成的差异；
F. 尚不能确定。

## 3. 严格范围

- 先复用 E6 已保存的重放输出、来源评估日志和所有 hash，不立即重跑 224 个 episode。
- 仅检查 L0 seed 0、场景 321101 的一条 48-step episode；如果源日志缺少逐步字段，最多追加一次受控重放。
- 使用来源报告记录的冻结 L0 checkpoint SHA-256：92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e。
- 通过 TRAIN/DEV registry 和 split guard 核实 321101 的实际 split；不得访问 reserved/test。
- 在运行前记录 git commit、工作树状态、环境配置、Python/PyTorch/NumPy 版本、CPU/GPU deterministic flags、所有相关代码文件 SHA-256、checkpoint SHA-256 和 seed 状态。
- 不训练 L1/L2，不更改任何模型权重、reward、action、tag config、queue order、SafetyProjector 或评估协议。
- 不放宽差异阈值；若无法从保存日志精确比较，明确记录缺失字段，不猜测。

## 4. 必须执行的检查

### A. 来源完整性
核对来源实验提交 f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb 中的 checkpoint、评估 runner、环境配置和代码文件 hash，与当前诊断分支的对应文件是否完全一致。检查 checkpoint 加载方式、模型处于 eval/deterministic 模式、观测维度/归一化参数、环境初始化和配置默认值。

### B. 首次分叉定位
对 scenario 321101，逐步记录并对比：
- reset 后完整关键状态 hash；
- 每步 observation hash/关键数值；
- actor 原始输出、裁剪后动作、SafetyProjector 输入/输出；
- UAV 位置、目标/任务关联；
- q_bits、队列容量、每步真实 arrivals/admitted/dropped/served；
- float snap、ledger 桶总量；
- SINR、干扰、终止条件；
- 每步 payload 累计和 path-length 增量。
报告第一项不一致发生的 step index、字段、绝对差和该差是否足以解释后续 divergence。

### C. 随机与浮点来源
- 检查 Python/NumPy/Torch/environment RNG 是否在 reset 时按同样方式初始化；
- 检查是否有依赖全局 RNG 的调用、候选/字典迭代顺序、并行环境调度；
- 检查 float32/float64 转换、float snap 和累计求和顺序；
- 不通过四舍五入或放宽容差来“修复”差异。

### D. 控制试验
若有足够源日志，先仅比较日志，无需运行。
若必须重放，先用同一 commit/依赖/配置运行一次未包装的原评估路径，再用同样设置运行 instrumentation 路径；两个路径均与保存结果比较。若两者都与保存结果存在同样差异，继续查环境/依赖/版本来源，不要把原因归到 instrumentation。

## 5. 成功标准

只有在以下条件全部满足时，Gate 才能为 REPLAY_DIVERGENCE_LOCALIZED：
- 找到首次分叉步骤及其可验证的直接原因；
- 解释为什么原始路径和 instrumentation 路径出现同样差异；
- 能在不放宽容差、不改模型参数的情况下，复现保存结果，或证明保存结果缺少必要运行环境元数据而无法精确复现；
- 正式 checkpoint、正式代码和历史结果没有被修改。

如果只定位到“差异很小”但没有可验证原因，Gate 为 REPLAY_DIVERGENCE_INCONCLUSIVE。若来源代码/hash、状态日志或 split 无法核实，Gate 为 REPLAY_DIVERGENCE_BLOCKED。

## 6. 交付物

- docs/research/HTW_UrgencyAware_Replay_Reproducibility_Report_CN.md
- results/phase6i_priority_tag_replay_repro/summary.json
- results/phase6i_priority_tag_replay_repro/step_diff.csv
- results/phase6i_priority_tag_replay_repro/schema.json

报告应包含首次 divergence、逐字段差异、来源 hash、依赖版本、执行耗时、容差判定、原因证据和下一步建议。所有新文件只提交到当前诊断分支。

## 7. 停止规则

- 不继续完整 224-episode attribution replay，除非本阶段先证明单场景重放可在 1 bit 容差内复现。
- 不重新训练、不调 lambda/rho/beta/tag_rate、不修改模型。
- 不合并来源实验分支或诊断分支。
- 完成后停止，等待后续自动规划；若 Gate 为 INCONCLUSIVE/BLOCKED，下一步仍是修复复现证据，不是继续解释年龄与标签贡献。
