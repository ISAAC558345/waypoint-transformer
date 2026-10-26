# D6 状态核查与结果推送指令

- 仓库：ISAAC558345/waypoint-transformer
- 分支：feature/maritime-wt-stage1a
- 计划依据：docs/research/HTW_UrgencyAware_ArrivalAware_Expansion_Instructions_CN.md
- 当前已知状态：远端最新提交仍为 D6 方案提交 `327eb05376b2e1f62581e09526e5a9b0bd782fe2`；尚未在该分支找到 D6 报告或结果文件。
- 本任务不是新实验计划，不授权重跑已完成 rollout。

## 1. 先核查，不要重复运行

1. 检查当前工作分支、HEAD commit、git status，以及本地是否已有未提交/未推送的 D6 结果。
2. 检查以下文件是否在其他本地路径、其他分支或未提交状态：
   - `docs/research/HTW_UrgencyAware_ArrivalAware_Expansion_Report_CN.md`
   - `results/phase6i_urgency_arrival_aware_expansion/selected_candidates.csv`
   - `results/phase6i_urgency_arrival_aware_expansion/labels.csv`
   - `results/phase6i_urgency_arrival_aware_expansion/summary.json`
   - `results/phase6i_urgency_arrival_aware_expansion/schema.json`
3. 若 D6 已成功完成且结果可信，核对实际场景 ID、时刻、候选/rollout 数、耗时、状态/候选哈希、模型与配置 SHA-256、split registry、旧 J15 复现、arrival 配对、ledger/q_bits 守恒和 payload accounting。
4. 若结果已经存在且通过核查，不要重新运行；只补齐缺少的报告/summary/schema，保留原始结果，并将必要文件提交到 `feature/maritime-wt-stage1a`。
5. 如果结果在其他分支，先核实来源与 commit，不要直接覆盖当前分支已有文件；将其 cherry-pick/合并到当前分支仅在无冲突且来源可验证时进行。
6. 如果 D6 未运行，或只有不完整日志，按 D6 原方案执行，但严格遵守其 20 分钟总预算、每状态最多 128 个候选、8 个登记训练场景和四个决策时刻。先跑一个状态估算时间，超过预算就停止并记录 BLOCKED，不得扩容。
7. 如果由于环境、registry、状态恢复或计账问题无法完成，提交一份明确的 BLOCKED 报告，说明真实原因和已完成的工作；不能伪造结果或把计划文档当成执行报告。

## 2. 禁止事项

- 不读取 validation 或 reserved test。
- 不训练 HGNN/MAPPO。
- 不修改正式 Planner、奖励、动作、SafetyProjector、checkpoint 或旧数据集。
- 不覆盖 D0–D5 历史结果。
- 不因远端缺文件就假设本地任务已完成；必须检查实际工作区和日志。
- 不重新运行已经完成且结果完整的状态/候选。

## 3. 交付与报告

将核查结果写入：
- `docs/research/HTW_UrgencyAware_ArrivalAware_Expansion_Status_CN.md`

报告必须明确：
- 核查的 branch/HEAD/worktree；
- D6 是未运行、部分完成、已完成但未推送，还是已完成且推送；
- 实际找到的结果文件路径与哈希；
- 如有实验，实际场景、时刻、候选数、rollout 数、耗时和所有 Gate 检查；
- 如未完成，具体阻塞原因；
- 后续是否可以进入高层训练目标设计。

完成后只提交状态报告及必要的已完成结果，停止等待后续指令。不要自动进入 D7，也不要训练高层模型。
