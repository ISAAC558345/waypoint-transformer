# 旧系统 L5：固定 idle 参照的离线选择对照

更新时间：2026-11-05。工作分支 `feature/legacy-system-maintenance`；已审查L4结果 `d25e97fec756028a394a1cca67edb6d29c47c137`。原L4指令归档于 `archive/Legacy_System_L4_Instructions_20261105_CN.md`，不再重复执行。

## 1 本阶段目标与预算

回答：保持同一模型、同一128候选和原始分数，改用一个预先固定的idle相对参照后，B10的t15稀疏选择和真实payload是否改变？同时修正L4已发现的两项统计解释问题。

先读 `AGENTS.md`、`README_BRANCH_SCOPE.md`、`Legacy_Only_Workflow_CN.md` 和 [L4审阅](Legacy_System_L4_Push_Review_CN.md)。只完善旧系统；新方案继续隔离，不执行或合并。

本包只读已有缓存，**模型前向0、训练更新0、反传0、环境步0、低层推理0**。不要加载checkpoint、运行diagnose或run-pair、重新生成数据。只做一次下面的固定对照及直接相关检查，不跑历史全套或无关安全专项。

## 2 固定输入

- L4 `raw_matrices.npz`，原Git blob `6bfcfe7dc08f9c22a7d442ce4585586f7acb8b5e`；
- L4 `raw_scores.npz`，原Git blob `c56c25d17ba9359a1c0d85fb2b429666e715f6fd`；
- L4 `state_records.json`，原Git blob `feadba3e397234d46d1a6f8ac842dd6498e60ccd`；
- L2 `paired/rows.csv` 及分组索引，用于候选身份和最后的标签读数。

文件均来自上述固定L4结果提交。矩阵和分数共用键，如 `B10_322016_t015`。保留E6/A10/B10三模型×16状态，每状态128行。沿用原12训练/4探索分组，分别呈现train/explore × t15/t30。两探索场景已经多次打开，只作探索与机制消融，不称新的独立验证。

新增独立NumPy/标准库缓存读取入口，例如 `wt.legacy_arrival_aware_ab.compare_selectors`；不通过会加载模型的旧诊断函数重算。明确根目录、缓存、配对表和输出目录参数，输出到新的 `results/legacy_arrival_aware_ab_l5/`。

## 3 先更正 L4 口径，保留原记录

1. 主要边统计从每状态 `candidate_edges` 或缓存矩阵中128候选实际使用的唯一有效(uav,buoy)边重算，不混入未使用边、不按候选重复出现次数加权。当前每状态36条，原完整矩阵72条。
2. 修正诊断汇总代码中主要统计的字段来源；明确命名“候选实际使用边”与“完整矩阵辅助统计”。本包只从缓存输出 `l4_corrected_summary.json`，不调用模型、不覆盖L4原manifest/report/缓存。
3. 报告组内regret时同时列组候选数。单例组regret=0不得作为排序能力证据；保留L4其他有效结论。更正说明放入L5报告，不改写历史运行。

## 4 唯一的新选择规则：每 UAV 的行均值 idle 参照

对每个模型、状态独立固定如下定义，不能依据标签改变：

- `M[u,b]` 是缓存原边分数；`S(a)` 是缓存原候选总分。
- `E_u` 是当前128候选中，UAV u实际被分配过的**唯一合法浮标**集合；不含idle、不含矩阵中的未使用边。
- `mu_u = mean(M[u,b] for b in E_u)`。若E_u为空，mu_u=0；该UAV在当前候选中也不活动。
- 原规则 `RAW_SUM`：沿用缓存S取最大。
- 新规则 `ROW_MEAN_IDLE`：`Q(a)=float64(S(a))-sum(mu_u for u active in a)`，按Q取最大。

等价解释：在原边分数尺度上，把idle参考值设为该UAV的合法边均值，再减去所有行的共同常数。它保留当前合法idle候选；同一active_mask内仅减相同常数，内部排序不变；不同mask或活动数之间可改变。

这是新选择规则和受控消融，不是原模型必须遵守的数值不变量，也不是已经证明idle真实收益等于mu。它移除了模型的行级参与偏好，可能损害原本合理的idle决策。唯一边均值减少重复计数影响，但仍受固定子集覆盖限制，不代表全13,327候选的最优。

数值实现：
- 原规则始终用原缓存float32分数与L3并列规则，不重新用矩阵累加来替换S。
- mu用float64计算；每个active_mask按UAV固定顺序求和一次，供该mask所有候选共用。Q用float64(S)减该常数，保留原并列关系，避免另一次求和顺序改变旧分数。
- Q完全并列时仍取较小candidate_id；不按payload破并列。
- 不做正值裁剪、不除以活动数、不加可调offset/系数、不改温度，不并列尝试中位数、最小值或活动数边际MAP，不增删候选、不强制全部UAV活动。

选择函数只接受matrix、缓存scores、assignments和candidate IDs，不能接收两列payload或标签排名。先保存全部规则选择和mu/E_u清单及代码身份，再关联已有标签计算结果。不得根据结果改规则或选择一个“最好模型”。

## 5 公平读数与必要检查

主比较预先固定为B10的RAW_SUM与ROW_MEAN_IDLE；E6/A10使用同一新规则作对照，检查它是否普遍改变参与行为。不能把“选择变化”当作payload收益。

逐状态记录两规则所选候选/assignment、活动数/mask、原分数与Q、真实payload/regret、零到达payload，以及新减旧的真实payload差。按train/explore × t15/t30、每场景、全部探索状态汇总，保留每个负差，不把训练与探索混成泛化指标。

只做直接检查：
1. 48个缓存键和各128行的身份对应；RAW_SUM应复现L4记录的48个候选与payload。该步骤为纯数组读数，不是新前向。
2. mu只使用E_u，数值有限；相同mask的Q-S偏移一致。用本次真实候选验证，不建立繁杂模拟或故障矩阵。
3. 所选项仍在原128候选内，标签只在选完后作评价。一次全包计算后保存原始结果，不重复运行搜好结果。

解释区分：
- 稀疏现象改变且payload改善：支持继续研究参照问题，但仍是已见小样本上的后验消融，不能宣布采用。
- 活动数增加但payload不改善或其他时刻变差：不能把“多派UAV”当作修复成功。
- 选择不变或结果混合：保留结果，不扫描mu系数、训练顺序或新场景。

不计算伪泛化置信区间，不自动追加模型推理或训练。下次是否需要独立冻结对照/模型改造，依据本包全部结果另定。

## 6 交付与停止

交付：
- 独立缓存对照入口、实际可复制命令及诊断主要边统计修复；
- `l4_corrected_summary.json`、预先保存的选择/参照清单、逐状态和分组差值；
- `results/legacy_arrival_aware_ab_l5/manifest.json`，登记固定来源提交、输入Git blob/文件身份、代码身份及五项消耗全部0。区分仓库原字节哈希与本机工作区换行哈希；
- `docs/research/Legacy_System_L5_Report_CN.md`，先说明L4口径更正，再报告唯一新规则的完整结果、局限和一项最有依据的后续建议。

原L2/L3/L4记录与模型保留，正式Planner/权重不替换，新方案不合并。论文编译等旧遗留项保持登记，本包不反复检查编译器。提交推送当前维护分支，报告SHA后停止。
