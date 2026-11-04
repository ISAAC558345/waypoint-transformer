# R1 原系统正式复现与证据归档指令

更新时间：2026-11-04。本轮先完善当前正式系统的可复现交付，提供后续方法比较的共用入口。依据 [发展路线](Development_Roadmap_20261104_CN.md) 和 [必要测试规则](Research_Priority_and_Necessary_Testing_CN.md) 执行。用户已明确原系统冲突优先，不需要重复确认常规实现选择。

## 1 来源与工作分支

正式算法来源 `e58e5d190109e5a0e860cb5c6a412321c0d74e31`。M0结果 `7e9e372e6ab603342f59c334301cdcc42ccc2e61` 与E19结果 `f9e9a9e007d7d84046232181fdab6f9a732a4a05` 作为只读证据。

从包含本指令的最新 `docs/ocean-observation-design-20261103` 提交建立或继续 `experiment/legacy-formal-repro-r1`。已有本地工作先保存并检查，不重置。遵守快照、清单和独立输出目录规则，Git分支不代替轮次管理。按用户最新测试规则处理旧分支中全套测试要求。

## 2 先解决已有证据支持的问题

已有206项失败中172项缺文件、11项历史状态错误、23项断言失败；未见当前queue/observation接口失配依据。不得为了清零改动正常正式行为或历史锁。

交付一个轻量独立运行器：通过实验配置副本的 `policy_path` 调用 `run_active_episode`。保留正式 `active_policy.json`，支持 `scheme_c` 与 `g1_rate`，两臂均为G1_RATE k=6、hold=15、同一个冻结MAPPO。不要调用仍为k=5的历史run_episode。

每次记录源码版本、依赖、实际读取的模型身份、配置副本及哈希、场景和种子的作用、决策时刻、各步served与累计payload、完成状态及运行耗时。以机器可读结果和一个中文摘要输出。已有异常字段自然保留，不为它们新增专项场景。

整理实际可运行环境与最小依赖说明；当前研究使用的Python/NumPy/torch版本与旧requirements不一致，新增独立的正式运行依赖说明或文件，保留历史requirements。不得为此批量升级全部历史模块。

## 3 必要A/B对照

预先固定TRAIN 322000、322001。每场景：
- A：Scheme C，正式k=6。
- B：G1_RATE，正式k=6。
- 每臂48步，总计2×2×48=192环境步，上限即192，不补场景、不换种子、不额外完整重复。
- 每臂重置进程种子1000000+scenario_id、torch单线程且开启确定性算法，沿用AB/AC语义；环境reset仍以scenario_id为种子。不要混用E19的历史进程seed2语境。
- 禁止读取或重跑旧DEV/holdout作为本包比较。原模型、奖励、FIFO、SafetyProjector、候选和动作语义保持不变。

192步是当前可运行与对照接口复现，不是新方法泛化评测。逐场景报告payload、差值和运行成本，不因为某臂结果较差换样本，也不必为这两条场景做bootstrap或多种子矩阵。总训练更新0。

若当前入口因实际错误无法运行，先修复最小根因并做针对性检查；尚未开始的对照可在修复后运行。已消耗研究步计入192；超出预算前停下报告，不以调试名义追加。

## 4 原始证据归档

只在已知项目运行目录和备份中追溯：
- results/phase6iab_planner_stability/summary.json
- results/phase6iac_independent_holdout/summary.json
- results/phase6if_ranking_loss/summary.json
- 与这些汇总直接对应的原始episode或训练记录。

原样恢复到独立归档目录，附来源、创建信息、文件哈希和统计口径。找不到就保留MISSING，不根据论文数字重造summary，不重新开启旧holdout。恢复的源文件与新192步结果清楚区分。

从既有日志给历史失败做一次影响当前正式路径的索引，列明缺档、历史状态、真实当前功能问题；不重新运行全套来再次获得206这个数字。发现其他未完成工作时更新继承清单。

## 5 只运行以下必要检查

1. `test_phase6iaa_formal_planner.Phase6IAAFormalPlannerTests.test_live_graph_uses_the_raw_edge_contract`：实时图与冻结模型接口一致。
2. `test_phase6b_bit_queue.Phase6BTests.test_accounting_cases_pass`：实际服务记账。
3. 新运行器的配置副本传递和输出字段最小检查；与上述192步复用输出核对决策[0,15,30,45]、实际served与payload一致。

只有实际修复触及另一个直接调用者时才增加对应回归，并在报告解释原因。现有测试内步数单独记录，不用于本包方法证据。没有默认全历史套件、故障注入、独立碰撞压力或广泛安全专项，不改容差掩盖问题。

## 6 输出与停止

交付运行器、配置副本、依赖说明、192步逐场景与逐步结果、原始证据索引、失败影响索引及必要检查结果。总报告 `docs/research/Legacy_Formal_Repro_R1_Report_CN.md` 链接准确产物位置。

报告可运行状态、已解决问题和保留缺档。满足“正式链路可运行、输出可追溯、必要相关检查通过”即可完成R1，不要求所有历史产物恢复，也不宣称系统已发布或全套通过。

成果推送到 `experiment/legacy-formal-repro-r1`，报告提交SHA。R1存在当前功能阻碍时优先处理；达到本包条件后再按独立指令执行M0A。R1不启动训练、标签权重搜索或正式分支模型替换。
