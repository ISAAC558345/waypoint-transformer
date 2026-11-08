# 旧系统 L6：正式运行入口与探索结案

更新时间：2026-11-05。工作分支：`feature/legacy-system-maintenance`；已审查L5结果 `30c880bedd9da4adff0b2de1d6b3fbef4bc27554`。L5原指令保存在 `archive/Legacy_System_L5_Instructions_20261105_CN.md`，不再重复运行。

## 1 当前目标与范围

先把旧正式系统交付为可直接运行、结果单独保存的命令入口，并结案L3–L5这组未采用的探索。当前不再在同16状态上试新选择器、按t15/t30拼接规则或继续训练。

先读 `AGENTS.md`、`README_BRANCH_SCOPE.md`、`Legacy_Only_Workflow_CN.md`、[L5审阅](Legacy_System_L5_Push_Review_CN.md) 及R1/R1A的命令与结果说明。新方案在 `research/ocean-observation-isolated` 保持原样，不执行或合并；正式分支也不合并。

本包训练更新0；只新增一次48步正式SchemeC运行，作为新入口的直接回归。没有新方法性能对照，不跑历史全套或无关安全专项。

## 2 新增独立正式运行命令

建议入口 `wt.legacy_system.run`，或选择等价清晰名称并登记。不要重新开放R1旧CLI的 `--rerun/--finalize`；它们继续保护原始证据。

默认只预览配置和将写入的目录，不建输出、不加载模型，标准库层即可工作；显式 `--run` 才调用模型和环境。建议实际命令：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
py -3 -m wt.legacy_system.run --scenario 322000 --output results/legacy_system_l6
py -3 -m wt.legacy_system.run --scenario 322000 --output results/legacy_system_l6 --run
```

以实现后的真实命令为准。本包只提供和验证现行正式SchemeC路径，不顺便实现L3/L5策略切换或新算法配置。

实现要求：
1. 调用现有 `run_active_episode` / `FormalPlanner` 正式链路。复用R1的 `prepare_policy_copy`、`run_arm`、进程种子和逐步served记录能力，避免复制Planner算法。不能调用R1的main/finalize/run_registered，也不能调用多场景audit程序。
2. 为本次运行生成独立策略配置副本，置于新输出目录；正式配置是G1_RATE、k=6、hold=15、scheme_c。保留历史字段的来源解释，不能把historical_formal_k=5当作现行k。
3. 只使用原正式SchemeC和冻结MAPPO-V2权重及原decoder，不使用E6作为正式高层替代、L3 A/B权重或ROW_MEAN_IDLE。
4. 预览与读结果不安装torch。显式运行使用当前已有可用依赖，记录实际版本；不重装大型依赖、不把当前运行说成新机器安装验证。
5. 运行产物写入新目录，不覆写R1/R1A、L2–L5和旧研究产物。配置、episode、steps和manifest之间使用明确的输出相对路径，机器上的原绝对路径如需记录放入运行上下文，不能作为迁移后唯一查找依据。
6. 已有 `run_closed_episode` 可接收fault_log，但 `run_active_episode` 未透传。若复用路径需要隔离本次异常日志，可做最小、默认行为不变的可选参数透传，并把本次路径放进新目录；不改Planner、SafetyProjector或故障判定逻辑，不为此做故障注入测试。也可直接复用相同FormalPlanner+run_closed_episode并显式指定路径，只要保持同一正式行为和种子约定。
7. 已存在的完整输出不得自动覆盖或删除；异常时记录实际消耗和错误后停止，不自动换场景/模型或重试环境回合。

## 3 唯一一次48步回归

先运行一次默认预览，确认它只展示配置；再显式执行一次下面的正式回合：

- scenario=322000，属于既有训练场景；allow_holdout=false；
- process_seed=1000000+322000=1322000，在构造Planner前重置random/NumPy/torch；
- 环境reset_seed=322000；CPU单线程、确定性算法，沿用R1真实约定；
- SchemeC、k=6、hold=15、stop_time=48；
- 期望完整48步，正式决策时刻0/15/30/45，scheme_c_calls=4；
- 训练0；正常执行约48个低层动作步，实际步数/调用按记录报告。

总新增环境步上限 **48**。不要再跑322001、G1对照、原4臂192步或48步新策略测试。这个回归覆盖新CLI的正式全链路，不是复核历史性能或补holdout数据。

参考原件：
- `results/legacy_formal_repro_r1/episodes/322000_scheme_c.json`；
- `results/legacy_formal_repro_r1/configs/322000_scheme_c.json`。

对比complete、terminal_time、种子、决策时刻、SchemeC调用数、payload/served及48条逐步served。参考payload为 **63785945.28984571 bit**。与原件的payload/逐步served绝对差以1 bit为比对容差；payload与served的运行内一致性仍沿用正式契约。路径、时间戳、耗时不要求相同。

若业务字段或数值不一致，保留新结果和具体差异，追查配置/运行版本等实际原因；不能改种子、替换权重、放宽约束或再跑一批结果选匹配者。发生部分运行也计入本包48步预算，写明剩余未完成项后停止，不自动补跑。

## 4 交付产物与状态

新目录至少保存：策略副本、episode.json、steps.csv、run_manifest.json和与R1原件的regression_comparison.json。清单区分本次实际运行与历史参考，登记实际runtime、正式所用权重/策略文件身份、输入来源、代码身份、实际步骤和错误。

正式模型参考：
- SchemeC Epoch1 SHA-256：`f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864`；
- MAPPO-V2 SHA-256：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`；
- decoder SHA-256：`de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5`。

以实际代码读取路径为准，不为列清单额外加载不用的权重。源模型、官方active_policy、旧结果身份保留；比对使用已有文件和本次回合，不额外执行回合。记录原始anomaly/错误，不用全量测试绿灯或性能提升标签代替本次功能回归结论。

## 5 同步命令与研究结案索引

1. 更新 `Legacy_System_Commands_CN.md` 与分支入口说明，清楚列出只读inspect、显式导出、配置预览和新正式运行命令。旧R1受保护命令不再被描述成唯一主动运行方式。
2. 输出 `Legacy_ArrivalAware_L3_L5_Closure_CN.md`，引用L3/L4/L5固定提交，写清240训练更新、48诊断前向、缓存消融的来源与实际结论；A/B模型与固定参照补丁均未采用。只结案该配置链，不关闭整个payload/标签服务或到达感知总目标。
3. 研究索引可从论文README链接，但不要冒称当前论文正文/PDF已包含新结案；本包不扩写论文正文、不重查编译器，不补造ranking-loss summary。明确这些仍是遗留项。
4. 输出 `Legacy_System_L6_Report_CN.md`，区分已完成的工程完善、48步回归结果、失败/未完成项及研究结案。

本包只做一次预览和一次实际全链路回归。不要新增同实现互相印证的mock测试、大规模参数矩阵或历史全套。与当前改动直接相关的实际失败可以修，但48步总预算不能通过重跑绕过。

完成后提交推送当前维护分支，报告SHA后停止。本次根聊天只发布指令，没有执行这48步。后续结构/目标改造须先有明确假设和证据预算，不由L5的局部改善自动启动。
