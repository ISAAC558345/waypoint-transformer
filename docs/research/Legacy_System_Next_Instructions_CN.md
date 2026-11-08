# 旧系统下一步指令：L10 按场景正确运行与比较

日期：2026-11-08。结果基线：bbb7a0eb329fc83da7cd59d91310da198def3b7b。执行分支：feature/legacy-system-maintenance。先读本次提交的Legacy_System_L9_Push_Review_CN.md。

## 1. 本轮范围

L9经复核保持NOT_PROMOTED，L8也保持NOT_PROMOTED。两轮均不启动闭环、不接入正式Planner，不改变门槛，不补训练或调参。本轮只修正正常入口对场景和参考的绑定，训练、反向均为0。

正式Scheme C epoch1、G1_RATE k6、hold15、冻结MAPPO-V2、SafetyProjector及served定义保持当前配置。海洋隔离分支不修改、不执行、不合并。lambda_H、delta_payload_max仍不取值。

项目辅助文件和临时文件留在项目内，临时项使用backup；新结果使用results/legacy_system_l10。保留旧结果、用户其他文件和目录，不新建或搬移工作树。

## 2. 已确认的问题与目标

目前：
- wt.legacy_system.run只接受322000，preview中的参考文件也总是322000；
- execute.run_once读取322000参考；
- completion.compare_episodes把322000的REFERENCE_PAYLOAD_BITS常量纳入所有数值比较；
- 新执行manifest还沿用L6包名和旧指令号。

底层run_arm已支持scenario参数及1_000_000+scenario的过程种子，无需改正式运行器或模型。

改为一处共享的标准库参考注册/解析逻辑，集中提供scenario_id、reference_episode、reference_config、process_seed、reset_seed和默认输出路径。当前仅登记R1已有完整参考的训练场景322000与322001，不自动开放整个训练池、validation或reserved test。

参考相对路径为：
~~~text
results/legacy_formal_repro_r1/episodes/{scenario}_scheme_c.json
results/legacy_formal_repro_r1/configs/{scenario}_scheme_c.json
~~~

注册表保存相对路径而非历史机器绝对路径。预览、执行和manifest共用这一处解析，不在三个位置重复写两套常量。可用一个小的标准库模块实现，不引入新依赖或大型配置框架。

## 3. 预览、输出与比较行为

预览仍只用标准库，不加载torch/模型，不建立目录。两场景各自打印正确参考、种子及run_accepted=true。对未登记场景明确表示未接受，实际--run在加载模型/创建输出前返回2；不要声称已有完整回归支持。

未显式--output时，默认输出随scenario分开，例如results/legacy_system/scenario_322001。显式指定路径的行为保持兼容，包括已文档化的322000命令。保持L7已完成的空目录接收、非空目录拒绝与退出码约定，不再改其逻辑。

比较器必须：
- 从本次对应reference读取payload/served参考值，去除322000常量对其他场景的约束；
- 校验reference/current都具有必要的scenario_id和policy字段，并一致；不能用缺失字段的None==None作为身份通过；
- 保留原有逐步served、步数/时刻、种子、决策时刻、Scheme C调用数及两个内部served一致性检查，数值容差仍为1bit；
- reference_payload_bits如继续输出，必须是该场景的实际参考值。如保留旧“公布值”字段，也须标注同一R1来源，不能当成另一份独立证据；
- 保持conclude的原语义：执行完成、matched、模型身份正确、正式策略未变才成功返回0；执行错误或不一致返回1；输入/输出拒绝返回2。无参考时不伪标matched或成功。

新manifest记录所选场景的参考episode/config、来源哈希、对应种子、实际执行代码及本次指令来源；包名不再误写为legacy_system_l6。保留原L6/L7 manifest，不回写历史元数据。

不要修改R1受保护入口、R1原始episode/config、run_arm、FormalPlanner、策略配置或任何权重。

## 4. 最小验证：先保存结果，再一次48步

只进行当前改动所需的以下验证，不追加测试矩阵。

第一步：标准库检查，模型前向和环境步均为0，结果保存到新L10目录，不能运行会回写旧L7目录的历史check_completion入口。

- 预览322000和322001：参考路径、种子、默认输出各自正确，不写预览目录、不导入torch。
- 原R1 322000与已保存L6 322000比较为matched，证明旧场景仍兼容。
- 原R1 322001与其自身比较为matched，证明不再受322000固定payload约束。
- 322000参考与322001记录交叉比较为different。
- 检查两套参考文件身份、48步完整性和来源，仅使用保存文件。

第二步：只实际执行322001一次，完整48步，命令示例：
~~~powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
py -3 -m wt.legacy_system.run --scenario 322001 --output results/legacy_system_l10/run_322001 --run
~~~

预期与对应R1结果一致：
~~~text
scenario_id = 322001
process_seed = 1322001
reset_seed = 322001
stop_time = terminal_time = environment_steps = 48
decision_times = [0, 15, 30, 45]
scheme_c_calls = 4
payload_bits = 68952073.58225587
served_bits = 68952073.58225589
payload_matches_served = 1
step_served_matches_episode = true
error = null
regression_status = matched
success = true
exit_code = 0
~~~

两个总量和全部48条逐步served均按原1bit容差比较；不要四舍五入后再比较。此验证是入口一致性，不是性能收益实验。

新增实际环境步预算48，训练/反向0。低层推理和正式高层调用记录实际值，正常预期分别48和4。失败/作废步也计入48预算。若出现错误，保留原始失败与消耗，不自动重新执行另一轮48步；先报告具体原因，不能改参考让结果通过。

不实际重跑322000，不重跑原192步，不重做L8/L9，不重复L7目录冲突检查，不运行安全/压力/故障注入矩阵，不重编译论文。原有输出保护没有改动时无需再测一遍。

## 5. 文档与交付

更新docs/research/Legacy_System_Commands_CN.md，说明当前可运行的两个参考场景、预览/显式运行、按场景默认输出、显式路径兼容和结果比较来源。保持R1只读/导出及其受保护入口的说明。

保存results/legacy_system_l10/中的标准库检查记录、run_322001的配置/episode/逐步served/比较/执行manifest、实际消耗，以及必要的错误记录。引用旧参考原件，不复制或改写R1证据。不要为本轮引入新的模型研究代码。

提交docs/research/Legacy_System_L10_Report_CN.md，说明：
- 哪些场景/参考硬编码已解除；
- 保存结果检查及322001新48步的结果与退出码；
- 实际环境步、低层推理、高层调用、更新和反向；
- L8/L9仍未晋级，正式模型没有改变；
- 当前仍仅支持登记参考的两个训练场景，不能写成全场景验证。

在维护分支提交并推送本轮代码、结果、命令文档和报告，提供提交号后停止。本轮不自动启动新的架构或目标研究；后续如投入另一种研究问题，先明确假设和预算。

正式分支e58e5d190109e5a0e860cb5c6a412321c0d74e31、海洋隔离分支4d8622ef183e734f2d9a890d60d735ed716f5d74保持不变。
