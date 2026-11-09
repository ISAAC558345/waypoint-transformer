# 论文代码仓库结构与迁移要求

日期：2026-11-09。源提交：8fc6fda3d50d51b414cb609b7a8953951788eb18。用户要求按论文代码仓库组织，去掉round1/round2等目录；同时选择先收尾旧系统，不新增模型研究。

## 目标不是外观包装

当前Git树有36个顶层round_目录、33个src/wt/roundN包。README列出的src/envs、src/models等主要是转发接口，实际实现仍在按阶段命名的树中。必须真正迁移实现和资源，不只改README、不再添加一层空壳、不把round_目录原样藏到archive下，也不换名stage01/phase01后继续按顺序组织。

借鉴两个真实论文代码仓库的职责划分，而不照搬其算法或依赖：
- MAPPO：https://github.com/marlbenchmark/on-policy，代码按algorithms、envs、runner、utils组织。
- Guided Diffusion：https://github.com/openai/guided-diffusion，实际代码包、scripts、datasets、evaluations分开。

保留wt作为Python顶层包，减少无意义的包名变化。下面是目标职责，不要求机械复制空目录：

~~~text
README.md
LICENSE
AGENTS.md
pyproject.toml                 # 小型包元数据；不引入新训练框架
requirements.txt
environment.yml
configs/                       # 当前环境、模型、正式策略、资源索引
src/wt/
  envs/                        # 当前位队列通信环境
  models/                      # HGNN、matching decoder、MAPPO网络真实实现
  planning/                    # 候选生成、FormalPlanner、TaskOption
  control/                     # 冻结MAPPO执行、安全投影
  evaluation/                  # episode、served统计、重放及参考比较
  data/                        # 构图、快照、数据加载代码；不是二进制数据仓库
  algorithms/                  # 已有损失、PPO公共函数
  common/                      # 根路径、资源解析、seed等公共实现
  cli/                         # evaluate、inspect等当前入口
  research/                    # 已完成研究的语义包，不按L/round编号建目录
  legacy/                      # 非当前主链的旧实现，按主题归类
data/                          # 已有数据集、快照、样本；不生成新数据
checkpoints/                   # 正式/初始化/历史研究权重，按用途区分
results/
  benchmarks/                  # 已有正式汇总
  reproducibility/             # R1参考、命令一致性、这次迁移回归
  research/                    # 已有未采用研究结果，按主题归类
scripts/                       # 当前可用的薄命令入口
tests/                         # 功能测试与历史测试分开
docs/
  SYSTEM_OVERVIEW_CN.md
  USAGE.md
  development/
  paper/
  research/
archive/
  experiments/                 # 历史非运行材料，以语义主题命名
  provenance/                  # 原锁、版本台账、来源记录
  upstream/                    # 原Waypoint/RvS说明、历史入口与归属
~~~

当前主链实际实现在上述功能包中，不能仍全部转发到原round/phase实现。htw_phase*、benchmark_v2/phase*和lowlevel_v2中的当前依赖按实际功能迁移；历史实现进入语义legacy/research包。保留算法、类/函数行为和参数，不重写模型。

根README改为当前系统说明和最短可运行步骤；上游长README、旧轮次流程归档并保留归属与LICENSE。不要把安装依赖的结构迁移表述为已验证新机器安装；本轮不重建环境。当前可验证的运行版本来自已有manifest。

## 必须处理的真实依赖

- HGNN使用round30.models.masked_mean；迁入models/common或common/masking等真实公共模块。
- MAPPO/PPO使用round30.ppo中的PPOConfig、compute_gae、tanh_log_prob、seed_everything。按实际职责迁入algorithms/common/control，更新全部导入。
- round18/19的旧环境和工具仍被一些历史模块使用，不能误删。round35/36是早期关系基准，不是当前正式L3B环境。
- 正式checkpoint使用state_dict/model/actor字典加载，不需要旧round包目录才能加载。不能修改权重字节迁就新代码。
- 禁止保留roundN Python目录、旧目录软链接、sys.modules包别名来绕过真实迁移；更新Python导入、模块入口和相对导入。历史文本中的旧路径无需抹掉。
- 不再以Path(__file__).parents[固定层数]解析迁移模块的资源根。建立一个从项目标识定位根目录的common.paths和资源注册表，不依赖当前工作目录，不依赖.git一定存在。正式配置、权重、参考episode、数据集根在同处按语义ID解析。
- 不要把历史current-round治理脚本继续接入当前入口。

## 历史语义映射

以下作为迁移基线；同一主题内按configs、records、reports、checkpoints等职责拆分。src/wt/round4至round36的历史代码按同一slug进入legacy，但被当前系统使用的公共实现先提取到正式功能包。前三轮的代码在wt.maritime，不能凭空创建round1至3对应包。

| 原根目录 | 语义主题 |
| --- | --- |
| round_01 | maritime_environment |
| round_02 | offline_baseline_collection |
| round_03 | queue_task_dataset |
| round_04 | spatial_waypoint_pretraining |
| round_05 | causal_offline_policy |
| round_06 | waypoint_closed_loop |
| round_07 | offline_dataset_audit |
| round_08 | spatial_waypoint_retraining |
| round_09 | waypoint_mechanism_diagnosis |
| round_10 | task_waypoint_labeling |
| round_11 | task_waypoint_predictor |
| round_12 | task_waypoint_policy |
| round_13 | service_action_controllability |
| round_14 | task_to_motion_mapping |
| round_15 | static_planning_headroom |
| round_16 | buoy_drift_planning |
| round_17 | predictable_channel_planning |
| round_18 | multi_uav_coordination |
| round_19 | shared_spectrum_planning |
| round_20 | multi_uav_task_dataset |
| round_21 | hgnn_task_waypoint |
| round_22 | deepsets_task_policy |
| round_23 | task_policy_seed_audit |
| round_24 | task_policy_multiseed |
| round_25 | supervision_ambiguity |
| round_26 | action_ambiguity |
| round_27 | conditional_planner_utility |
| round_28 | planning_need_representation |
| round_29 | cross_scale_planning_trigger |
| round_30 | deepsets_ppo |
| round_31 | ppo_failure_analysis |
| round_32 | horizon_semantics |
| round_33 | long_episode_benchmark |
| round_34 | benchmark_complexity |
| round_35 | relational_benchmark_environment |
| round_36 | relational_planning_validation |

顶层CURRENT_ROUND.txt、ROUND_VERSIONING.md、round_manifest.json等陈旧治理文件移到archive/provenance，改用语义文件名。旧锁/记录内容保持原字节，不再作为当前工作状态。

历史AGENTS内容应保留为历史说明，例如historical_instructions.md，并在映射中保留原文件身份，不让归档后的旧指令成为当前运行规则。根AGENTS更新为用户当前范围和最小检查政策。

本地忽略目录中若也存在round05_formal等按轮次命名的子目录，要纳入同一语义迁移；不能只检查GitHub可见文件。

## 数据、权重和历史来源

正式4份资源集中登记并保持SHA-256：
- Scheme C：f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864
- MAPPO seed0：92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e
- matching decoder：de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5
- E6初始化/启动身份：7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde

当前策略JSON只定义scheme_c、G1_RATE、k6、hold15等，原字节应保留；路径迁移由注册表解决。原R1参考按新reproducibility布局读取，当前322000/322001注册与种子语义保持不变。

所有受迁移的数据、checkpoint、原结果、锁、原日志、历史PDF/正文保持字节。不要对历史JSON/log/PDF中的旧路径做全局替换，也不要改写旧锁哈希。新派生文档、活动配置/代码和新的资源索引可以使用新路径。

生成docs/development/path_migration_20261109.json，至少记录old_path、new_path、kind、原blob/字节SHA、是否改内容、必要的旧/新Python模块名。建立旧逻辑路径到新物理路径的只读解析，让历史manifest中的路径仍能定位原内容；这不同于创建旧物理目录。路径可用前缀映射，但要处理最长匹配及冲突，不能错误合并两个同名文件。

## 本地范围与迁移方式

先清点当前主检出的已跟踪、未跟踪和忽略文件，尤其旧round outputs、数据和checkpoint。保留未提交修改；不能reset或覆盖冲突。只处理项目主检出，不遍历/移动.git、虚拟环境、项目外路径、backup/snapshots、backup/baseline_worktrees或其他独立Git检出。历史备份中的旧树不属于本轮论文代码布局。

验证源/目标绝对路径都在本项目内再移动；追踪文件优先git mv，未追踪工件用同一PowerShell或Python文件操作流程。移动可逆，不删除历史内容。目标有冲突时按来源语义区分，不覆盖。

原.gitignore忽略docs/archive及round输出；迁移后同步新模式。已追踪证据继续追踪；原本仅本地的大文件不应因目录改变意外全量提交。原.git属性的过滤/换行规则也须核对，二进制和历史原件以原字节保存。

先完成结构迁移与一次必要回归，再更新论文中活动路径和交付索引。最终报告必须列出实际改动，不把本设计文档或一层转发视为重构完成。
