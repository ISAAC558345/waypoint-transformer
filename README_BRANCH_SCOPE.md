# 原系统独立维护分支

当前工作分支：`feature/legacy-system-maintenance`。正式基线：`e58e5d190109e5a0e860cb5c6a412321c0d74e31`。

本分支从原正式基线建立，只保留旧系统R1/R1A完善、原系统论文及必要交付文件，不从含新方案文档的分支整包继承。新方案保存在 [独立隔离分支](https://github.com/ISAAC558345/waypoint-transformer/tree/research/ocean-observation-isolated)，后续是否合并由用户另行决定。当前只完善原系统。

正式Planner、环境、队列、SafetyProjector和模型blob与原正式基线一致。R1/R1A原始运行结果和40份归档原件保留。唯一现行运行辅助修复是历史日志从本地镜像读取，原提交/路径仍作为来源；不需要获取新方案Git对象。原R1登记指令移入历史记录目录。研究任务入口是 [旧系统下一步](docs/research/Legacy_System_Next_Instructions_CN.md)。现行正式单场景命令、只读检查和显式导出写在 [命令说明](docs/research/Legacy_System_Commands_CN.md)。旧 R1 的 `--rerun` 与 `--finalize` 仍返回 2，不再是唯一主动运行方式。

分支隔离清单为 `BRANCH_ISOLATION_MANIFEST.json`。本次没有新环境回合、训练或策略推理；只做分支文件身份核对，以及一项本地日志读取检查。必要测试偏好继续有效，不默认跑历史全套或无关安全专项。

新方案现有分支和旧研究分支均保留，没有删除、重置或合并。后续工作使用本维护分支，原正式分支仍为固定基线。
