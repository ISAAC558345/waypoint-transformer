# 原系统当前可用命令

以下命令在本维护分支的仓库根目录执行。只读工具使用Python标准库；本分支为src目录布局，需要先让Python能找到src。

PowerShell示例：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
py -3 -m wt.legacy_formal_repro_r1a.inspect --artifact-root results/legacy_formal_repro_r1
```

该命令只读取原192步记录，不加载模型、不写原清单。

如需显式导出派生视图，选择一个尚不存在的新目录：

```powershell
py -3 -m wt.legacy_formal_repro_r1a.export --artifact-root results/legacy_formal_repro_r1 --output results/legacy_formal_repro_r1a_view
```

目标文件存在时工具拒绝覆盖。代码与来源说明见R1A报告；其中原运行环境是历史运行记录，不是新机器安装验证。

当前 `wt.legacy_formal_repro_r1.run --rerun` 和 `--finalize` 均受保护而返回2，不能将它们作为目前可执行的重跑方式。主动运行入口要在有独立输出和明确预算的后续任务中实现。本轮不重跑原192步。
