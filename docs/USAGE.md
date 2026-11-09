# 使用说明

在仓库根目录执行。Python 需要能导入 `src` 下的 `wt` 包。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
py -3 -m wt.cli.evaluate --scenario 322000
py -3 -m wt.cli.evaluate --scenario 322001
```

上面两条是预览。它们打印正式合同，不导入 PyTorch，也不写结果。未注册场景在 `--run` 时会在加载模型前返回退出码 2。

已核对的一次运行命令是：

```powershell
py -3 -m wt.cli.evaluate --scenario 322001 --output results/reproducibility/layout_refactor_20261109/run_322001 --run
```

该次结果与 `results/reproducibility/formal_scheme_c/episodes/322001_scheme_c.json` 一致：payload `68952073.58225587`，served `68952073.58225589`，`process_seed` `1322001`，`reset_seed` `322001`，决策时刻 0/15/30/45，高层 4 次，低层 48 步，退出码 0。

`py -3 scripts/benchmark.py` 只查找已保存的验证摘要和保留集摘要。2026-11-09 检查时这两个摘要都不在检出中，脚本报告缺失。

资源位置由 `configs/resources.json` 登记。`wt.common.resources.resolve_logical` 用最长前缀把旧逻辑路径指到迁移后的文件，不会为了迁移重建旧目录。
