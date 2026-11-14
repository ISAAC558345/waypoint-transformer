# 使用说明

在已有Python环境中从仓库根目录设置PYTHONPATH=src。下面的python可替换为实际解释器路径。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m wt.cli.evaluate --scenario 322000
python -m wt.cli.evaluate --scenario 322001
```

两条均为只读预览，不导入PyTorch、不写输出。显式运行使用空目录：

```powershell
python -m wt.cli.evaluate --scenario 322001 --output results/reproducibility/manual_run --run
```

当前仅登记322000/322001参考，不开放其他场景。已保存的322001结果为48环境步、48低层调用、4高层调用，payload=68952073.58225587、served=68952073.58225589，过程/reset种子1322001/322001；退出0、matched。此次目录整理不再运行模型。

```powershell
python scripts/benchmark.py
```

该入口只读以下已存在的历史原件：
- results/reproducibility/formal_scheme_c/evidence_archive/results/phase6iab_planner_stability/summary.json
- results/reproducibility/formal_scheme_c/evidence_archive/results/phase6iac_independent_holdout/summary.json

之前的“摘要缺失”来自旧路径映射遗漏；文件一直存在，无需重跑。

configs/resources.json记录当前权重和参考位置。wt.common.resources.resolve_logical解析旧文件逻辑路径；旧目录已拆分为不同功能目录，请按具体工件路径或迁移索引查找，不再依赖旧轮次目录。
