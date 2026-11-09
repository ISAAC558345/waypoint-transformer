# 交付状态

当前正式模型仍是方案 C Epoch 1 加冻结 MAPPO-V2 seed 0。策略文件是 `configs/formal_scheme_c_policy.json`，内容为 G1_RATE、scheme_c、k=6、hold=15。

当前命令：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
py -3 -m wt.cli.evaluate --scenario 322001
```

加 `--run` 才会执行。已注册场景是 322000 和 322001。

最新论文是 `docs/paper/Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261109.md` 和同名 PDF。L8 与 L9 维持 NOT_PROMOTED，没有接入正式规划器。验证集 322048–322063 和保留集 322064–322079 的摘要文件不在当前检出中。

这次工作按用户选择完成现有系统的目录整理和论文收口，不开始新的模型研究。
