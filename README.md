# 当前正式系统

这是多无人机海上数据采集的正式规划代码。高层用方案 C 的异构图网络对合法分配打分，低层用冻结的 MAPPO-V2 执行，再经过安全投影。正式策略是 `G1_RATE`、方案 C、`k=6`、`hold=15`。已注册并可运行的场景只有 322000 和 322001。

预览不加载模型，也不创建结果目录：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
py -3 -m wt.cli.evaluate --scenario 322001
```

显式运行会执行 48 步。2026-11-09 的目录迁移后，场景 322001 已用下面的命令复核一次，结果与该场景自己的参考一致：

```powershell
py -3 -m wt.cli.evaluate --scenario 322001 --output results/reproducibility/layout_refactor_20261109/run_322001 --run
```

`scripts/evaluate.py` 只调用同一个入口。`scripts/benchmark.py` 只读取已有摘要；当前验证集 322048–322063 和保留集 322064–322079 的摘要文件不在检出中，脚本会报告缺失，不会重跑。

正式权重：

| 用途 | 路径 | SHA-256 |
| --- | --- | --- |
| 方案 C Epoch 1 | `checkpoints/formal/scheme_c_epoch1.pt` | `f77c0bcef3f359e01de35b75753f3776a9e8332bf65fd191f112236df074f864` |
| MAPPO-V2 seed 0 | `checkpoints/formal/mappo_v2_seed0.pt` | `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e` |
| matching decoder | `checkpoints/formal/matching_decoder.pt` | `de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5` |
| Epoch 6 初始化 | `checkpoints/initialization/joint_utility_epoch6.pt` | `7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde` |
| 正式策略 | `configs/formal_scheme_c_policy.json` | `3fc2f0b1277188e6f12084370c9c1d5acb70e897f6a4fba8b47bedf031b25ebb` |

L8 和 L9 的固定预算对照没有采用，正式模型没有更换。记录到的运行环境是 Python 3.14.2、NumPy 2.4.1、PyTorch 2.12.0+cpu。这不是一次新机器安装验证。

上游 Waypoint Transformer 的说明、许可证归属和旧实验材料在 `archive/upstream` 与 `archive/experiments`。当前规则见 `AGENTS.md`，使用说明见 `docs/USAGE.md`。
