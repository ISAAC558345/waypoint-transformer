# 多无人机通信与数据采集

本仓库实现高层任务分配与低层运动控制：高层采用冻结的 Scheme C Epoch 1 HGNN 对合法分配评分，低层采用冻结的 MAPPO-V2，并经过安全投影。当前策略为 G1_RATE、k=6、hold=15；默认分支提供按功能组织的代码。

## 目录

```text
configs/                    当前配置与历史配置
src/wt/
  envs/                     通信与位队列环境
  models/                   HGNN、matching decoder及低层网络
  planning/                 候选生成、正式规划器、任务接口
  control/                  冻结MAPPO执行与安全投影
  evaluation/               回合运行、统计与参考比较
  algorithms/               排序损失及已有算法公共函数
  data/                     构图、快照和数据加载代码
  common/                   项目根与资源路径解析
  cli/                      当前命令入口
  research/                 已完成的研究实现
  legacy/                   历史实现
data/                       输入数据与数据包说明
checkpoints/                正式、初始化及历史权重
results/                    已有基准、复现记录和研究输出
scripts/                    当前命令及历史脚本
tests/                      功能测试与历史测试
docs/                       使用、系统说明、论文和历史来源
```

原轮次目录中的文件已按内容拆入配置、权重、结果、文档和来源记录目录。历史材料没有删除；旧路径通过[迁移索引](docs/development/functional_layout_20261114.json)和资源解析器定位。源码目录不存放实验 checkpoint。

## 快速开始

在已有可运行的 Python 环境中，从仓库根目录执行；将下面的 python 替换为所用解释器。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m wt.cli.evaluate --scenario 322001
```

默认仅预览，不加载模型、不创建输出目录。目前登记的参考场景为322000和322001。显式运行需要新的或空的输出目录：

```powershell
python -m wt.cli.evaluate --scenario 322001 --output results/reproducibility/manual_run --run
```

该命令运行48步。已有322001回归的payload为68,952,073.58225587 bit，逐步served与其R1参考一致。已记录的模型运行环境为Python 3.14.2、NumPy 2.4.1、PyTorch 2.12.0+cpu；这是已有环境记录，不是新机器安装验证。

## 已有结果与论文

```powershell
python scripts/benchmark.py
```

这条命令只读取已归档的验证与保留集摘要，不重新执行实验。历史场景级配对均值分别为+2.644991 Mbit和+3.502192 Mbit；它们不是本次目录整理产生的新收益。

- [使用说明](docs/USAGE.md)
- [系统说明](docs/SYSTEM_OVERVIEW_CN.md)
- [交付状态](docs/DELIVERY_STATUS_CN.md)
- [论文、LaTeX与PDF](docs/paper/README.md)
- [分支及历史基线](README_BRANCH_SCOPE.md)

正式权重在checkpoints/formal，E6启动/初始化权重在checkpoints/initialization；[资源配置](configs/resources.json)记录其位置。L8、L9研究模型均未采用，正式模型未更换。按用户当前选择，本次只整理旧系统，不训练新模型、不生成新标签、不合并海洋观测方案。

上游Waypoint Transformer/RvS说明保存在docs/archive/upstream，原启动脚本归入scripts/legacy/upstream；许可证与归属保留。
