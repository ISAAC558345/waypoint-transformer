# 原系统当前可用命令

以下命令在 `feature/legacy-system-maintenance` 的仓库根目录执行。原 192 步不重跑，历史全套不运行。现行正式单场景入口和受保护的 R1 入口分开列出。

## 路径

仓库根目录是包含 `src` 和 `results` 的目录。Python 需要能从 `src` 导入：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
```

R1 的结果根目录是包含其 `run_manifest.json` 的 `results/legacy_formal_repro_r1`。配置按场景和策略名读取 `configs/{scenario_id}_{policy}.json`，不把保存的绝对路径当作唯一查找位置。正式单场景结果另写在 `results/legacy_system_l6`，查找用其中的相对路径。

## 只读

```powershell
py -3 -m wt.legacy_formal_repro_r1a.inspect --artifact-root results/legacy_formal_repro_r1
```

这个入口只使用 Python 标准库。读取已有结果不需要安装 torch，也不加载 checkpoint。它只打印摘要，不写原清单、原报告或原 episode。

## 显式导出

```powershell
py -3 -m wt.legacy_formal_repro_r1a.export --artifact-root results/legacy_formal_repro_r1 --output results/legacy_formal_repro_r1a_view
```

导出目录必须是新目录。目标文件已经存在时，命令拒绝覆盖。派生视图与原 192 步记录分开保存。

## 正式配置预览

预览只使用标准库，不建输出目录，不加载模型。

```powershell
py -3 -m wt.legacy_system.run --scenario 322000 --output results/legacy_system_l6
```

预览打印 Scheme C、G1_RATE、k=6、hold=15、stop_time=48。`historical_formal_k=5` 只保留旧候选数的来源，现行候选数是 k=6。

## 正式单场景运行

显式 `--run` 才调用正式 `run_active_episode`。当前回归只接受场景 322000，输出目录里已有文件时拒绝覆盖。

```powershell
py -3 -m wt.legacy_system.run --scenario 322000 --output results/legacy_system_l6 --run
```

L6 已把一次 48 步结果写入 `results/legacy_system_l6`。再执行同一输出目录不会重跑。换一次新的运行需要另一个空目录，并且不在本包内执行。

成功结束时进程返回 0，保存的 JSON 使用同一个退出码。运行失败、回归不一致、模型身份不符或正式策略被改动时返回 1。输入被拒绝时仍返回 2。已有的空目录可以准备；非空目录仍拒绝覆盖，并且不写入该目录。L7 用已保存记录核对了这条约定，没有重跑 48 步。

## 受保护的 R1 入口

不带参数执行 `py -3 -m wt.legacy_formal_repro_r1.run` 只打印边界并返回 0，不写文件。

```text
py -3 -m wt.legacy_formal_repro_r1.run --rerun
py -3 -m wt.legacy_formal_repro_r1.run --finalize
```

这两条仍然返回 2，继续保护原始 192 步证据。它们不是当前唯一的主动运行方式；现行单场景命令是上面的 `wt.legacy_system.run`。

## 依赖

标准库读取与模型运行分开。已记录的模型运行版本、历史 `requirements.txt` 钉扎，以及“新机器安装未验证”的说明见 [依赖说明](Legacy_Formal_Repro_R1A_Dependencies_CN.md)。不要为了阅读结果或编写文档而安装 torch。
