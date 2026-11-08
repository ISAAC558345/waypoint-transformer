# 原系统当前可用命令

以下命令在 `feature/legacy-system-maintenance` 的仓库根目录执行。原 192 步不重跑，历史全套不运行。现行正式单场景入口和受保护的 R1 入口分开列出。

## 路径

仓库根目录是包含 `src` 和 `results` 的目录。Python 需要能从 `src` 导入：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
```

R1 的结果根目录是包含其 `run_manifest.json` 的 `results/legacy_formal_repro_r1`。配置按场景和策略名读取 `configs/{scenario_id}_{policy}.json`，不把保存的绝对路径当作唯一查找位置。正式单场景结果按场景分开：未指定 `--output` 时写入 `results/legacy_system/scenario_{scenario_id}`。显式路径仍然有效，已完成的 322000 结果留在 `results/legacy_system_l6`。查找用相对路径。

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

预览只使用标准库，不建输出目录，不加载模型。当前登记了两套已有完整 R1 参考的训练场景，322000 和 322001。这不是整个训练池、validation 或 reserved test 的回归。

```powershell
py -3 -m wt.legacy_system.run --scenario 322000
py -3 -m wt.legacy_system.run --scenario 322001
py -3 -m wt.legacy_system.run --scenario 322000 --output results/legacy_system_l6
```

前两条使用各自的默认输出 `results/legacy_system/scenario_322000` 和 `results/legacy_system/scenario_322001`。第三条沿用已文档化的显式路径。每条预览打印该场景自己的参考 episode、参考 config、`process_seed=1000000+scenario` 和 `reset_seed=scenario`。`run_accepted` 只在场景已登记时为 true。未登记场景打印未接受，参考路径为空。

预览同时打印 Scheme C、G1_RATE、k=6、hold=15、stop_time=48。`historical_formal_k=5` 只保留旧候选数的来源，现行候选数是 k=6。

## 正式单场景运行

显式 `--run` 才调用正式 `run_active_episode`。场景、参考文件、种子和默认输出由同一处标准库注册表解析。比较读取该场景的 R1 episode，不再把 322000 的 payload 常量套到其他场景。未登记场景在加载模型和创建输出前返回 2。

```powershell
py -3 -m wt.legacy_system.run --scenario 322000 --output results/legacy_system_l6 --run
py -3 -m wt.legacy_system.run --scenario 322001 --output results/legacy_system_l10/run_322001 --run
```

L6 已把 322000 的一次 48 步结果写入 `results/legacy_system_l6`。再执行同一输出目录不会重跑。L10 已把 322001 的一次 48 步结果写入 `results/legacy_system_l10/run_322001`。换一次新的运行需要另一个空目录。

成功结束时进程返回 0，保存的 JSON 使用同一个退出码。执行完成、比较 matched、模型身份正确且正式策略未变，才算成功。运行失败或结果不一致时返回 1。输入或输出被拒绝时返回 2。没有对应参考时不标成 matched。已有的空目录可以准备；非空目录仍拒绝覆盖，并且不写入该目录。L7 用已保存记录核对了目录约定，没有重跑 48 步。

## 受保护的 R1 入口

不带参数执行 `py -3 -m wt.legacy_formal_repro_r1.run` 只打印边界并返回 0，不写文件。

```text
py -3 -m wt.legacy_formal_repro_r1.run --rerun
py -3 -m wt.legacy_formal_repro_r1.run --finalize
```

这两条仍然返回 2，继续保护原始 192 步证据。它们不是当前唯一的主动运行方式；现行单场景命令是上面的 `wt.legacy_system.run`。

## 依赖

标准库读取与模型运行分开。已记录的模型运行版本、历史 `requirements.txt` 钉扎，以及“新机器安装未验证”的说明见 [依赖说明](Legacy_Formal_Repro_R1A_Dependencies_CN.md)。不要为了阅读结果或编写文档而安装 torch。
