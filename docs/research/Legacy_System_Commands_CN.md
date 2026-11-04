# 原系统当前可用命令

以下命令在 `feature/legacy-system-maintenance` 的仓库根目录执行。本轮不重跑原 192 步，也不运行历史全套。

## 路径

仓库根目录是包含 `src` 和 `results` 的目录。Python 需要能从 `src` 导入：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
```

结果根目录是包含 `run_manifest.json` 的目录，当前为 `results/legacy_formal_repro_r1`。配置按场景和策略名读取 `configs/{scenario_id}_{policy}.json`，不把保存的绝对路径当作唯一查找位置。

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

## 主动运行

不带参数执行 `py -3 -m wt.legacy_formal_repro_r1.run` 只打印上述边界并返回 0，不写文件。

```text
py -3 -m wt.legacy_formal_repro_r1.run --rerun
py -3 -m wt.legacy_formal_repro_r1.run --finalize
```

这两条现在都返回 2，不能当作当前可执行的重跑或覆写入口。主动运行模型需要以后另定独立输出目录和步数预算。本维护交付不执行它们。

## 依赖

标准库读取与模型运行分开。已记录的模型运行版本、历史 `requirements.txt` 钉扎，以及“新机器安装未验证”的说明见 [依赖说明](Legacy_Formal_Repro_R1A_Dependencies_CN.md)。不要为了阅读结果或编写文档而安装 torch。
