# R1A 命令边界

只读入口使用 Python 标准库，不导入 torch，也不加载 checkpoint。下面的路径相对于仓库根目录。

## 读取已有结果

```text
py -3 -m wt.legacy_formal_repro_r1a.inspect --artifact-root results/legacy_formal_repro_r1
```

这个命令只打印摘要。它不写清单、不写报告、不重跑 192 步。

## 显式导出新视图

```text
py -3 -m wt.legacy_formal_repro_r1a.export --artifact-root results/legacy_formal_repro_r1 --output results/legacy_formal_repro_r1a
```

导出目录必须和原结果目录分开。目标文件已经存在时，命令拒绝覆盖。派生清单写在新目录，原 `run_manifest.json` 和原报告保持原字节。

## 当前主动重跑状态

`--rerun` 与 `--finalize` 仍然返回 2，继续保护原始 192 步证据。现行正式单场景命令是 `wt.legacy_system.run`，默认只预览配置；显式 `--run` 才执行。命令和已完成的 48 步输出见 [当前可用命令](Legacy_System_Commands_CN.md)。

从干净检出开始的完整命令与src路径设置见 [当前可用命令](Legacy_System_Commands_CN.md)。
