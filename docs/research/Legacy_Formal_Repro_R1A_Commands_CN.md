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

## 以后主动重跑

```text
py -3 -m wt.legacy_formal_repro_r1.run --rerun
```

本包不执行这条命令。不带参数运行 `wt.legacy_formal_repro_r1.run` 也只打印边界，不再因为发现已有 episode 就重写清单。
