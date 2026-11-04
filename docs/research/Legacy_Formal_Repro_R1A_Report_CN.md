# 原系统正式复现 R1A

本包只修正已保存结果的读取方式。环境步、训练更新、策略推理、模型加载和大体积下载都是 0。原 192 步、40 个归档文件和 `results/phase6if_ranking_loss/summary.json` 的 MISSING 状态都留在 `results/legacy_formal_repro_r1`。

## 读取行为

修复前，`wt.legacy_formal_repro_r1.run` 在发现已有 episode 时调用 `finalize_saved()`。该函数重写 `run_manifest.json`、证据索引、失败索引、运行时说明和中文报告，把当前机器的依赖、路径和模型身份写进原运行记录。配置核对读取臂记录里的绝对 `policy_path`，结果目录换到另一份检出后会继续指向原机器路径。

修复后，不带参数运行该入口只打印三条命令并返回 0，不写文件。`--rerun` 和 `--finalize` 返回 2，本包没有执行重跑。只读命令是：

```text
py -3 -m wt.legacy_formal_repro_r1a.inspect --artifact-root results/legacy_formal_repro_r1
```

显式导出命令是：

```text
py -3 -m wt.legacy_formal_repro_r1a.export --artifact-root results/legacy_formal_repro_r1 --output results/legacy_formal_repro_r1a
```

以后的主动重跑命令是 `py -3 -m wt.legacy_formal_repro_r1.run --rerun`。本包没有执行它。命令说明见 `docs/research/Legacy_Formal_Repro_R1A_Commands_CN.md`。

配置按 `scenario_id` 和策略名映射到 `configs/{scenario_id}_{policy}.json`，策略名只接受 `scheme_c` 和 `g1_rate`。读取后核对 `policy_sha256_after`。文件缺失时报告这条相对路径。原绝对路径保留在派生视图的 `original_path`，查找不再只靠它。

## 来源分层

| 角色 | 记录 |
| --- | --- |
| 原运行源码 | `ed2c87c83dcdaccee1420ee3e14a8dfa19982e92` |
| 结果整理源码 | `1abcf1b264b6df1763cbcf2cfe410923cdc57dae` |
| 当前检查源码 | `src/wt/legacy_formal_repro_r1a` 下的四个读取文件 |
| 原运行环境 | Python 3.14.2，NumPy 2.4.1，torch 2.12.0+cpu |
| 本次检查环境 | Python 3.14.2，`torch_imported` 为 false |

依赖说明见 `docs/research/Legacy_Formal_Repro_R1A_Dependencies_CN.md`。`new_machine_install_verified` 为 false，本次没有在新机器上安装并核验。当前检查环境写在派生清单里，原清单的 `runtime` 保持原字节。

## 三项检查

`tests/test_legacy_formal_repro_r1a.py` 在一次临时目录流程里完成，结果为 3 项通过，用时 10.009 秒。记录见 `results/legacy_formal_repro_r1a/necessary_checks.json`。

1. 清单和四份配置被复制到另一个临时根。四份配置都从该根的显式相对路径解析，哈希与原 `policy_sha256_after` 一致。检查没有加载模型。
2. 只读检查、一次导出，以及默认入口、`--finalize`、`--rerun` 之后，原清单、四份 episode、四份配置和 `docs/research/Legacy_Formal_Repro_R1_Report_CN.md` 的字节都不变。对同一导出目录再导出一次时，命令拒绝覆盖。
3. 派生视图的四项 payload、两项配对差值和 192 步计数与原清单一致。`source_steps_are_a_new_run` 为 false，读取端环境步为 0。

## 原结果摘要

| 场景 | Scheme C（bit） | G1_RATE（bit） | 差值（bit） |
| --- | ---: | ---: | ---: |
| 322000 | 63785945.28984571 | 64731141.622913174 | -945196.3330674618 |
| 322001 | 68952073.58225587 | 63445121.872805715 | 5506951.7094501555 |

四个臂各 48 步，合计 192 步。这个计数属于原运行。派生清单在 `results/legacy_formal_repro_r1a/`。

## 剩余事项

`results/phase6if_ranking_loss/summary.json` 仍是 MISSING。本包没有再搜索磁盘，也没有重跑训练或 holdout。历史套件的 23 个失败和 183 个错误保持原分类，没有重跑。原标签服务和算法改进目标仍开放。本交付只完成读取修复，不表示这些科学目标已经达成。
