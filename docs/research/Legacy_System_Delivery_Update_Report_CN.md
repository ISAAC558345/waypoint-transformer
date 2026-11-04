# 旧系统交付收尾

更新时间：2026-11-04。分支是 `feature/legacy-system-maintenance`，起点是 `bb408f854d012850a4a3dbe3d0246be8ccbcf9ad`。新方案仍在 `research/ocean-observation-isolated`，本包没有修改、执行或合并它。新环境步、训练更新和策略推理都是 0。原 192 步没有重跑，历史全套没有运行。

## 命令与依赖

只读、导出和主动运行已经分开写在 `docs/research/Legacy_System_Commands_CN.md`。只读入口使用标准库，不安装 torch。结果目录是包含清单的目录，配置使用相对路径。不带参数的旧运行入口只打印边界；`--rerun` 和 `--finalize` 返回 2，本包没有执行。

依赖说明区分三层：历史 `requirements.txt` 仍钉住 NumPy 1.21.6 和 torch 1.13.0；R1 原运行记录是 Python 3.14.2、NumPy 2.4.1、torch 2.12.0+cpu；`new_machine_install_verified` 仍为 false。没有为写文档重装这些依赖。

一次只读检查通过：`wt.legacy_formal_repro_r1a.inspect` 返回 0，打印的 192 步标记为原运行，原清单和原 R1 报告的字节不变。

## 论文证据

`docs/paper/evidence_20261104.json` 记录 2 份汇总和 32 份 episode。它们的 SHA-256 与归档提交 `8a34efc718665a7cfe4375b662e8f5249263f03d` 的原始 blob 一致。表中的兆比特数值按三位小数与汇总一致。开发汇总带有 `validation_is_development_set`。保留集没有重跑，两个负差仍是 322065 的 -8.162 Mbit 和 322071 的 -1.871 Mbit。episode payload 与汇总行一致。ranking-loss 的 `summary.json` 仍是 MISSING。

E19 引用固定结果 `f9e9a9e007d7d84046232181fdab6f9a732a4a05`，没有把该实验复制进本分支。16 个状态上 T 与 M 选择相同，机制证据是 `NO_TAG_DECISION_EFFECT`，科学门是 `FIXED_TAG_RULE_DIAG_COMPLETE`。performance pass、采用、发布和合并都是 false。

Markdown 与 LaTeX 源码已同步这些结论，论文只写旧系统。两份旧 PDF 保持原字节。本机没有 pdflatex、xelatex、latexmk 或 tectonic，所以没有新 PDF，也没有核对编译页面。源码更新因此仍待编译验证。

## 后续协议

到达感知高层对照写在 `docs/research/Legacy_ArrivalAware_HighLevel_AB_Protocol_CN.md`。唯一变量是离线标签来源。D5 的完成结论和 D6 的 inconclusive 结论都保留；`lambda_H` 与 `delta_payload_max` 仍未定。协议没有执行。

## 仍未完成

- 旧标签服务和算法改进目标仍开放。E19 没有被采用。
- ranking-loss 原 summary 仍缺。
- 论文源码尚未在本机编译，旧 PDF 不含这次 E19 收尾文字。
- 新机器安装仍未验证。
- 历史失败签名没有重跑，全量套件不是绿灯。
- 到达感知 A/B 还没有获准执行。
