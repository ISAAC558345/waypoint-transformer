# 论文源码收尾（2026-11-04）

本次只更新旧系统论文源码。两份已有 PDF 保持原字节：原六页 PDF，以及 2026-11-02 的修订 PDF。本机没有 pdflatex、xelatex、latexmk 或 tectonic，所以没有生成新 PDF，也没有核对编译页面。源码中的新句子尚未经过编译验证。

已核对的内容：

- 归档提交 `8a34efc718665a7cfe4375b662e8f5249263f03d` 的 AB/AC 汇总和 32 份 episode。表中的兆比特数值按三位小数与汇总一致，episode 的 payload 与汇总行一致。
- 保留集的两个负差仍是 322065 的 -8.162 Mbit 和 322071 的 -1.871 Mbit。
- `results/phase6if_ranking_loss/summary.json` 仍然缺失，没有补造。
- E19 使用固定结果提交 `f9e9a9e007d7d84046232181fdab6f9a732a4a05`。16 个状态上 T 与 M 的选择相同，机制证据是 `NO_TAG_DECISION_EFFECT`，没有 performance pass，也没有采用。

论文没有加入新方案内容。本次没有新的仿真、训练或 holdout。

L2 把当前正文另存为 `Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261104.md`，并把 `Revised_20261102.md` 恢复为提交 `b30200cf8fd9fda0c5dece7bceb52edaa918f9bc` 的 11 月 2 日稿。两份 PDF 仍是先前独立排版，不是本 LaTeX 工程已核验的编译结果。正式重放 1488 步，另有 30 步作废计数，合计 1518。
