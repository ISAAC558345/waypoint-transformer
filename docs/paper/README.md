# 论文与证据

当前可编辑正文是 **2026-11-08** 稿。11 月 2 日和 11 月 4 日正文保留原内容。入口是带 11 月 8 日日期的 Markdown。`Revised_20261102.md` 只保存从提交 `b30200cf8fd9fda0c5dece7bceb52edaa918f9bc` 恢复的 11 月 2 日同名稿。

源码已更新，并且这次用项目内的 Tectonic 0.17.0 实际编译出了 7 页 PDF。7 页都已查看。旧 PDF 仍是先前的独立排版，不是这次编译的产物。

- [2026-11-08 可编辑正文](Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261108.md)
- [2026-11-08 编译 PDF](Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261108.pdf)
- [2026-11-04 可编辑正文](Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261104.md)
- [2026-11-02 历史正文](Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261102.md)
- [2026-11-02 修订论文 PDF](Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261102.pdf)
- [LaTeX 主入口](latex/main.tex) · [编译与修改说明](latex/README_CN.md)
- [环境与数据流图片](figures/README_CN.md)
- [2026-11-02 中文修订说明](REVISION_NOTES_20261102_CN.md)
- [2026-11-04 源码收尾说明](REVISION_NOTES_20261104_CN.md)
- [2026-11-08 修订说明](REVISION_NOTES_20261108_CN.md)
- [2026-11-02 数值与来源核查](evidence_20261102.json)
- [2026-11-04 AB/AC 与 E19 索引](evidence_20261104.json)
- [2026-11-08 Scheme C 聚合索引](evidence_20261108.json)
- [原六页论文 PDF](Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV.pdf)

E19 的完成结论来自固定提交 `f9e9a9e007d7d84046232181fdab6f9a732a4a05`：T 与 M 在 16 个状态上选择相同，没有采用。正式重放是 1488 步，另有 30 步作废计数，合计 1518。AB/AC 汇总和 episode 已按归档核对。Scheme C 在 best epoch 1 的原保存聚合已核对；ranking-loss 的 summary.json 和基线侧格值仍然缺失。本次未运行新实验。正式 Planner、环境和奖励没有修改。

## 维护研究索引

[L3–L5 到达标签探索结案](../research/Legacy_ArrivalAware_L3_L5_Closure_CN.md) 记录这条已测试配置。2026-11-08 正文和这次编译的 PDF 写入了该结案的短述。11 月 2 日、11 月 4 日正文和更早的 PDF 没有改写。编译状态、工具和产物哈希见 [2026-11-08 源码清单](latex/source_manifest_20261108.json)。
