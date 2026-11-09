# L11 目录迁移与一致性复核

场景 322001 的一次 48 步运行与它自己的参考一致。payload 为 68952073.58225587 bit，served 为 68952073.58225589 bit，process_seed 为 1322001，reset_seed 为 322001，决策时刻为 0/15/30/45，高层调用 4 次，低层推理 48 次，环境步 48，训练更新 0，反向传播 0，比较结果 matched，退出码 0。

主检出已经按功能目录放置。正式入口是 `python -m wt.cli.evaluate`。方案 C、MAPPO-V2、decoder 和 Epoch 6 的 SHA-256 与迁移前合同一致。正式策略哈希仍是 `3fc2f0b1277188e6f12084370c9c1d5acb70e897f6a4fba8b47bedf031b25ebb`。L8 和 L9 仍是 NOT_PROMOTED。

## 迁移范围

36 个原 round 材料目录进入 `archive/experiments`。当前正式链路进入 `src/wt` 下的环境、模型、规划、控制、评估和数据目录。未采用的研究代码进入 `src/wt/research`，其余历史包进入 `src/wt/legacy`。权重离开 `src`，正式权重复核通过。根目录不再有 round 编号目录；嵌套的 round 编号目录也已改名。`.git`、虚拟环境、`backup/snapshots` 和 `backup/baseline_worktrees` 没有纳入改名。

`docs/development/path_migration_20261109.json` 记录目录移动和索引中的重命名。`git_blob` 是重写前的索引对象。`content_changed` 标出工作区已改写的文件。历史 JSON、日志和 PDF 里的旧路径字符串没有改写。

## 必要检查

预览 322000 和 322001 没有加载 PyTorch。正式规划器可以导入，四个正式权重和策略文件都在新路径上。已保存的 L6 与 322000 参考比较为 matched，322001 参考与自身比较为 matched，322000 与 322001 的交叉比较为 different。`scripts/benchmark.py` 报告验证摘要和保留集摘要缺失，没有重跑。

PDF 由 Tectonic 0.17.0 编译，退出码 0，共 7 页，七页都已查看。新表的数值完整，公式和引用没有被裁切。

## 消耗

这次新的环境步是 48。训练更新和反向传播是 0。没有重跑 322000、L8、L9 或保留集。

## 未完成项

验证集 322048–322063 与保留集 322064–322079 的历史摘要仍不在检出中。没有开始下一项模型研究。
