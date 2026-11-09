# 2026-11-09 目录结构

主检出按功能放置当前代码，不再保留 round 编号目录。

- `src/wt/envs`、`models`、`planning`、`control`、`evaluation`、`data`、`algorithms` 放正式链路和仍被它导入的模块。
- `src/wt/research` 放未采用的到达、优先级和海洋观察研究代码。
- `src/wt/legacy` 放其余历史包，包括原 round 包和上游 Waypoint Transformer。
- `checkpoints/formal` 与 `checkpoints/initialization` 放正式权重和 Epoch 6 初始化。其他权重在 `checkpoints/research`。
- `results/reproducibility` 放正式参考和 2026-11-09 的一次一致性运行。研究对照在 `results/research`。
- `archive/experiments` 放原 round 材料。`archive/upstream` 放上游说明和实验启动脚本。`archive/provenance` 放旧轮次状态。
- 根目录的 `src/envs`、`src/models`、`src/policies`、`src/algorithms`、`src/utils` 转发壳已删除。`src/wt/benchmark_v2` 和 `src/wt/lowlevel_v2` 只剩说明性 `__init__.py`，也已删除。

对应关系、Git blob 和内容是否改变记在 `path_migration_20261109.json`。历史原文里的旧路径字符串可以保留；当前入口通过 `configs/resources.json` 解释它们。
