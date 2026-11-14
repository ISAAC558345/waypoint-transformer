# 权重

本分支不再保存旧通信系统的权重。方案 C 与 MAPPO-V2 权重仍在默认分支 `feature/maritime-wt-stage1a`（`e58e5d190109e5a0e860cb5c6a412321c0d74e31`），也仍在本分支清理前的提交 `e44e5227caa71c2bb5150fe7a1ab62c6fd7fbd05`。

海洋仿真不加载这些权重。`src/wt/ocean_observation_m0/parameters.py` 里的 `FORMAL_CHECKPOINT` 仍是原来的路径字符串，只写入来源记录，运行时不打开该文件。
