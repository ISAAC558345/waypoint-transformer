# 论文修订（2026-11-08）

这一版从 11 月 4 日正文复制后修改。11 月 2 日和 11 月 4 日正文、以及更早的 PDF，保持原字节。11 月 4 日的 `main.tex` 和 `body.tex` 在修改前归档到 `latex/archive/`。

源码改动：

- ListNet 目标与代码一致：分母是 `max(max y - min y, 1e-6)`，softmax 先减去最大 quality，目标脱离计算图。
- 写入 L3–L5 结案。两个已见训练池场景、四个 15 步窗口、候选子集限制都保留。B 的原均值是 20.150 Mbit，A 是 25.905，E6 初始化是 25.863。行均值 idle 参照把 B 的描述性均值升到 24.404 Mbit，但 t=30 有负差并降低对照组，两条规则都没有采用。这不是新的 48 步策略收益，也不是泛化通过。边统计使用 L5 更正后的 36 条边。单例组的 0 regret 不是排序证据。
- L6 的正式命令完成了 48 步，并与原 R1 的逐步 served 和 payload 匹配。这是工程回归，不是新性能实验。固定提交是 `77c73c00462ad2bb3d778d12a2ad98f0e4ffd576`。
- Scheme C 列改为核对原保存聚合 `runs/c/result.json` 的 best epoch 1。基线侧、顶层 summary 和逐状态序列仍缺失。原 result 的 loss 是非数值，没有改成 0。
- 标题日期不再写“本次未编译”。构建结果写在 README 和源码清单里。

编译：项目内 Tectonic 0.17.0 退出码 0，生成 7 页 PDF。7 页都已查看。日志没有 Overfull，引用已解析。GitHub 发布包下载返回 504，实际使用的是同一安装说明中的 conda-forge 0.17.0 预编译包。细节见 `latex/TECTONIC_L7_CN.md`。

没有新的仿真、训练或 holdout。
