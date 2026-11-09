# 2026-11-09 修订说明

这次修订发生在目录迁移和一次场景 322001 的 48 步一致性运行之后。2026-11-08 及更早的正文、TeX 和 PDF 字节没有被覆盖；11 月 8 日正文的归档副本是 `docs/paper/latex/archive/main_20261108.tex` 与 `body_20261108.tex`。

正文增加了 L8、L9 的未采用对照和 L10 的分场景参考说明。ListNet 对照是 L3B10，不是方案 C Epoch 1。两组对照都是历史训练集上的 128 候选、15 步离线比较，不是独立泛化，也不是 48 步正式增益。L8 因 t30 未过门槛为 NOT_PROMOTED，L9 因低于 E6 为 NOT_PROMOTED。正式模型没有更换。

Tectonic 0.17.0 编译退出码为 0，PDF 为 7 页。七页都已查看。构建日志在 `backup/local_build/l11/tex/main.log`，不提交。
