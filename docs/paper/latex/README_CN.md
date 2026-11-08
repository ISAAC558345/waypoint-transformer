# LaTeX 论文源码

本目录的当前源码是 2026-11-08 稿。修改前的 11 月 4 日 `main.tex` 和 `body.tex` 字节保存在 `archive/main_20261104.tex` 与 `archive/body_20261104.tex`。它从修订正文转写而来，不是从 PDF 反推的作者原始工程。

`docs/paper/` 中 11 月 8 日之前的 PDF 保持原字节。它们来自先前的独立排版，不是这次编译的产物。2026-11-08 的 PDF 由下面的项目内 Tectonic 命令生成。

## 文件

- `main.tex`：主入口、包、标题和版式。
- `body.tex`：正文、公式、四张结果/范围表。
- `references.tex`：参考文献；使用手工 bibliography，无需 BibTeX。
- `figures/formal_architecture.tex`：可编辑 TikZ 正式架构图。
- `figures/system_schematics.tex`：新环境图和数据流图的可选插图入口。
- `../figures/`：两张中文 PNG 示意图及生成说明。
- `.latexmkrc`：pdfLaTeX 编译设置。

## 编译

在本目录执行：

```sh
latexmk -pdf main.tex
```

也可执行两遍 `pdflatex -interaction=nonstopmode -halt-on-error main.tex`。
在 Overleaf 中上传整个 `latex` 目录，主文件选择 `main.tex`，编译器选择 pdfLaTeX。
若启用新中文插图，还需保留相邻的 `figures` 目录，或相应修改图片路径。
这些中文文字已经包含在 PNG 内，不需要安装 CJK 字体。

## 插入本次两张图片

取消 `main.tex` 中 `\input{figures/system_schematics}` 前的 `%`。
默认保留正文原有的 TikZ 图，避免在未决定版式前改变论文内容。

## 2026-11-08 编译

在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File docs\paper\latex\build_l7_tectonic.ps1
```

脚本使用 `backup/local_tools/tectonic-0.17.0/bin/tectonic.exe`，把 `TECTONIC_CACHE_DIR` 指到 `backup/local_build/l7/tectonic-cache`，输出到 `backup/local_build/l7/tex`。仓库路径对 Tectonic 的格式文件写入过长，脚本在进程内临时映射一个盘符，结束后删除。工具和缓存不提交。来源、504 记录和实际产物见 [Tectonic 说明](TECTONIC_L7_CN.md) 与 [2026-11-08 清单](source_manifest_20261108.json)。

这次编译退出码为 0，得到 7 页 PDF。日志只有 Underfull hbox，没有 Overfull，也没有未定义引用。7 页 PNG 已逐页查看。

## 核查状态

11 月 4 日清单仍是 `source_manifest.json`。2026-11-04 的 L2 只做了一次 PATH 检查：当时 pdflatex、xelatex、lualatex、latexmk、tectonic 都不在 PATH。那次没有编译。上面的 2026-11-08 编译是另一次实际构建。
引用仍使用手工 bibliography，没有 BibTeX。
本次未运行实验、训练、保留集或仓库测试。
