# LaTeX 论文源码

本目录的源码是 2026-11-04 稿，证据覆盖至完成的 E19。它从修订正文转写而来，不是从 PDF 反推的作者原始工程。

`docs/paper/` 中的两份 PDF 保持原字节。它们来自先前的独立排版，本包没有核验它们是这个 LaTeX 工程的编译结果，也不能用它们代替一次成功编译。

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

## 核查状态

正文、公式和表格属于 2026-11-04 源码。引用仍使用手工 bibliography，没有 BibTeX。
旧的结构检查结果只属于被归档的清单，不能转到当前稿。当前清单由本包重新计算。
2026-11-04 的 L2 只做了一次可用性检查：pdflatex、xelatex、lualatex、latexmk、tectonic 都不在 PATH，TeX Live 和 MiKTeX 的常见安装目录也不存在。因此没有编译，分页和引用解析仍未完成。
本次未运行实验、训练、保留集或仓库测试；未修改正式系统代码。
