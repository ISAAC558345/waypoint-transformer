# LaTeX 论文源码

本目录的源码已在 2026-11-04 更新到完成的 E19。2026-11-02 的修订 PDF 仍是当时的编译件，不含这次源码更新。
它是从现有修订正文转写的可编辑源码，不是从原 PDF 恢复出来的作者原始工程。

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

正文、公式和表格已随 2026-11-04 源码更新。引用仍使用手工 bibliography，没有 BibTeX。
本机没有 pdflatex、xelatex、latexmk 或 tectonic，本次没有实际编译，也没有核对分页或引用解析。
仓库中已存在的两份 PDF 保持原字节。LaTeX 源码现在与这两份 PDF 不一致，不能把旧 PDF 当成已包含 E19 完成结论的编译件。
本次未运行实验、训练、保留集或仓库测试；未修改正式系统代码。
