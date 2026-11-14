# LaTeX论文源码

当前main.tex/body.tex对应2026-11-09稿，同日PDF在上级目录。archive子目录按日期保存历史论文源码，不用开发轮次组织当前代码。

现有构建入口：
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File docs/paper/latex/build_l11_tectonic.ps1
```

脚本使用项目内backup/local_tools/tectonic-0.17.0与TeX缓存，临时产物在backup内。保存的构建退出0、7页，页面查看记录见source_manifest_20261109.json。2026-11-14没有重新编译；仅更正清单的Git对象/换行身份及11月8日归档源码原字节。

修改正文时同步同日Markdown和body.tex，再做必要编译与页面检查。references.tex使用手工bibliography，无需BibTeX。旧PDF与实验原始证据保留。
