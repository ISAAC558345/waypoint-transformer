# 2026-11-08 Tectonic 构建

优先路线是 Tectonic 官方 GitHub 预编译包：

`https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip`

发布页登记的 sha256 是 `f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`。直接请求和经本地代理的请求都由 GitHub 边缘返回 HTTP 504，响应正文是 “504 Gateway Time-out”。这个 zip 没有下载到，所以上面的 sha256 只是发布页登记值，不是本机文件哈希。0.15.0 的同类地址同样返回 504。

实际编译使用官方安装说明里的 conda-forge 预编译包，版本同为 0.17.0：

`https://conda.anaconda.org/conda-forge/win-64/tectonic-0.17.0-h8a4d0b3_2.conda`

文件大小 5684028，sha256 `0e50ad47a61b38b7b7545b7b2357aff73c581be3a18b22390189fb820855a4c8`，与 Anaconda 清单一致。可执行文件解到 `backup/local_tools/tectonic-0.17.0/bin/tectonic.exe`，sha256 `298339ebfa4f9cb33f0fae20e9eb8edb96dfc6bd8b401967b765e9b1a649b144`。它的 fontconfig、ICU、HarfBuzz 等运行库也放在同一 `bin` 目录，没有安装系统级 TeX，也没有改系统 PATH。`tectonic --version` 输出 `Tectonic 0.17.0`。

仓库根路径长 168 字符，缓存文件路径超过 Tectonic 格式文件写入所能接受的长度，第一次报 `cannot write format file texput.fmt`（os error 3）。构建脚本因此在进程内临时映射一个空闲盘符，编译结束后删除。缓存目录仍是 `backup/local_build/l7/tectonic-cache`。

成功命令的形式：

```text
tectonic -X compile --keep-logs --keep-intermediates --outdir backup/local_build/l7/tex main.tex
```

实际调用使用项目内绝对路径，工作目录是 `docs/paper/latex`。进程环境设置 `TECTONIC_CACHE_DIR`、`TEMP` 和 `TMP`。退出码 0。日志是 `l7_build.log`。产物 `backup/local_build/l7/tex/main.pdf` 已另存为 `docs/paper/Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261108.pdf`，共 7 页。

日志有 Underfull hbox，没有 Overfull box，也没有未定义引用。7 页都渲染成 PNG 并逐页查看过：公式、四张表、架构图、列宽、引用和页码可见，没有缺字成方框。页面 PNG 留在 `backup/local_build/l7/pages/`，不提交。
