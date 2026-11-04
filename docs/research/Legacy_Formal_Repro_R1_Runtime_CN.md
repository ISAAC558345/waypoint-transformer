# R1 正式入口运行依赖

本文件记录这次正式入口实际导入的版本。历史 `requirements.txt` 保持原样，不按本文件批量升级旧模块。

- Python：3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)]
- NumPy：2.4.1
- torch：2.12.0+cpu
- 历史 requirements.txt 中的 NumPy 行：numpy==1.21.6
- 历史 requirements.txt 中的 torch 行：torch==1.13.0

当前研究环境的 Python、NumPy 和 torch 与历史 requirements.txt 的钉扎不一致。正式入口使用当前已安装的版本运行，没有另装一套依赖。
