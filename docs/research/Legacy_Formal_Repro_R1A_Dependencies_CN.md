# R1A 依赖说明

只读入口 `wt.legacy_formal_repro_r1a` 只使用 Python 标准库。读取已有结果不安装 torch，检查记录里的 `torch_imported` 为 false。模型运行使用下面的历史记录，不把这次文档整理当成一次新安装。

原运行环境保存在 `results/legacy_formal_repro_r1/run_manifest.json` 的 `runtime` 字段，并原样复制到派生清单的 `provenance.original_runtime`：

- Python 3.14.2
- NumPy 2.4.1
- torch 2.12.0+cpu
- 平台 Windows-11-10.0.26200-SP0

仓库里的历史 `requirements.txt` 仍钉住 `numpy==1.21.6` 和 `torch==1.13.0`。本包没有修改这个文件，也没有按它重装环境。

本次检查进程的 Python 是 3.14.2，平台与原记录相同。派生清单把这次检查环境单独放在 `check_environment`。`new_machine_install_verified` 为 false：没有在另一台机器上安装依赖并完成核验。
