# 实验报告：冻结 L0 场景 321101 的重放复现

- Gate：`REPLAY_DIVERGENCE_LOCALIZED`
- 原因字母：`A`
- 诊断分支：`experiment/priority-tag-replay-repro`
- 指令提交：`6e58a6dd4bc2beb4c644841dfde809d96cfa9b3b`
- 来源实验提交：`f4ec753d7611d72dc25ace3aeaa8e877d8ef92bb`
- 来源诊断提交：`51adade38d8f3da5402e67506cf875d80224e82d`
- 是否合并：否
- 是否重新训练：否

## 1. 范围

只检查 L0、模型种子 0、development 场景 321101 的 48 步。
没有重训 L1/L2，没有改奖励、动作、标签、队列顺序或 SafetyProjector。
记账容差保持 1 bit。路径相等检查是 1e-09 米，用来判断轨迹是否同一条，不代替 1 bit 容差。

## 2. 来源与环境

正式 checkpoint SHA-256：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。
来源实验提交中的相关文件哈希一致：True。
场景 split 校验通过：True。
Python：3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)]
NumPy：2.4.1；仓库 requirements 写的是 1.21.6。
PyTorch：2.12.0+cpu；仓库 requirements 写的是 1.13.0。
CUDA 可用：False。
重放开始前 torch 线程数：12；确定性算法开关：False。
已保存的评估 CSV 和 summary 没有逐步动作、观测、snap 或上述进程开关。

## 3. 对照

fresh 是新进程的默认 torch 状态，走未包装的原评估路径。
instrumented 使用同一默认状态和归因包装。
inherited 在同一进程中补上训练结束时留下的 `torch.set_num_threads(1)` 和 `seed_everything(2)`，再走未包装路径。
这三趟都只跑场景 321101，不扩大到 224 条。

reset 状态一致：True。
包装路径和 fresh 逐步一致：True。
fresh 相对已保存结果进入 1 bit 且路径相等：False。
inherited 相对已保存结果进入 1 bit 且路径相等：True。

- fresh 差：payload 1.553167 bit，普通服务 1.291123 bit，tagged 服务 0.262044 bit，普通丢弃 1.494478 bit，tagged 丢弃 0.058690 bit，紧急度 0.135255 bit，路径 0.000006 米。
- instrumented 差：payload 1.553167 bit，普通服务 1.291123 bit，tagged 服务 0.262044 bit，普通丢弃 1.494478 bit，tagged 丢弃 0.058690 bit，紧急度 0.135255 bit，路径 0.000006 米。
- inherited 差：payload 0.000000 bit，普通服务 0.000000 bit，tagged 服务 0.000000 bit，普通丢弃 0.000000 bit，tagged 丢弃 0.000000 bit，紧急度 0.000000 bit，路径 0.000000 米。

## 4. 首次分叉

fresh 相对 inherited 的第一处差异在 step 0，字段 `mu`。
该步的观测哈希一致，分叉发生在同一观测上的后续数值。
- mu 绝对差 0.000000035390
- desired 绝对差 0.000000008846
- projected 绝对差 0.000000006988
- executed 绝对差 0.000000006988
- position 绝对差 0.000000153729
- q_bits 绝对差 0.000430169865
- served 绝对差 0.001025728183
- payload 绝对差 0.001025728183
- path_increment 绝对差 0.000000000121

## 5. 判定

原因字母 A：同一套代码和同一个 checkpoint 下，重放进程没有继承训练结束时留下的 torch 线程数和确定性算法开关。
线程数和确定性算法开关是一起恢复的，本阶段没有拆开单独测试。
当前安装的 PyTorch 在恢复这些开关后能够对上已保存结果，因此这次 1 bit 以外的差不是 requirements.txt 里旧版本号本身造成的。
包装路径如果和 fresh 逐步相同，E6 里两条路径一起偏离保存结果就不是包装造成的。
下一步只记录并恢复评估进程的 torch 线程数和确定性算法开关，再对这一条场景复核 1 bit 容差。不要据此重训，不要改奖励，也不要恢复 224 条归因。

正式 checkpoint 和正式代码没有改。历史评估文件没有改。诊断分支不合并。
