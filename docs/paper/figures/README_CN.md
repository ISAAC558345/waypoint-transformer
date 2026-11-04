# 当前正式系统示意图

- [环境示意图](system_environment.png)：6 架 UAV、12 个浮标，两个浮标簇；仅示意数量和分布，不是实际轨迹。区域 160 m × 80 m，horizon 48，hold 15，服务半径 16 m。
- [数据流图](system_dataflow.png)：八节点闭环，高层每 15 步任务决策，低层每步执行；正式 G1_RATE k=6 + Scheme C + 冻结 MAPPO-V2。
- [可编辑数据流逻辑](system_dataflow.mmd)：Mermaid 源码。
- [生成提示词与来源](generation_prompts_20261102.json)：使用内置 image_gen，保存最终提示词和核对信息。

图片采用概念位置，不绘制海洋、山地等装饰。环境图没有服务连线，避免把任务关联画成实际成功通信。
流程中的“任务 + 实时状态”表示低层每步重新构建观测，hold 内目标保持不变。实际 served_bits 累加为 payload。
仅供展示当前系统，不表示新优先级策略已采用或有新增实验结果。图中 top-6 是每个 UAV 的候选目标数。

LaTeX 可通过 ../latex/figures/system_schematics.tex 插入这两张图；默认论文源码保留原有的可编辑 TikZ 正式架构图。
