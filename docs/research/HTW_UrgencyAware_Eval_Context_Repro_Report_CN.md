# 实验报告：显式评估运行时上下文的单场景复核

- Gate：`EVAL_CONTEXT_REPRO_PASS`
- 分支：`experiment/priority-tag-eval-context`
- 指令提交：`88ceeab4dc2ac6f0e19fc2603d9fdf915ee0590d`
- 来源 E7 提交：`64ae1907eae2354d64625361c0a0afd866982d42`
- 运行时 HEAD：`88ceeab4dc2ac6f0e19fc2603d9fdf915ee0590d`
- 是否合并：否
- 是否重新训练：否

## 1. 种子角色

报告里的模型种子 0 是 L0 checkpoint 和已保存 CSV 的 model_seed 列。
环境 reset seed 是场景编号 321101。分配随机数种子是评估 worker 使用的 7000000+321101。
全局 Python/NumPy/Torch seed 的数值 2 只代表历史进程里最后一次 `train_variant` 留下的继承上下文，不是评估协议宣布的通用种子。
规范化的显式评估全局种子没有在历史协议里找到，因此 manifest 里该字段是 null，没有用另一个新种子冒充历史设置。

## 2. 容差与完整性

payload 和记账容差是 1 bit。路径容差是 1e-09 米。
正式 checkpoint SHA-256：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。
运行后 checkpoint SHA-256：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。
manifest SHA-256：`6f82124f8b977dbcc675a795151dc68d105f0f7e4094d8e9ef2a27a33fb6132e`。
manifest 保持历史继承标签：True。
DEV split 校验：True。

## 3. 两次独立进程

逐步轨迹在原容差内一致：True。
进程间 payload 最大差 0.000000 bit，路径差 0.000000 米。
reset 哈希一致：True。
- run_a 相对已保存结果：payload 0.000000 bit，普通服务 0.000000 bit，tagged 服务 0.000000 bit，普通丢弃 0.000000 bit，tagged 丢弃 0.000000 bit，紧急度 0.000000 bit，路径 0.000000 米，进入容差：True。
- run_b 相对已保存结果：payload 0.000000 bit，普通服务 0.000000 bit，tagged 服务 0.000000 bit，普通丢弃 0.000000 bit，tagged 丢弃 0.000000 bit，紧急度 0.000000 bit，路径 0.000000 米，进入容差：True。
账本闭合：run_a=True，run_b=True。

## 4. 判定

下一步才可以制定带有这份显式 manifest 的完整归因重放方案。本阶段不启动那次重放，不重训，不合并。
