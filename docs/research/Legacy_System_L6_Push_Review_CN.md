# L6 推送审阅与 L7 交付安排

审阅日期：2026-11-08。结果提交：`77c73c00462ad2bb3d778d12a2ad98f0e4ffd576`；父指令提交：`2b2fa2f3ca66d64688a122d939e2b9bdb5f6605a`。当前只完善原系统，新方案保持隔离。

## L6 已完成且有效的结果

正式单场景命令已新增，默认预览使用标准库，不建目录、不加载torch。显式运行通过原正式链路，配置副本和相对路径清单位于独立输出目录；fault_log透传正确，默认None保持历史行为。

实际场景322000完整48步，process_seed=1322000，reset_seed=322000，决策时刻0/15/30/45，SchemeC调用4次。payload为63785945.28984571 bit，环境步48、训练0、异常0。

本次直接读取新episode与原R1 episode：48条时刻一致，逐步served最大绝对差0，payload与served的跨记录差也均0；complete、两项运行内一致性标志都为真，error为空。L6成功结果有效，不需要重跑。原模型和官方策略身份保留，L3–L5未采用索引已完成。

## 三处明确的命令可用性缺口

1. `execute.py::run_once`前置允许已有空输出目录，但随后mkdir没有exist_ok=True且位于try外；传入现成空目录会报错。
2. 该函数末尾无条件return0。运行错误、回归different、权重身份不符或官方策略变化，尚未统一反映到退出码。
3. `_compare`记录payload_matches_served与step_served_matches_episode，却不将它们纳入matched判定。

这些不推翻已保存的L6成功回归。L7只修完成语义，用现存L6/R1记录和空目录验证；不做故障注入、不新增环境回合。

## 论文仍需完成的实际交付

当前正文是11月4日稿、旧PDF由独立排版生成；真正由当前LaTeX工程产生并核对的新版PDF仍缺。L7允许在项目内部配置官方便携构建工具，实际编译、渲染并检查全部页面，不再重复将PATH检查作为交付。

还有两项可直接提高源码准确性：
- 论文ListNet分母现写为max-min+epsilon，实际代码是max(max-min,1e-6)，softmax前减去质量最大值。新版应与真实实现一致。
- 原ranking-loss顶层summary仍缺，但R1归档的runs/c/result.json含48状态的原保存聚合。其best_epoch=1的payload为19.240091242 Mbit，Top1为0.6666667，Top3/Top5为0.7083333，normalized_regret为0.0354734337。可以核对论文对应C格，不能冒称恢复整个summary、基线逐状态数据或独立重算。原loss:NaN必须保留来源区别，派生索引不写成0。

## L7 的范围

先修上述命令语义，再同步日期明确的新论文源码、L3–L5未采用结案和L6工程回归证据，完成项目内便携LaTeX编译与逐页核对。新环境步、模型前向、训练和反传均0；不继续在16个已见状态上调参，不重开validation/holdout。

用户已要求整理上级目录。后续辅助脚本、提交消息、日志、工具和缓存都留在项目内部；基线worktree集中在backup/baseline_worktrees，不再散落E:\srdp。其他文件和目录位置保持不变。本次同时在AGENTS中固定此整理约定。

完整执行范围见 [L7指令](Legacy_System_Next_Instructions_CN.md)。本次根聊天只核对远端结果和发布指令，没有执行新回合或生成新版PDF。
