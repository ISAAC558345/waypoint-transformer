# 旧系统 L7：命令完成语义与论文真实交付

更新时间：2026-11-08。工作分支 `feature/legacy-system-maintenance`；审阅结果提交 `77c73c00462ad2bb3d778d12a2ad98f0e4ffd576`。L6原指令归档于 `archive/Legacy_System_L6_Instructions_20261105_CN.md`，不要重跑。

## 1 本包目标与范围

完成两项交付：修正正式命令的完成语义；将当前旧系统、已结案研究和可核对证据同步到新版论文，实际编译LaTeX并检查页面。新环境步、模型前向、训练更新、反传全部0。

阅读AGENTS、README_BRANCH_SCOPE、Legacy_Only_Workflow及 [L6审阅](Legacy_System_L6_Push_Review_CN.md)。新方案保持在research/ocean-observation-isolated，不修改、执行或合并；正式分支也不合并。L3–L5已结案，不再调选择器、权重或时刻门控，不重开旧validation/holdout。

所有辅助文件留在当前项目内。建议便携工具位于 `backup/local_tools/`，构建/缓存/逐页PNG位于 `backup/local_build/l7/`，最终文档和小型证据记录沿用docs/paper与results。不要再向E:\srdp根目录写脚本、提交说明或日志，也不移动或重写其他项目文件夹。已有9个基线worktree仅按其现有Git关联使用，不重新创建或搬回上级。

## 2 修复三个已确认的入口缺口，0环境步验证

仅修 `wt.legacy_system` 的完成语义，不改Planner、环境、奖励、动作、权重或fault判定：
1. 接受已存在但为空的输出目录；目录准备进入适当异常处理，非空目录仍不覆盖。
2. 将运行错误、回归是否匹配、实际完成状态、所用模型身份及官方策略是否变化统一纳入最终成功结论。成功才退出0；执行失败/不匹配返回明确非0；输入拒绝可沿用2。保存的JSON状态与退出码一致。
3. `_compare` 的matched同时要求payload/served两项运行内一致性，不能只记录标志却忽略。

原L6结果满足成功条件，原episode/manifest/对照文件保持原件，不用新代码覆写原运行身份。不要为验证返回码再跑一次48步。

最小必要验证：一次现成空目录准备；以保存的L6匹配记录作为成功例，以已有另一份R1 episode作为不匹配例检查结果判定/退出语义。必要时抽出标准库判定函数，使检查不导入模型。只验证这次改动，不构建逐字段故障矩阵或模拟环境测试。结果记录到新 `results/legacy_delivery_l7/`。

## 3 用原保存记录补全证据口径

使用现有R1归档，不做新的模型评估：
- `results/legacy_formal_repro_r1/evidence_archive/results/phase6if_ranking_loss/runs/c/result.json`；
- 同目录体系中的 `validation_loss_backfill.json`，仅用于注明其原保存字段及来源；
- 已有AB/AC证据索引、L3/L4/L5结案和L6结果固定提交。

按源文件best_epoch=1定位C的原保存聚合，核对19.240091242 Mbit、Top1 0.6666667、Top3/5 0.7083333、normalized_regret 0.0354734337对应的论文格。逐项登记源路径、原blob/原字节哈希、JSON定位、样本单位48状态、舍入口径和claim类型。

这些格可标“原保存聚合已核对”，不是重新推理得到的统计。缺失的原顶层summary、基线/best逐状态数据仍标MISSING，不从差值倒推补造，不新建伪装成原件的 `results/phase6if_ranking_loss/summary.json`。其他未有原件支持的历史表格格值保留reported-only状态。

原result含loss:NaN。原文件不改；派生JSON用null及原因，若引用backfill必须分别注明源字段，不把NaN变0或自动嫁接成原result。新版证据索引须是标准JSON，不输出NaN/Infinity。沿用已有AB/AC原件核对，不重做整轮历史证据审计。

## 4 同步新的旧系统论文版本

从11月4日稿生成日期明确的新稿，例如 `Execution_Grounded_Hierarchical_Graph_Learning_MultiUAV_Revised_20261108.md`，同步当前LaTeX。保留11月2日/4日正文和所有旧PDF；当前TeX修改前按原字节归档一次到dated archive，避免混淆版本。

必要修改限定为：
- ListNet公式与代码一致：`q_i=(y_i-min y)/max(max y-min y,1e-6)`，`p=softmax(30*(q-max q))`，交叉熵目标脱离计算图；不改真实算法。
- 增加精简的L3–L5探索结案。清楚标注两个已见训练池场景、4个15步探索窗口、候选子集限制；B原均值20.150 Mbit，A25.905，E6初始化25.863；行均值idle消融使B升到24.404但有t30负差并损害对照组，均未采用。不能写成新的48步策略收益或泛化通过。L4统计采用L5已更正口径，单例组0regret不作排序证据。
- 加入L6正式CLI已完成48步、与原R1逐步及payload匹配的可复现说明；它是工程回归，不是新性能实验。
- 更新C列原保存聚合可追溯状态，保留其他真实缺口。正式benchmark数值、原负差案例、未采用结论和物理通信约定不变，不加入新方案内容。
- 摘要、讨论、结论、制品说明和引用状态一致。对读者的论文标题/日期不再含已过时的“本次未编译”说明；具体构建状态放README/清单，依据实际结果填写。

固定引用L3=8052e96267aa0699e30b6b37927f4971c8ba51b4，L4=d25e97fec756028a394a1cca67edb6d29c47c137，L5=30c880bedd9da4adff0b2de1d6b3fbef4bc27554，L6=77c73c00462ad2bb3d778d12a2ad98f0e4ffd576；正式证据继续使用原固定来源，不能只链mutable branch。

## 5 在项目内部实际编译，不再停于PATH探测

已知本机没有系统LaTeX编译器。本包允许按需获取官方便携构建工具及所需TeX资源到项目backup目录，不安装系统级大套件、不改系统PATH、不移动其他文件。优先一条固定路线：Tectonic官方预编译程序，登记实际版本、官方来源URL与下载文件哈希；不从源码编译Rust工具链。

[官方安装说明](https://tectonic-typesetting.github.io/book/latest/installation/)
[官方独立文档编译说明](https://tectonic-typesetting.github.io/book/latest/v2cli/compile.html)

用过程内 `TECTONIC_CACHE_DIR` 将资源缓存指向项目 `backup/local_build/l7/tectonic-cache`。在当前latex目录运行实际工具；独立文档命令形式：
```text
tectonic -X compile --keep-logs --keep-intermediates --outdir <项目内构建目录> main.tex
```
实际调用可使用项目内程序绝对路径。保存完整命令、工具版本、日志与退出状态。工具/资源不作为大体积仓库产物提交，保留便携构建脚本与来源说明即可。

修正实际出现的TeX/字体/引用/版式错误并重新构建，直至得到本工程的PDF；不要用旧独立排版PDF或其他生成器代替LaTeX编译。若官方资源下载或权限真的阻断，记录具体URL/错误、已完成工作和可复现源码包，不虚称编译成功；也不能只重复“没有编译器”而不尝试这条已授权路线。

## 6 实际页面检查与来源身份

编译成功后，用可用PDF渲染工具生成全部页面PNG并逐页查看，核对公式、表格、图、列宽、链接/引用、分页和字形。文本提取用于查缺字与引用，不替代视觉核对。只重建/重看因实际修改受影响的必要产物，不运行模型或仓库全套测试。

另存新的日期明确PDF，记录页数与实际视觉检查状态；没有完成页面检查的状态须如实保留。更新论文README、LaTeX README、revision notes、source_manifest和新版evidence索引，准确区分“源码更新”“实际编译”“页面已看”。

来源清单计算当前源码对应Git blob的原字节SHA-256，并登记Git blob身份；工作区CRLF哈希如有区别另列，不混用。避免为了把最终提交SHA写回而无限重编译；源文件、构建工具、命令和产物哈希足以关联。原清单和旧PDF保持可追溯。

## 7 交付与停止

交付：
- CLI完成语义修复及上述最小保存数据检查结果；
- 新版Markdown、LaTeX、实际可生成的新PDF、构建脚本与日志；
- 新证据索引、版本/构建来源清单及可复现说明；
- `Legacy_System_L7_Report_CN.md`，分别报告命令修复、证据核对、编译/逐页检查与尚未完成事项，五项研究消耗均0。

不重跑L6、R1、L3–L5、validation/holdout或历史全套，不开始新模型训练或新方案。完成后提交推送当前维护分支，报告SHA后停止。本次根聊天只发布指令，没有执行这些修复或生成PDF。下一轮算法研究再依据明确目标和独立证据预算安排。
