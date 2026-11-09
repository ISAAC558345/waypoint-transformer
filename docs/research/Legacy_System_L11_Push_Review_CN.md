# L11 新推送复核：正式链路通过，交付尚有收尾项

日期：2026-11-09。结果提交：fa845dfda498b8ba18134505a9cf3ab03f61ba6d；迁移指令提交：b55981aad49e58366870ed3391217af427786c90。

## 已完成的真实迁移

主检出的Git树没有round编号目录。HGNN、FormalPlanner、ResidualActor、环境等已在功能包内具有真实实现，不是全部转发到旧目录；未发现wt.roundN导入或round包别名。源码顶层现在按envs/models/planning/control/evaluation/data/common/cli等职责组织。

322001的一次48步与对应R1参考完全一致：payload 68952073.58225587 bit、served 68952073.58225589 bit、逐步served差为0，过程/reset种子1322001/322001，高层4、低层48、环境48、更新/反向0，matched、success=true、exit0。正式权重和策略未变。

对迁移前Git树中的390份pt/pth/npz/npy/pdf逐一按Git内容对象核对，全部在新树中找到相同对象，未发现这些权重/数据/历史PDF丢失或字节改写。新版11月9日正文/PDF已上传，记录为Tectonic退出0、7页均已查看。

## 不能直接宣布完全交付的原因

1. 源码内仍有199份.pt，共316,947,845字节（约302MiB），包括planning/task_option/outputs/handoff中5份旧DeepSets-util权重。它们不是现行正式推理资源，应移出src。当前4份正式启动/推理权重已经在checkpoints内，路径保持不变即可不重跑48步。
2. 历史路径resolver只做一次前缀替换，不能组合“目录迁移+嵌套改名”。例如round_13/amendments/round13_completion_amendment_20260825_0010尚不能一次解析到实际completion_amendment目录；round_11/AGENTS.md也未解析到historical_instructions.md。需要旧逻辑路径到最终物理位置的闭合映射。
3. AB/AC摘要并未丢失。它们分别在results/reproducibility/formal_scheme_c/evidence_archive/results/phase6iab_planner_stability/summary.json及phase6iac_independent_holdout/summary.json。benchmark入口缺少对应资源映射，因而错误报告absent，README/USAGE/DELIVERY_STATUS及新evidence也沿用错误缺失说明。
4. 论文最新索引仍指11月8日；新正文新增表没有定义R12/R28、期望归一化regret、每臂120更新、12/28训练状态及L8比较数据转为L9训练的关系。Sec.VII“两场景四窗口”应仅指L3–L5；结论首句仍声称仅推进至E18。
5. 新源码清单中main/body的git_blob是LF规范化身份，而实际提交为CRLF。两份旧11月8日TeX归档也发生LF转换。只需按实际Git对象修正/恢复，不代表实验数值有误。
6. 新48步manifest的instruction_commit仍是L10的204dd33，而本次迁移指令是b55981a。应保留原始记录，用派生更正说明纠正来源，不重新执行。

## 摘要的真实来源与数值

AB归档blob：73e01163aafb5edc771cd7f4aee6139fdd775b11；16个场景，paired mean=2644990.8087698035 bit，CI=[180100.0770928479,5016616.160644779]，11正0平5负。

AC归档blob：e23290d36258acc7d417a79d21c07a5502d95702；16个场景，paired mean=3502191.5265653464 bit，CI=[1270476.686997788,5395098.500613804]，14正0平2负。

这些是已归档历史统计，不是本次重新运行/重估。ranking-loss顶层summary和基线侧分状态等仍缺失，不能把两类缺失混为一谈。

## 下一步范围

按用户C选择，只做L12目录/资源/论文的最后收尾。199份旧权重及相关归档输出移出源码，映射与只读摘要可用，最新文档与论文自洽。全部可在0模型前向、0环境步、0训练、0反向下完成；论文正文变化后只做必要编译和逐页检查。

不重新做L11迁移、48步回归、L8/L9、历史全套或保留集；不新增模型研究。L8/L9仍NOT_PROMOTED，海洋方案继续隔离。通过后报告交付完成并停止，不为了“下一步”自动制造新的检查轮次。
