# 旧系统下一步指令：L9 固定预算的数据覆盖对照

日期：2026-11-08。结果基线：77eb078e54dde9e6ca0a55c3e47a3c482ff20d2c。执行分支：feature/legacy-system-maintenance。先读本次提交的 Legacy_System_L8_Push_Review_CN.md。

## 1. 当前结论与唯一问题

L8按NOT_PROMOTED结案。t30未达到原门槛，不改结论、不放宽容差、不补种子、不按时刻拼接选择，不运行L8闭环。

本轮新问题是：相同120次监督更新预算下，把真实到达训练状态从12个扩至28个，能否改善另一组场景上的最终payload选择？只改变训练数据集合及其固定访问分配；网络结构、损失、初始化、优化器、更新量和冻结执行器保持相同。

这是固定计算预算的数据分配对照。新数据覆盖增加时每状态训练次数会下降，不得声称每样本训练量也保持不变。数据覆盖不足只是待检验假设。

只推进旧系统，正式Planner与Scheme C保持当前配置，海洋隔离分支不修改、不执行、不合并。lambda_H、delta_payload_max仍不取值。所有辅助文件放在项目内；临时文件使用backup，结果使用新的results/legacy_arrival_coverage_l9。保留旧结果、用户其他文件和目录。

## 2. 两臂、参考与固定训练设置

主对照：

- R12_U120：复用 results/legacy_payload_objective_l8/r_epoch_010.pt。SHA-256为4f4ffa043f13b31b83f31508ef8ade138d96cee36a9addf977d33aeb555bc27f。原12状态各训练10次、共120更新。不要重训或继续训练。
- R28_U120：从同一E6完整模型和空Adam重新开始，用下面的28状态固定序列训练120次。不能从R12继续训练，不能加载它的优化器状态。

固定参考仅评分、不训练：

- E6：results/phase6id_joint_utility_training/epoch_006.pt，SHA-256 7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde。
- L3A10：results/legacy_arrival_aware_ab_l3/arm_a_epoch_010.pt，SHA-256 87f7c7673555518d80303a8226ae699057bcdaeacb4307512331b7dbad4e60d8。
- C_LISTNET：results/legacy_arrival_aware_ab_l3/arm_b_epoch_010.pt，SHA-256 6024f4c42cdf2de2e8452b49da6ed1824ff8d6e56e84204681a6a37161b81141。

编码器继续可训练，与L8相同。decoder固定为results/phase5k_joint_decoder/matching_decoder.pt，SHA-256 de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5，参数冻结、eval，仍向编码器传递梯度。

直接复用L8的expected_regret_loss，不另写或改历史损失：

~~~text
y = real_payload_bits.detach()
r = ((y.max() - y) / clamp(y.max() - y.min(), min=1e-6)).detach()
L = sum(softmax(candidate_scores, dim=0) * r)
~~~

温度固定1。Adam lr=3e-4、weight_decay=1e-4、clip_grad_norm=1.0，seed0、CPU、单线程、确定性，保持L8的dtype与模型构建口径。只运行这一个新臂，不调温度、偏置、模型结构、活动数奖励、标签权重或学习率。

R28保存为r28_u120.pt，并记录updates=120、完整pass数4、尾段更新8。它不是10轮训练，不要为了兼容旧loader伪写epoch=10。按实际updates与完整model严格加载。

## 3. 直接复用28个训练状态

原12：L2 train中的322016–322021，各t15/t30，每状态128候选。读取原有修订图、assignment和real_payload，不使用zero_payload作为监督。

新增16：L8的322024–322031，各t15/t30，每状态128候选。直接读取：

- results/legacy_payload_objective_l8/comparison/scenario_*/t*/graph.npz；
- results/legacy_payload_objective_l8/candidates.json；
- results/legacy_payload_objective_l8/real_labels.jsonl；
- 同状态state_meta.json及必要来源清单。

图均为相同7键修订输入，候选中已有local_position、row_index、candidate_id、assignment与状态/子集哈希。标签按scenario_id、time_step、candidate_id、assignment关联，不能用candidate_id直接当本地数组下标。保留各状态原128候选的本地顺序。

新增独立的L9 loader，仅拼上述28状态，不放宽L2 loader，也不改L8结果或范围。需要的图/标签全部已有，旧数据环境步、旧数据模型前向和旧数据重新标注均为0。不要重放L8，不恢复全套历史数据，不为不存在的zero标签填0。

总计28图、3584候选及对应真实payload值。L8的16个比较状态现在明确转为L9训练数据，不再作为L9验证结果。322022–322023旧探索状态不参与本轮训练、选参或主比较。

固定访问序列：按(scenario_id,time_step)升序排列上述28状态；update从0到119，每次使用states[update % 28]。因此120=4×28+8，前8个状态各使用5次，其余20个各4次。保存全部120个访问键与每状态次数；R12旧数据为每状态10次。这一访问分配在任何新比较标签揭示前固定，不按既有负差、payload大小、t15/t30或损失值重加权。

只使用第120次更新后的模型，不选择中间检查点，不跑训练集末轮评分或重算旧模型训练诊断。120行训练loss/梯度记录已足够说明执行完成；主结论来自下一组固定比较。

## 4. 新比较范围和数据生成

固定322032–322039，各t15/t30，共16状态。它们未在这条已执行到达标签研究中使用，但仍在历史TRAIN内；E6可能见过零到达数据，因此不能称独立验证、未见模型泛化或正式留出集。不打开历史validation或reserved test。

复用L8已经验证的30步重放、完整快照/账本保存、修订图构造、固定候选采样与单臂真实标签逻辑。为L9范围写适配层，不调用整个L8入口，不改L8全局常量后继续运行旧实验，不复制不必要的大段历史流程。

每场景沿MAX_RATE_INDEPENDENT与冻结MAPPO-V2 seed0重放一次至t30，保存t15/t30完整快照、ledger和当前图。共240环境步。这不是正式Scheme C轨迹。service_progress和uav_step_displacement_m必须来自同一真实轨迹；不能置零或额外跑一条图轨迹。

每状态128候选，沿用D6规则：旧零到达J15 top32 + 名次分层96、seed605。候选、顺序、并列口径、pool/subset/state哈希相同地提供给5个模型。不依照新方法分数或真实标签增删候选。

需要的旧候选来源为aadbb3c5c5954669dd335dce9afb3462d7de114c中的高层训练数据，修订图来源可用8a34efc718665a7cfe4375b662e8f5249263f03d核对。只读取所需对象，保留源路径/blob/字节哈希。不重新计算全候选J15。所需源对象实际缺失或口径不一致时记录具体缺口并停止，不另换采样方案。

先冻结R28权重，再在新状态上保存5模型全部原始分数与argmax选择、共同候选和输入哈希，最后才生成真实到达标签。未来到达、真实payload、名次、scenario_id特征和最优索引不得进入图或候选评分。scenario_id只作身份键。

真实标签仅计算一臂：每候选从共同快照恢复，zero_arrivals=False、horizon=15，采用现有served口径和相同冻结执行器/外生机制。共2048候选回合、30,720候选环境步。不得重做零到达对照、t0/t45、D6双臂研究或标签优先级权衡。

分数并列取较小candidate_id，不用payload破并列。保留idle合法性，不强制6机活动。子集最优只指128候选内最大payload，披露旧零到达J15采样造成的支持偏差，不宣称全候选最优。

## 5. 必要检查和固定预算

| 项目 | 计划/上限 |
| --- | ---: |
| 新方法更新、反向传播 | 各120 |
| 重训R12或其他参考 | 0 |
| 旧28状态重放/重新标注 | 0 |
| 新轨迹重放 | 240环境步 |
| 新候选标签 | 2048回合，30,720环境步 |
| 总环境步 | 上限30,960 |
| 高层评分前向 | 计划200，上限204 |

200前向=120训练+5模型×16新状态评分。额外最多4次仅供实际需要的接线验证，不默认跑。不要追加28状态训练末轮评分，也不要跑一次check-only后再自动重复同样模型检查。

只做与当前数据拼接和比较直接相关的检查：

1. 一次读取中核对28状态/3584条候选标签的身份、原本地顺序、图schema与有限值；确认训练范围和新比较范围不相交。保存来源字节哈希和访问次数，不用模型检查旧数据。
2. 确认各权重、初始化、空Adam、decoder冻结与120次更新一致。L8损失未改，不重新做已通过的小张量梯度测试。
3. 在正常新数据流程中检查相同步数、候选身份、当前观测口径和served统计；沿用已有恢复逻辑，不做额外重复标签回合。
4. 完成后读取确认正式权重、decoder、原始输入、旧研究模型与正式策略未被改写。

环境步、低层调用、前向、更新、反向都记录实际消耗；失败或作废部分也算预算。若每步一次MAPPO调用，应分别实测记录，不自动从环境步复制推理数。出现实际执行错误时保存原始记录，在剩余预算内修复明确错误；方法结果差不是重跑理由。

不跑历史全量测试、安全/压力/故障注入矩阵，不重新检查目录搬迁、不重放L6、不重编译论文。不修改当前论文实验结论。

## 6. 预注册指标与下一阶段门槛

主比较为新16状态上的R28_U120−R12_U120，指标是argmax所选真实payload配对差，报告bit与Mbit。

保存每状态5模型的候选编号、assignment、active_count、分数、payload、子集regret。按全16状态、t15、t30分别汇总；按场景先平均两个时刻，再报告8个场景的配对差和正/负/近零数，近零容差1bit。两个时刻不是两个独立场景。

同时报告R28相对C_LISTNET、E6、L3A10的差。soft期望regret和训练loss仅为辅助，不决定检查点。不能把L8已转训练的状态混入新比较。

只有以下全部满足，才标记CONTINUE_TO_CLOSED_LOOP_COMPARISON：

- R28−R12总体平均 >1bit，且t15/t30各自平均均 >=−1bit；
- R28−C_LISTNET总体平均 >1bit，且t15/t30各自平均均 >=−1bit，保留原L8相对ListNet的要求；
- R28总体payload不低于固定E6和L3A10，各自容差1bit。

这只是决定下次是否值得制定闭环对照，不自动执行闭环，不采用新模型，不重写L8的NOT_PROMOTED。完整结果不满足则NOT_PROMOTED；数据或执行不完整则INCOMPLETE。不能事后挑场景、挑时刻、换基线、调容差或再增训练量。本轮结束后停止。

## 7. 交付与推送

实现独立模块，例如wt.legacy_arrival_coverage_l9，默认仅显示固定配置，显式--run才执行。报告实际命令、输入来源和所用指令提交。

results/legacy_arrival_coverage_l9/至少保存：28状态来源/拼接索引与固定访问表、120行训练记录、r28_u120.pt与哈希、16个新状态图/完整快照/候选来源、预测先于标签的冻结记录、5模型原始分数与选择、2048条真实标签、配对明细/汇总、实际预算和必要检查/失败记录。旧训练图和标签引用原件，不复制大份旧结果。

提交docs/research/Legacy_System_L9_Report_CN.md，先写门槛结果，再写主A/B与固定参考指标、实际5项消耗、固定更新下每样本访问量不同及历史TRAIN限制。明确旧L8数据已经用于训练，不能作为新验证证据。

在当前维护分支提交并推送代码、结果、报告，提供提交号和主要结果后停止。正式分支e58e5d190109e5a0e860cb5c6a412321c0d74e31、海洋隔离分支4d8622ef183e734f2d9a890d60d735ed716f5d74均保持不变。
