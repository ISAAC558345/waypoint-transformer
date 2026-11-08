# 旧系统下一步指令：L8 单变量 payload 目标对照

日期：2026-11-08。结果基线：157c1c9fabfea6af4a23c9175bd736197923f27c。执行分支：feature/legacy-system-maintenance。先阅读本次提交的 Legacy_System_L7_Push_Review_CN.md。

## 任务与范围

完成一次“ListNet 与期望归一化 regret”公平对照，回答：在相同真实到达标签和相同训练量下，更接近 payload 的训练目标能否改善最终 argmax 任务选择。

只完善旧系统。正式 Scheme C、正式 Planner、候选生成 G1_RATE k6、hold15、冻结 MAPPO-V2、SafetyProjector 和 served 定义保持当前配置。海洋观测新方案不修改、不执行、不合并。本轮结果只用于研究，不接入正式入口。

L7 的命令修复和 7 页论文交付已结案。本次指令提交已直接纠正源码清单的四项换行身份；执行者只读确认当前提交含该更正，不必重做清单、编译 PDF 或重放 L6。

遵守项目内文件放置规则，辅助脚本、提交说明、工具和临时文件放入项目内 backup；结果放入新的 results/legacy_payload_objective_l8。保留用户其他文件和既有结果目录。不要新建基线工作树或搬移目录。

## 1. 唯一变量与固定两臂

为避免与 L3 的 A/B 命名混淆，本轮称为 C_LISTNET 与 R_EXPECTED_REGRET。

- C_LISTNET：直接复用 results/legacy_arrival_aware_ab_l3/arm_b_epoch_010.pt。该模型在同样的 12 个训练状态、真实到达标签和固定配置下从 E6 训练了 120 次。SHA-256 必须是 6024f4c42cdf2de2e8452b49da6ed1824ff8d6e56e84204681a6a37161b81141。不要重新训练。
- R_EXPECTED_REGRET：从相同 E6 完整 model 重置，空 Adam 状态，训练 120 次。唯一改变是下述损失。不能从 B10 继续训练，不能改变输入、候选、decoder、编码器结构或更新范围。
- 两个固定参考：未追加训练的 E6；L3 零到达标签 A10，路径 arm_a_epoch_010.pt，SHA-256 为 87f7c7673555518d80303a8226ae699057bcdaeacb4307512331b7dbad4e60d8。参考不参加选择或调参。

R 的每状态损失固定为：

~~~text
y = real_payload_bits.detach()
span = clamp(y.max() - y.min(), min=1e-6)
r = ((y.max() - y) / span).detach()
p = softmax(candidate_scores, dim=0)
L = sum(p * r)
~~~

softmax 温度固定为 1，不搜索温度、偏置、loss 系数、混合目标、active-count 惩罚或分时刻开关。所有 y 相同则 r 和梯度为 0。这个目标优化软选择的期望 regret，不能宣称已等价解决 argmax 问题。

在独立的新研究模块实现损失和入口，例如 src/wt/legacy_payload_objective_l8/。新增比较数据使用单独固定场景范围的 loader，原 L2 loader 保留其原有范围约束。复用已有加载/图/评分逻辑，不修改历史 LOSSES["c"]、旧实验代码或结果来适配新方法。

## 2. 固定训练配置

训练数据直接读取 results/legacy_arrival_aware_ab_l2 的 12 个 train 状态：322016–322021，各 t15、t30，每状态 128 候选。监督始终 real_payload_bits。既有 322022–322023 四个探索状态不用于选择配置或检查点，本轮无需重复其评估。

- E6 初始化：results/phase6id_joint_utility_training/epoch_006.pt，SHA-256 7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde；完整严格加载 model，epoch=6。
- decoder：results/phase5k_joint_decoder/matching_decoder.pt，SHA-256 de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5；参数冻结、eval，梯度仍传回编码器。
- 编码器保持可训练，结构、初始参数与 L3 相同。Adam lr=3e-4、weight_decay=1e-4，梯度裁剪 1.0。
- seed=0、CPU、单线程、确定性，与 L3 相同。顺序为场景升序，每场景先 t15 后 t30。
- 10 轮 × 12 状态 = 120 次更新与 120 次反向传播，只保存和使用 epoch10；不根据 loss 或新标签选择轮次，不另跑 seed1/2。
- 完成后只对 R 的 12 个训练状态评分一次，记录训练选择表现。C 对应的原始评分可复用 L4 缓存，不重算旧诊断。

冻结 R10 的权重及哈希之后，才揭示下面新增的真实到达标签。原始 L2 配对数据和所有旧权重不得被覆盖。

## 3. 另外 8 个场景的固定比较

固定场景 322024–322031，各 t15 和 t30，共 16 个状态，全部保持 15 步标签 horizon。本轮不打开既有 validation 或 reserved test。

这组仅是“本次到达标签研究此前未使用的比较场景”，仍在历史训练池内，且可能已用于 E6 的零到达学习。必须在 manifest、报告和结论写明这一限制。

每个场景只重放一次至 t30，复用 phase6i_urgency_highlevel_pilot.replay.replay_scenario 的 MAX_RATE_INDEPENDENT 路径、冻结 MAPPO-V2 seed0 与相同外生到达机制，保存 t15/t30 完整 env_snap 和 ledger。此轨迹不是正式 Scheme C 轨迹。不要调用会重放到 t45 的整套 D6 入口。

图按 L2 已验证的修订输入口径生成，仅使用决策时刻已存在的信息。记录真实 service_progress 和 uav_step_displacement_m；可以在这同一次 30 步重放中记录，不能置零或再多跑一遍。旧 revised 状态 NPZ 只可作为图来源或核对来源，它不含完整 rollout 快照，不能用它替代缺失 env_snap 字段。

每状态共同固定 128 个候选，沿用 D6 sampling.select_state / pilot.select_candidates 的规则：旧零到达 J15 前32 + 名次分层96，seed605，候选编号、排序与并列规则沿用现有实现。在看新真实标签之前保存候选、pool/subset/state 哈希和四个模型的全部原始分数与选择。禁止依照新方法分数增补、删减或重新抽样候选。

采样所需的旧 J15 / manifest 可以精确读取已有历史对象；只恢复所需文件，记录源提交与 blob。旧 revised 图的历史源为 8a34efc718665a7cfe4375b662e8f5249263f03d 下 revision_hgnn_input/dataset/train/scenario_*/t*_state.npz。不要重新生成全候选零到达标签，也不要恢复整套历史数据。缺少所需旧 J15 或无法保持相同状态/候选口径时，记录具体缺口并停止本轮，不改用另一种子集后继续声称同一协议。

每状态对同一 128 候选只生成真实到达 payload 标签，使用 D6 已有冻结执行与恢复口径；复用 score_label(zero_arrivals=False, horizon=15)，不要调用会同时计算两臂的 _score_rows。每个候选从相同快照独立恢复并运行 15 步，未来到达只进入离线标签，不进入图或选取逻辑。不再运行零到达对照、不扩展标签优先级指标、不重复整套 D6。用于采样的旧零到达 J15 会造成候选支持偏差，须与128子集限制一起披露。

候选最大分数并列时固定选较小 candidate_id，不能用真实 payload 打破并列。保留 local_position / row_index / candidate_id 的不同含义。保存四模型共用的候选与最终真实标签，随后按身份键关联比较。

## 4. 预算与必要检查

预算固定，不因结果不佳扩展：

| 项目 | 上限 |
| --- | ---: |
| 新方法监督更新 / 反向传播 | 各 120 |
| 重复训练 C 或参考模型 | 0 |
| 轨迹重放 | 8 × 30 = 240 环境步 |
| 真实到达候选 rollout | 16 × 128 = 2048 次 |
| 候选环境步 | 2048 × 15 = 30,720 |
| 总环境步 | 30,960 |
| 高层评分前向 | 200 |

高层评分的计划构成为：120 次训练 + R 的12次训练末轮评分 + 4模型×16状态的64次比较，共196；最多再留4次必要的接线检查。图生成和 MAX_RATE 选择不引入其他 HGNN 诊断前向。低层推理、环境步、训练、前向、反向都按实际计数，失败/作废消耗也计入，不仅计成功部分。候选评价若批量实现，另报告真实低层调用次数，不把调用数自动写成环境步数。

只做这些必要检查：

1. 对新增损失做一次小张量计算/梯度检查，确认高 payload 候选分数上升方向和全相等标签零梯度。它不需要模型或环境。
2. 读取输入/权重哈希和关键配置，确认 C 与 R 使用同一训练数据、初始模型、decoder 和更新范围。优先复用 L3 已保存检查，不重跑旧路径搬迁或双模型初始化诊断。
3. 在正常数据流程内检查候选身份与图不含未来量、恢复后的相同步数、served 总量口径、有限数值；遇到实际错误立即保留失败记录，修正明确实现错误后把消耗计入上限。不能把方法效果差当实现错误重跑。
4. 训练后确认 decoder、正式权重、原输入和正式策略没有被改写。

不跑历史全量测试、不增加安全/压力/故障注入矩阵、不重新验证目录搬迁、不重跑 L6、不重编译论文。不得用更宽容的 served 定义让结果通过。

## 5. 预先固定的比较与去向

主指标是新16状态上的 R10 相对 C_LISTNET 的 argmax 所选真实 payload 配对差，报告 bit 与 Mbit。regret 的最优值只取共同128候选内最大值，不能称全候选最优。

同时报告：

- 每状态4模型所选 candidate_id、assignment、score、active_count、真实 payload、子集 regret。
- 全16状态以及 t15/t30 分组均值；每场景先对两个时刻平均，再给8个场景的配对差、正/负/近零个数。两个时刻不是16个独立场景。
- R−C、R−E6、R−L3A10。软期望 regret 与训练 loss 只作辅助说明，不作为选择检查点的依据。
- 训练结果与新场景比较分开，不把原已见16状态并入主结果。

本轮仅在以下条件同时满足时标记 CONTINUE_TO_CLOSED_LOOP_COMPARISON：

1. R−C 的总体 payload 均值 > 1 bit；
2. R−C 在 t15 与 t30 各自均值均 >= −1 bit；
3. R 的总体 payload 不低于固定 E6 和 L3A10，各自容差 1 bit。

该门槛只决定是否值得另行制定闭环对照，不自动开跑下一阶段、不采用新模型。任一条件不满足则标记 NOT_PROMOTED 并如实结案；数据或执行不完整则标记 INCOMPLETE，而不是方法失败。禁止结果出来后改门槛、挑时刻、挑场景、换温度、换损失或补跑种子。

## 6. 交付

提交以下可复核产物：

- 新模块、固定配置、默认仅显示配置且显式 --run 才执行的入口，以及实际运行命令。
- results/legacy_payload_objective_l8/：来源和预算 manifest、R10 权重、120行训练记录、16状态图/完整快照与候选来源、预测先于标签的冻结记录、四模型原始分数和选择、2048条真实标签、配对明细与汇总、实际检查或失败记录。避免提交可由源码重建的大型重复中间产物。
- docs/research/Legacy_System_L8_Report_CN.md：先写结果和门槛，再写证据与限制；说明对照复用了 L3B10，新增训练只有 R 的120次。报告不能写正式闭环收益或独立泛化。
- 不修改当前论文实验结论，不因初步结果追加页数；有必要的论文更新留到之后结论明确时。
- 在当前维护分支提交并推送，提供提交号、主要指标、实际五项消耗、剩余缺口，随后停止。lambda_H、delta_payload_max 继续未定。

当前正式分支 e58e5d190109e5a0e860cb5c6a412321c0d74e31 和海洋隔离分支 4d8622ef183e734f2d9a890d60d735ed716f5d74 不因本轮改动。
