# HTW Phase-5ZL：True-H1 分层影子集成契约

**Gate = INT-C DECISION_OR_MODEL_SELECTION_UNRESOLVED**

阻塞项：['INT-C', 'INT-D']。

**planner_ready = False**

**deployment_verified = False**

**FROZEN_FOR_FRESH_CONFIRMATION = False**

本阶段只在 310110–310117 已消费状态上做独立影子重放。没有调用正式高层决策函数，没有修改正式 planner，没有打开 310118–310125，没有把这次重放称为新的独立确认。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。

## Q1. 当前正式高层使用哪一个 encoder 和 decoder？

正式高层是冻结的 DeepSets。权重来自 `final_artifacts/checkpoints/deepsets_util_seed0.pt`，模型类是 `OrderFreeAssignmentModel("deepsets")`。解码器是顺序的 `OrderFreeSetDecoder`：每次在未分配 UAV 上取一个浮标或空闲标记，直到六架 UAV 都有目标。正式入口把这个解码结果包成 `TaskOption`，`remaining_hold = 15`，`association` 等于 `targets`，`urgency` 为空。

Phase-5L HGNN 加 Matching decoder 的 Hungarian 分配是实验对照，不是当前正式高层。Stage-1 的 120 个互斥候选也不是正式高层的直接输出。

## Q2. optimized True-H1 应该插入哪个位置？

插入位置在 Stage-1 联合候选已经生成之后、冻结 Model-R1 评分之前。数据流是：当前状态、冻结 HGNN encoder、120 个联合候选、optimized True-H1 的一步响应、Model-R1、稳定降序 argsort、选中的联合分配、TaskOption。

True-H1 只前进一个仿真步。它不代替 option hold=15 的后续执行，也不预测完整 15 步。高层仍按每 15 个仿真步触发一次；低层在两次高层决策之间按仿真步长 1 s 执行。

## Q3. 它是否会修改真实环境状态？

不会。24 个决策状态上，影子路径前后的环境指纹差是 0.0，随机数状态包含在该指纹里。反事实的 `step` 发生在 deepcopy 后的副本上，每个候选执行前都恢复同一份快照。影子选出的 TaskOption 没有送入正式环境。

## Q4. 120 个候选是否正确对应全部 UAV 的联合分配？

是。每个状态 120 个候选，形状 [120, 6]，每一行是六架 UAV 的联合目标。空闲编码是 -1。候选全部通过排他性检查，非法分配数是 0。这些候选来自每架 UAV 的 top-3 浮标再做互斥组合，不是把每架 UAV 的单独最佳目标拼起来。

## Q5. Model-R1 的 value 是否对应候选的低层执行结果？

对应。响应形状是 [120, 6, 14]，与候选顺序对齐。把第 0 个候选的响应换成另一个候选的响应后，该位置的 value 发生变化：True；其余候选的 value 保持不变：True。五个 seed 相对已保存 v_r1 的最大绝对误差是 0.000e+00。

## Q6. 五个模型 seed 是否选择一致？

逐 seed 重放，没有做多种子聚合，也没有挑选最佳 seed。24 个状态里，五个 seed 选中同一分配的比例是 0.9583。不一致的状态只有 310111、t=15。seed 0 选 `-1;-1;4;-1;5;3`，seed 1–4 选 `3;-1;4;-1;5;-1`。两条分配的已保存 J15 都是 3.0000e6 bit，normalized regret 都是 0，Top-1 都是 1。因此差的是联合分配，不是 selected J15。该状态上选中 value 的标准差是 0.107，单个 seed 内部的第一名和第二名之差都大于 1e-6，所以这不是平局打破规则造成的。

全部 120 个状态-seed 上，接近平局的次数是 0，最高分完全相同的次数是 0。value 标准差在 24 个状态上的平均是 9.5286e-2，这是 Model-R1 分数的离散程度，不是 J15。稳定降序 argsort 只作为重放约定，不是已经批准的部署平局规则。

这一个状态已经使最终分配无法由现有五个 checkpoint 唯一确定。本阶段不采用多数 seed，也不指定 seed 0。

## Q7. 是否保留原始 R1 的 selected J15？

保留的是已保存标签上的重放，不是新确认。影子选中分配与已保存 v_r1 的 argmax 一致：True。五个 seed 的平均 selected J15 是 2.7917e+06 bit，Model-S 是 2.3417e+06 bit，差值是 4.5000e+05 bit。已报告的 R1 相对 S 的平均提升是 4.5000e5 bit。本次配对差值与该数字的差是 0.0000e+00 bit。原始串行 True-H1 与 optimized True-H1 的选中分配一致：True。响应逐位相同：True。

310114 t=15、seed 0：最优向量 `-1;4;2;5;-1;-1` 的名次是 2，预测向量 `2;5;-1;4;-1;-1` 的名次是 4。

## Q8. TaskOption 和 hold=15 是否保持原语义？

保持。影子 TaskOption 的 `remaining_hold` 全是 15，`association` 等于 `targets`，覆盖全部六架 UAV，`urgency` 为空。`refuse_rematch` 全部通过。继续或切换是相对当前 `env.assignment` 逐 UAV 记录的，不改变 hold 的执行机制。本阶段没有把一步响应当成 15 步预测，也没有跑新的 15 步闭环。

## Q9. 完整高层推理延迟是多少？

Phase-5ZK 的统一中位耗时 357.55 ms 包含 encoder、候选准备、True-H1、一个 Model-R1 和 argmax，不包含 TaskOption 构建。本阶段把 TaskOption 构建放进同一计时窗口。

影子链路的状态均值之中位数是 342.09 ms，均值是 370.39 ms，P95 是 468.77 ms，P99 是 617.85 ms。候选数是 120。TaskOption 构建的平均耗时是 0.0497 ms，选择本身约 0.06 ms。310110 t=0 的分量均值大约是 encoder 6.26 ms、候选准备 115.39 ms、True-H1 189.69 ms、Model-R1 4.34 ms。tracemalloc 峰值是 756.2 KB，这是 Python 分配峰值，不是进程 RSS。场景聚类自助法给出的是均值的 95% 区间 [350.79, 396.09] ms，不是中位数。

342.09 ms 低于 357.55 ms，原因不是 TaskOption。TaskOption 只有 0.05 ms。本次重复次数是 3，Phase-5ZK 是 5，两次都是状态均值的中位数，不能把这个差值解释成加速。t=30 的状态更慢，310110 t=30 的状态均值是 661.59 ms。

仿真步长是 1.0 s，option hold 是 15，高层在决策时刻触发，而不是每个仿真步都触发。这台机器不是部署目标。deployment_latency_requirement = NOT_SPECIFIED。仿真步长不是在线推理时限。

## Q10. 是否存在未定义的 fallback、模型选择或部署时限？

存在。部署时限没有书面规定。五种 seed 没有预先批准的聚合或部署选择规则。十类异常都只有待人工批准的接口说明：空候选池、全部空闲、非法关联、True-H1 非有限、Model-R1 非有限、安全投影失败、推理超时、checkpoint 缺失、单个反事实 step 异常、value 完全相同。本阶段没有把 DeepSets、B2 或 Oracle 写成新的部署回退。

## Q11. 是否具备下一阶段正式集成准备条件？

主 Gate 是 INT-C，同时记录 INT-D。接口技术检查通过：反事实隔离、响应逐位一致、TaskOption 映射和已保存 J15 都成立。INT-A 才表示影子契约已核验，并且也只授权下一阶段制定正式集成规范，不表示 planner_ready。当前还没有批准的部署 seed、部署时限和回退规则，所以不具备把该链路写入正式 planner 的条件。planner_ready、deployment_verified、FROZEN_FOR_FRESH_CONFIRMATION 都保持 False。本阶段不开始 Phase-4F，也不开始下一份集成规范。

## 对照

DeepSets 选中分配落在已保存 120 候选池内的比例是 0。Phase-5L Hungarian 直接分配的对应比例也是 0。因此这两条路径没有已保存 J15，不能和 C、D、E 比较收益。例如 310110 t=0，DeepSets 给出 `9;-1;0;-1;-1;-1`，其中浮标 9 不在该 UAV 的 top-3 里；HGNN 直接分配给出 `4;5;2;3;1;0`，六架 UAV 全部被指派，也不在 Stage-1 互斥池中。影子相对这两条路径的分配一致比例都是 0。池外结果没有重新 rollout。

D 与 E 在全部状态和全部 seed 上的选中分配一致，响应逐位相同。E 相对 C 的平均 selected J15 差是 4.5000e5 bit，与已报告数字相同。这个差值只出现在 5 个决策时刻：310110 t=0 为 8.0000e5，310111 t=15 为 3.0000e6，310113 t=15 为 2.0000e6，310114 t=15 为 3.0000e6，310115 t=15 为 2.0000e6。其余 19 个时刻的差是 0。这是已消费标签上的重放，不是新的独立确认。
