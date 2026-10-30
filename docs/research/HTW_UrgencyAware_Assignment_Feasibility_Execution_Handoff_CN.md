# E17A 执行续办清单：先完成已授权诊断，再判断下一阶段

- 核对日期：2026-10-30（Asia/Shanghai）
- 仓库：ISAAC558345/waypoint-transformer
- 执行分支：experiment/priority-tag-next-direction
- 本次核对的分支 HEAD：161f0d73e66cdc5473f12225506988b0bbbbc25b
- 既有注册指令：docs/research/HTW_UrgencyAware_Next_Direction_Instructions_CN.md
- 既有选择记录：results/phase6i_priority_tag_assignment_feasibility/direction_decision.json
- 用户已明确选择 A，direction_status=CHOSEN_A；不要再次询问同一方向或常规实现参数

## 1. 当前已核实状态

截至本次连接所见的上述提交，E17 已有激活后的执行指令和用户选择记录，尚未见：
- src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/ 实现；
- docs/research/HTW_UrgencyAware_Assignment_Feasibility_Report_CN.md；
- 本阶段 summary、candidate_rows、baseline_rows 或 snapshot_manifest。

因此，目前没有 E17 诊断结果可供解释。本清单是执行交接，不是实验结果，不新增 E18，也不推断任务分配有效或无效。

执行前先读取远端最新提交与本地工作状态。若已经有另一份可验证的 E17 实现或结果，先提交/对账其来源和实际预算，不因本清单重复采样。若本清单发布后已有更新，以更新的真实数据为依据，不能重复执行已经完成的登记预算。

## 2. 立即落实的工作

1. 阅读既有完整 E17 指令和 CHOSEN_A 记录，核对冻结 L0、G1_RATE k=5 抽样分配上下文以及与正式 Scheme C 的区别。
2. 在独立目录完成诊断实现和注册配置，保持正式 Planner、奖励、SafetyProjector、FIFO、旧模型及历史结果不变；不训练新模型或更新参数。
3. 完成候选合法性、完整 capture/restore、分配 RNG 推进、基准回放、桶级已知/未来标签归因及无未来信息泄漏的必要测试。
4. 先提交真实实现并固定 source/runtime/config 哈希，再执行已登记自然场景仿真。不要只返回新的计划或再次创建用户选择记录。
5. 场景仅为 TRAIN 321000–321007，共 8 条冻结 L0 参考轨迹；t=0、15、30，共 24 个真实决策前快照。
6. 每快照比较原 G1_RATE k=5 池内最多 5 个唯一合法分配，每个完整执行 15 步，并额外重复一次基准。含参考轨迹在内最多 2544 个环境 steps，不扩池、不补跑有利样本。
7. 分别记录快照时已入队的 tagged 服务与之后新接纳的 tagged 服务；不能把未来标签收益当成当时可见信息可以支持的策略收益。
8. 按原 S0/S1/S2 条件统计服务、安全及通信机会，保留候选/快照/场景层次。事后最佳候选仍是 hindsight_diagnostic，不是可部署的新高层策略。

所有公式、容差、候选生成、场景、预算和判定仍以既有完整注册指令为准。本清单不修改任何科学协议。

## 3. 必须提交的可核对产物

- 真实实现、注册配置及必要测试；
- HTW_UrgencyAware_Assignment_Feasibility_Report_CN.md 中文报告；
- results/phase6i_priority_tag_assignment_feasibility/summary.json；
- baseline_rows.csv、candidate_rows.csv、snapshot_manifest.json；
- interface_identity.json、桶级归因明细；
- source/runtime/seed/冻结 checkpoint 哈希与检查日志；
- 实际环境 steps、有效快照/候选、缺失覆盖和证据限制。

若正常完成，报告 ASSIGNMENT_FEASIBILITY_COMPLETE 和实际 opportunity_evidence。
若契约、来源、基准复现或归因失败，报告 ASSIGNMENT_FEASIBILITY_BLOCKED、失败点及已消耗预算。即使被阻断，也应提交已有可核查结果与日志，不能把“尚未执行”写成“没有观察到机会”。

## 4. 停止与后续

不读新的 DEV/保留/测试数据，不启用 B，不训练，不合并，不修改正式 Scheme C，不重启已结案的裁剪路线。

执行端应在预算内完成 E17A 或给出有证据的 BLOCKED 交付，再推送结果。只有读到这份诊断结果后，才选择是否需要下一阶段；不凭指令激活提交虚构实验进展。
