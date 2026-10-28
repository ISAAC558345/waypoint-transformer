# 标签信号可见性、控制能力与安全退化诊断报告

- Gate：`DIRECT_TAG_SIGNAL_AUDIT_COMPLETE`
- 本 Gate 不是性能验收。E10 的 `DIRECT_TAG_REWARD_BENCHMARK_FAIL` 保持不变。
- 本阶段没有训练、没有搜索系数、没有合并。
- 合并状态：False
- 标签是合成标签，不是外部业务优先级。
- seed 2 只作为历史复现时重新施加的进程上下文。模型种子、场景 reset 种子和任务分配种子彼此分开。
- 诊断轨迹使用评估分配种子 `7000000 + scenario_id`，不是训练分配种子 `321000000 + model_seed * 10000 + env_index`。

## 1. E10 结果与来源

- 场景行数 320，唯一键 320，问题 无。
- 重算 Gate `DIRECT_TAG_REWARD_BENCHMARK_FAIL`，与已保存 FAIL 一致：True。
- 通过种子数 0。区间差异 无。
- 训练实现字节：missing。指令提交只含指令：True。
- checkpoint 问题：无。

重算后的 tagged 与 payload 均值仍以保存的场景表为准，负均值不是显著性结论。三个 tagged 区间是否跨零，以 E10 保存区间和本次重算差异为准。

- L0 / 321101 预检匹配 True，最大 bit 差 0.000000，路径差 0.000000000000 m。

## 2. 标签是否进入 actor 和控制窗口

暴露率的分子和分母来自 8 条完整轨迹。生成标签的抽样率不作为可见率。

- L0 seed 0 / 321000：可见 tagged UAV 步 19/288 = 0.065972；其中策略掩码为正 19/19 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 54/323 = 0.167183；入队 tagged 桶 16 / 生成 tagged 事件 27；正 tagged 服务步 13/48 = 0.270833；可见延迟中位数 7.000（未删失 10，未可见 6）；服务延迟中位数 11.000（未删失 15，未服务 1）；tagged 服务 14536967.034 bit，bonus_tag 合计 0.000000。
- L3 seed 0 / 321000：可见 tagged UAV 步 17/288 = 0.059028；其中策略掩码为正 17/17 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 67/302 = 0.221854；入队 tagged 桶 12 / 生成 tagged 事件 27；正 tagged 服务步 12/48 = 0.250000；可见延迟中位数 11.000（未删失 7，未可见 5）；服务延迟中位数 11.000（未删失 9，未服务 3）；tagged 服务 9000000.000 bit，bonus_tag 合计 0.008529。
- L3 seed 1 / 321000：可见 tagged UAV 步 14/288 = 0.048611；其中策略掩码为正 14/14 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 64/303 = 0.211221；入队 tagged 桶 13 / 生成 tagged 事件 27；正 tagged 服务步 11/48 = 0.229167；可见延迟中位数 5.500（未删失 6，未可见 7）；服务延迟中位数 10.500（未删失 10，未服务 3）；tagged 服务 10000000.000 bit，bonus_tag 合计 0.009793。
- L3 seed 2 / 321000：可见 tagged UAV 步 21/288 = 0.072917；其中策略掩码为正 21/21 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 56/308 = 0.181818；入队 tagged 桶 17 / 生成 tagged 事件 27；正 tagged 服务步 17/48 = 0.354167；可见延迟中位数 11.000（未删失 11，未可见 6）；服务延迟中位数 5.000（未删失 15，未服务 2）；tagged 服务 15000000.000 bit，bonus_tag 合计 0.013436。
- L0 seed 0 / 321001：可见 tagged UAV 步 12/288 = 0.041667；其中策略掩码为正 12/12 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 37/292 = 0.126712；入队 tagged 桶 12 / 生成 tagged 事件 23；正 tagged 服务步 14/48 = 0.291667；可见延迟中位数 1.000（未删失 9，未可见 3）；服务延迟中位数 1.000（未删失 12，未服务 0）；tagged 服务 12000000.000 bit，bonus_tag 合计 0.000000。
- L3 seed 0 / 321001：可见 tagged UAV 步 18/288 = 0.062500；其中策略掩码为正 18/18 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 43/284 = 0.151408；入队 tagged 桶 12 / 生成 tagged 事件 23；正 tagged 服务步 14/48 = 0.291667；可见延迟中位数 1.000（未删失 8，未可见 4）；服务延迟中位数 1.000（未删失 10，未服务 2）；tagged 服务 10000000.000 bit，bonus_tag 合计 0.009522。
- L3 seed 1 / 321001：可见 tagged UAV 步 15/288 = 0.052083；其中策略掩码为正 15/15 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 55/291 = 0.189003；入队 tagged 桶 12 / 生成 tagged 事件 23；正 tagged 服务步 15/48 = 0.312500；可见延迟中位数 4.500（未删失 6，未可见 6）；服务延迟中位数 3.000（未删失 11，未服务 1）；tagged 服务 10658537.945 bit，bonus_tag 合计 0.009643。
- L3 seed 2 / 321001：可见 tagged UAV 步 15/288 = 0.052083；其中策略掩码为正 15/15 = 1.000000；掩码范围 [1.000, 1.000]；队首为 tagged 的浮标步 55/282 = 0.195035；入队 tagged 桶 12 / 生成 tagged 事件 23；正 tagged 服务步 13/48 = 0.270833；可见延迟中位数 1.000（未删失 9，未可见 3）；服务延迟中位数 1.000（未删失 9，未服务 3）；tagged 服务 8896686.283 bit，bonus_tag 合计 0.008391。

策略掩码在 worker 中是全 1。可见且策略有效因此等于可见 UAV 步，它不是学出来的活动门。
新生成的 tagged 桶在生成步之后才可能进入下一步的 actor 输入。未成为队首或没有 UAV 关联的桶记为未可见，不补插延迟。

## 3. 冻结 L3 对 tag 特征是否敏感

比较只把 actor 输入的 tag 列置零。队列、年龄、分配和物理状态不变。动作不变只说明这些快照上不敏感。

- L0 seed 0 / 321000：不接受 tag 输入，未伪造置零。
- L3 seed 0 / 321000：对齐步 48 / 48，对齐失败 0；对齐且 tag 输入非零的步 17；其中原始残差变化 8，执行动作变化 8；残差差中位数 0.000000，最大 0.001899；执行动作差中位数 0.000000。
- L3 seed 1 / 321000：对齐步 48 / 48，对齐失败 0；对齐且 tag 输入非零的步 14；其中原始残差变化 8，执行动作变化 8；残差差中位数 0.000538，最大 0.004726；执行动作差中位数 0.000114。
- L3 seed 2 / 321000：对齐步 48 / 48，对齐失败 0；对齐且 tag 输入非零的步 17；其中原始残差变化 5，执行动作变化 5；残差差中位数 0.000000，最大 0.001143；执行动作差中位数 0.000000。
- L0 seed 0 / 321001：不接受 tag 输入，未伪造置零。
- L3 seed 0 / 321001：对齐步 48 / 48，对齐失败 0；对齐且 tag 输入非零的步 14；其中原始残差变化 7，执行动作变化 7；残差差中位数 0.000019，最大 0.002167；执行动作差中位数 0.000000。
- L3 seed 1 / 321001：对齐步 48 / 48，对齐失败 0；对齐且 tag 输入非零的步 14；其中原始残差变化 3，执行动作变化 3；残差差中位数 0.000000，最大 0.001815；执行动作差中位数 0.000000。
- L3 seed 2 / 321001：对齐步 48 / 48，对齐失败 0；对齐且 tag 输入非零的步 14；其中原始残差变化 4，执行动作变化 4；残差差中位数 0.000000，最大 0.001167；执行动作差中位数 0.000000。

## 4. 短时域动作反事实

预定规则：每条轨迹 8 个时间等分左端点，共 64 个快照；每个快照 5 个候选、各 4 步；扰动幅度 0.1。候选是加在每架 UAV 残差上的同一偏移，经过同一裁剪和 SafetyProjector。
基线可复现快照 64 / 64。在可复现快照上，tagged 服务增加超过 1 bit、payload 下降不超过 1 bit、安全和约束失败不增加的非零扰动候选行数：18。
结论：在预定短时域筛选中观察到 18 个候选行满足诊断阈值。4 步筛选不是长期可控性的证明，也不是新策略的性能结果。

## 5. 奖励与离线学习信号

以下比值和 GAE 都是诊断估计，不是 300 次更新期间的优势或梯度。分母为零的步单独计数。中位数使用全部分母非零的步，包含 bonus 为 0 的步，因此可以为 0；正 tag bonus 的步数和 float32 是否仍能分辨另计。

- L3 seed 0 / 321000：bonus_age 合计 0.005492，bonus_tag 合计 0.008529，payload 项合计 51.719438，运动惩罚合计 0.004359；payload 项为零的步 19，运动惩罚为零的步 0；正 tag bonus 步 12，其中 float32 仍能分辨 12；tag/payload 中位数 0.000000，tag/运动惩罚中位数 0.000000。
- L3 seed 1 / 321000：bonus_age 合计 0.005357，bonus_tag 合计 0.009793，payload 项合计 53.218614，运动惩罚合计 0.004043；payload 项为零的步 19，运动惩罚为零的步 0；正 tag bonus 步 11，其中 float32 仍能分辨 11；tag/payload 中位数 0.000000，tag/运动惩罚中位数 0.000000。
- L3 seed 2 / 321000：bonus_age 合计 0.005917，bonus_tag 合计 0.013436，payload 项合计 52.083931，运动惩罚合计 0.002893；payload 项为零的步 17，运动惩罚为零的步 0；正 tag bonus 步 17，其中 float32 仍能分辨 17；tag/payload 中位数 0.000170，tag/运动惩罚中位数 0.000000。
- L3 seed 0 / 321001：bonus_age 合计 0.005463，bonus_tag 合计 0.009522，payload 项合计 57.767572，运动惩罚合计 0.003700；payload 项为零的步 11，运动惩罚为零的步 0；正 tag bonus 步 14，其中 float32 仍能分辨 14；tag/payload 中位数 0.000000，tag/运动惩罚中位数 0.000000。
- L3 seed 1 / 321001：bonus_age 合计 0.006388，bonus_tag 合计 0.009643，payload 项合计 57.831926，运动惩罚合计 0.003393；payload 项为零的步 10，运动惩罚为零的步 0；正 tag bonus 步 15，其中 float32 仍能分辨 15；tag/payload 中位数 0.000000，tag/运动惩罚中位数 0.000000。
- L3 seed 2 / 321001：bonus_age 合计 0.005548，bonus_tag 合计 0.008391，payload 项合计 55.896686，运动惩罚合计 0.002580；payload 项为零的步 11，运动惩罚为零的步 0；正 tag bonus 步 13，其中 float32 仍能分辨 13；tag/payload 中位数 0.000000，tag/运动惩罚中位数 0.000000。

- L0 seed 0 / 321000：L0 has no experiment critic; its value trace is not used as an L3 baseline
- L3 seed 0 / 321000 诊断估计：归一化优势里 tag 增量的平均绝对差 0.000115，最大 0.000339；年龄增量的平均绝对差 0.000062；原始 GAE 的 tag 平均绝对差 0.002337。
- L3 seed 1 / 321000 诊断估计：归一化优势里 tag 增量的平均绝对差 0.000128，最大 0.000367；年龄增量的平均绝对差 0.000060；原始 GAE 的 tag 平均绝对差 0.002683。
- L3 seed 2 / 321000 诊断估计：归一化优势里 tag 增量的平均绝对差 0.000129，最大 0.000350；年龄增量的平均绝对差 0.000054；原始 GAE 的 tag 平均绝对差 0.003845。
- L0 seed 0 / 321001：L0 has no experiment critic; its value trace is not used as an L3 baseline
- L3 seed 0 / 321001 诊断估计：归一化优势里 tag 增量的平均绝对差 0.000151，最大 0.000338；年龄增量的平均绝对差 0.000041；原始 GAE 的 tag 平均绝对差 0.002801。
- L3 seed 1 / 321001 诊断估计：归一化优势里 tag 增量的平均绝对差 0.000111，最大 0.000291；年龄增量的平均绝对差 0.000056；原始 GAE 的 tag 平均绝对差 0.002893。
- L3 seed 2 / 321001 诊断估计：归一化优势里 tag 增量的平均绝对差 0.000133，最大 0.000277；年龄增量的平均绝对差 0.000039；原始 GAE 的 tag 平均绝对差 0.002446。

历史训练日志没有逐步 bonus_tag。不能从 mean_bonus 回推。
- 训练 seed 0，300 次更新：KL 均值 0.005051，最大 0.042079；clip fraction 均值 0.060237；value loss 均值 56.861；explained variance 均值 0.333982；动作标准差均值 0.119184；投影率均值 0.275434；裁剪前梯度均值 277.731，最大 6392.045；安全频率均值 0.000208。较大的裁剪前梯度本身不构成失稳结论。
- 训练 seed 1，300 次更新：KL 均值 0.005371，最大 0.016674；clip fraction 均值 0.064978；value loss 均值 57.947；explained variance 均值 0.330284；动作标准差均值 0.115395；投影率均值 0.272292；裁剪前梯度均值 235.505，最大 7138.613；安全频率均值 0.000191。较大的裁剪前梯度本身不构成失稳结论。
- 训练 seed 2，300 次更新：KL 均值 0.004948，最大 0.016029；clip fraction 均值 0.061112；value loss 均值 58.065；explained variance 均值 0.319201；动作标准差均值 0.117640；投影率均值 0.244427；裁剪前梯度均值 212.082，最大 3138.564；安全频率均值 0.000260。较大的裁剪前梯度本身不构成失稳结论。

## 6. 两处安全事件

计数器在执行后的真实间距上使用 1e-9；投影器在预测间距上使用 1e-6。阈值都是 1.0 m。同一间距可以同时通过投影器检查并被计数器记为违规。证据标签可以并列，不改 SafetyProjector。

- L3 seed 1 / 321123：与 E10 保存行匹配 True，解释状态 reported，当前 bit 差 {'payload_bits': 0.0, 'served_tagged_bits': 0.0, 'served_ordinary_bits': 0.0, 'dropped_bits': 0.0, 'safety_violations': 0.0, 'constraint_failures': 0.0}，路径差 0.000000000000。
  - 步 3，UAV 0 与 1，间距 0.999999999 m，前一步间距 1.515808057 m，阈值 1.000，预测间距 0.999999999，投影状态 ok，约束失败 0。
    证据标签：counter_tolerance_1e-9_and_projector_tolerance_1e-6, policy_changed_positions_at_event, accepted_projection_then_realized_violation, predicted_safe_realized_violation, near_numeric_boundary。
- L3 seed 2 / 321104：与 E10 保存行匹配 True，解释状态 reported，当前 bit 差 {'payload_bits': 0.0, 'served_tagged_bits': 0.0, 'served_ordinary_bits': 0.0, 'dropped_bits': 0.0, 'safety_violations': 0.0, 'constraint_failures': 0.0}，路径差 0.000000000000。
  - 步 1，UAV 3 与 5，间距 0.999999999 m，前一步间距 3.490255825 m，阈值 1.000，预测间距 0.999999999，投影状态 ok，约束失败 0。
    证据标签：counter_tolerance_1e-9_and_projector_tolerance_1e-6, policy_changed_positions_at_event, accepted_projection_then_realized_violation, predicted_safe_realized_violation, near_numeric_boundary。

## 7. 测试、证据缺口与下一步建议

- 相关测试命令：`py -3 -m unittest tests.test_priority_tag_signal_audit tests.test_priority_tag_direct_service_reward tests.test_priority_tag_prototype tests.test_priority_tag_eval_context tests.test_priority_tag_reward_attribution tests.test_priority_tag_attribution_full_replay`，退出码 0。
- 相关测试失败：无。
- 全量测试命令：`py -3 -m unittest discover -s tests`，退出码 1。
- 全量测试失败 17 项：['ERROR: test_round22_original_lock_and_narrow_amendment_verify (test_governance_state_sync.GovernanceStateSyncTests.test_round22_original_lock_and_narrow_amendment_verify)', 'ERROR: test_15_no_nan_inf_and_packed_config (test_htw_phase11.Phase11Tests.test_15_no_nan_inf_and_packed_config)', 'ERROR: test_unnamed_limits_remain_explicit (test_phase6iae_fault_handling.Phase6IAEContractTests.test_unnamed_limits_remain_explicit)', 'ERROR: test_graph_batch_excludes_labels_and_keeps_checkpoint_widths (test_phase6ic_hgnn_smoke.Phase6ICDatasetTests.test_graph_batch_excludes_labels_and_keeps_checkpoint_widths)', 'ERROR: test_round20_registered_hashes_unchanged (test_round21_hgnn_task_waypoint.Round21HGNNTests.test_round20_registered_hashes_unchanged)', 'ERROR: test_round21_amendment_overlay_is_narrow_and_exact (test_round22_governance.Round22GovernanceTests.test_round21_amendment_overlay_is_narrow_and_exact)', 'ERROR: test_round22_governance_state_sync_amendment_is_narrow_and_exact (test_round22_governance.Round22GovernanceTests.test_round22_governance_state_sync_amendment_is_narrow_and_exact)', 'ERROR: test_round22_frozen_lock_amendment_predictor_and_checkpoints_verify (test_round23_audit.Round23AuditTests.test_round22_frozen_lock_amendment_predictor_and_checkpoints_verify)', 'ERROR: test_round23_lock_hashes_its_sources_outputs_figures_and_report (test_round23_audit.Round23AuditTests.test_round23_lock_hashes_its_sources_outputs_figures_and_report)', 'ERROR: test_round24_lock_and_frozen_preconditions (test_round25_supervision_ambiguity.Round25SupervisionAmbiguityTests.test_round24_lock_and_frozen_preconditions)', 'ERROR: test_round25_lock_and_frozen_preconditions (test_round26_action_supervision_ambiguity.Round26ActionSupervisionAmbiguityTests.test_round25_lock_and_frozen_preconditions)', 'ERROR: test_round26_lock_and_frozen_preconditions (test_round27_conditional_planner_utility.Round27ConditionalPlannerUtilityTests.test_round26_lock_and_frozen_preconditions)', 'ERROR: test_01_round32_original_lock_sha_is_unchanged (test_round33_governance_compatibility.Round33GovernanceCompatibilityTests.test_01_round32_original_lock_sha_is_unchanged)', 'ERROR: test_02_successor_amendment_verifies (test_round34_governance_lifecycle.Round34GovernanceLifecycleTests.test_02_successor_amendment_verifies)', 'ERROR: test_round34_lock_rehashes_every_registered_artifact (test_round34_lock.Round34LockTests.test_round34_lock_rehashes_every_registered_artifact)', 'FAIL: test_all_required_outputs_figures_and_completion_records_exist (test_round23_audit.Round23AuditTests.test_all_required_outputs_figures_and_completion_records_exist)', 'FAILED (failures=1, errors=15)']。这些失败来自日志中的历史锁定文件缺失，审计相关测试本身退出码为 0。失败项已保留。
- 环境：3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)]；NumPy 2.4.1；Torch 2.12.0+cpu；CUDA False。
- 证据缺口：['E10 training checkpoint has no implementation digest; current source is not the training-time proof.', 'Historical training logs do not record per-step bonus_tag.', 'A four-step residual screen does not prove long-horizon controllability.', 'The synthetic generation rate is not the actor visibility rate.', 'L0 has no tag input, so it is an interface control rather than a tag-sensitivity baseline.', 'policy_active_mask is identically one in ExperimentWorker.', 'The repository test suite reported failures. They are listed and were not deleted.']

若继续做实验，应单独预先声明，并以本次未回答的问题为对照，而不是直接调大系数。短时域筛选看到了局部机会。这只说明这些冻结动作附近存在诊断意义上的差异，不能当作新策略已经更好。两处安全事件只有在重放匹配后才有证据标签；标签可以同时成立，不能收成单一原因。历史逐步 tag bonus 和训练当时的实现字节仍然缺失。

本报告到此停止，不启动下一阶段实验。
