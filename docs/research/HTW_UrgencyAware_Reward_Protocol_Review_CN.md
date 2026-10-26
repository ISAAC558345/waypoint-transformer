# E4 报告：紧急奖励准入标准与训练前协议审查

- 分支：`feature/maritime-wt-stage1a`
- 指令提交：`fe2967115cbb86e97e1b2d92c2c6bc47399681d0`
- E2 结果提交：`5210dd7611befceed37c897a998cc1c5bd2a76bd`，Gate：`URGENCY_REWARD_SCALE_INCONCLUSIVE`
- E3 结果提交：`6492e9960af094dc28e71748ea4a5b6c8d988f86`，Gate：`URGENCY_REWARD_FORM_INCONCLUSIVE`
- 本阶段 Gate：`URGENCY_REWARD_PROTOCOL_WAITING_FOR_OWNER`
- E3 Gate 保持原值：是。没有改写成 READY，也没有放宽预注册标准。
- 条件候选：Form B，`lambda_u = 0.0005`，`rho = 0.001`，状态 `CONDITIONAL_CANDIDATE_NOT_APPROVED`
- `lambda_selected = null`，`rho_selected = null`，`lambda_H = NOT_SELECTED`
- 性质：只复核已提交的报告、summary、schema 和正式奖励代码。没有重跑环境，没有重抽 tag，没有训练，没有修改正式 Planner、奖励、动作、checkpoint 或旧数据集。

## 1. 直接结论

审查材料可以核对，E3 的原结论保持 `URGENCY_REWARD_FORM_INCONCLUSIVE`。两种奖励形式都没有在预注册标准下通过。Form B 的 `lambda_u=0.0005, rho=0.001` 只作为后续受控试验的条件候选，状态是 `CONDITIONAL_CANDIDATE_NOT_APPROVED`。它没有被选定，也没有获准训练。

这是固定轨迹上的奖励尺度审查。它不表示训练后的策略会保持同样的 payload、服务量或排序。

复核问题：无。

## 2. 源文件与正式奖励

E2 与 E3 的数据、报告都对上各自的结果提交。正式 `worker.py`、正式 Planner 和冻结 checkpoint 的当前哈希与 E2 summary 中记录的 watched hash 一致。checkpoint 记录值是 `92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`。

### E2

- `results/phase6i_priority_tag_reward_scale_e2/step_rows.csv`：`7e5d5ab0ca360eee075ee637fe575a07603fe72947188ec792dec7b9086fa49f`，提交 `5210dd7611befceed37c897a998cc1c5bd2a76bd`
- `results/phase6i_priority_tag_reward_scale_e2/episode_rows.csv`：`876e162dc6ed6706e5d99a1cad147ea263e1e2ec9ddcc6d6570dc65f9830d719`，提交 `5210dd7611befceed37c897a998cc1c5bd2a76bd`
- `results/phase6i_priority_tag_reward_scale_e2/summary.json`：`0d5a2645e39a38673f7411f828ebd3d08b004f0d0968e70f312953595ae48995`，提交 `5210dd7611befceed37c897a998cc1c5bd2a76bd`
- `results/phase6i_priority_tag_reward_scale_e2/schema.json`：`e6cc0e20a10a88d208ad5169cec5a73956b5236b98a5822b6bd07676c7176358`，提交 `5210dd7611befceed37c897a998cc1c5bd2a76bd`
- `docs/research/HTW_UrgencyAware_Reward_Scale_Replication_Report_CN.md`：`06c1f2c939f3fd6154f10cb225269bb4303ddd7e64ddfa8e6754872c1e6c5d7f`，提交 `5210dd7611befceed37c897a998cc1c5bd2a76bd`

### E3

- `results/phase6i_priority_tag_reward_form_e3/form_rows.csv`：`826d7a3b3f9a8f73be1db9b6c16c1d91e7f35adfda524efdaa616a5407d97a14`，提交 `6492e9960af094dc28e71748ea4a5b6c8d988f86`
- `results/phase6i_priority_tag_reward_form_e3/summary.json`：`78891052f1aef466b38a86a71fb3f6ce94e08a1b641248721f10e4ae49f94061`，提交 `6492e9960af094dc28e71748ea4a5b6c8d988f86`
- `results/phase6i_priority_tag_reward_form_e3/schema.json`：`125f31665dc3393aa8baba97a9fd915004b4b62afcb21ccc50786c7efbcf3413`，提交 `6492e9960af094dc28e71748ea4a5b6c8d988f86`
- `docs/research/HTW_UrgencyAware_Reward_Form_Comparison_Report_CN.md`：`845d9161ea7bf302e42db8bcc7cc2ea234d6b9c8396a927bd07ec8a238f0f6db`，提交 `6492e9960af094dc28e71748ea4a5b6c8d988f86`

### E1 schema 与正式代码

- `results/phase6i_priority_tag_prototype/schema.json`：`79db45bfb9cf25664500f56de59d47800ec5aee42017ec4729d73899f67fad22`，提交 `827f2ad51654ab984cb0326a2670693cd2793c7a`
- `src/wt/lowlevel_v2/phase1/protocol.py`：`149292d9feccb64c62f01c5df617959f469a5b8cdabf42b7079db9cf7c4226a4`，提交 `8a34efc718665a7cfe4375b662e8f5249263f03d`
- `src/wt/benchmark_v2/phase6g_lowlevel_mappo/protocol.py`：`9b1c90a57607cbbc512a82663d20e21e8df0be158f5fceda70373159b825194e`，提交 `8a34efc718665a7cfe4375b662e8f5249263f03d`
- `src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py`：`79fc576924aaabc74df7b5badb4f23cd4fbbba915162ca646e6b819855f425b3`，提交 `8a34efc718665a7cfe4375b662e8f5249263f03d`
- `src/wt/benchmark_v2/phase6iaa_formal_planner/planner.py`：`dc67c0402a273400fc5490f72b0d8fda34713a0baadb1005de06f17e3be8bf9a`，提交 `8a34efc718665a7cfe4375b662e8f5249263f03d`
- `src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt`：`92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e`，提交 `8a34efc718665a7cfe4375b662e8f5249263f03d`

正式低层奖励仍是：

`r_base = served_bits/1e6 - 0.05*c_project - 0.01*c_residual`

其中 `payload_term = served_bits / 1e6`，投影系数是 0.05，残差系数是 0.01，`ACTION_SCALE = 1`。`phase6g` 协议只引用这些常数，没有另写一套系数。本阶段没有改这个函数。

## 3. E3 的原门槛

门槛在 E3 之前已经固定，本阶段原样保留：

- 每个场景的 bonus/payload 均值不超过 1%；
- bonus/penalty 的 P95 不超过 1；
- payload 项相差超过 1 bit 时，奖励排序没有实质变化；
- Form B 相对未封顶 bonus 的保留比例不低于 0.5。

E3 的记录是：384 条 step、8 条 episode，Form A 复现了 E2 的 lambda 表，各形式累计 payload 相同，Form B 的 cap 没有超过对应 rho，`usable_candidates` 为空。因此 E3 Gate 继续是失败于保守标准，不是通过。

## 4. 两种形式的尺度

Form A：`r_A = r_base + lambda_u * U_t`。`U_t = urgency_served_bits / 1e6`。lambda 只取 0、0.0005、0.001、0.002。0.05 是高风险参照。

Form B：`r_B = r_base + min(lambda_u * U_t, rho * payload_term)`。rho 只取 0.001、0.0025、0.005，也就是本步成功服务 payload 项的 0.1%、0.25%、0.5%。本步 `payload_term = 0` 时 bonus 为 0。

下表数字来自 E3 已保存的 summary，不是本阶段重新计算的。

| 形式 | lambda_u | rho | bonus/payload 均值 | bonus/penalty P95 | payload 实质场景 | base 实质场景 | 保留比例 | 通过 E3 标准 |
|---|---|---|---|---|---|---|---|---|
| A | 0 | 无 | 0 | 0 | 无 | 无 | 不适用 | 是 |
| A | 0.0005 | 无 | 0.000130 | 11.613369 | 无 | 321000 | 不适用 | 否 |
| A | 0.0010 | 无 | 0.000260 | 23.226738 | 321000 | 321000 | 不适用 | 否 |
| A | 0.0020 | 无 | 0.000519 | 46.453477 | 321000、321007 | 321000、321007 | 不适用 | 否 |
| A | 0.0500 | 无 | 0.012977 | 1,161.336924 | 321000、321001、321002、321003、321004、321005、321006、321007 | 321000、321001、321002、321003、321004、321005、321006、321007 | 不适用 | 否 |
| B | 0.0005 | 0.0010 | 0.000130 | 11.613369 | 无 | 321000 | 1.000000 | 否 |
| B | 0.0005 | 0.0025 | 0.000130 | 11.613369 | 无 | 321000 | 1.000000 | 否 |
| B | 0.0005 | 0.0050 | 0.000130 | 11.613369 | 无 | 321000 | 1.000000 | 否 |
| B | 0.0010 | 0.0010 | 0.000260 | 23.226738 | 321000 | 321000 | 1.000000 | 否 |
| B | 0.0010 | 0.0025 | 0.000260 | 23.226738 | 321000 | 321000 | 1.000000 | 否 |
| B | 0.0010 | 0.0050 | 0.000260 | 23.226738 | 321000 | 321000 | 1.000000 | 否 |
| B | 0.0020 | 0.0010 | 0.000481 | 37.444649 | 321000 | 321000、321007 | 0.905814 | 否 |
| B | 0.0020 | 0.0025 | 0.000519 | 46.453477 | 321000、321007 | 321000、321007 | 1.000000 | 否 |
| B | 0.0020 | 0.0050 | 0.000519 | 46.453477 | 321000、321007 | 321000、321007 | 1.000000 | 否 |

E2 对 lambda 0.0005 的记录是 bonus/payload 均值约 0.000130，bonus/penalty P95 为 11.613369，payload 排序的实质重排场景数是 0。0.001、0.002、0.05 的 payload 实质重排场景数是 1、2、8。E3 读到的 payload 实质场景数与这些记录一致。

运动惩罚是 `0.05 * c_project + 0.01 * c_residual`，用来惩罚执行动作相对期望动作、以及残差动作本身。它是正则项。payload 项才是主要任务收益。bonus/penalty P95 大于 1，表示在冻结轨迹的许多步骤上，紧急度 bonus 已经大于这个正则项；对 0.0005 而言，总体 P95 是 11.613369，同时它只约占 payload 项的 0.000130。这不自动等于“系统目标失败”，因为研究目标看的是 payload 是否仍是主项、紧急信息是否只拿到小比例额外奖励。它仍然是策略风险：正则项原来用于限制动作变化，bonus 在大量步骤上超过它，训练时仍可能把动作从原来的低惩罚方向拉开。这个风险保留。门槛本身不放宽，E3 也不改成通过。

## 5. 为什么有界 bonus 不能保证训练后的总 payload 不下降

Form B 的 cap 是单步约束：`0 <= bonus_t <= rho * payload_term_t`，并且 `payload_term_t = 0` 时 bonus 为 0。它限制的是“这一步已经成功服务的 payload 项”里面，紧急度最多再加多少奖励。

它没有约束策略会选择哪一条轨迹。训练优化的是新策略下的累计 `r_B`。新动作会改变服务量、惩罚和紧急度服务。cap 用的是新轨迹自己的本步 payload，所以一条总 payload 更低的轨迹仍然可以满足 cap。例如少服务一块较大的普通数据、多服务一块较小的高紧急度数据时，单步 bonus 仍不超过该步新的 payload 项，但 episode 总 payload 可以下降。cap 也不会奖励没有成功交付的步骤，因为它在零 payload 上是 0；这只避免给未交付数据发奖励，并不惩罚“本可以交付但没有交付”的 payload。

E2/E3 的累计 payload 在各形式之间相同，是因为动作没有重新执行。固定轨迹重算不能推出训练后的 payload、轨迹或服务量。只有训练后的独立 benchmark 能回答闭环 payload 是否下降。本阶段不把 cap 写成性能保证。

在已保存的 384 步上，lambda 0.0005 和 0.001 的未封顶 bonus 已经低于最小 rho 0.001，所以这些 Form B 行的保留比例是 1，数字与对应 Form A 相同。cap 的设计约束存在，但在这批轨迹上没有实际截掉 bonus。只有 lambda 0.002、rho 0.001 的保留比例低于 1，截断后惩罚 P95 仍大于 1。

## 6. 条件候选

建议后续若做小规模、独立的低层试验，优先使用 Form B，而不是无上限的加法奖励。理由只是设计：bonus 被限制在本步成功服务 payload 项的固定小比例内。这不是性能结论，也不是从固定轨迹里选出的最优组合。

`lambda_u=0.0005, rho=0.001` 是这条设计里的首个条件候选，因为 cap 是 payload 项的 0.1%，而且 E3 记录中没有 payload 项实质重排。它没有通过 E3：bonus/penalty P95 是 11.613369，大于 1；base reward 实质重排场景是 321000；保留比例是 1.000000。状态因此是 `CONDITIONAL_CANDIDATE_NOT_APPROVED`。网格不扩大。不允许同时改 observation、正式 reward、动作、SafetyProjector 或任务分布。新模型若将来获准训练，也只能是新 checkpoint，不能覆盖 L0。

高层 Scheme C 保持冻结，继续 payload-first。低层 bonus 不加入高层目标。紧急度沿用 E1 的 `synthetic_priority_bucket_v1`，tag 必须标成 synthetic。`lambda_H` 保持 `NOT_SELECTED`。本协议不要求另定高层 `delta_payload_max`，因为高层不允许为了紧急度牺牲实质 payload。

## 7. 需要负责人确认

1. 是否接受这条准则：训练后的独立 benchmark 如果出现总 payload 下降，就停止并回滚，不把该模型当成可用策略。
2. 是否批准把 `lambda_u=0.0005, rho=0.001` 作为唯一的首个受控低层训练候选。批准也只允许训练新模型，绝不覆盖 L0。
3. 若第 2 项未批准，则维持不训练。本阶段现在就处于这个状态。

以上三项都还没有答案。没有负责人的明确批准，就不能进入独立 observation/reward builder 的 smoke，也不能训练。

## 8. 停止

本阶段不进入 E5，不启动 L1/L2，不修改正式奖励。E3 Gate 保持 `URGENCY_REWARD_FORM_INCONCLUSIVE`。
