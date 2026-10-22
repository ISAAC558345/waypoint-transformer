# HTW Phase-6D：时间抽象与 hold-aware 联合规划审计

Gate 是 TEMP-F（TEMPORAL_CANDIDATE_COVERAGE_FAILURE）。

Phase-6A 仍是 BV2-E，Phase-6B 仍是 BITQ-A，Phase-6C 仍是 DYN-A。环境是 `L3BBitQueueEnvV2`。低层执行器始终是 deterministic fly-toward。320100–320131 没有打开。没有训练 HGNN、DeepSets、MLP、MAPPO、Transformer 或 value network。`planner_ready = False`，`deployment_verified = False`。

shadow assignment evaluator 只用于诊断、计算宏动作价值和判断未来监督目标。它不是正式系统里的第三层。若以后训练高层，结构必须是 state → HGNN → joint assignment，由 HGNN 直接学习宏动作累计收益。不能做成 HGNN → top-3 → 第二个决策层再选目标。

## Q1. 为什么 one-step true-bit 即时收益高，闭环收益却低？

Phase-6C 已经在同一初始状态上看到 one-step joint served bits 高于 independent max-rate。hold=15 的闭环复现结果是 max-rate 均值 7.61185e+07 bit，joint true-bit 均值 1.5e+07 bit。按 48 个 episode step 统计，joint 的零服务比例是 0.902018。它用一拍目标选中当前就能传完的近端队列。服务在移动之后结算，而步长是 22 m、服务半径是 16 m：已经贴着浮标时，一步 fly-toward 会先冲出半径。hold 把这个目标锁住，队列传完后的剩余步不再服务，一拍规则也不会为了远端尚未进入半径的浮标出发。

## Q2. 92% 无服务的主要来源是什么？

joint true-bit 在 hold=15 的零服务时间步里，出半径 A 占 0.10646，其中过冲（移动前在半径内、服务时已出半径）占 0.10646。队列提前耗尽 B 占 0.710574，干扰 C 占 0，决策时目标已空 D 占 0，其他 F 占 0.182966。A、B、C、D、F 加总为 1。E（hold 锁定在空目标上）占 0.710574，与 B 是同一批步。过冲是 A 的子集。F 主要是仍有正队列时一拍规则选择 idle，没有开始飞向远端。

## Q3. 缩短 hold 后，joint true-bit 是否恢复？

诊断用的最佳短 hold 是 8，其 joint 均值是 1.5e+07 bit，hold=15 是 1.5e+07 bit。配对差均值 0 bit，95% CI [0, 0]，正收益比例 0。idle-after-depletion 从 0.640951 变到 0.25651。HOLD_MISMATCH_CONFIRMED = False。缩短 hold 没有提高 joint true-bit 的 payload。hold=1 更差，因为每一步都在半径外重新决策，UAV 无法在覆盖圈内停留。这不是正式 hold 选择。

## Q4. hold=1/3/5/8/10/15 的诊断趋势在哪里？

joint_true_bit：hold=1 531250，hold=3 1.33105e+07，hold=5 1.49342e+07，hold=8 1.5e+07，hold=10 1.5e+07，hold=15 1.5e+07 bit。 max_rate_independent：hold=1 8.21998e+07，hold=3 9.16579e+07，hold=5 9.24379e+07，hold=8 8.08556e+07，hold=10 7.62626e+07，hold=15 7.61185e+07 bit。 曲线只说明诊断趋势，不作为正式 hold。

## Q5. J15_CURRENT 是否稳定优于 one-step true-bit？

B6 − B5 的均值是 5.5054e+07 bit，95% CI [5.43276e+07, 5.57543e+07]，正收益比例 1。TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED = True。

## Q6. J15_CURRENT 是否稳定优于 independent max-rate？

B6 − B3 的均值是 -6.06452e+06 bit，中位数 -5.34731e+06，标准差 6.58571e+06，95% CI [-7.64847e+06, -4.45773e+06]，正收益比例 0.109375。稳定优于 max-rate：False。B6 只是 instantaneous-rate top-3 的 candidate-pool optimum。Q9 显示这个 pool 对 J15 覆盖不足，所以这里不能把“B6 没有超过 max-rate”写成联合规划没有价值，也不能把它写成 TEMP-D。

## Q7. 如果不能，J15_FUTURE 是否可以？

B7 是 NON_CAUSAL_UPPER_BOUND，不能用于训练、checkpoint、正式控制器或以后的 confirmation。B7 − B3 的均值是 -1.46295e+06 bit，95% CI [-3.14161e+06, 275173]，正收益比例 0.4375，稳定优于 max-rate：False。

## Q8. 未来 arrival 是否是当前主要缺失信息？

B7 − B6 的均值是 4.60156e+06 bit，95% CI [3.92857e+06, 5.31018e+06]，正收益比例 0.96875。FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED = True。即使这个差很大，本阶段也不把 realized future arrival 放进任何特征。以后若要建模，只能使用到达统计、到达历史、队列趋势或时间索引这类因果量。

## Q9. top-3 candidate pool 在 J15 下是否仍足够？

预注册子集 320000–320015 上，max-rate 决策状态的 candidate-pool optimum 均值是 J15@3 1.14026e+07、J15@4 1.25853e+07、J15@5 1.35808e+07 bit。top-4 相对 top-3 的 scenario 均值相对增益是 0.108219，95% CI [0.077951, 0.147511]。top-5 pool optimum 的非 idle 边落在 top-3 候选里的比例是 0.748438。TEMPORAL_CANDIDATE_COVERAGE_FAILURE = True。这些是 candidate-pool optimum，不是 global Oracle。

## Q10. 宏动作尺度上的 interference 是否仍改变 assignment？

HOLD15 current-state pool optimum 在真实干扰和无干扰反事实下的 assignment disagreement 是 0.4375，对应的 ΔJ15 均值是 1.19616e+06 bit。MACRO_RELATION_VALUE_CONFIRMED = True。Phase-6C 的 0.55 只说明关系影响一拍分配；这里的数字才是宏动作尺度。

## Q11. queue、distance、rate、interference 哪些因素影响 J15 optimum？

queue_removed 的同状态 assignment disagreement 是 0.421875，ΔJ15 均值 46006.3 bit，闭环 payload 相对 B6 的均值差是 -200483 bit。distance_removed 的同状态 assignment disagreement 是 0.421875，ΔJ15 均值 1.05303e+06 bit，闭环 payload 相对 B6 的均值差是 -3.31118e+07 bit。interference_removed 的同状态 assignment disagreement 是 0.375，ΔJ15 均值 1.03252e+06 bit，闭环 payload 相对 B6 的均值差是 -8.32291e+06 bit。instantaneous_rate_removed 的同状态 assignment disagreement 是 0.8125，ΔJ15 均值 1.10132e+06 bit，闭环 payload 相对 B6 的均值差是 -4.4847e+06 bit。 同状态消融在 320000–320015 的 B6 决策点上；闭环差覆盖全部 64 个 development scenario。没有修改正式 benchmark。去掉 queue 会改变分配，但 ΔJ15 和闭环 payload 几乎不动。distance、interference，以及用 queue top-3 代替 instantaneous-rate top-3，都会降低 J15；闭环里下降最大的是去掉 distance。

## Q12. 现在是否可以开始训练 HGNN？

NO

## Q13. 如果下一阶段训练 HGNN，监督目标应该是什么？

现在不建立监督数据。不能使用 one-step assignment similarity，也不能使用 one-step served bits。当前 Gate 是候选池覆盖失败，下一阶段应先修正 Stage-1 candidate generator，而不是用 one-step 标签训练 HGNN。

## Q14. shadow evaluator 是不是最终系统中的中间决策层？

NO。它只是 diagnostic / teacher / label generator。最终高层应直接学习该价值。

## Q15. 是否需要现在重训 MAPPO？

NO。

## Q16. 是否需要现在加深 HGNN？

NO。

## Q17. 是否需要现在加入 continuity penalty？

NO。本阶段没有把 continuity penalty 立为新的 bit-level 因果项目。

## Q18. 320100–320131 是否仍完全 untouched？

YES。

## Q19. 最终 Gate 是哪一个？

TEMP-F（TEMPORAL_CANDIDATE_COVERAGE_FAILURE）。

## max-rate 为什么在闭环里更强

hold=15 的窗口对比：max-rate 的决策距离均值 29.5003 m，目标队列 2.49908e+06 bit，窗口服务 1.90296e+07 bit，服务利用率 0.741146，距离缩短 16.3572 m，空目标锁定步 1.05469。joint true-bit 的决策距离均值 1.37327 m，目标队列 625000 bit，窗口服务 3.75e+06 bit，服务利用率 0.0783854，距离缩短 -17.8882 m，空目标锁定步 7.69141。零服务分解上，max-rate 的旅行/耗尽/干扰/空目标/其他比例是 0.653846 / 0.334988 / 0 / 0.0111663 / 0。max-rate 的优势来自它按瞬时信道把 UAV 派向还能持续服务的链路，飞向过程本身变成了跨步的正则：目标在到达后仍有队列。joint 的一拍比特把 UAV 留在马上传完的近端目标上，宏动作后半段没有服务。

附加标志：HOLD_MISMATCH_CONFIRMED = False；TEMPORAL_OBJECTIVE_MISMATCH_CONFIRMED = True；FUTURE_ARRIVAL_INFORMATION_VALUE_CONFIRMED = True；TEMPORAL_CANDIDATE_COVERAGE_FAILURE = True；MACRO_RELATION_VALUE_CONFIRMED = True。
