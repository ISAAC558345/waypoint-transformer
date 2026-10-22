# HTW Low-Level V2 Phase-4B：通信干扰可控性与动作相关性审计

本阶段 **training=false**。未改 reward / Actor / Critic / Neighbor Encoder / SafetyProjector / d_safe / 高层 / hold。未加 interference penalty 或 SINR reward。未重做 collision distillation。未访问 `223000–223063`、`222000–222063`、`106000–106063`。未创建新 independent validation。候选幅度训练前冻结：`0.5 * alpha_residual = 0.125`（无网格）。H=3 唯一。

恢复：Phase-3 **LLV2-MS2-A**，`stable_residual_advantage=true`，**REL2-0**。Phase-4A **COLREL-C**，`collision_relation_internalized=false`。碰撞继续由 SafetyProjector 硬保证。

开发数据：仅 `LLV2_train` + `LLV2_design` 的 L0 冻结 rollout。Actor 可自由规划的样本（`policy_active_mask=1`，排除 ServiceResidualLock / idle / invalid / empty queue）一共 **30** 个 agent-state。这本身说明：真正允许 Residual 的时刻大多还不在服务链路内。HIGH/LOW 按该时刻 **team total interference** 的 25% 分位划分：HIGH=8，LOW=14。

Actor 通信通道：`[SINR, rate, total_interference, INR]`，由环境原样拷贝，**无 log / clip / scale**。无有效 target 时全零。SINR 与 rate 单调等价（Spearman≈1），属于特征重复，不是第二套公式。

---

## 22 问正式回答

1. **Interference/INR 进入 Actor 前后的尺度？**  
   完全相同（identity）。interference：min=0.001（噪声底），median=0.001，mean≈0.00409，max≈0.0938，std≈0.0166。INR：min=0，median=0，mean≈3.09，max≈92.8。多数 active 飞行状态只有噪声，不是“数值被压成 0”，而是 **几何上尚未进入共信道服务**。

2. **SINR/rate 尺度？**  
   同样 identity。SINR mean≈0.065，多数为 0，max≈1.95。rate 为 bps，mean≈5.19e4，max≈1.56e6，std≈2.80e5。rate 动态范围远大于 interference，线性层会被 rate 主导。

3. **Jacobian 多大？**  
   Phase-3 五 seed、design 状态上：mean ||∂μ/∂SINR||≈0.00312，||∂μ/∂interference||≈0.00256。量级接近、都很小。shuffle≈0 主要是 **权重几乎不读这些通道**，不只是输入尺度问题。

4. **HIGH interference state 数量？** **8**（team-interf top 25%；active 全集 30）。

5. **每 state 平均多少 safe alternative？** HIGH mean feasible（含 C0）= **4.5**。

6. **>1% team-rate headroom 比例？** HIGH **0.375**

7. **>5%？** HIGH **0.375**

8. **>10%？** HIGH **0.375**（改善集中在少数状态，不是平滑的 1%/5%/10% 阶梯）

9. **HIGH best-vs-base median team-rate improvement？** **0**（均值相对改善≈0.220，被右尾拉动）

10. **LOW 对应 median？** **0.0090**。HIGH median 并不大于 LOW，`high_gt_low=false`。

11. **H=3 是否仍有通信收益？** 部分有：HIGH 中 37.5% 的 best H=3 team-rate 相对改善 >0；**median=0**。

12. **next-step / H3 payload 是否改善？** HIGH next-step payload+ = **0.125**；H3 payload+ = **0.125**。payload 几乎不随这些小横移改善。

13. **RL 通信 action regret？** HIGH mean team-rate regret 五 seed 约 **1.07e6–1.38e6** bps（相对 best feasible）。说明局部候选里常有更高 team-rate，但冻结 RL 没有吃到。

14. **RL residual 与 best communication 方向是否对齐？** cosine mean 跨 seed **0.004–0.63**（n=3 个 HIGH 且 best≠C0 的状态）。正方向比例高但样本极小，**不能**说 RL 在稳定做干扰感知。

15. **zero/shuffle 通信特征是否改变通信 utility 方向？** interference zero/shuffle：动作 L2≈1e-6，team-rate 变化≈-3.5。SINR shuffle：动作 L2≈0.005，偶发 utility 变化。interference 消融几乎不改动作。

16. **干扰与几何是否相关？** Spearman(interf, d_nearest)≈**-0.02**；Spearman(interf, d_target)≈**0.31**。弱相关，不作因果。

17. **COMM Gate？** **B**

18. **communication_trajectory_headroom？** **false**（未达 COMM-A：HIGH median≤0 且 HIGH 不优于 LOW；payload 头寸弱）

19. **COMFEAT Gate？** **B**（有限 rate headroom + 特征未做归一化但并非 almost-zero；Actor 未利用。**不**把“只改 normalization”当成本阶段下一步，因为 COMM 未达 A。）

20. **下一步唯一建议？** **现在不要训练，不要加 interference auxiliary。** Rate 上存在局部可行改进，但 HIGH/LOW 无稳定差异，payload/H3 收益弱。碰撞维持 SafetyProjector。

21. **本阶段是否训练？** **否。**

22. **是否重新碰 collision 模块？** **否。**

本阶段停止。
