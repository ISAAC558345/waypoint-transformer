# HTW Phase-4E Failure Attribution Audit

本阶段只对已经完成的 Phase-4E 失败做事后归因。没有重训 B2，没有修改 τ、Gate-1、Candidate-C、planner 或 decide()，没有删除或重生成 holdout。

**status = DEPLOYMENT_VERIFICATION_FAILED**

**deployment_verified = False**

**planner_ready = False**

**criterion = CONSERVATIVE_WITH_FALLBACK_V1**

**tau = 1534933.5944128654**

## 结论

四个 degradation 都发生在 Gate-1 按冻结规则放行 B2 的 scenario 上，而且 margin 与真实 ΔR 方向相反。B2 的 top-1 在这四场都不是 oracle。Candidate-C 的失败由 B2 ranking error 和具有误导性的 margin 共同造成。

failure is not merely a threshold-boundary effect.

230091 的 margin/τ = 1.013，靠近阈值。230070、230077、230078 的 margin/τ 分别是 1.427、1.265、2.482。其中 230078 落在 margin ≥ 2τ 的桶里，真实 ΔR 仍是 -0.182。

是否应该修改 B2、τ 或 Candidate-C：**NOT ESTABLISHED**。本审计不提出新阈值。

## Batch-1 / Batch-2

Candidate-C 的直接统计原因在 Batch-2：

| 项 | Batch-1 | Batch-2 |
| --- | --- | --- |
| n | 64 | 64 |
| Gate-1 选择 B2 | 20 | 30 |
| degradation | 0/64 | 4/64 = 0.0625 |
| mean ΔR | 0.101687 | 0.113262 |
| worst-case ΔR | 0 | -0.200 |
| C1 ≤ 0.02 | PASS | FAIL |
| C2 ≥ -0.125 | PASS | FAIL |
| C3 ≥ 0.02 | PASS | PASS |
| C4 ≥ 0 | PASS | PASS |

Batch-1 PASS，Batch-2 FAIL。Combined 不能抵消 Batch-2。`independent_batches_passed = 1`。

## 四个失败 scenario

ΔU 的单位是 L3-B 的 `block_payload_bits = 1e6`。128 场已实现的 ΔU 全部是这个步长的整数倍。四场失败分别是 -2、-2、-2、-1 个 payload block。这是 evaluation reward 的离散结构，不是连续残差被四舍五入。

| scenario | margin | margin/τ | 桶 | ΔR | ΔU | B2 regret | LOCAL regret | B2–ORACLE gap R | B2 top-1 = oracle | oracle 在 B2 top-3 | LOCAL top-1 = oracle | B2 的 U_traj 排名 | LOCAL 的 U_traj 排名 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 230070 | 2.190856e6 | 1.427 | 1.25τ–1.5τ | -0.181818 | -2e6 | 5e6 | 3e6 | 0.454545 | 否 | 是 | 否 | 6 | 2 |
| 230077 | 1.941896e6 | 1.265 | 1.25τ–1.5τ | -0.200000 | -2e6 | 4e6 | 2e6 | 0.400000 | 否 | 否 | 否 | 11 | 4 |
| 230078 | 3.810212e6 | 2.482 | ≥ 2τ | -0.181818 | -2e6 | 4e6 | 2e6 | 0.363636 | 否 | 否 | 否 | 9 | 7 |
| 230091 | 1.555024e6 | 1.013 | τ–1.25τ | -0.090909 | -1e6 | 1e6 | 0 | 0.090909 | 否 | 是 | 是 | 2 | 1 |

Gate-1 四场都选择 B2。LOCAL candidate / B2 candidate / oracle 分别是：

- 230070：6 / 20 / 19
- 230077：22 / 20 / 2
- 230078：16 / 5 / 6
- 230091：7 / 5 / 7

四场的 B2 candidate 在 U_single 上的排名是 3、3、2、3。也就是说 B2 放行的是 U_single 的第 2 或第 3 名，不是一个 U_single 很差的远端 candidate。它的 U_single 低于 LOCAL，C_hat 明显高于 LOCAL，于是 U_hat margin 超过 τ。真实 U_traj 上，这四个被选中的 candidate 分别排第 6、11、9、2，LOCAL 分别排第 2、4、7、1。

LOCAL 在四场上的 regret 都小于 B2。只有 230091 的 LOCAL 正好是 oracle。其余三场 LOCAL 也没有选中 oracle，只是比 B2 更接近。

## margin-risk

按已实现的 Gate-1 结果分桶。margin < τ 时选择 LOCAL，所以该桶的 ΔR 和 U gain 为 0。

| 桶 | n | n_B2 | degradation | rate | mean ΔR | median ΔR | worst ΔR | mean U gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| margin < τ | 78 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| τ ≤ margin < 1.25τ | 13 | 13 | 1 | 0.0769 | 0.1702 | 0.2000 | -0.0909 | 2.000e6 |
| 1.25τ ≤ margin < 1.5τ | 4 | 4 | 2 | 0.5000 | 0.1045 | 0.1091 | -0.2000 | 1.000e6 |
| 1.5τ ≤ margin < 2τ | 14 | 14 | 0 | 0 | 0.3818 | 0.4167 | 0.1667 | 4.571e6 |
| margin ≥ 2τ | 19 | 19 | 1 | 0.0526 | 0.3042 | 0.3571 | -0.1818 | 4.000e6 |

失败没有集中在 τ 边界。靠近边界的只有 230091。1.25τ–1.5τ 有两场失败，margin ≥ 2τ 还有 230078。高 margin 桶的 mean ΔR 仍然为正，失败是桶内的尾部，不是整个高 margin 区域都变差。

## B2 ranking

这里的 top-1 / top-3 是 argmax U_hat 相对 argmax U_traj，不是 Gate-1 最终选择。

| 集合 | B2 top-1 recovery | B2 top-3 recovery | LOCAL top-1 recovery |
| --- | --- | --- | --- |
| Batch-1 | 0.421875 | 0.765625 | 0.203125 |
| Batch-2 | 0.296875 | 0.656250 | 0.140625 |
| Combined | 0.359375 | 0.710938 | 0.171875 |

四个失败场景上，B2 top-1 都不等于 oracle。230077 和 230078 的 oracle 也不在 B2 top-3。230070 和 230091 的 oracle 在 B2 top-3，但被选中的仍是另一个 candidate。230091 的 LOCAL top-1 就是 oracle。

因此这不是“Gate-1 单独放错、B2 排序正确”，也不是“只有阈值边界上的一次误放”。冻结规则确实放行了 B2；放行之后，B2 排序在这四场上相对真实 U_traj 明显偏差，而 margin 没有反映出随后的负 ΔR。

## LOCAL fallback 的事后对照

Batch-2 中 margin < τ 的 scenario 有 34 个。它们都被 Gate-1 选为 LOCAL：degradation = 0，mean ΔR = 0，worst ΔR = 0。

Batch-2 中被放行的 30 个 B2 scenario：degradation = 4，mean ΔR = 0.241626，worst ΔR = -0.200。

当前 τ 没有把这 4 个高风险 scenario 留在 LOCAL。它们的 margin 都大于 τ。若在这 30 场上事后改成强制 LOCAL，ΔR 会变成 0，但那也会拿掉这 30 场目前为正的平均 ΔR。这是对照，不是新的 Gate-1。本阶段不据此修改 τ。

## deployment-time 共同模式

四场失败在 deployment-time 可观测量上的共同点是：B2 candidate 的 U_single 低于 LOCAL，C_hat 明显高于 LOCAL，margin 仍大于 τ。和 Batch-2 里其余 26 个被放行且未退化的 B2 scenario 相比，失败样本的 margin 均值是 2.374e6，未失败样本是 2.804e6，失败样本的 margin 并不更大。Phase-3E geometry 差值方向也相同，例如 B2 candidate 的 `mean_uav_target_dist` 都低于 LOCAL。

事后才能分开它们的量是真实 U_traj 排名：失败样本的 B2 candidate 平均排第 7，未失败的 B2 candidate 平均排第 2.27。U_traj 排名不是 deployment-time 输入。这里只把它记为 failure-associated observable pattern，不把它变成新特征，也不据此改模型。

## distribution shift

candidate 数两边都是 24。描述统计：

| 量 | Batch-1 mean / median | Batch-2 mean / median |
| --- | --- | --- |
| margin | 1.249e6 / 1.310e6 | 1.599e6 / 1.512e6 |
| B2 使用率 | 0.3125 | 0.46875 |
| U_single(LOCAL) | 4.453e6 / 4.400e6 | 4.353e6 / 4.400e6 |
| R_LOCAL | 0.743 / 0.727 | 0.746 / 0.739 |
| U_traj(LOCAL) | 9.297e6 / 8.000e6 | 8.828e6 / 8.000e6 |
| candidate U_hat | 5.716e6 / 6.037e6 | 5.797e6 / 6.175e6 |
| candidate geometry | 均值接近；`competition_count` 两边都是 0 | 同左 |

Batch-2 的 margin 分布更高，所以 B2 使用率更高。U_single、R_LOCAL 和 Phase-3E geometry 没有同等幅度的分离。不能把 Candidate-C 失败单独写成环境 distribution shift。

## 十一个问题

1. 四个 degradation 是否都发生在 B2-selected scenario？是。
2. 它们的 margin 是否都 ≥ τ？是。
3. failure 是否集中在 τ 附近？否。只有 230091 靠近 τ。
4. 高 margin 是否仍有 failure？是。230078 的 margin ≥ 2τ。
5. B2 ranking 在 failure scenario 上是否明显错误？是。四场 top-1 都不是 oracle；两场 oracle 不在 top-3；被选 candidate 的真实排名为 6、11、9、2。
6. LOCAL 是否更接近 oracle？是，四场 LOCAL regret 都更小。只有 230091 的 LOCAL 正好是 oracle。
7. Batch-2 相对 Batch-1 是否有 distribution shift？margin 和 B2 使用率更高。U_single、R_LOCAL 和 geometry 接近。
8. Candidate-C 失败的直接统计原因是什么？Batch-2 的 degradation rate 0.0625 > 0.02，且 worst-case ΔR -0.200 < -0.125。
9. 能否据此判断应该修改 B2？**NOT ESTABLISHED**。
10. 能否据此判断应该修改 τ？**NOT ESTABLISHED**。
11. 能否据此判断应该修改 Candidate-C？**NOT ESTABLISHED**。

## 下一步

允许继续做只读审计，或由项目负责人决定是否单独立项。

禁止：重训 B2，修改 feature、hyperparameter、seed、τ、Gate-1、C1–C8，修改 planner / decide()，删除或替换 scenario，重生成 holdout，根据这次失败调参，宣布 deployment PASS，设置 planner_ready = True，进入 Phase-4F，进入 planner integration。

本阶段只完成 failure attribution。
