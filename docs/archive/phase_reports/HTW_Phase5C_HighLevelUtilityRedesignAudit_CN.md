# HTW Phase-5C High-Level Utility Redesign Audit

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练，没有重训 MAPPO，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C 或 planner，没有使用 230000–230127，也没有进入联合训练。

## 1. 当前 high-level 目标

在线推理读取冻结的 DeepSets order-free 模型，输出联合贪心分配。这一步没有 loss，也没有 payload reward。

该检查点的训练目标在 `utility_losses`：

- 输入：UAV: normalized position, velocity, progress, assigned flag, assigned queue, energy slot fixed at 0; Buoy: normalized position, queue/capacity, last arrival, progress, occupied, buoy-count ratio; UAV-Buoy edge: distance, hypothetical rate, ETA; UAV-UAV edge: distance, active-link interference ratio
- 标签：softmax over a frozen candidate pool of H=15 fly-toward payload increments
- loss：`L_util + 0.25 L_rank + 0.1 L_ref`。τ = 1.0，本阶段只读取、不修改。
- reward：none; the high-level has no reward and no payload term inside decode

它不直接最大化整段 payload。`L_util` 只让候选集合上的对数概率去贴近已经算好的 15 步 fly-toward payload 分布。候选集合之外的目标不会被这个 loss 看见。解码时也不再计算 payload。

Phase-5 的 HGNN 用的是另一套目标：标签是 `queue * 当前假设速率 / (distance + 1)` 的排他匹配，loss 是这个单一分配的 set NLL。这是 assignment similarity，不是 payload utility。

## 2. Trajectory-aware utility 接口

只定义接口，不写入模型。

`U(assignment) = α * expected_payload_gain + β * urgency - γ * travel_cost - δ * interference_cost - η * energy_cost`

系数本阶段不拟合。

| term | 现在能否计算 | 说明 |
| --- | --- | --- |
| expected_payload_gain | yes, by offline shadow, not as a network input | H=15 fly-toward payload increment of the candidate assignment. SINR is already inside this number. |
| urgency | queue ratio only | q_k / (max q + 1e-8) only. Waiting time, AoI, and deadline are absent. |
| travel_cost | yes | ETA already stored on the UAV-Buoy edge: distance / step length. |
| interference_cost | edge exists; value is 0 when no link is active | Active-link interference is already folded into the H=15 payload. |
| energy_cost | no | L3-B does not expose energy. The UAV energy channel is a structural zero. |

若 `expected_payload_gain` 就是 15 步通信 payload，干扰已经进了这个数，`δ` 应为 0，否则干扰会被减两次。`η` 现在必须为 0，因为没有能耗变量。`urgency` 仍然只有队列比例。

## 3. HGNN 输入

没有改 encoder。

| feature | current | needed |
| --- | --- | --- |
| UAV-to-target distance | yes, UAV-Buoy edge channel 0 | keep |
| estimated travel time | yes, ETA on UAV-Buoy edge channel 2 | keep |
| predicted communication gain | instantaneous hypothetical rate only | use as the training label, not a new input feature |
| interference graph | yes, UAV-UAV edge, active links only | keep |
| energy consumption | slot filled with 0 | do not add |
| queue | yes | keep |
| channel | yes, rate_if_assigned | keep |

距离、旅行时间和当前信道已经在 UAV–Buoy 边上。不要把 15 步 payload 再拼进输入，否则标签泄漏。干扰边保留。能耗不加。

## 4. 新的训练标签

旧标签是一条排他匹配。探针先用这条匹配和 fly-toward 走到 t=15，不调用 `decide()`。然后固定其他无人机，替换一名无人机的未占用浮标，再用冻结 fly-toward 走 15 步，记录 payload 增量。没有调用 MAPPO。

匹配分数和 15 步 payload 的 Spearman = -0.3929，Kendall = -0.1508，Top-1 相同比例 = 0.0000，平均 payload 极差 = 1083333.3333 bit，候选被拉开至少一个服务块的比例 = 1.0000，组数 = 12。

因此新标签应是每个状态下候选分配的 15 步 payload 排序，而不是匹配是否一致。正式训练时仍然不能把 shadow 的未来轨迹写进网络输入。

## 5. 路线

推荐 Option-2：HGNN 保持现有边，把监督从 assignment match 换成 trajectory-aware utility ranking。

Option-1 只换 encoder。Phase-5 已经做过，HGNN 更像距离/队列匹配，payload 下降。

Option-3 是高低层反馈对齐。Phase-5B 里 oracle 目标交给当前 MAPPO 仍能产生 payload，低层不是当前瓶颈。联合训练本阶段禁止开始。

Phase-5C 停在规格。`planner_ready = False`，`deployment_verified = False`。
