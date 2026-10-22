# HTW Phase-5B Hierarchical Objective Alignment Diagnosis

**planner_ready = False**

**deployment_verified = False**

本阶段没有训练，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C 或 planner，没有使用 230000–230127，也没有进入 Phase-5C。

## 1. 当前系统结构

High-level: 冻结的 DeepSets order-free joint greedy。Phase-5 的 HGNN 只作为已经保存的有界检查点参加对照，不重新训练。

Low-level: Phase-5 基线 Residual MAPPO 检查点 `mappo_original.pt`。Oracle 轨迹是给定分配后的 fly-toward，不是新的策略。

对照上界是现有 `oracle_option_assignment` 加 fly-toward。这个 oracle 用整段 shadow payload 选近/远簇分工，诊断只把它当标签，不把它写进网络。

## 2. 三个诊断实验

闭环种子 [310100, 310101]。排序种子 [310100, 310101, 310102, 310103]，排序只在 t=0 读取 logits，不执行轨迹。

| 实验 | payload | trajectory reward | regret | completion | oracle agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oracle high + fly-toward | 4500000.0000 | 4.5000 | 0.0000 | 48.0000 | 1.0000 |
| 实验1 Oracle high + current MAPPO | 6000000.0000 | 6.0000 | -0.3333 | 48.0000 | 1.0000 |
| 实验2 DeepSets + fly-toward | 7000000.0000 | 7.0000 | -0.5556 | 48.0000 | 0.3333 |
| 实验2b Phase-5 HGNN + fly-toward | 0.0000 | 0.0000 | 1.0000 | 48.0000 | 0.2083 |

实验3 不执行轨迹。Oracle 排序把 oracle 选中的目标放在第一，其余浮标按当前队列、剩余到达和距离排序。

| 模型 | Spearman | Kendall | Top-1 | Top-3 | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSets | -0.6987 | -0.6624 | 0.4167 | 0.4167 | 24 |
| Phase-5 HGNN | 0.5586 | 0.3478 | 0.2917 | 0.4583 | 24 |

## 3. mismatch 来源

负的 regret 表示 payload 高于 oracle+fly-toward。这个簇划分 oracle 在这两组种子上不是 payload 上界。

实验1：oracle 分配交给当前 MAPPO，payload 是 6.0e6，高于同一分配规则加 fly-toward 的 4.5e6，regret 是 -0.33。目标锁定以后，当前低层能够完成服务，而且飞得更短。低层不是“给对目标也不会执行”。

实验2：DeepSets 分配交给 fly-toward，payload 是 7.0e6，regret 是 -0.56，和 oracle 分配的重合度只有 0.33。DeepSets 并没有在模仿这个 oracle，但它的分配在 fly-toward 下收益更高。Phase-5 HGNN 用同一个 fly-toward 执行，payload 是 0，路径长度 6336 m，重合度 0.21。高层把目标选成飞不到、也服务不完的浮标时，换成 oracle 轨迹也补不回来。

实验3 不飞轨迹。DeepSets 相对 oracle 排序的 Spearman 是 -0.70，Top-1 和 Top-3 都是 0.42，说明错的时候不在前三名里。HGNN 的 Spearman 是 0.56，但 Top-1 只有 0.29。HGNN 更像队列/距离排序，而这个排序本身不是 payload 更高的那个分配。Phase-5 用排他匹配标签训练 HGNN，会把高层拉向一个不能转化成服务的目标。

## 4. HGNN 特征对照

没有修改模型。

| feature | current | required |
| --- | --- | --- |
| UAV-UAV edge | yes | yes |
| UAV-Buoy edge | yes | yes |
| interference feature | yes, UAV-UAV channel | yes |
| channel quality | yes, rate_if_assigned on UAV-Buoy | yes |
| energy cost | slot exists but is filled with 0 | not available in L3-B |
| queue state | yes, buoy queue/capacity and assigned-queue on the UAV | yes |
| urgency | queue ratio appended only on the Phase-5 full checkpoint; measured assignment did not change when it was zeroed | a shared score that can change the assignment |

UAV–UAV 和 UAV–Buoy 边已经存在。干扰是当前正在服务的链路干扰，没人进入服务半径时这路特征是 0。信道是当前的假设速率。能耗槽位恒为 0，因为 L3-B 没有能耗状态。Queue 已经在浮标特征里。Urgency 只出现在 Phase-5 完整检查点的额外通道上，上一阶段把该通道置零后分配不变。

## 5. Shared urgency 设计

当前 `u_k(t) = q_k(t) / (max_j q_j(t) + 1e-8)` 只是同一时刻的队列比例。它不包含等待了多久、信息年龄或截止时间。

可以写成以后的接口，但现在不能填进代码：

`urgency_k(t) = α q̃_k(t) + β w̃_k(t) + γ ã_k(t) + δ d̃_k(t)`

- `q̃_k`：现有队列比例，是唯一已经可观测的项。
- `w̃_k`：队列保持为正的步数 / 48。环境没有这个计数器。
- `ã_k`：AoI / 48。环境没有 AoI。
- `d̃_k`：距离截止时间的剩余比例。环境没有 deadline。

在这三个时钟被环境真正记下来之前，β、γ、δ 只能是 0。把它们设成非零会凭空制造特征。权重先不拟合。共享方式仍然是决策时算一次，写入 `TaskOption`，15 步内高低层读同一份，不进入 reward。

## 6. 下一阶段路线

推荐 Option-1，对象是高层分配，不是当前 Residual MAPPO。Oracle 目标交给 MAPPO 后 payload 高于 fly-toward，所以低层不是执行瓶颈。Phase-5 HGNN 交给 fly-toward 后 payload 为 0，DeepSets 相对这个簇 oracle 的 Top-1 也低于 0.5。下一步要改的是高层目标如何排序，而且不能把这个簇 oracle 当成 payload 上界。本阶段不实现。

Option-1 要补的是能改变排序的 UAV 交互、有效干扰、以及一个真正被解码器使用的 urgency。能耗要等环境提供变量。

Option-2 要补的是轨迹对干扰和通信的执行，不是再换一个 encoder。能耗约束同样要等环境变量。

Option-3 只作为方案：joint reward、reachability feedback、HAC/HIRO。禁止在本阶段实现。

Phase-5C 没有开始。`planner_ready = False`，`deployment_verified = False`。
