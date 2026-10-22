# HTW Phase-6I：独立 HGNN Dataset 与训练 smoke test

## 结论

Gate 为 **JOINT_UTILITY_SMOKE_READY**。

独立入口已经把一个训练状态和一个验证状态送进现有 `JointUtilityHGNN.condition`、冻结的 matching decoder、未改动的 `joint_rank_loss`、一次 backward 和一次 `optimizer.step`。输入、输出、loss 和全部 78 个参数梯度都是有限值。被观察的 UAV 编码层权重发生了更新。原始 checkpoint、Phase-5L 源文件、冻结的 322000–322063 分片和 revision 分片都没有改写。没有启动正式训练。322064–322079 没有读取。`planner_ready = False`，`deployment_verified = False`。

## 旧入口为什么读不了 revision

`run_phase5l.py` 没有 Dataset，也没有 DataLoader。它调用 `load_states()`，而该函数只打开 `phase5k_joint_decoder/assignment_cache.pt`。缓存里是 scenario 310100 和 310101 的 6 个状态。`assert_phase5l_seed` 只接受这两个 scenario。

每个旧状态要有 `urgency_batch`、`assignments` 和 `joint`。缓存中 scenario 310100、t=0 的记录是：

| 张量 | shape |
| --- | --- |
| `urgency_batch["uav"]` | `(1, 6, 8)` |
| `urgency_batch["buoy"]` | `(1, 12, 8)` |
| `urgency_batch["edge"]` | `(1, 6, 12, 3)` |
| `urgency_batch["edge_uu"]` | `(1, 6, 6, 2)` |
| `assignments` | `(43, 6)` |
| `joint` | `(43,)` |

`joint` 是旧的 J(A)。`append_queue_edge` 在运行时把浮标队列通道接到 3 维边上，得到 checkpoint 的 4 维 UAV–浮标边。revision 文件里没有 `urgency_batch`，也没有 `joint`。它保存的是原始观测量、`service_progress`、`uav_step_displacement_m`，以及每个状态 13327 行的 `assignment` 和 `j15_bits`。

因此旧 Dataset 不能直接复用。新入口在 `src/wt/benchmark_v2/phase6ic_hgnn_smoke/`，没有修改 Phase-5L 的任何文件。

## 新 Dataset 的字段与 shape

读取目录只有 `revision_hgnn_input/dataset/`。划分是训练 322000–322047、验证 322048–322063。每个 scenario 使用 t=0、15、30。训练 144 个状态、1,919,088 条候选；验证 48 个状态、639,696 条候选。候选数来自各状态自己的 manifest，smoke 读到的两个状态都是 13327，没有截断到 43。

状态文件包含冻结分片原有的观测量，并额外带有 revision 写入的两个字段：

| 字段 | 角色 |
| --- | --- |
| `uav_position_m`、`buoy_position_m`、`association` | 节点位置与当前关联 |
| `uav_buoy_distance_m`、`uav_uav_distance_m`、`uav_buoy_gain` | 几何与增益 |
| `uav_buoy_hypothetical_rate`、`uav_buoy_in_range` | 链路速率与覆盖 |
| `q_bits`、`current_sinr`、`current_rate_bps`、`current_mean_interference`、`time_index` | 队列、当前链路和时间 |
| `service_progress` `(6,)` | UAV 服务进度 |
| `uav_step_displacement_m` `(6, 2)` | 一步位移，单位米 |

候选文件的行与 `assignment` 对齐：

| 字段 | shape | 用途 |
| --- | --- | --- |
| `assignment` | `(13327, 6)` | 候选分配，送入冻结 decoder |
| `j15_bits` | `(13327,)` | 监督标签 |
| `utility_rank`、`tie_group_id`、`candidate_id` | `(13327,)` | 审计，不进入模型 |

`j15_bits` 是冻结分片里的 `J15_MAPPO_CURRENT`。它由 `batch_j15_mappo` 对每个候选累加 hold 期间的 `payload_increment_bits`，单位是 bit。revision 按字节复制候选文件，322000 t=0 的 manifest 记录 `j15_max_abs_error_vs_frozen = 0`。它和旧 J(A) 不是同一个量。

送进 `condition` 的张量在 `append_queue_edge` 之后是：

| 输入 | shape |
| --- | --- |
| UAV | `(1, 6, 8)` |
| 浮标 | `(1, 12, 8)` |
| UAV–浮标边 | `(1, 6, 12, 4)` |
| UAV–UAV 边 | `(1, 6, 6, 2)` |
| global | `(1, 3)` |

拼接前的边是 3 维。拼接结果与 `hgnn_tensors` 的第 4 通道逐元素一致，容差 `1e-6`。DataLoader 的 batch 是一个决策状态：冻结的 `PairScore` 要求 batch 轴等于 1，所以没有把多个状态堆成一个 batch。

## 输入和标签的对应关系

UAV 8 维按 Phase-5L 的 `feature_dict_l3b` 顺序组装：归一化坐标、一步速度、`service_progress`、是否已分配、所分配浮标的队列比例，以及能量通道。L3-B 不提供能量，这个通道在原特征定义里就是常数 0。浮标 8 维是归一化坐标、队列比例、上一拍到达、浮标上的服务进度、占用、`K/12` 和紧急度。边的前三维是归一化距离、截断速率和 ETA；第 4 维是 `append_queue_edge` 从浮标队列通道复制的比例。UAV–UAV 边是归一化距离和由增益、关联、覆盖与队列重建的干扰。

`assignment` 的每一行对应 `j15_bits` 的同一个下标。`utility_rank` 不在 graph batch 里。标签数值本身也不进入节点或边；`joint_rank_loss` 只用 `j15_bits` 的严格大小关系决定哪一对候选产生损失。

## 监督目标和 checkpoint

`condition` 返回的是节点嵌入 `h_u (1, 6, 128)` 和 `h_b (1, 12, 128)`，不是每个候选一个分数，也不是整个状态的一个效用。Phase-5L 原训练把这两个嵌入交给冻结的 `JointAssignmentDecoder.score_assignments`。matching 模式把该候选所选 UAV–浮标配对的 logit 相加，空闲目标（小于 0）贡献 0，得到长度等于候选数的分数向量。这次 smoke 的分数 shape 是 `(13327,)`，与 `j15_bits` 对齐。

原 loss 是

`mean(-logsigmoid(s_i - s_j))`，只对 `j15_i > j15_j` 的配对取平均。

相等的标签不产生配对。这个严格比较和数据集里按 1 bit 合并的 `utility_rank` 分组不是同一条规则；smoke 没有改用 `utility_rank`。13327 阶的全配对矩阵在这次 CPU 步骤里可以放下，工作集约 3417 MiB，所以没有改 loss，也没有抽样配对。

`joint_utility_encoder.pt` 以 `load_state_dict` 载入 `JointUtilityHGNN(buoy_dim=8)`。`matching_decoder.pt` 载入后 `requires_grad = False`，不在 Adam 参数组里。优化器只含 HGNN，学习率 `3e-4`，权重衰减 `1e-4`，梯度裁剪 1.0。没有调用保存，原 checkpoint 文件没有被覆盖。

## Smoke 实测

设备是 CPU。当前进程里 `torch.cuda.is_available()` 为假，所以没有 CUDA 峰值。

| 项目 | 训练 322000 t=0 | 验证 322048 t=0 |
| --- | --- | --- |
| 候选数 | 13327 | 13327 |
| loss | 0.390592 | 0.379606 |
| forward + loss | 3.300 s | 2.473 s |
| backward | 2.285 s | 未做 |
| `optimizer.step` | 0.0157 s | 未做 |

训练状态的一次完整更新是 5.601 s。梯度范数 0.1060，低于裁剪阈值。被观察的 `encoder.uav_enc[0].weight` 最大绝对变化是 `2.9999e-4`，与 Adam 第一步、学习率 `3e-4` 的更新幅度一致。验证只做了无梯度 forward 和 loss。

按每个状态 5.601 s 线性外推，144 个训练状态一轮约 807 s（13.4 min）。若每轮再对 48 个验证状态做一次 forward 和 loss，另加约 119 s。这是单个 CPU 状态的外推，不是实测的完整 epoch。Phase-5L 历史上的 40 个 epoch 若原样套用，训练步约 9.0 h，加上每轮验证约 10.3 h。正式训练的 epoch 数还没有确定，这里不作为训练计划。

## 尚未解决的问题

正式训练没有启动。下一步需要单独指令。

当前计时来自 CPU。若之后在 GPU 上训练，单步时间和内存要重新测量。全配对矩阵是这一步的主要开销；保持原 loss 时，一个状态一步仍然要构造约 `13327²` 的比较。冻结 decoder 继续要求每次只给一个决策状态。`j15_bits` 完全相等才视为平局，1 bit 以内的 `utility_rank` 分组没有进入 loss。
