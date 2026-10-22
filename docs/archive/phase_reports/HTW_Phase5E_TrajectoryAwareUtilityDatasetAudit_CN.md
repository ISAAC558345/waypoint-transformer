# HTW Phase-5E Trajectory-Aware Utility Dataset Audit

**planner_ready = False**

**deployment_verified = False**

本阶段只构造离线 utility 标签并比较两种排序。没有训练 HGNN，没有重训 Residual MAPPO，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C 或 planner，没有使用 230000–230127，也没有进入 Phase-5F。

## 1. 当前 assignment supervision 的问题

Phase-5 HGNN 的标签是当前时刻的排他匹配：`queue * 当前假设速率 / (distance + 1)`。这个分数不含未来轨迹。Phase-5C 在 t=15 上已经看到它和 15 步 payload 的 Spearman 为 -0.39，Top-1 相同比例为 0。

冻结 DeepSets 检查点用的是另一套候选池上的 15 步 payload 软标签。本数据集不读取那个检查点，也不改它的 τ。这里要验证的是：候选级的 assignment similarity 是否该换成 15 步 trajectory utility 排序。

## 2. Offline rollout

场景只用已有重置 310100 和 310101。决策时刻是 t=0、15、30，对应冻结的 option hold。时刻之间用当前排他匹配加 fly-toward 推进，不调用 `decide()`，也不调用 MAPPO。

在每个决策时刻，先记下当前状态，再固定其他无人机的当前排他 assignment。对一名无人机的每个未占用浮标，只替换它的目标，在环境副本上用 fly-toward 走 15 步，读取通信 payload 增量。副本轨迹不写回特征。

输入只含当前量：UAV 位置、由上一步位移得到的速度（t=0 为 0）、当前是否已分配、被分配浮标的队列、当前速率；浮标位置、队列、容量、当前假设速率；UAV–UAV 干扰和 UAV–Buoy 的 distance 与 ETA。没有读取 `remaining_arrivals`。

## 3. Utility 公式

`U_k = α * payload_gain + β * urgency - γ * travel_cost - δ * interference_cost - η * energy_cost`

系数没有拟合。baseline 写在 `phase5e_utility_dataset/utility_config.json`：α=1，β=0，γ=0，δ=0，η=0。因此这条 baseline 的 `utility_score` 等于 `payload_gain`。

β 保持 0，因为 urgency 只有当前队列比例，没有可标定的第二套时钟。γ 保持 0，因为 ETA 定义为 distance / step_length，所以 distance/ETA 等于固定步长，不能区分候选。δ 保持 0，因为 15 步 payload 已经包含 SINR。η 保持 0，因为环境没有能耗状态；本数据集里 energy_cost 全部是 0。

实测 travel_cost 的标准差是 0.0000，组内 interference 是否随候选变化的比例是 0.0000。

## 4. Dataset 字段

文件：`phase5e_utility_dataset/trajectory_utility_dataset.csv`。

标识：scenario_id、time_step、uav_id、candidate_id。

当前特征：queue、channel、distance、ETA、interference、urgency。

标签：payload_gain、utility_score、utility_rank。utility_rank 从 1 开始，1 是最高 utility，相同分数并列。

行数 = 252。决策组数 = 36。payload 被拉开至少一个服务块的组数 = 23。

## 5. Feature 与 label 隔离

特征在调用 15 步 shadow 之前从当前 `env.info()` 取出。payload_gain、utility_score、utility_rank 只出现在标签列。特征列与标签列的交集为空，检查结果是通过。

完整未来位置和动作没有进入 CSV。15 步结果只允许出现在标签里。

## 6. Ranking 指标

两种分数都去对 payload_gain 排序。Top-1 用最高分集合是否相交，避免并列时 argmax 碰到第一个下标。Top-3 recovery 看 payload 最高的候选里是否有 utility 或 assignment 名次不差于 3。

| ranking | subset | n | Spearman | Kendall | Top-1 | Top-3 |
| --- | --- | --- | --- | --- | --- | --- |
| assignment similarity | 全部 | 36 | -0.2391 | -0.1534 | 0.3611 | 0.9444 |
| trajectory utility | 全部 | 36 | 1.0000 | 0.2328 | 1.0000 | 1.0000 |
| assignment similarity | payload 已分开 | 23 | -0.4658 | -0.2402 | 0.0000 | 0.9130 |
| trajectory utility | payload 已分开 | 23 | 1.0000 | 0.3644 | 1.0000 | 1.0000 |

trajectory utility 在 baseline 下等于 payload_gain。Spearman、Top-1 和 Top-3 都是 1。Kendall 低于 1，是因为现有实现把并列候选对留在分母里。payload 已经分开的组里，utility 与 payload 的反向候选对是 0，assignment similarity 的反向候选对是 146。

因此更接近 payload 最优目标的是 trajectory utility ranking。assignment similarity 在这些组上的 Top-1 是 0；payload 最高候选仍经常落在它的前 3 名里，所以它作为短名单还留有重叠，但不能当作最优目标。

## 7. Failure case

文件：`phase5e_utility_dataset/utility_failure_cases.csv`。

assignment 选错 = 23。utility 选错 = 0。payload 最高但 utility 名次低于 1 = 0。

assignment 选错表示当前匹配分数的最高候选和 15 步 payload 的最高候选不是同一个。utility 选错和 payload-utility 倒挂在这条 baseline 上是 0，因为 utility 就是 payload，而且没有再减 travel、interference 或 energy。若以后把 β、γ、δ 改成非零，这三列就是用来检查新系数会不会把 payload 最高的候选排低。当前不改系数。

## 8. 能否进入 HGNN 重新训练

标签验证支持把 high-level 监督从 assignment similarity 换成这份 trajectory utility ranking。特征里没有 15 步结果，低层也不是这次要动的部分。

本阶段不满足开始训练的执行条件。HGNN 没有训练，Phase-5F 没有开始，系统优化没有完成。

`planner_ready = False`

`deployment_verified = False`
