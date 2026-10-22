# HTW Phase-5F HGNN Trajectory Utility Retraining

**planner_ready = False**

**deployment_verified = False**

本阶段只重训现有 high-level HGNN 的监督目标。没有重训 Residual MAPPO，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C、低层策略或 planner，没有使用 230000–230127，也没有进入 Phase-5G。

## 1. 模型

结构保持 Phase-5 的 `Phase5HGNNEncoder` 和 `OrderFreeSetDecoder`。没有增加新的网络模块。UAV–Buoy 边和 UAV–UAV 边保留。新模型的浮标输入多一个当前队列比例 urgency，这是原 encoder 已经支持的通道。

对照不重新训练：冻结 DeepSets 检查点，以及 Phase-5 用 assignment set NLL 训练的 `hgnn_orderfree.pt`。

## 2. Loss

assignment similarity 的 set NLL 没有用于新模型。每个无人机在其他无人机分配固定时，decoder 对合法浮标给出分数 s。标签是 Phase-5E 的 `utility_rank`，1 为最好。

对所有 rank(i) < rank(j) 的候选对，loss 是 `-log sigmoid(s_i - s_j)` 的平均。并列候选不产生配对。这是成对排序损失，系数没有再搜索，τ 没有读取或修改。

## 3. 训练数据

标签来自 `phase5e_utility_dataset/trajectory_utility_dataset.csv`。状态由同一套 310100、310101 重置重放得到，并逐项核对 queue、channel、distance、ETA、interference、urgency 和 15 步 payload。没有新的 scenario。

Phase-5E 只有两个场景，不能再切出第三个互不重叠的场景。训练是场景 310100 的 t=0 和 t=15。验证是同一场景的 t=30，只用于选择 epoch。测试是整个场景 310101，不参与训练或 early stopping。

## 4. Feature isolation

送入 encoder 的键只有 uav、buoy、mask、global、edge、edge_uu。`payload_gain`、`utility_score` 和 `utility_rank` 不在输入里。urgency 是当前队列比例。15 步轨迹没有进入输入。加载数据时如果这些标签出现在图里会直接报错。

## 5. 训练曲线

| epoch | train loss | validation Spearman |
| --- | --- | --- |
| 1 | 0.2228 | 0.4524 |
| 2 | 0.1040 | 0.4524 |
| 3 | 0.1040 | 0.2143 |
| 4 | 0.1040 | 0.2143 |
| 5 | 0.1040 | 0.2143 |
| 6 | 0.1040 | 0.1310 |
| 7 | 0.1040 | 0.2143 |

验证 Spearman 用的是 payload 已经分开的组。检查点保留验证 Spearman 最高的那一轮参数。

## 6. 测试场景 ranking

Spearman 和 Kendall 比较的是预测分数与 utility 名次：名次越好，目标分数越高。Top-1 看最高分是否落在最好的 utility 名次上。Top-3 看最好的 utility 候选是否进入预测前 3。payload 和 regret 用该候选在 Phase-5E 里已经核对过的 15 步 fly-toward shadow。

| model | subset | n | Spearman | Kendall | Top-1 | Top-3 | 15-step payload | regret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assignment similarity | all | 18 | -0.2321 | -0.1323 | 0.3889 | 0.8889 | 333333.3333 | 0.4722 |
| assignment similarity | payload_separated | 11 | -0.4545 | -0.2165 | 0.0000 | 0.8182 | 454545.4545 | 0.7727 |
| DeepSets checkpoint | all | 18 | 0.1310 | 0.0529 | 0.7222 | 0.8889 | 722222.2222 | 0.1944 |
| DeepSets checkpoint | payload_separated | 11 | 0.3442 | 0.0866 | 0.5455 | 0.8182 | 1090909.0909 | 0.3182 |
| HGNN assignment similarity | all | 18 | -0.2698 | -0.1323 | 0.3889 | 0.8889 | 333333.3333 | 0.4722 |
| HGNN assignment similarity | payload_separated | 11 | -0.4643 | -0.2165 | 0.0000 | 0.8182 | 454545.4545 | 0.7727 |
| HGNN utility ranking | all | 18 | 0.3929 | 0.1376 | 0.8333 | 0.8889 | 833333.3333 | 0.1389 |
| HGNN utility ranking | payload_separated | 11 | 0.3929 | 0.2251 | 0.7273 | 0.8182 | 1272727.2727 | 0.2273 |

payload 已经分开的 11 组才拉开了候选。新 HGNN 的 Spearman 是 0.3929，Top-1 是 0.7273，平均 15 步 payload 是 1272727.2727 bit，regret 是 0.2273。旧 assignment HGNN 的 Spearman 是 -0.4643，Top-1 是 0.0000，payload 是 454545.4545 bit，regret 是 0.7727。冻结 DeepSets 检查点原来学的是候选池上的 payload 分布，不是 assignment NLL；它在这些组上的 payload 是 1090909.0909 bit。

全部 18 组里的 Top-1 会把 payload 没有差别的组算成命中，因为那些组的 utility 名次全部并列。分开的 11 组才是排序是否有效的结果。

## 7. Payload

15 步 payload 是选中候选后、其他无人机保持当前排他分配、低层固定为 fly-toward 的通信增量。没有调用 MAPPO，也没有调用 `decide()`。regret 是该组内最大 payload 减去选中 payload，再除以最大 payload。

## 8. Failure cases

文件：`phase5f_utility_rank/utility_rank_failures.csv`。测试场景中新模型 Top-1 没有命中 utility 最优候选的组数是 3。

## 9. Phase-5G

测试场景上，新 HGNN 的排序和 15 步 payload 同时好于旧的 assignment HGNN。

这个提高只来自一个隔离测试场景上的 11 个有效组。它说明现有 HGNN 可以在不看未来轨迹的情况下学到一部分 utility 排序，并提高 fly-toward 的 15 步 payload。它不构成 deployment 结论，也不启动 Phase-5G。

本阶段不进入 Phase-5G，也不做 deployment verification。系统优化没有完成。

`planner_ready = False`

`deployment_verified = False`
