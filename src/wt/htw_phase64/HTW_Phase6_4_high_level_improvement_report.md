# HTW Phase-6.4 High-level Option Policy Improvement

只优化 π_H(z|s)。冻结 Phase-6.2/6.3 低层，不改环境、不加 HGNN、不用 Task-Waypoint、不改低层权重。
高层预算：80 updates，rollout=16，num_envs=4，seed 0。

## 1. 为什么 6.3 高层失败

逐步 `Δpayload` + 每步重选 z，使 π_H 偏向就近、立刻能进服务圈的 buoy（completion 高、accuracy≈0.02）。远簇延迟到达需要 **hold 住 option** 并看到队列/ETA/到达预测，而不是每步被稀疏通信奖励拉回近簇。

## 2. 实验设置

- A：option hold ∈ {5,10,15}，reward=`delta`，观测=6.3 DeepSets。
- B：hold=5，reward ∈ {episode payload, Δpayload+service, Δpayload+service−delay}。
- C：hold=5，B 的 delay shaping，buoy 特征增加 queue/distance/ETA/progress/persistence arrival（仍为 DeepSets，无图）。

## 3. 结果

| name | hold | reward | enhanced | payload | return | completion | service | option accuracy |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| State-only (6.3) | — | — | no | 2875000 | 2.88 | 0.108 | 0.012 | 0.462 |
| Oracle + frozen LL | 1 | oracle | no | 10250000 | 10.25 | 0.586 | 0.043 | 1.000 |
| Learned 6.3 | 1 | delta | no | 2625000 | 2.62 | 0.803 | 0.055 | 0.020 |
| A_hold5 | 5 | delta | no | 6000000 | 6.00 | 0.914 | 0.083 | 0.060 |
| A_hold10 | 10 | delta | no | 8500000 | 8.50 | 0.777 | 0.111 | 0.087 |
| A_hold15 | 15 | delta | no | 11000000 | 11.00 | 0.692 | 0.087 | 0.105 |
| B_episode | 5 | episode | no | 3250000 | 3.25 | 0.906 | 0.064 | 0.020 |
| B_delta_service | 5 | delta_service | no | 5000000 | 5.00 | 0.796 | 0.065 | 0.078 |
| B_delta_service_delay | 5 | delta_service_delay | no | 5250000 | 5.25 | 0.792 | 0.082 | 0.049 |
| C_enhanced_delay | 5 | delta_service_delay | yes | 6000000 | 6.00 | 0.854 | 0.094 | 0.059 |

**最佳 6.4：** `A_hold15` payload 11000000，accuracy 0.105。
相对 State-only 差 8125000；相对 6.3 learned accuracy +0.085。

注意：hold=15 的逐步 Oracle 一致率仍只有 0.105，低于 State-only 的 0.46（贪心每步对齐 Oracle）。这不矛盾——冻结低层在长 hold 下更能把 option 执行完，payload 甚至略高于逐步重选的 Oracle+LL。accuracy 标签是逐步 Oracle，与 15-step option 不对齐。

## 4. 三个问题

1. **原高层失败原因：** 每步重选 + 稀疏 Δpayload 奖励近视近簇可完成目标，学不到 Oracle 的跨簇长期分配。
2. **哪个改进有效：** **option horizon 最有效。** hold=5/10/15 的 payload 为 6.00e6 / 8.50e6 / **1.10e7**。hold=5 上的 reward shaping 与增强观测（B/C）都没有超过单纯 hold=5 的 `delta`（6.00e6）；episode 级稀疏奖励最差（3.25e6）。
3. **是否准备加入 HGNN？可以列入下一阶段，本阶段仍未加入。** Learned（hold=15）已高于 State-only，accuracy 相对 6.3 从 0.02 升到 0.11。但相对逐步 Oracle 的 1.0 仍很低，关系型匹配（图）仍是候选，不是现在实现。

JSON：`src/wt/htw_phase64/outputs/high_level_improvement/summary.json`
