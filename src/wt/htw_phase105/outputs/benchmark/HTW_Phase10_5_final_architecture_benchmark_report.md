# HTW Phase-10.5 Final Architecture Freeze and Unified Hierarchical Benchmark

Pre-Final Independent Benchmark. Not Final Untouched Test. No training.

FINAL-HIER = DeepSets-util + Order-Free decoder + hold=15 + fly-toward + InRangeZeroMotionGate.
UOBJ-A / GRAPH-U-C / LLEXEC-C recorded. M2 legal=True.

1. 最终冻结架构：FINAL-HIER（高层 seed0 SHA=dbd6196b23c1…）。
2. 不再使用 HGNN：GRAPH-U-C，没有稳定优于 DeepSets。
3. 不再使用低层 PPO：Phase-10.4 LLEXEC-C，stable_low_level_advantage=false。
4. 最终 Task Option：每机目标 buoy/idle，hold=15，低层不改 association。
5. Primary baselines：M0 合法 myopic、M1 exclusive、M2 DeepSets-old（若磁盘兼容）、M3 FINAL-HIER。
6. Flat PPO 不可作为严格主 baseline：决策权限不同（只控 movement / 外部 association / 旧环境）。
7. 64 场景与历史交集：0。
8. payload M0/M1/M2/M3 = 8406250.0 / 1890625.0 / 6890625.0 / 10609375.0。
9. FINAL-HIER delta vs M0/M1/M2 = {'M3_minus_M0': 2203125.0, 'M3_minus_M1': 8718750.0, 'M3_minus_M2': 3718750.0}。
10. 95% CI = {'M3_minus_M0': {'low': 1468359.375, 'high': 2890625.0}, 'M3_minus_M1': {'low': 8155859.375, 'high': 9250000.0}, 'M3_minus_M2': {'low': 2921484.375, 'high': 4484375.0}}。
11. completed blocks M3=10.609375 vs M0=8.40625 / M1=1.890625。
12. residual/drop M3=30.6875/75.703125。
13. flight M3=442.75。
14. SYS=A。FINAL-HIER shows a confirmed system advantage on this pre-final set (not untouched).
15. system_advantage_confirmed=True。
16. 当前 L3-B 支持：DeepSets encoder、utility-aligned 高层、order-free decoder、gated fly-toward 执行。
17. 仅历史机制：Phase-6.4 hold 消融（L2-A）；Flat PPO 对照。
18. 是否支持进入 final untouched：仅建议后续 Phase-10.6，本轮不自动执行。
19. 本阶段有没有重新训练任何模型？否。
20. 下一阶段唯一建议：建议 Phase-10.6 Final Untouched Evaluation 一次。本轮不自动执行。

| Method | Payload | Gain vs FINAL-HIER | Completed Blocks | Residual Queue | Drop | Flight Distance | Legality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Myopic | 8406250 | 0.2621 | 8.406 | 30.594 | 78.000 | 81.1 | 1.000 |
| Exclusive | 1890625 | 4.6116 | 1.891 | 37.125 | 77.984 | 145.4 | 1.000 |
| DeepSets-Old | 6890625 | 0.5397 | 6.891 | 33.422 | 76.688 | 300.4 | 1.000 |
| FINAL-HIER | 10609375 | — | 10.609 | 30.688 | 75.703 | 442.8 | 1.000 |

墙钟 68.1s。图：{'payload': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase105\\outputs\\benchmark\\phase105_main_payload.png', 'paired': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase105\\outputs\\benchmark\\phase105_paired_delta.png', 'blocks': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase105\\outputs\\benchmark\\phase105_completed_blocks.png', 'queue': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase105\\outputs\\benchmark\\phase105_residual_queue.png', 'flight': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase105\\outputs\\benchmark\\phase105_flight_distance.png'}

