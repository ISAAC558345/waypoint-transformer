# HTW Phase-10.6 Final Untouched Evaluation

FINAL UNTOUCHED TEST. One-shot. No training. No retuning.

1. 最终架构：FINAL-HIER（DeepSets-util seed0 + order-free + hold=15 + fly-toward + InRangeZeroMotionGate）。SHA=dbd6196b23c111a8…
2. Final untouched 场景数量：64。
3. 与全部历史 scenario 交集：0。
4. final_untouched_consumed=True。
5. payload M0/M1/M2/M3 = 8984375.0 / 1718750.0 / 6703125.0 / 10218750.0。
6. M3-M0 Δ=1234375.0，相对提升=0.1373913043478261。
7. M3-M2 Δ=3515625.0，相对提升=0.5244755244755245。
8. M3-M1 Δ=8500000.0（M1 仍为 Phase-10.5 冻结的 sequential exclusive，未改定义）。
9. 95% CI M0/M1/M2 = {'low': 531250.0, 'high': 1969140.624999998} / {'low': 7875000.0, 'high': 9109765.624999998} / {'low': 2703125.0, 'high': 4281250.0}。
10. completed blocks M0/M1/M2/M3 = 8.984375 / 1.71875 / 6.703125 / 10.21875。
11. residual queue = 30.015625 / 37.328125 / 33.609375 / 30.96875。
12. drop = 78.0 / 77.953125 / 76.6875 / 75.8125。
13. flight distance = 80.4375 / 129.9375 / 249.90625 / 456.5。
14. legality = 1.0 / 1.0 / 1.0 / 1.0。
15. FINAL Gate=A。Final untouched confirmation of FINAL-HIER vs primary frozen baselines.
16. final_system_confirmed=True。
17. 可写结论：Utility-Aligned high-level assignment is effective on final L3-B; learned assignment beats legal myopic/exclusive on payload; order-free decoder, Task Option hold=15, and InRangeZeroMotionGate remain as the frozen stack.
18. 不能写：HGNN优于DeepSets；PPO低层优于规则低层；Transformer贡献显著；全面优于所有baseline（忽略飞行距离）。
19. 是否重新训练或调过参数？否。
20. 是否允许继续根据 final test 修改方法？否。

| Method | Payload | Relative Gain | Completed Blocks | Residual Queue | Drop | Flight Distance | Legality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Myopic | 8984375 | 0.1374 | 8.984 | 30.016 | 78.000 | 80.4 | 1.000 |
| Exclusive | 1718750 | 4.9455 | 1.719 | 37.328 | 77.953 | 129.9 | 1.000 |
| DeepSets-old | 6703125 | 0.5245 | 6.703 | 33.609 | 76.688 | 249.9 | 1.000 |
| FINAL-HIER | 10218750 | — | 10.219 | 30.969 | 75.812 | 456.5 | 1.000 |

| Comparison | Mean Delta | Median Delta | Positive Fraction | 95% CI |
| --- | --- | --- | --- | --- |
| M3-M0 | 1234375 | 1000000 | 0.547 | [531250, 1969141] |
| M3-M1 | 8500000 | 9000000 | 1.000 | [7875000, 9109766] |
| M3-M2 | 3515625 | 4000000 | 0.812 | [2703125, 4281250] |

FINAL-HIER improves communication payload and service completion, at the cost of increased UAV travel distance. FINAL-HIER提高了累计数据采集量与任务完成量，但需要更大的无人机运动距离。Flight distance is not an energy model.

墙钟 75.2s。图：{'payload_bits': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase106\\outputs\\untouched\\phase106_final_payload.png', 'paired': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase106\\outputs\\untouched\\phase106_final_paired_delta.png', 'completed_blocks': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase106\\outputs\\untouched\\phase106_final_completed_blocks.png', 'queue_drop': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase106\\outputs\\untouched\\phase106_final_queue_drop.png', 'flight_distance_m': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase106\\outputs\\untouched\\phase106_final_flight.png', 'n_uav_completed': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase106\\outputs\\untouched\\phase106_final_service_completion.png'}

