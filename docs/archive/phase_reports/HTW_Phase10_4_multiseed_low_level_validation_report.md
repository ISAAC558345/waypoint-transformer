# HTW Phase-10.4 Multi-Seed Independent Validation of Gated Low-Level PPO

Independent ordinary validation, not final untouched test. Methods frozen from Phase-10.1/10.3.
Phase-10.3 design 8-15 (not this set): B0=8.50e6, B1=7.25e6, B2=12.875e6, B3=10.75e6, B2-B3=+2.125e6.

1. 新增低层 seeds：[1, 2, 3, 4]；seed0 复制 Phase-10.1 final，未重训。
2. 每个新 seed 预算：100 updates × 48 × 4 = 19200 transitions，同 Phase-10.1。
3. 独立 scenario 数：64。
4. 与历史 scenario 交集：0（必须为 0）。
5. B3 mean payload：10671875.0。
6. B2 五 seed mean payload：[12640625.0, 13406250.0, 10890625.0, 9046875.0, 4453125.0]。
7. 每 seed B2-B3 delta：[1968750.0, 2734375.0, 218750.0, -1625000.0, -6218750.0]。
8. 正向 seed 比例：3/5。
9. overall mean delta：-584375.0。
10. hierarchical 95% CI：{'high': 1965859.3749999988, 'low': -3550156.25}。
11. seed-cluster 95% CI：{'high': 1925000.0, 'low': -3512734.375}。
12. B2 是否稳定优于 B3：False。
13. B2-B1 跨 seed 正向比例：5/5。
14. 完成块 B2/B3 mean=10.0875/10.671875；换目标丢失 B2/B3=1.640625/0.703125；飞行距离 B2/B3=752.9971760182711/485.71875。
15. 明显失败 seed：[3, 4]。
16. LLEXEC=C。Current learned low-level does not stably beat gated fly-toward.
17. stable_low_level_advantage=False。
18. 是否支持进入最终完整系统 benchmark：本轮不支持自动进入最终 untouched/系统 benchmark。
19. 是否修改过 reward/gate/high-level？否。
20. 下一阶段唯一建议：保留 DeepSets 高层 + fly-toward + InRangeZeroMotionGate 作为当前可靠系统。不要为保留 PPO 无限训练。

训练墙钟 118.6s（seed1–4），评估墙钟 145.4s，总墙钟 267.0s。seed0 未重训。
图：{'per_seed_delta': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase104\\outputs\\validation\\phase104_per_seed_b2_b3_delta.png', 'family_payload': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase104\\outputs\\validation\\phase104_family_payload.png', 'paired_delta': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase104\\outputs\\validation\\phase104_paired_delta_hist.png', 'completed_blocks': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase104\\outputs\\validation\\phase104_completed_blocks.png', 'gate_unfinished': 'E:\\srdp\\NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference\\waypoint_transformer_official\\src\\wt\\htw_phase104\\outputs\\validation\\phase104_gate_and_unfinished.png'}

