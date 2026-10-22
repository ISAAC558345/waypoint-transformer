# HTW Low-Level V2 Phase-1 Pilot Report

单 seed 0，100 updates，LLV2_design 验证。非 confirmation，非 final untouched。未改 FINAL-HIER。未访问 106000–106063。

1. d_safe 冻结：1.0 m（仿真中心距，非真实机身直径）
2. alpha_residual：0.25
3. ServiceSafeEnvelope margin：0.5 m
4. MAPPO：gamma=0.99, gae_lambda=0.95, clip=0.2, lr=0.0003, value_coef=0.5, entropy=0.01, max_grad=0.5, rollout=48, num_envs=4, minibatches=4, update_epochs=4, 来自 Phase-10.1 PPOConfig。log_std_init=-2.2。
5. 是否从旧 PPO 加载权重？**否。**
6. Actor：self, task, communication(SINR/rate/interference/INR), neighbor embedding, d_nearest（hold 与剩余时间在 task 内）。
7. Critic：188 维 global state。
8. Reward：r = delta_payload_bits/1e6 - 0.05*C_project - 0.01*C_residual；lambda_progress=lambda_near=lambda_interference=0。
9. environment transitions：19200
10. agent transitions：115200
11. 训练耗时：116.3s（含 design eval）
12. update0 payload：10375000.0
13. update 20/40/60/80/100 payload：{'u0': 10375000.0, 'u20': 11875000.0, 'u40': 11750000.0, 'u60': 10125000.0, 'u80': 11500000.0, 'u100': 9250000.0}
14. L0 payload：10375000.0
15. update100 vs L0 delta：-1125000.0
16. completed blocks u0/u100/L0：10.375 / 9.25 / 10.375
17. collision count (u100)：0.0
18. minimum separation worst (u100)：0.9999999992100349
19. projector intervention rate u0/u100：0.0026041666666666665 / 0.328125
20. projector failure (u100)：0.0
21. neighbor shuffle disagreement：7.962626136090019e-10；interference shuffle：0.001194311048746811
22. 高层 SHA 不变：True `dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109`
23. 是否访问旧 final untouched？**否。**
24. Gate：LLV2-PILOT-C。No useful learning trend, shield takeover, or optimizer issue. SAFETY_FAIL=False shield_dependence_high=False
25. 是否建议 LL-V2.2 多 seed：否（本阶段不自动执行）

init residual RMS=0.02726655878359452 (limit 0.05)
smoke env transitions=384（不计入正式 pilot）
primary checkpoint=update100，禁止按 design 曲线选点。
