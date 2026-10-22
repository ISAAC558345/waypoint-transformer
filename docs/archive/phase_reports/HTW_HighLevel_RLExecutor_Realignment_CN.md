# HTW Integration Phase-2：RL 执行器对齐的高层重训练

本阶段只改高层 supervision 的 utility label。未改 DeepSets / Order-Free Decoder / Task Option / hold=15 / 低层 RL / SafetyProjector / reward / candidate generator / L=24。未恢复 HGNN。未做 collision / interference 增强。未访问 confirmation / untouched / `106000–106063` / `222000–222063` / `223000–223063`。OLD-HL checkpoint 未覆盖。

恢复：COMPAT-B，`high_level_retraining_needed=true`。LLV2-MS2-A。COLREL-C。COMM-B。五低层 seed 全部 `update100`。U_new = 五冻结 Residual MAPPO 的 H=15 payload 均值。

---

## 24 问正式回答

1. **训练 utility state 数？** 64（validation 32）
2. **每 state candidate 数？** 24
3. **是否使用 5 低层 seed mean？** 是。禁止 best-seed。
4. **utility dataset SHA？** `ff493cf0828e3fa8232b4d5d54cfcee7d34dddc1f42976bc6e9670beaefa3f57`
5. **高层架构是否改变？否。**
6. **训练了几个 high-level seed？** 5（全部从头初始化）
7. **checkpoint selection rule？** `best_validation_mean_candidate_regret`（seed0 epoch=60; seed1 epoch=70; seed2 epoch=80; seed3 epoch=80; seed4 epoch=30）
8. **OLD-HL median normalized regret？** 0.2500
9. **NEW-HL 五 seed 各自 median regret？** seed0=1.0000, seed1=0.6548, seed2=0.9750, seed3=0.5657, seed4=1.0000
10. **NEW 聚合 median regret？** 0.9750
11. **OLD probability-U_new Spearman？** 0.3513
12. **NEW 五 seed Spearman？** seed0=0.5483, seed1=0.5261, seed2=0.5117, seed3=0.5626, seed4=0.4843（mean=0.5266）
13. **OLD NOM5？** 0.2913
14. **NEW NOM5？** seed0=0.4023, seed1=0.4346, seed2=0.3933, seed3=0.4631, seed4=0.3896（mean=0.4166）
15. **legality？** OLD conflict=0.0000；NEW conflict=0.0000
16. **OLD-HL + RL low-level closed-loop payload？** mean=1.091250e+07；per-LL=1.088e+07, 1.119e+07, 1.075e+07, 1.088e+07, 1.088e+07
17. **NEW-HL × RL low-level closed-loop payload？** mean=5.587500e+06；H0=2.312e+06/2.375e+06/2.375e+06/2.438e+06/2.438e+06; H1=8.438e+06/8.500e+06/8.250e+06/8.438e+06/8.438e+06; H2=4.625e+06/5.375e+06/4.188e+06/4.875e+06/4.688e+06; H3=1.056e+07/1.038e+07/1.006e+07/9.750e+06/1.031e+07; H4=2.188e+06/2.000e+06/2.375e+06/2.250e+06/2.062e+06
18. **high-level seed variance？** std=3.223595e+06
19. **low-level seed variance？** std=9.013878e+04
20. **HLALIGN Gate？** C
21. **high_level_rl_alignment_confirmed？** false
22. **是否支持进入新的 End-to-End Independent Validation？** false
23. **本阶段是否修改低层？否。**
24. **是否重新开启 collision/interference 增强？否。**

New utility supervision did not reduce decoded policy regret, or design closed-loop payload dropped systematically.

Stop. Do not retune tau/lambda/network. Diagnose high-level seed instability or utility objective issues.

机制诊断（非调参）：池内 top-1 candidate regret 相对 OLD-HL 下降（mean top1_regret 9.875e5 → 2.75e5–3.88e5），P vs U_new Spearman 与 NOM5 也上升。但 Order-Free 自由解码 assignment 多数不在 L=24 池内，对解码轨迹做 5-seed H=15 shadow 后，NEW 的 decoded policy NR 全面差于 OLD（OLD median 0.25，NEW 0.57–1.00）。design 闭环 payload 从 1.091e7 掉到均值 5.588e6，且高层 seed 方差极大：seed3 接近 OLD，seed0/seed4 崩溃到约 2e6。NEW 池内 idle 约 0.61–0.64，高于 OLD 0.53。这是高层 seed 不稳 + 自由解码与池监督错位，不是低层或 collision/interference 模块问题。

本阶段停止。不要自动进入下一阶段。
