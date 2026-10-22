# HTW Integration Phase-2B：Absolute Set-Likelihood Utility Supervision

本阶段 **只改** \(L_{utility}\)：candidate-conditional softmax → absolute set log-probability。未改 q/tau/lambda、ranking、reference、网络、candidate、L=24、优化器、checkpoint rule、低层、SafetyProjector、collision/interference。未覆盖 OLD/COND checkpoint。未访问 confirmation/untouched。

---

## 25 问正式回答

1. **L_abs 真实数学式？** \(L_{abs}=-\sum_j q_j \log P_\theta(\Gamma_j\mid s)\)，\(L=L_{abs}+0.25 L_{rank}+0.10 L_{ref}\)。
2. **L_abs=L_cond-logPoolMass 是否 PASS？** true
3. **全局 set probability normalization 是否 PASS？** true（toy M=3 K=4 sumP=1.000000；production restricted sumP=1.000000）
4. **shift sensitivity 是否符合理论？** true（conditional 不变，absolute 变为 \(L-c\)）
5. **训练了几个 ABS high-level seed？** 5（从头初始化）
6. **selected epochs？** seed0 epoch=80; seed1 epoch=60; seed2 epoch=80; seed3 epoch=80; seed4 epoch=30
7. **support penalty 训练中占比？** 0.4034
8. **OLD/COND/ABS PoolMass？** OLD mean=0.000255094；COND mean=0.000199429；ABS mean=0.0018461
9. **decoded-in-pool rate？** OLD=0.1562；COND seed0=0.000, seed1=0.031, seed2=0.000, seed3=0.000, seed4=0.031；ABS seed0=0.062, seed1=0.000, seed2=0.062, seed3=0.031, seed4=0.062
10. **logP_decode-max_pool？** COND median=1.0904；ABS median=0.2828
11. **ABS 五 seed DecodeNR？** seed0 median=0.3258, seed1 median=0.2500, seed2 median=0.4000, seed3 median=0.2250, seed4 median=0.9250
12. **ABS 聚合 median DecodeNR？** 0.3258
13. **within-pool Spearman/NOM5？** COND spearman=0.4134 nom5=0.4166；ABS spearman=0.3944 nom5=0.4808；kept=True
14. **beam32 pool overlap？** COND=0.0035；ABS=0.0082
15. **pool 外低 utility 高概率解？** COND top8_ood_low=6.681；ABS=5.812
16. **idle/cardinality bias？** idle_ok=True
17. **五 seed 自由解码稳定性？** ABS|0 vs ABS|1 exact=0.000; ABS|0 vs ABS|2 exact=0.094; ABS|0 vs ABS|3 exact=0.094; ABS|0 vs ABS|4 exact=0.094; ABS|1 vs ABS|2 exact=0.094; ABS|1 vs ABS|3 exact=0.000; ABS|1 vs ABS|4 exact=0.031; ABS|2 vs ABS|3 exact=0.062; ABS|2 vs ABS|4 exact=0.031; ABS|3 vs ABS|4 exact=0.062
18. **OLD-HL+RL closed-loop payload？** 1.091250e+07
19. **COND-HL closed-loop payload？** 5.587500e+06
20. **ABS-HL closed-loop payload？** 8.440000e+06
21. **ABSSUP Gate？** B
22. **absolute_support_supervision_confirmed？** false
23. **是否支持进入新的 End-to-End Independent Validation？** false
24. **本阶段是否修改低层？否。**
25. **是否重新开启 collision/interference？否。**

Support improved and decode regret improved, but not to OLD level or seed/closed-loop still insufficient.

Stop. Do not retune absolute-loss weight, candidate pool, decoder, or checkpoint rule. Diagnose remaining mechanism.

本阶段停止。不要自动进入下一阶段。
