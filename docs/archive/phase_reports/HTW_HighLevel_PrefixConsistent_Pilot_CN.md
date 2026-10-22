# HTW Integration Phase-2D：Prefix-Consistent Utility Supervision Pilot

本阶段 **只把** \(L_{\mathrm{abs}}\) **换成** \(L_{\mathrm{prefix}}\)。未改 q/tau/lambda、ranking、reference、网络、L=24、优化器、checkpoint、低层。正式推理仍是 joint greedy。未用 beam / Pool-MAP。训练 PREFIX seed0/1 从头初始化。

---

## 27 问正式回答

1. **L_prefix？** \(L_{\mathrm{prefix}}=-\sum_j q_j \mathbb{E}_{\pi\sim\mathrm{Unif}(S_M)}[\log P_\theta(\pi(\Gamma_j)\mid s)]\)，等价 subset 公式 \(\sum_{k,S,e} q_j\frac{1}{C(M,k)}\frac{1}{M-k}\log\pi_\theta(e\mid S,s)\) 的负值。\(L=L_{\mathrm{prefix}}+0.25 L_{\mathrm{rank}}+0.10 L_{\mathrm{ref}}\)。
2. **enum vs subset？** true（worst=1.431e-06）
3. **permutation tests？** true
4. **target 合法？** true
5. **PREFIX seed 数？** 2（seed0, seed1）
6. **selected epochs？** seed0 epoch=80; seed1 epoch=70
7. **L_prefix vs L_abs 尺度？** prefix mean=21.415；abs mean=14.819；未手工缩放
8. **ABS step1 mass？** median=0.3460
9. **PREFIX step1？** seed0=0.2609, seed1=0.2788
10. **ABS step2？** median=0.1089
11. **PREFIX step2？** seed0=0.1520, seed1=0.1478
12. **first divergence 是否后移？** ABS early(<=2)=0.875；PREFIX=0.641
13. **PREFIX DecodeNR？** seed0 median=0.2250, seed1 median=0.2500
14. **是否有 seed <=0.10？** false
15. **decoded-in-pool？** seed0=0.094 seed1=0.094
16. **PoolMass？** PREFIX seed0 mean=0.000758394 seed1=0.00050275；ABS mean=0.00196007
17. **within-pool？** seed0 spearman=0.4347 nom5=0.4598；seed1 spearman=0.4132 nom5=0.4147
18. **toy greedy MAP？** before random logits=0.500（机制指标，不选 checkpoint）
19. **set/path gap？** ABS OMB greedy median=2.1955；PREFIX=4.4598
20. **OLD-HL+RL closed-loop？** 1.091250e+07
21. **ABS-HL+RL closed-loop？** 8.440000e+06
22. **PREFIX-HL+RL closed-loop？** 1.128750e+07（seed0=1.136250e+07 seed1=1.121250e+07）
23. **PREFIX Gate？** B
24. **prefix_consistent_supervision_supported？** false
25. **是否支持 5-seed 正式训练？** false
26. **本阶段是否修改低层？否。**
27. **是否重新碰 collision/interference？否。**

Prefix metrics and DecodeNR moved the right way, but closed-loop or seed stability is insufficient.

Stop. Do not add L_abs, retune prefix weight, change pool, decoder, beam, or checkpoint rule.

本阶段停止。不要自动进入下一阶段。
