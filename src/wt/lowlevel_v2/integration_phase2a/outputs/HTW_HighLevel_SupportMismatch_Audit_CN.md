# HTW Integration Phase-2A：Candidate-Support 与自由解码错配审计

本阶段 **training=false**。未改 tau/lambda/utility/DeepSets/decoder/candidate generator/L=24/低层/高层 checkpoint。未开 HGNN。未碰 collision/interference。未访问 confirmation / untouched / 独立验证。OLD-HL 与 NEW-HL seed0–4 只读，checkpoint 为 Phase-2 正式 selection。

Phase-2 冻结：HLALIGN-C。COMPAT-B。

---

## 23 问正式回答

1. **当前 utility loss 真实数学式？** \(L=L_{utility}+0.25 L_{rank}+0.10 L_{ref}\)，其中 \(L_{utility}=-\sum_j q_j \log p_j\)，\(p=\mathrm{softmax}_{L=24}(\log P_\theta(\Gamma_j|s))\)，\(q=\mathrm{softmax}(\mathrm{median\text{-}MAD}(U)/\tau)\)，\(\tau=1\)。详见 `HL_RL_SUPPORT_LOSS_EQUATIONS.md`。
2. **是否 candidate-conditional？** 是。\(L_{utility}\) 在 L=24 上 softmax。
3. **candidate 统一 logP 平移后 utility loss 是否不变？** true（c=-5 与 +5）。`candidate_uniform_logprob_shift_invariant=True`。
4. **OLD PoolMass？** mean=0.000255094，median=0.000150964，p10=3.36723e-05，p90=0.000632556
5. **NEW 五 seed PoolMass？** seed0 mean=0.0002143 median=0.0001113, seed1 mean=0.000157 median=0.0001, seed2 mean=0.0001243 median=6.787e-05, seed3 mean=0.0002727 median=0.0001384, seed4 mean=0.0002289 median=0.0001445
6. **OLD decoded-in-pool rate？** 0.1562
7. **NEW 五 seed decoded-in-pool rate？** seed0=0.000, seed1=0.031, seed2=0.000, seed3=0.000, seed4=0.031
8. **NEW pool 外 decoded assignment utility？** median=0；池内解码 median=5e+06
9. **Decode normalized regret？** OLD median=0.2500；NEW seed0 median=1.0000, seed1 median=0.6548, seed2 median=0.9750, seed3 median=0.5657, seed4 median=1.0000
10. **free decode 相对 pool candidate 的 logP？** OLD logP_decode-max_pool median=0.2565；NEW median=1.0904
11. **beam32 有多少 assignment 在 pool 内？** OLD 比例=0.0205；NEW=0.0035（相对 width=32）
12. **beam top8 中 pool 外高 utility 多不多？** NEW 每 state 均值 top8_ood_high=1.244，ood_low=6.681
13. **candidate pool 是否缺好解？** BeamBestGain>0 的 NEW 比例=0.044；pool_still_has_good_solutions=True
14. **idle/cardinality bias？** OLD-HL old0 idle=0.625; NEW-HL 0 idle=0.635; NEW-HL 1 idle=0.656; NEW-HL 2 idle=0.630; NEW-HL 3 idle=0.630; NEW-HL 4 idle=0.667
15. **为什么 seed3 较好而 seed0/4 崩？** 闭环：seed3≈1.021e7，seed0≈2.39e6，seed4≈2.18e6。五 seed 自由解码 exact agreement 几乎为 0（典型 Hamming≈2.5–3.2）。差异在 **pool 外解码**，不在池内排序。seed3 的 PoolMass 在 NEW 中最高（mean 2.73e-4），DecodeNR 最低（median 0.566）。
16. **OLD-HL 为什么更稳定？** 不是因为 OLD 有明显更高的 PoolMass（OLD mean 2.55e-4，与 NEW 同为 ~1e-4 量级）。OLD 更稳是因为：decoded-in-pool 15.6% vs NEW 0–3%；logP_decode−max_pool median 0.26 vs NEW 1.09（NEW 更常选比任何 pool candidate 绝对概率更高、utility 更差的 pool 外 assignment）。
17. **primary mechanism？** SUPPORT
18. **secondary mechanism？** DECODER
19. **Mechanism Gate？** HLSUP-A / A
20. **下一步唯一 minimal change？** 将 \(L_{utility}\) 从 candidate-conditional softmax CE 改为绝对 set-likelihood：\(L_{abs}=-\sum_j q_j \log P_\theta(\Gamma_j\mid s)\)。暂不改 ranking / reference。本阶段不训练。
21. **本阶段是否训练？否。**
22. **是否修改低层？否。**
23. **是否重新碰 collision/interference？否。**

标签：SUPPORT, DECODER, SEED, OBJECTIVE。PoolMass 对 OLD/NEW 都极小；NEW 的失败是条件监督不压 pool 外质量，且逐步解码选中比 pool 更高 logP、更低 U 的 assignment（OOD U median=0；top8 中 ood_low≈6.68 vs ood_high≈1.24）。BeamBestGain>0 仅 4.4%，L=24 仍含较好解，故不是 coverage 主因。本阶段停止。不要自动训练。
