# HTW Integration Phase-2C：Set-Likelihood vs Free-Decoding Search Mismatch Audit

本阶段 **不训练**。不改 \(L_{\mathrm{abs}}\)、网络、checkpoint、低层、collision/interference。只诊断冻结 OLD / COND / ABS 在 development val states 上的 search。

---

## 23 问正式回答

1. **free decoder 实际是什么 search？** 联合 greedy \(\arg\max\) over legal \((m,k)\) logits，平局用 `canonical_keys`。不是 beam，也不是按 UAV 下标的顺序 softmax。
2. **是否最大化 set probability？** toy greedy MAP recovery=0.500；ABS 中 beam-set-best \(>\) greedy set-logP 的比例=0.838。`training_inference_objective_mismatch`=true
3. **greedy set logP 比 beam-set-best 低多少？** ABS median gap=0.3468（mean=0.7704）
4. **beam 变宽后 DecodeNR 是否下降？** ABS median DecodeNR by width: w=1:0.440, w=2:0.440, w=4:0.400, w=8:0.500, w=16:0.438, w=32:0.500, w=64:0.500
5. **多少 state 在 beam<=32 恢复（DecodeNR<=0.10）？** 0.438；各 width 即时恢复率：w=1:0.306, w=2:0.331, w=4:0.350, w=8:0.369, w=16:0.369, w=32:0.369, w=64:0.356
6. **set vs path？** ABS best-pool OMB median=2.5427；greedy OMB median=2.0746
7. **高 utility candidate 的 order entropy 是否更大？** best-pool \(N_{\mathrm{eff}}\) median=62.915；greedy 42.446
8. **first divergence？** 分布 {'1': 0.35625, '2': 0.51875, '3': 0.0875, '4': 0.0375, '5': 0.0, '6': 0.0}；step<=2 比例=0.875
9. **best-pool-compatible prefix mass（step1, 全排列平均）？** median=0.4043
10. **Pool-MAP DecodeNR？** median=0.0000 mean=0.1611
11. **Beam64 path DecodeNR？** median=0.5000；set-best median=0.5000
12. **Greedy DecodeNR？** median=0.4400（OLD greedy median=0.2500）
13. **Pool-MAP 是否恢复 seed4？** seed4 greedy median=0.9250；pool-MAP 0.0000
14. **Beam 是否恢复 seed4？** path 1.0000；set-best 1.0000
15. **idle 是否主错误分支？** idle choice fraction=0.635；close idle calls=20
16. **toy exact set-MAP recovery？** greedy=0.500；beam contains MAP {'beam1': 0.5, 'beam16': 1.0, 'beam2': 0.9, 'beam4': 0.95, 'beam8': 1.0}
17. **training-inference mismatch？** true
18. **primary mechanism？** SEARCH
19. **secondary mechanism？** SETPATH
20. **Search Gate？** B (HLSETPATH-B)
21. **下一步唯一 minimal change？** Next allowed: prefix-consistent utility supervision research. Do not train in this phase.
22. **本阶段是否训练？否。**
23. **是否修改低层？否。**

ABS greedy per seed: s0=0.326, s1=0.250, s2=0.400, s3=0.225, s4=0.925
ABS pool-MAP per seed: s0=0.000, s1=0.000, s2=0.000, s3=0.000, s4=0.000
ABS beam64-path per seed: s0=0.399, s1=0.500, s2=0.125, s3=0.359, s4=1.000

Beam expansion does not recover most states; prefix/set-path inconsistency remains.

Next allowed: prefix-consistent utility supervision research. Do not train in this phase.

本阶段停止。不要自动进入下一阶段。
