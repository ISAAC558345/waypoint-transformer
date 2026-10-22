# HTW Integration Phase-3：Executor-Conditioned Utility Supervision

本阶段 **只改标签**：用冻结 Low-Level executor 的 \(U_{\mathrm{LL}}\) 线性归一化得到 \(q\)，损失为 \(L_{\mathrm{exec}}=-\sum_i q_i\log P_\theta(\Gamma_i\mid s)\)。未改低层 MAPPO、SafetyProjector、collision/interference、reward、actor/critic、decoder、DeepSets、ranking/reference、hold、Horizon、candidate pool。正式推理仍是 joint greedy。PREFIX 作为 baseline，未覆盖旧实验。

---

## 审计

1. **executor utility label？** mean=1.64523e+06；std=1.52169e+06；min=0；max=7e+06；n=2304
2. **HL 概率 vs \(U_{\mathrm{LL}}\) Spearman？** PREFIX mean=0.4239；EXEC mean=0.3976（seed0=0.3798, seed1=0.4069, seed2=0.3684, seed3=0.4139, seed4=0.4193）
3. **DecodeNR？** PREFIX aggregate=0.2375；EXEC aggregate=0.5000（seed0 median=0.5600, seed1 median=0.4800, seed2 median=0.7500, seed3 median=0.3591, seed4 median=0.5000）
4. **normalized regret？** 即上列 DecodeNR（\((U_{\mathrm{best}}-U_{\mathrm{decode}})/(|U_{\mathrm{best}}|+\epsilon)\)）
5. **closed-loop payload？** OLD=1.091250e+07；ABS=8.440000e+06；PREFIX=1.128750e+07；EXEC=8.692500e+06（seed0=5.550000e+06, seed1=1.056250e+07, seed2=7.275000e+06, seed3=1.057500e+07, seed4=9.500000e+06）
6. **seed 稳定性？** PREFIX std=1.060660e+05；EXEC std=2.212316e+06

## 其余记录

- **selected epochs？** seed0 epoch=reused; seed1 epoch=reused; seed2 epoch=reused; seed3 epoch=reused; seed4 epoch=70
- **q 构造？** U/sum(U); if any U<=0 then U-min(U)+eps then normalize；artificial_q=false；prefix_label=false
- **数据集 SHA？** ff493cf0828e3fa8232b4d5d54cfcee7d34dddc1f42976bc6e9670beaefa3f57
- **Gate loop/NR/var/Spearman？** False/False/False/False
- **EXEC Gate？** C
- **confirmed？** false
- **失败原因？** closed-loop 8.692e+06 < OLD 1.091e+07; DecodeNR 0.5000 > PREFIX 0.2375; seed std 2.212e+06 not < PREFIX 1.061e+05; Spearman 0.3976 not > PREFIX 0.4239
- **本阶段是否修改低层？否。**
- **是否重新碰 collision/interference？否。**

Executor-conditioned supervision did not pass. closed-loop 8.692e+06 < OLD 1.091e+07; DecodeNR 0.5000 > PREFIX 0.2375; seed std 2.212e+06 not < PREFIX 1.061e+05; Spearman 0.3976 not > PREFIX 0.4239

Stop. Do not retune q, pool, decoder, beam, or checkpoint rule. Do not start the next phase.

本阶段停止。不要自动进入下一阶段。
