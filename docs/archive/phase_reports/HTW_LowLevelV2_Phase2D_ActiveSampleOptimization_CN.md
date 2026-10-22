# HTW Low-Level V2 Phase-2D：Active-Sample-Aware Actor Optimization Pilot（LL-V2.2D）

本阶段是 development pilot。**唯一方法改动**是 Actor 的 active-sample-aware 优化。未改 reward、lr、clip、entropy 系数、update epochs、alpha、beta、epsilon_base、d_safe、ServiceResidualLock、BaseZeroResidualFreeze、Actor/Critic/Neighbor、SafetyProjector、高层、hold、训练预算。beta=0.25 的 cap **保留但不得写成贡献**。未访问 222000–222063。只训练 seed0 与 seed1，均从头初始化，未加载 Phase-2B 权重。Primary=update100。

正式结论保持：**LLV2-MS-B**，`stable_residual_advantage=false`，**REL-1**。Phase-2C：MASK-B，BASEZERO-A。测试 20/20 PASS 后才训练。

保留机制应分开写：ServiceResidualLock、BaseZeroResidualFreeze、Active-Sample-Aware Actor Optimization。不要称 Residual Trust Region 是成功机制。

---

## 实现

Actor 专用 advantage：只在 `policy_active_mask==1` 的 UAV-step 上计算 `mu_active`、`std_active`，再 `A_actor=(A-mu)/(std+eps)`。locked 样本不进 surrogate、entropy、也不进该归一化统计。

Actor minibatch：每个 PPO epoch 取全部 active indices，shuffle，分成最多 4 个近似等大的非空 batch（例如 80→20×4，11→3/3/3/2）。每个 active sample 每 epoch 恰好一次，不 oversample。没有 zero-active Actor batch。

Critic：仍 4 minibatches × 4 epochs，使用全部 env-step 的原始 GAE/return 与 value loss。log-prob 仍针对 raw residual（pre_tanh），不用 r_used。

---

## 1. 本轮唯一修改是什么？

**Active-Sample-Aware Actor Optimization**：active-only advantage 归一化 + active-only minibatching。

## 2. Actor advantage 现在如何归一化？

只对 `policy_active_mask==1` 的 UAV-step 统计 mean/std，再归一化。训练日志：归一化后 mean≈0、std≈1（两 seed 均成立）。

## 3. Actor minibatch 如何构建？

只从 active indices 构造，每 epoch shuffle 后最多 4 个非空近似等大 batch；不重复采样。

## 4. Critic 是否变化？

**否。** 公式、4×4 batch、全样本 value target 不变。仅把 Actor/Critic 的 optimizer.step 拆开以便分开记梯度。

## 5. Reward / network / lr 是否变化？

**否。** `r = Δpayload/1e6 − 0.05 C_project − 0.01 C_residual`。entropy_coef=0.01，clip=0.2，lr=3e-4。

## 6. seed0/seed1 训练预算？

各 **100 updates**，rollout=48，num_envs=4。每 seed **19200 env / 115200 agent** transitions。未给失败 seed 加训。

## 7. active fraction 分别多少？

训练均值：**seed0 7.25%**，**seed1 7.33%**（min 约 3.2% / 2.6%）。与 Phase-2C 的 ~7% 一致。

## 8. 每 update actor minibatch 实际大小？

因 ~80 个 active UAV-step，4 个 batch 均值约 **20.9 / 21.1**。范围：seed0 min 9 max 39；seed1 min 7 max 40。每 update actor optimizer steps=16（4 epoch × 4 batch）。

## 9. 是否还有 zero-active 或极小 active batch？

**无 zero-active。** 无 `<5` 的 actor minibatch。最小为 7（seed1），不再出现旧实现混入 locked 后的 7/288≈2.4% 那种“整批几乎全锁”的 Actor 更新。

## 10. seed0 payload 曲线（primary u100）？

u0=10.375e6，u20=10.188e6，u40=10.812e6，u60=10.875e6，u80=10.875e6，**u100=10.875e6**。

## 11. seed1？

u0=10.375e6，u20=10.625e6，u40=11.000e6，u60=11.438e6，u80=11.125e6，**u100=11.188e6**。

## 12. L0 payload？

**10.375e6**（design）。

## 13. new seed0 − L0？

**+0.500e6**（10.875 − 10.375）。seed0 ≥ L0。

## 14. new seed1 − L0？

**+0.8125e6**（11.188 − 10.375）。seed1 ≥ L0。

## 15. 相对 Phase-2B seed0/seed1？

Phase-2B u100：seed0=10.3125e6（低于 L0 0.0625e6），seed1=12.1875e6。

本阶段：seed0 **从低于 L0 转为高于 L0**（+0.5625e6 vs P2B seed0）。seed1 仍明显高于 L0，但低于 P2B seed1（约 −1.00e6）。曲线在 u40 后振荡小于 seed1 的 P2B 冲高，但仍有 u60→u80 回落。

## 16. Actor KL / clip / gradient 是否更稳定？

相对 Phase-2B seed0 出现过 ~387–487 的联合 grad 尖峰：本阶段 **actor grad 均值 7.81 / 8.57，max 12.7 / 17.0**，无 >50 异常。KL 均值 0.014 / 0.012，clip 0.163 / 0.150。更稳定，但不能把 cap 写成原因。

## 17. arrival time？

u100：new0 **5.219**，new1 **5.708**，L0 6.062，P2B seed1 5.604。两 seed 均快于 L0；NAV 未崩。

## 18. completed blocks？

new0 **10.875**，new1 **11.188**，L0 10.375，P2B seed0/1 = 10.3125 / 12.1875。SERVICE 未崩（lock violation=0）。

## 19. collision？

**0 / 0**。min_sep_worst ≥ 1.0m（float 容差内）。

## 20. projector failure？

**0 / 0**。service_lock_violation=0。

## 21. cap 是否仍未触发？

**是。** design u100 cap_activation=0。beta norm cap 在已观测开发轨迹上仍 inactive/redundant。

## 22. P2D Gate？

**P2D-A。** `active_sample_optimization_supported=true`。两 seed ≥ L0，安全通过，advantage 归一化正确，无空 Actor batch，seed0 相对 P2B 无损失且越过 L0，seed1 相对 L0 仍明显为正。

## 23. 是否支持重新进入正式 5-seed 验证？

门控允许：**是**（需全新 independent validation，禁止再用 222000–222063）。**本阶段不执行。未训练 seed2/3/4。**

---

## 图

`src/wt/lowlevel_v2/phase2d/outputs/figures/`：payload curves、active fraction、actor minibatch size、Actor KL/clip/grad、arrival steps、payload vs L0。

Phase-2D 停止。
