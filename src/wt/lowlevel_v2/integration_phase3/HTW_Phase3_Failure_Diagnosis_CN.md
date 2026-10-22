# HTW Phase-3 Failure Diagnosis

本报告只做离线审计。未训练、未改现有方法、未改 checkpoint、未重跑 Phase-3、不进入下一阶段。

审计输入：冻结的 `hl_exec_align_{train,val}.npz`（SHA `ff493cf0828e3fa8232b4d5d54cfcee7d34dddc1f42976bc6e9670beaefa3f57`）、`deepsets_exec_seed{0-4}.pt`、`deepsets_prefix_seed{0,1}.pt`、Phase-3/2D/2B 的 Decode / Within-pool / closed-loop / training history CSV，以及 `shadow_h15` / `evaluate_design` / `decide` / `executor_targets` / `utility_losses_*` 源码。

已确认：`repeated_low_level_rollout_in_training = false`，`code_changed = false`。失败不归因于重复 rollout 或 cache。

---

## 1. Phase-3 失败的事实

Gate = C；`confirmed = false`。四项门控全部失败。

| 方法 | closed-loop payload | DecodeNR | Spearman | Kendall | NOM5 | payload std |
|---|---:|---:|---:|---:|---:|---:|
| OLD-HL + RL | 1.09125e7 | 0.2500 | 0.1593 | 0.1853 | 0.2913 | （单 checkpoint） |
| ABS-HL + RL | 8.44e6 | 0.3258 | 0.3944 | 0.3386 | 0.4808 | 高于 PREFIX |
| PREFIX-HL + RL | **1.12875e7** | **0.2375** | **0.4239** | **0.3640** | 0.4373 | **1.061e5** |
| EXEC-HL + RL | 8.6925e6 | 0.5000 | 0.3976 | 0.3385 | 0.4372 | 2.212e6 |

EXEC 五 seed closed-loop：5.55e6 / 1.05625e7 / 7.275e6 / 1.0575e7 / 9.50e6。seed0 崩塌到 5.55e6。

结论（已冻结）：executor-conditioned supervision **没有**同时改善 closed-loop payload、DecodeNR、seed stability 和 utility alignment。

---

## 2. Candidate utility 与 High-Level policy alignment

问题：DecodeNR=0.5 是因为 (A) 网络没学到 \(U_{LL}\)，还是 (B) \(U_{LL}\) 对最终 assignment 的排序本身不稳定？

在 val 的 32 个 state、每个 state 的 L=24 pool 上，用冻结 checkpoint 的 \(P_\theta=\log P_{\mathrm{set}}(\Gamma\mid s)\) 对 \(U_{LL}\) 计算 alignment。正式 Spearman/Kendall/NOM5 仍以 Phase-3 报告为准（`pool_metrics`）。本诊断额外计算 top-1 / top-3 / top-25% / NDCG / 归一化 regret。

定义：`regret = (U_best - U_selected) / max(1, |U_best|)`，`U_selected = U[argmax P]`。

| 模型 | top-1 agree | top-3 overlap | top-25% overlap | NDCG | regret mean / median / p90 |
|---|---:|---:|---:|---:|---:|
| EXEC seed0–4 均值 | 0.4125 | 0.489 | 0.534 | 0.899 | 0.145 / **0.0** / ~0.39 |
| PREFIX seed0–1 均值 | 0.4063 | 0.510 | 0.555 | 0.897 | 0.143 / **0.0** / ~0.48 |

分 seed（EXEC）：

| seed | top-1 | top-3 | top-25% | NDCG | regret mean | 正式 Spearman | 正式 NOM5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.469 | 0.510 | 0.536 | 0.911 | 0.117 | 0.380 | 0.460 |
| 1 | 0.375 | 0.479 | 0.542 | 0.879 | 0.190 | 0.407 | 0.432 |
| 2 | 0.406 | 0.458 | 0.505 | 0.906 | 0.118 | 0.368 | 0.377 |
| 3 | 0.438 | 0.500 | 0.536 | 0.906 | 0.147 | 0.414 | 0.462 |
| 4 | 0.375 | 0.500 | 0.552 | 0.891 | 0.154 | 0.419 | 0.455 |

**对 A/B 的判定：**

- **不是单纯的 A。** 网络在 pool 内学到了中等偏上的 \(U_{LL}\) 排序：正式 Spearman≈0.40，NOM5≈0.44，诊断 NDCG≈0.90，regret 中位数为 0。EXEC 与 PREFIX 的 **pool 内 alignment 几乎相同**（top-1 0.41 vs 0.41）。
- **B 在标签层是真的，但不是 EXEC 独有的 DecodeNR 原因。** 同一 assignment hash 在不同 `(scenario,t)` 上的 \(U_{LL}\) 方差很大（第 4 节）。PREFIX 用的是同一套 \(U_{LL}\)，DecodeNR 却是 0.2375。
- **DecodeNR=0.5 的直接原因是 official decode 几乎不在被监督的 24-pool 里选点。**

| 模型 | in_pool | OOD DecodeNR median | \(U_{\mathrm{decode}}\) mean | \(U_{\mathrm{best,pool}}\) mean |
|---|---:|---:|---:|---:|
| EXEC seed0 | 0.031 | 0.600 | 1.50e6 | 3.51e6 |
| EXEC seed1 | **0.000** | 0.480 | 1.93e6 | 3.51e6 |
| EXEC seed2 | 0.094 | **0.750** | **1.07e6** | 3.51e6 |
| EXEC seed3 | 0.031 | 0.400 | 2.13e6 | 3.51e6 |
| EXEC seed4 | 0.031 | 0.600 | 1.86e6 | 3.51e6 |
| PREFIX seed0 | 0.094 | 0.250 | **2.58e6** | 3.51e6 |
| PREFIX seed1 | 0.094 | 0.250 | 2.25e6 | 3.51e6 |
| OLD | **0.156** | 0.250 | 2.23e6 | 3.51e6 |

代码：`eval_model` 用 `greedy_trace` 得到 assignment，再用 `u_new`（pool 命中则读标签，否则 shadow）算 \(U_{\mathrm{decode}}\)；`DecodeNR=(U_{best,pool}-U_{decode})/(|U_{best,pool}|+\epsilon)`。`in_pool = assignment_hash(greedy) in pool_hashes`。

EXEC 五 seed 平均 in_pool = **0.0375**。因此 DecodeNR 比较的是 **pool 外 greedy assignment 的 \(U_{LL}\)** 与 **pool 内最优 \(U_{LL}\)**，不是 `argmax P` 对 `argmax U` 的 pool 内 regret。PREFIX 同样大量 OOD，但其 OOD \(U_{\mathrm{decode}}\) 均值约 2.41e6，EXEC 约 1.70e6。

**本节结论：** EXEC 的 DecodeNR=0.5 主要不是 “没学到 pool 内 \(U_{LL}\)”（否决纯 A），也不是 “\(U_{LL}\) 排序完全不可用”（否决纯 B 作为 EXEC 特有原因）。它是 **监督在 24-pool 的 set-logP 上，推理却是全空间 joint greedy**，且 EXEC 的 OOD greedy 比 PREFIX 差。

---

## 3. 训练监督 \(U_{LL}\) 与最终 closed-loop payload 的一致性

这是本诊断最重要的一条。只报告代码实际在算什么。

### 3.1 \(U_{LL}\) 实际是什么

`dataset.py` 标签：`U_LL_payload_increment`。来源是 Phase-2 `generate_state_utilities`：

1. `replay_to_hold(scenario, t)`：从 `env.reset(scenario)` 用 **`oracle_option_assignment` + `fly_toward`** 步进到时刻 `t`（不是 EXEC / PREFIX / RL 闭环轨迹）。
2. 对每个 candidate（完整 6-UAV assignment）调用 `shadow_h15`：
   - `restore_worker` 回到该 hold snapshot；
   - `bind_assignment` 冻结该 assignment，`pending=HOLD_STEPS=15`；
   - 走最多 15 步 Low-Level（`zero_residual=False`，即真实 RL residual）；
   - `U = env.payload_bits - start_payload`。
3. 对 5 个 Low-Level seed 取均值。

因此 \(U_{LL}\) **不是**：

- 单个 UAV–浮标边的 immediate increment；
- 完整 H=48 / 整局 closed-loop payload；
- 多个 hold 的累计 assignment 序列效用。

它是：**在 oracle 回放到的某一 hold 状态上，把某一个完整 assignment 冻结 15 步后的 payload 增量。**

### 3.2 最终 closed-loop 实际是什么

`evaluate_design`（Phase-2B eval，Phase-3 复用）：

1. `worker.reset(seed)`，从 **t=0** 开始，不是 hold snapshot；
2. 循环到 `episode_horizon`（设计集每局 `n_steps=48`，16 scenario × 48 = CSV 中的 768）；
3. `payload = payload_bits - payload0`（整局累计）；
4. High-Level 在 `pending<=0` 时调用 `decide` → `decode_assignment`（joint greedy），**不是**从 L=24 pool 取 `argmax P`；
5. `CompatWorker.frozen_assignment` 在闭环里保持 `None`，因此每个 hold **重新 decode**，不是 `shadow_h15` 那种冻结 15 步。

闭环 CSV 实测：所有方法 `high_level_decision_count = 4`；assignment 切换次数均值 EXEC 2.525、PREFIX 2.375、ABS 2.53、OLD 2.825。一局有 **4 个会互相改变后续状态的 assignment**。

### 3.3 中间断层（代码证据，不是理论猜测）

```
oracle 回放到达的 hold snapshot
        ↓
candidate-level U_LL = 冻结该 assignment 的 15 步 payload 增量
        ↓
L_exec 只提高这 24 个 set 的加权 log P（绝对 set-logP，不是 pool 上的 log_softmax）
        ↓
official / 闭环 decode = 全空间 joint greedy（in_pool ≈ 3–9%）
        ↓
闭环从 t=0 走 48 步、重规划 4 次
        ↓
最终 payload = 整局 payload_bits 增量
```

已确认的不一致：

| 断层 | 证据 |
|---|---|
| candidate-conditioned 执行 ≠ 实际选中 assignment | 监督的是 pool 内 24 个 assignment；闭环 `decide()` 走 greedy，EXEC in_pool 均值 3.75%。 |
| 单 candidate utility ≠ 组合 assignment 序列 utility | \(U_{LL}\) 只标一个 hold；闭环 4 次决策、约 2.5 次切换。没有序列标签。 |
| 状态转移依赖 | 后一个 hold 的状态由前一个 assignment + RL 执行产生；标签状态由 **oracle + fly_toward** 产生（`replay_to_hold`）。 |
| 后续 15 步 interaction | `shadow_h15` 冻结 assignment 15 步；闭环每个 hold 结束就重绑。 |
| 非加性 | 没有 “(A then B)” 的联合 \(U\)。4 × 均值 \(U_{LL}\)（1.645e6）≈ 6.58e6，低于 PREFIX/OLD 整局 ~1.1e7，也不同于 EXEC 的 8.69e6；数量级对不上简单相加。 |
| candidate 竞争 | 每个 candidate 已是完整 6-UAV assignment。闭环竞争发生在 **时间上相继的 4 个 assignment** 之间，不在 24-pool 的一次选择里。 |

补充：val snapshot 上的 \(U_{\mathrm{decode}}\) 与闭环 payload **不是单调关系**。EXEC seed2 的 \(U_{\mathrm{decode}}\) 均值最低（1.07e6）但闭环 7.275e6；seed0 的 \(U_{\mathrm{decode}}\) 1.50e6 反而闭环 5.55e6。这是 “单 hold 标签效用 ≠ 整局闭环效用” 的直接数据。

---

## 4. Context / assignment dependence

每个 candidate 就是一个完整 6-UAV assignment（hash = assignment digest）。比较同一 hash 在不同 `(scenario, t)` 上的 \(U_{LL}\)。

**Train（64 state × 24 = 1536）：**

- unique hashes = 1103；reused hashes = 196（17.8%）；reused 观测 = 629
- 全局 var = 2.50e12
- between-state（state 均值的 var）份额 ≈ 0.53
- within-state（state 内 24 个 candidate 的 var 的均值）份额 ≈ 0.50
- **同一 hash 跨 state 的 within-var 均值 = 1.40e12，中位数 = 3.67e11**
- 同一 scenario、不同 t 的 within-var 均值 = 1.96e12

**Val（32 × 24 = 768）：** reused 18.4%（105/571）；within-reused var 均值 8.76e11。

\(\sqrt{1.40\times 10^{12}}\approx 1.18\times 10^{6}\)，与 \(U_{LL}\) 总体 std 1.52e6 同量级。

**判定：** \(U_{LL}\) **不能** 写成 \(U(\mathrm{candidate})\)。数据支持

\[
U_{LL}=U(\mathrm{candidate}\mid \mathrm{state},\ \mathrm{oracle\text{-}replayed\ context},\ t)
\]

其中 context 至少包括：时刻 t、队列/进度/payload（snapshot 内）、UAV/浮标几何。同一 assignment 在 t=0 与 t=15 可以差一个数量级。

因此，把 \(q_i=U_i/\sum U\) 当作 “这个 assignment 的固有好坏” 做 candidate-level 监督，存在结构性错位：网络看到的是 **条件效用**，目标却被写成 **与后续闭环状态无关的静态 q**。

这一点 PREFIX/ABS/EXEC **共享**，它解释 “为何 candidate-level 真 \(U_{LL}\) 仍推不动闭环”，不单独解释 PREFIX > EXEC。

---

## 5. Utility 稀疏性与尺度

全量 train+val：n=2304，mean=1.645e6，std=1.522e6，min=0，max=7e6，非正个数=629。

| 统计 | 值 |
|---|---:|
| \(U_{LL}=0\) 比例 | **0.2730** |
| near-zero (\(\lvert U\rvert<1\)) | 0.2730（与零重合，没有介于 0 与 1 的碎值） |
| top 5% utility 占总正效用 | 0.1609（全局并不极端尖） |
| 每 state utility range 均值 | 3.731e6 |
| 每 state CV 均值 | 1.010 |

OLD / ABS / PREFIX / EXEC **共用同一套 \(U_{LL}\)**（Phase-2 SHA 一致）。四者的差异在 **q 的构造** 和 **likelihood**，不在原始 U 的分布。

监督目标分布对比（同一 U，不同 q）：

| q 构造 | 使用方法 | entropy 均值 | \(n_{\mathrm{eff}}=e^{H}\) | \(q_{\max}\) 中位数 |
|---|---|---:|---:|---:|
| 线性 `U/sum(U)`，非正则 shift | **EXEC** | 2.667 | **17.05** | 0.095 |
| 不 shift 的 `U/sum(U+)` | 对照 | 2.469 | — | — |
| `soft_targets`（MAD + τ=1 softmax） | OLD / ABS / PREFIX | **1.840** | **8.69** | **0.344** |

EXEC 的 q 把有效支持摊到约 17/24 个 candidate；PREFIX/ABS 的 q 更尖（有效约 9 个，`q_max` 中位数 0.34）。`q_i>0.05` 的 EXEC 支持均值只有 7.2，但长尾仍占大量质量。

---

## 6. \(q_i\) normalization 是否扭曲监督

代码（`exec_loss.executor_targets`）：

- 默认 \(q_i=U_i/\sum U\)；
- 若 `any(U<=0)`：\(U'=U-\min(U)+\epsilon\)，再归一化。

审计结果：

- **98.96% 的 state 触发了 shift**（几乎每个 state 都有至少一个 \(U=0\)）。
- 全部非正值都是 **精确 0**，没有负效用。因此 shift 就是 \(U'=U+\epsilon\)。
- **shift 前后 ranking 不变：`rank_unchanged_rate=1.0`。** `U-\min(U)+\epsilon` 是严格保序变换，不会把优劣颠倒。
- shift 后 entropy 从 2.469 升到 2.667：原来 \(q=0\) 的零效用 candidate 获得 \(\epsilon/\sum\) 的微小质量，分布更平。
- 与 PREFIX/ABS 的 `soft_targets` 相比：`L1(q_exec,q_soft)` 均值 **0.736**；约 10.4% state 的 `soft_targets` 排序与原始 U 不完全一致（近并列 + MAD/softmax 数值），但这不是 shift 造成的。

**会不会 “把接近的 utility 拉开/压扁”？**

- 拉开/压扁的是 **相对质量**，不是名次。
- 线性归一化 + 给零效用发 \(\epsilon\)，系统性把 q **压扁**（\(n_{\mathrm{eff}}\) 17 vs 8.7）。高 U candidate 的 \(q_{\max}\) 中位数从 soft 的 0.34 降到 0.095。
- 这会让 \(L_{\mathrm{exec}}=-\sum q_i\log P_{\mathrm{set}}(\Gamma_i)\) 接近 “对几乎整个 pool 做加权 NLL”，而不是 “集中推高少数高 \(U_{LL}\) assignment”。

**但是：** ABS 用尖的 `soft_targets` + **同一条** 绝对 set-logP，闭环 8.44e6；EXEC 用扁的线性 q + 同一条 set-logP，闭环 8.69e6。两者接近。因此 **q 压扁不是 PREFIX（1.13e7）对 EXEC 的主因**；它解释监督更散，不解释闭环比 PREFIX 低 2.6e6。

---

## 7. High-Level 训练过程到底在优化什么

可审计的完整 epoch 曲线只有 **EXEC seed4**（`HL_EXEC_TRAINING_HISTORY.csv`）。checkpoint 规则：`best_validation_mean_candidate_regret`，seed4 选中 epoch 70。

| epoch | \(L_{\mathrm{exec}}\) | val Spearman | val NOM5 | val top1 regret |
|---:|---:|---:|---:|---:|
| 1 | 14.454 | — | — | — |
| 10 | ~11.3 量级（loss 已降） | 0.257 | 0.236 | 756250 |
| 70（选中） | ~10.32 | 0.419 | 0.455 | 318750 |
| 80（last） | 10.851 | 0.432 | 0.425 | 325000 |

PREFIX 对照（同一套 \(U_{LL}\) + ranking/ref）：seed0 的 \(L_{\mathrm{prefix}}\) 21.18→17.15，val Spearman 0.235→0.435，选中 epoch 80，regret 325000；seed1 选中 epoch 70。

**训练过程发生了什么：**

1. \(L_{\mathrm{exec}}\) 下降，pool 内 Spearman / NOM5 上升、regret 下降。**训练目标在自己的 val 指标上有效。**
2. seed4 闭环 payload 仍是 9.50e6 < PREFIX 1.129e7，且 DecodeNR median=0.50。
3. 其他 seed 的 pool 指标与 seed4 接近（Spearman 0.37–0.42，NOM5 0.38–0.46），闭环却从 5.55e6 到 1.06e7。**pool 排序改进 ≠ 闭环改进，也不保证 seed 稳定。**

这就是

\[
\text{training objective improvement}\ \ne\ \text{closed-loop execution improvement}
\]

的具体证据：优化的是 “oracle-hold 上 24 个 set 的加权绝对 logP + pairwise rank”；闭环要的是 “从 t=0 起 4 次 greedy 重规划的整局 payload”。

---

## 8. OLD / ABS / PREFIX / EXEC 失败机制对比

四者 **标签 \(U_{LL}\) 相同**。差异只在 q 与 likelihood。

| 方法 | supervision target | likelihood | DecodeNR | Spearman | NOM5 | payload | seed std | 主要失败机制 |
|---|---|---|---:|---:|---:|---:|---:|---|
| OLD | `soft_targets(U,τ=1)` | **pool 上 log_softmax**（条件） | 0.250 | 0.159 | 0.291 | 1.091e7 | n=1 | pool 排序最弱；但 in_pool 最高（15.6%），idle 最低（0.553），闭环仍高 |
| ABS | 同上 soft q | **绝对 set-logP** \(-\sum q\log P_{\mathrm{set}}\) | 0.326 | 0.394 | **0.481** | 8.44e6 | 高（seed4 DecodeNR median 0.925） | 排序变好，greedy 不稳定，闭环掉到 8.44e6 |
| PREFIX | 同上 soft q | **路径/前缀局部 \(\log\pi\)** | **0.238** | **0.424** | 0.437 | **1.129e7** | **1.06e5** | pool 外 greedy 的 \(U_{\mathrm{decode}}\) 最高；闭环最好 |
| EXEC | **线性 \(q=U/\sum U\)**（零值 shift） | **绝对 set-logP**（与 ABS 同构） | 0.500 | 0.398 | 0.437 | 8.69e6 | 2.21e6 | 与 ABS 同族；q 更扁；OOD greedy 更差；seed0 崩塌 |

**为什么 PREFIX 的 1.12875e7 高于 EXEC 的 8.6925e6，即使 EXEC “直接使用真实 Low-Level utility”？**

数据不允许把答案写成 “EXEC 没用真 utility”。EXEC **没有引入新的 executor 标签**，只是把同一套 \(U_{LL}\) 的 q 从 `soft_targets` 改成线性归一化，likelihood 仍是 ABS 那种绝对 set-logP。

对照：

1. **EXEC ≈ ABS（8.69e6 vs 8.44e6）**，二者 likelihood 相同、q 不同 → 线性 executor q **不是**相对 PREFIX 掉 2.6e6 的主因。
2. **PREFIX ≫ ABS ≈ EXEC**，PREFIX 与 ABS 的 q 相同、likelihood 不同 → **路径一致的局部 \(\log\pi\)（对齐 official greedy）才是三条 \(U_{LL}\) 方法里闭环差距的主对照。**
3. EXEC/PREFIX 的 **pool 内** top-1 / NDCG / Spearman 接近，但 PREFIX 的 OOD \(U_{\mathrm{decode}}\) 高约 40%，DecodeNR 低一半。
4. “真实 \(U_{LL}\)” 本身只是单 hold、oracle 状态上的冻结增量，不是闭环 4-hold 轨迹效用。把它线性塞进 set-NLL，并不能让 greedy 在整局里选出高 payload 序列。

OLD 的 Spearman 最差（0.16）但闭环高于 EXEC：再次说明 **pool 内 \(P\)–\(U\) alignment 不是闭环的充分统计量**。

---

## 9. 最可能的结构性原因（必须有数据/代码证据）

不是单一口号，而是三条互相咬合、都有证据的结构：

**结构 1 — 监督对象不是闭环对象。**  
\(U_{LL}=\) oracle-hold 上冻结 15 步的 payload 增量；闭环 \(=\) t=0 起 48 步、4 次 greedy 重规划的整局 `payload_bits`。标签状态分布（`replay_to_hold` + oracle）≠ EXEC 闭环状态分布。seed0/seed2 的 \(U_{\mathrm{decode}}\) 与闭环交叉，说明单 hold 效用排不好整局。

**结构 2 — 监督的似然不是推理的 decode。**  
\(L_{\mathrm{exec}}=-\sum q_i\log P_{\mathrm{set}}(\Gamma_i\mid s)\) 不在 24-pool 上做 `log_softmax`，也不训练 greedy 的逐步 \(\log\pi\)。官方推理是 `decode_assignment` / `greedy_trace`。EXEC in_pool 均值 3.75%，PoolMass 中位数 \(3\times10^{-4}\) 量级。PREFIX 用同一 \(U_{LL}\) 却训练局部路径，闭环与 DecodeNR 都更好。ABS 与 EXEC 同构，闭环同档。

**结构 3 — candidate-level \(q_i\) 假定了 \(U(\Gamma)\)，数据是 \(U(\Gamma\mid s,t)\)。**  
同一 assignment hash 跨 state 的 \(U_{LL}\) 方差与全局方差同量级。简单的 \(q_i=U_i/\sum U\)（再 shift）不能表达 context。q 压扁（\(n_{\mathrm{eff}}\approx17\)）加重了 “对一池条件效用做平均拟合”，但对照 ABS 可知这不是 PREFIX 差距的主项。

这三条合在一起回答标题问题：

> 为什么已经用 candidate-level 真实 Low-Level closed-loop utility \(U_{LL}\) 做 High-Level supervision，EXEC-HL 仍然无法稳定产生高 closed-loop payload？

因为所谓 “真实 closed-loop utility” 在代码里 **并不是** 最终评估的那条闭环轨迹效用；监督也 **没有** 约束最终用来产生轨迹的 greedy decoder；而 \(U_{LL}\) 还是 **强依赖 state/time 的条件量**。Phase-3 只改了 q 的归一化，没有补上这三处错位。

不归因于：重复 Low-Level rollout、cache、改代码、没学到任何 pool 排序。

---

## 10. 下一阶段候选方向（只提机制，不实现、不训练）

原则：先由诊断决定机制，不为了继续实验而强行加 loss。

| 方向 | 诊断是否支持 | 依据 | 是否值得现在就做 |
|---|---|---|---|
| A. state/context-conditioned utility | **支持** | 同 hash 跨 `(scenario,t)` 的 U 方差 ≈ 全局方差 | 作为标签定义问题，是的 |
| B. assignment-**序列** / 整局 utility | **支持** | 4 hold、~2.5 次切换；单 hold \(U_{\mathrm{decode}}\) 与闭环交叉 | **最值得先验证** |
| C. pairwise preference | 弱 | `L_rank` 已存在且权重相同；EXEC 与 PREFIX 的 pool 排序接近 | 否，不是缺口 |
| D. listwise 再改 q | 弱 | Phase-3 已经 listwise 线性 q；ABS 尖 q 闭环仍差 | 不要再为实验改 q |
| E. counterfactual executor utility | 部分已做 | \(U_{LL}\) 已是 hold 上的反事实冻结增量；缺的是 **策略诱导状态 + 多 hold** | 要扩展定义，不是重做 hold-15 |
| F. utility + ranking 联合 | 已在用 | `0.25 L_rank + 0.10 L_ref` 四方法共用 | 不是新方向 |
| G. decode-path / greedy-consistent supervision | **支持（PREFIX 已是对照）** | PREFIX ≫ ABS≈EXEC，q 相同、likelihood 不同 | 不要再发明一条 set-NLL 变体；若做，应先 **测量** greedy 轨迹的整局 U，而不是先写新 loss |

不自动进入下一阶段。不实现上述任何一项。

---

## 已确认的问题

1. Phase-3 Gate C：EXEC 闭环 8.6925e6 < OLD 1.09125e7 与 PREFIX 1.12875e7；DecodeNR 0.50 > PREFIX 0.2375；seed std 2.212e6；Spearman 0.3976 < PREFIX 0.4239。
2. 失败与 `repeated_low_level_rollout_in_training` / cache / `code_changed` 无关。
3. \(U_{LL}\) 是 oracle-hold 上冻结 15 步的 payload 增量，不是整局闭环 payload。
4. 闭环是 48 步、4 次 `decide()`/`decode_assignment` 重规划。
5. EXEC 与 PREFIX/ABS 共用同一 SHA 的 \(U_{LL}\)；Phase-3 只改了 q，likelihood 与 ABS 同构。
6. EXEC 与 PREFIX 的 **pool 内** alignment 接近；DecodeNR 差在 OOD greedy 的 \(U_{\mathrm{decode}}\)。
7. EXEC in_pool 均值 3.75%；官方 decode 不是 `argmax_{pool} P`。
8. 同一 assignment 的 \(U_{LL}\) 随 state/t 大幅变化。
9. `U-\min U+\epsilon` **不改变名次**，但把 q 压扁；对照 ABS，压扁不是 PREFIX 差距的主因。
10. seed4 上 \(L_{\mathrm{exec}}\)↓ 且 val 排序↑，闭环仍低于 PREFIX；seed 间闭环方差由 seed0 崩塌主导。

## 尚未确认的问题

1. 4 个相继 assignment 的效用是否非加性、交互有多大（没有序列标签，不能量化）。
2. seed0 崩塌是 decoder 局部模式还是闭环状态分布上的倒霉轨迹（没有逐步闭环 assignment 日志）。
3. 若强制闭环从 24-pool 取 `argmax P`（而不是 greedy），EXEC payload 会不会接近 PREFIX——未做，也不在本阶段做。
4. q 压扁对 DecodeNR 的因果贡献（只有 ABS 对照，不是析因实验）。
5. 诊断 Spearman（phase1 stats）与正式 `pool_metrics` Spearman 数值不同；Kendall 与正式一致。正式比较以 Phase-3 报告为准。

## 下一阶段最值得验证的一个假设

**不要先写新 loss。**

假设（单一）：

> EXEC 无法稳定产生高闭环 payload，是因为监督的 \(U_{LL}\) 只标了 **oracle 状态上单个 hold 的冻结 assignment**，而正式闭环的 payload 来自 **策略自己走出的 4-hold greedy 轨迹**；在这套错位下，把 q 改成线性真实 utility 不会优于已经用同一 \(U_{LL}\) 的 ABS，也打不过训练了 greedy 局部路径的 PREFIX。

最小验证（仅提案，不执行）：在冻结 EXEC/PREFIX checkpoint 上，记录闭环每一次 `decide()` 的 assignment，并对其 **整条 4-hold 轨迹** 计算 payload；看该轨迹效用与 (i) 各 hold 的 \(U_{LL}\) 之和、(ii) pool 内 `argmax U`、(iii) greedy 的单 hold \(U_{\mathrm{decode}}\) 是否同序。若 (i)(ii)(iii) 都与整局 payload 不同序，则下一阶段的对象应是 **轨迹级 / 状态诱导的 executor utility**，而不是另一条 candidate-level q。

本阶段停止。
