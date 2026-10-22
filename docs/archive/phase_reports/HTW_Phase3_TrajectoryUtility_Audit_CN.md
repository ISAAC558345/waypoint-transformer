# HTW Phase-3 Trajectory Utility Consistency Audit

本报告只做离线评估。未训练、未改模型、未改 checkpoint、未改 pool / decoder / Low-Level / reward / collision、未改 Phase-3。

**定义（与正式代码一致，不重定义 payload）：**

- `U_traj`：`evaluate_design` 同一公式，`payload_bits(end) - payload_bits(reset)`。
- `U_LL / U_single`：官方 `shadow_h15` 的 `payload_increment`（冻结该 assignment，最多 HOLD=15 步；撞到 `episode_horizon=48` 则提前停）。自然轨迹与第一 hold 池上的 \(U_{LL}\) 为 5 个 Low-Level seed 的均值。后续 hold 的 pool 排序用 Low-Level seed=0 的单次 shadow（只用于 percentile，不进入 \(U_{\mathrm{traj}}\)）。
- 设计集：正式 `LLV2_design` 16 个 scenario（221000–221015）。
- High-Level：OLD 冻结 util seed0；ABS / PREFIX / EXEC 各用正式 **seed0**。闭环执行 Low-Level seed=0。
- 第一 hold 候选池：官方 `generate_pool(env, seed_key=scenario*1009+0)`，L=24，不改训练池。

EXEC 本审计的自然轨迹是 **seed0**（Phase-3 中闭环崩到 5.55e6 的那个）。正式五 seed 均值仍是 8.6925e6。对照表同时给出本审计 \(U_{\mathrm{traj}}\) 与正式闭环均值。

---

## 判定（按数据，不预设）

同时成立两件不同的事：

1. **沿已实现轨迹：Case A（加性）。**  
   \(\sum_t U_{\mathrm{single}}(\Gamma_t\mid s_t^{\mathrm{actual}})\) 与 \(U_{\mathrm{traj}}\) 几乎重合（EXEC Spearman=0.988，median gap=0，median ratio=1）。  
   原因：\(s_t\) 是策略真实状态，`shadow_h15(\(\Gamma_t\mid s_t\))` 与该 hold 实际增量一致。`payload_bits` 沿时间可加。  
   **这不是“oracle 单 hold 排序能预测 4-hold 收益”。**

2. **第一 hold 的 oracle \(U_{LL}\) vs 固定该 hold 后的 \(U_{\mathrm{traj}}\)：Case D。**  
   EXEC counterfactual：Spearman **0.539**，top-1 **0.125**，regret **0.242**。  
   同一 \(U_{LL}\) 的 \(\Gamma_1\) 可以走出差数倍的 \(U_{\mathrm{traj}}\)。  
   **这才是标题问题的答案：单 hold oracle 排序不能可靠预测 4-hold 闭环收益。**

另外，EXEC seed0 的 greedy **第一 hold 本身就很差**（pool 内 percentile 中位数 0.40，DecodeNR 中位数 0.90，\(U_{\mathrm{single}}\) 均值 0.69e6）。PREFIX 第一 hold percentile 0.96、\(U_{\mathrm{single}}\) 均值 4.28e6。  
因此 EXEC 失败是 **greedy 选点差 + 第一 hold \(U_{LL}\) 不是轨迹目标（Case D）** 叠在一起，不是“没学会 pool 内 \(U_{LL}\)”。

---

## 1. 轨迹级 ground truth

每个 (method, scenario) 记录 4 个 hold 的 \(\Gamma_t\)、\(U_{LL}(\Gamma_t\mid s_t)\)、该 hold 实际增量、以及整局 \(U_{\mathrm{traj}}\)。均为 4 个 hold（t=0,15,30,45；最后一段只剩 3 步，shadow 在 horizon 处截断）。

本审计自然轨迹 \(U_{\mathrm{traj}}\) 均值（LL seed=0）：

| method | n | \(U_{\mathrm{traj}}\) mean | official 5×5 / 多 seed 闭环 |
|---|---:|---:|---:|
| OLD | 16 | 1.0875e7 | 1.09125e7 |
| ABS seed0 | 16 | 9.8125e6 | 8.44e6（五 seed） |
| PREFIX seed0 | 16 | 1.15625e7 | 1.12875e7 |
| EXEC seed0 | 16 | 5.8125e6 | 8.6925e6（五 seed；seed0 正式为 5.55e6） |

与正式定义一致。EXEC seed0 低闭环在本审计复现。

---

## 2. \(\sum U_{\mathrm{single}}\) vs \(U_{\mathrm{traj}}\)

| method | Pearson | Spearman | Kendall | ratio median | abs error median | rel error median |
|---|---:|---:|---:|---:|---:|---:|
| OLD | 0.999 | 0.997 | 0.892 | 1.000 | 0 | 0 |
| ABS | 1.000 | 0.994 | 0.892 | 1.000 | 0 | 0 |
| PREFIX | 0.961 | 0.982 | 0.692 | 1.000 | 0 | 0 |
| EXEC | 0.999 | 0.988 | 0.883 | 1.000 | 0 | 0 |

沿**已实现**的 \((\Gamma_1,\ldots,\Gamma_4)\)，单 hold \(U_{LL}(\Gamma_t\mid s_t^{\mathrm{actual}})\) **能够加总成** 整局 payload。  
它回答不了：“在 \(s_0\) 上按 \(U_{LL}\) 排序，能不能选出高 \(U_{\mathrm{traj}}\) 的 \(\Gamma_1\)？”

---

## 3. 非加性 / interaction_gap

`interaction_gap = U_traj - sum_t U_single`

| method | mean | median | std | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| OLD | 3.75e4 | 0 | 1.09e5 | 0 | 1.00e5 |
| ABS | −2.50e4 | 0 | 6.83e4 | −1.00e5 | 0 |
| PREFIX | 1.38e5 | 0 | 2.50e5 | 0 | 6.00e5 |
| EXEC | 3.75e4 | 0 | 1.31e5 | 0 | 2.00e5 |

相对 \(U_{\mathrm{traj}}\)，|gap| 约 0.6%（EXEC）。**沿已实现轨迹，没有大的加性缺口。**

按 hold index（EXEC，\(U_{\mathrm{single}}\) 与实际 hold 增量几乎相同）：

| hold | t | \(U_{\mathrm{single}}\) mean | \(U_{\mathrm{hold,actual}}\) mean |
|---:|---:|---:|---:|
| 0 | 0 | 6.88e5 | 6.88e5 |
| 1 | 15 | 2.49e6 | 2.50e6 |
| 2 | 30 | 2.16e6 | 2.19e6 |
| 3 | 45 | 4.38e5 | 4.38e5 |

PREFIX 第一 hold 均值 **4.28e6**，OLD 3.68e6，ABS 3.78e6，EXEC **0.69e6**。  
EXEC 的损失主要发生在 **第一 hold 选了一个低即时效用的 \(\Gamma_1\)**，不是 sum 对不上 traj。

按 candidate type（idle 桶）：自然轨迹里多数是 half_idle。EXEC 几乎全是 half_idle，且该桶 \(U_{\mathrm{single}}\) 均值只有 1.43e6；PREFIX half_idle 为 2.80e6。

跨 hold 的“交互”不表现在 \(U_{\mathrm{traj}}-\sum U_{\mathrm{single}}\)，而表现在：**换 \(\Gamma_1\) 会改变 \(s_2,s_3,s_4\)，从而改变后面的 \(U_{LL}\) 和总收益**（第 5 节）。

---

## 4. Oracle hold vs 实际 greedy

在 \(t=0\) 用官方 24-pool 的 \(U_{LL}\) 作 oracle。greedy 几乎都不在池内（OLD/PREFIX/EXEC 的 first_in_pool=0；ABS=0.1875），与 Phase-3 DecodeNR 审计一致。

| method | top-1 agree（精确命中 pool argmax） | DecodeNR median | \(U_{LL}\) percentile median | first-hold \(U_{\mathrm{single}}\) mean |
|---|---:|---:|---:|---:|
| OLD | 0.00 | 0.225 | **0.917** | 3.68e6 |
| ABS | 0.19 | 0.104 | **0.938** | 3.78e6 |
| PREFIX | 0.00 | **0.091** | **0.958** | **4.28e6** |
| EXEC | 0.00 | **0.900** | **0.396** | **0.69e6** |

**greedy 的 \(U_{LL}\) 并不总是“还行”：**

- PREFIX / OLD / ABS：assignment 虽 OOD，但 \(U_{LL}\) 落在池子顶部（percentile≈0.92–0.96），即时效用不差。
- EXEC seed0：经常明显偏离 oracle best（DecodeNR 0.90，percentile 0.40）。

后续 hold（PREFIX / EXEC，官方 generate_pool 在**策略诱导状态**上，1-seed shadow）：

| method | later top-1 | later top-3 | later percentile | later regret |
|---|---:|---:|---:|---:|
| PREFIX | 0.00 | 0.00 | 0.88 | ≈0（中位数） |
| EXEC | 0.00 | 0.00 | 0.74 | 0.40 |

后续 greedy 也不在新生成的 24-pool 里（top-3=0）。PREFIX 的 OOD \(U_{LL}\) 仍接近池顶；EXEC 较差。

---

## 5. 最重要：第一 hold counterfactual

同一 \(s_0\)：固定 \(\Gamma_1\) 为 24-pool 中某一个，后 3 个 hold 仍由该 High-Level 的 official greedy 决定，记录 \(U_{\mathrm{traj}}\)。比较第一 hold \(U_{LL}(\Gamma_1\mid s_0)\) 与这条轨迹的 \(U_{\mathrm{traj}}\)。

按 scenario 计算再平均（16 个 scenario × 24 个 \(\Gamma_1\)）：

| method | Spearman | Kendall | top-1 agree | regret |
|---|---:|---:|---:|---:|
| OLD | 0.619 | 0.450 | 0.188 | 0.150 |
| ABS | 0.511 | 0.372 | 0.250 | 0.153 |
| PREFIX | 0.391 | 0.271 | **0.000** | 0.278 |
| EXEC | **0.539** | 0.387 | **0.125** | 0.242 |

实例（EXEC，scenario 221015）：\(U_{LL}=5.0\times10^6\) → \(U_{\mathrm{traj}}=1.4\times10^7\)；\(U_{LL}=1.0\times10^6\) 可以是 \(1.0\times10^7\) 或 \(1.0\times10^6\)；\(U_{LL}=0\) 可以是 \(7.0\times10^6\) 或 \(1.0\times10^6\)。

**第一 hold 的 oracle argmax 与最终 4-hold 最优 \(\Gamma_1\) 不一致。** 这是 long-horizon / future-state effect，不是加性 gap。

PREFIX 的 first-hold→\(U_{\mathrm{traj}}\) Spearman **更低**（0.39），但它的 greedy 稳定选高即时 \(U_{LL}\) 的 \(\Gamma_1\)，闭环仍然最高。说明：

- “会选高 \(U_{LL}\) 的第一 hold”足以拉开 EXEC seed0 与 PREFIX 的闭环差距；
- “\(U_{LL}\) 排序 = 轨迹最优”并不成立，四条方法都不是。

---

## 6. Utility 真正处在哪一层

| Case | 含义 | 本审计 |
|---|---|---|
| A | \(U_{\mathrm{single}}\approx U_{\mathrm{traj}}\)（沿已实现路径） | **成立（加性）** |
| B | 中等相关 + 大 interaction_gap | 不成立（gap 中位数 0） |
| C | \(U_{\mathrm{single}}\) 与 \(U_{\mathrm{traj}}\) 几乎无关 | 不成立（沿路径 Spearman>0.98） |
| D | 第一 hold \(U_{LL}\) 与 4-hold \(U_{\mathrm{traj}}\) 明显不一致 | **成立（counterfactual）** |

对标题问题应报 **Case D（主）+ 路径加性 Case A（辅）**。  
Phase-3 的监督目标（oracle-replay 状态上的单 hold \(U_{LL}\)）**不是** 最终 4-hold 轨迹目标；即便改到“真实 \(s_t\) 上的 \(U_{LL}\)，沿路径可以加总，**第一 hold 的排序仍选不对轨迹最优**。

---

## 7. OLD / ABS / PREFIX / EXEC

| method | first-hold DecodeNR | first-hold Spearman（\(U_{LL}\) vs \(U_{\mathrm{traj}}\)） | trajectory correlation（\(\sum U\) vs \(U_{\mathrm{traj}}\)） | interaction gap mean | closed-loop payload（正式） |
|---|---:|---:|---:|---:|---:|
| OLD | 0.225 | 0.619 | 0.997 | 3.75e4 | 1.091e7 |
| ABS | 0.104 | 0.511 | 0.994 | −2.50e4 | 8.440e6 |
| PREFIX | 0.091 | 0.391 | 0.982 | 1.38e5 | **1.129e7** |
| EXEC | 0.900 | 0.539 | 0.988 | 3.75e4 | 8.693e6 |

---

## 必须回答的五个问题

1. **\(U_{LL}\) 能否预测最终 4-hold closed-loop utility？**  
   沿已实现轨迹：能加总（Spearman≈0.99）。  
   作为第一 hold 的选择分数：不能可靠预测。EXEC counterfactual Spearman=0.54，top-1=0.125。  
   Phase-3 用的还是 **oracle 回放状态** 上的 \(U_{LL}\)，比本审计的真实 \(s_t\) 更远。

2. **单 hold 与 trajectory 是否存在显著 interaction？**  
   加性 gap 不显著（median 0）。  
   交互在 **状态转移**：\(\Gamma_1\) 改变 \(s_{2,3,4}\)，同一 \(U_{LL}(\Gamma_1\mid s_0)\) 对应不同 \(U_{\mathrm{traj}}\)。

3. **第一 hold 的 oracle 选择与最终 4-hold 收益是否一致？**  
   不一致。四条方法的 counterfactual top-1 均为 0–0.25。

4. **EXEC 失败是什么？**  
   - 不是“High-Level 完全没学 \(U_{LL}\)”（Phase-3 pool Spearman≈0.40）。  
   - **是 greedy decoder / 选点问题（EXEC seed0）：** 第一 hold \(U_{LL}\) 均值 0.69e6 vs PREFIX 4.28e6，DecodeNR 0.90。  
   - **也是 single-hold 目标错误 / long-horizon（Case D）：** 即便在池内按 \(U_{LL}\) 选 \(\Gamma_1\)，也只有 12.5% 机会选到最高 \(U_{\mathrm{traj}}\)。  
   两条同时成立。PREFIX 说明：只修 decoder、继续用单 hold \(U_{LL}\) 当选点启发，闭环可以很高，但 \(U_{LL}\) 仍不是轨迹最优标签。

5. **下一阶段应优化哪一层？**  
   不要再改 candidate-level q。  
   路径加性已成立，缺的是 **第一 hold（以及后续 hold）对 \(U_{\mathrm{traj}}\) 的 credit assignment / 轨迹级效用**。  
   EXEC seed0 的 greedy 选点应视为伴随现象，不是新 loss 的借口。

---

## 下一阶段只验证一个假设（不实现、不训练）

> 在固定后续 greedy 的条件下，第一 hold 的官方 \(U_{LL}\) 与 4-hold \(U_{\mathrm{traj}}\) 不同序（本审计 EXEC top-1=0.125，Spearman=0.54）。因此下一阶段的对象应是 **轨迹级回报，或 hold 对 \(U_{\mathrm{traj}}\) 的 credit**，而不是另一条 candidate-level \(q_i\)。

最小后续（仅提案）：在冻结 PREFIX 与 EXEC 上，用本审计已有的 counterfactual 表，比较 “argmax \(U_{LL}\)” 与 “argmax \(U_{\mathrm{traj}}\)” 的 assignment 差在哪些 UAV/浮标边上，确认 credit 应回传给第一 hold 的哪一部分。

本阶段停止。不要训练。不要实现新方法。不要改 checkpoint。
