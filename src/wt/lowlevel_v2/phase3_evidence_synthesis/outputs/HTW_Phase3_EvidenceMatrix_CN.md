# HTW Phase-3 Evidence Synthesis Audit

本文件只汇总已经完成的 Phase-3C～3H。未训练、未改 Low-Level / Actor-Critic / reward / decoder / pool / greedy / checkpoint。未重新 rollout。未进入 End-to-End Independent Validation。未进入 Phase-4。未设计新的 geometry / feasible-set proxy，未再 sweep predictor。

主判断家族：**EXEC**。数字一律取各阶段 JSON / 正式报告中的锁定值。`11.88e6` 与 `1.188e7` 都指 **11875000**。

---

## 1. 数据一致性检查

对照 Phase-3C～3H 的 LOCAL / TRAJ / NATURAL / FHAT / OOF Spearman。发现差异只做记录，不改旧实验。

| 量 | 锁定值 | 出现阶段 | 一致性 |
|---|---|---|---|
| EXEC NATURAL | 5.812e6 | 3C, 3D, 3E, 3F, 3G | 通过 |
| EXEC LOCAL-ORACLE | 9.000e6 | 3C–3H | 通过 |
| EXEC TRAJ-ORACLE | 11.875e6（写作 11.88e6） | 3C–3H | 通过 |
| EXEC FHAT 闭环 | 9.375e6 | 3E–3H | 通过 |
| ExtraTrees-C OOF Spearman | 0.553 | 3E 源；3F/3G/3H 引用 | 通过 |
| candidate geometry OOF | 0.546 | 3F；3G/3H 引用 | 通过 |
| 3D β=0 / β=1 | 必须等于 LOCAL / TRAJ | 3D sanity 通过 | 通过 |
| LOCAL DecodeNR | 0 | 3C/3D 定义 | 通过 |

**记录性差异（不改旧实验）：**

1. **`R_pred` 有两种合法聚合。** Phase-3E JSON 用均值比：`(9.375e6−9.000e6)/(11.875e6−9.000e6)=0.130`。Phase-3F/3G/3H 用 scenario 比率再平均，得到 **0.135**。闭环 `U_traj=9.375e6` 本身一致。后文写 `R_pred≈0.13`，两种算法都指向同一缺口。
2. **`U_single` 的 OOF Spearman 因估计器不同而不同。** Phase-3E Ridge-A=**0.193**；Phase-3F ExtraTrees-A=**0.295**。不是抄错。引用时必须写清模型。
3. **组内 vs 池化。** Phase-3C 池化 Spearman(`U_single`,F)=0.211；Phase-3F/3G 组内 median=0.259。指标定义不同。
4. **`F_hat` 与 `S_fhat`。** Phase-3F 组内 Spearman(`F_hat`,F)=0.577；Phase-3G/3H 的 FHAT 是 `U_single+0.25 F_hat`，组内 Spearman=0.353。分数不同，不是矛盾。
5. **书写圆整。** 10.81e6=10812500；11.44e6=11437500；11.88e6=11875000。

结论：主锚点 LOCAL / TRAJ / NATURAL / FHAT / 0.553 **相互一致**。没有需要回改的历史结果。

---

## 2. 最终证据矩阵

| Phase | 假设 | 方法 | 核心指标 | 结果 | 是否支持 |
|---|---|---|---|---|---|
| C | local utility 足以代表 trajectory utility | first-hold counterfactual | LOCAL vs TRAJ | 9.00 vs 11.88e6 | **否** |
| D | trajectory-aware credit 存在 | oracle `S_β=U_single+βF` | recovery | β=0.5 ≈ TRAJ 的 96% | **是** |
| E | first-hold 可观测信息可预测 F | OOF ExtraTrees | Spearman | 0.553 | **部分** |
| F | candidate geometry 提供 future signal | feature-group ceiling | OOF | 0.546～0.553 | **部分** |
| G | 简单连续结构 proxy 可解释 signal | STRUCT / `O_traj` | Spearman | ≤0.187 | **否** |
| H | 离散 feasible-set 可解释 signal | degree / Flex | Spearman | ≤0.041 | **否** |

CSV：`src/wt/lowlevel_v2/phase3_evidence_synthesis/outputs/phase3_evidence_matrix.csv`  
锁定数字：`src/wt/lowlevel_v2/phase3_evidence_synthesis/outputs/phase3_summary.csv`

---

## 3. Phase-3C～3H 证据链

### 3C — local ≠ long-horizon

只改 first-hold，hold 2/3/4 仍正式 greedy。

| 条件 | EXEC `U_traj` |
|---|---:|
| NATURAL | 5.812e6 |
| LOCAL-ORACLE | 9.000e6 |
| TRAJ-ORACLE | 11.875e6 |

LOCAL 相对 NATURAL 抬 +3.19e6；TRAJ 再多 +2.88e6。`argmax U_single` 与 `argmax U_traj` 的 mismatch 是结构事实，不是噪声。DecodeNR：LOCAL=0，TRAJ≈0.516。LOCAL 选中的 remaining F=4.388e6，TRAJ 选中的 F=9.625e6。

### 3D — future-return F 能解释 first-hold credit，但不能当在线输入

`S_β=U_single+βF`，F 来自冻结 counterfactual，不是观测。

| β | EXEC `U_traj` | 占 TRAJ | Spearman | top-1 |
|---:|---:|---:|---:|---:|
| 0 | 9.000e6 | 75.8% | 0.539 | 0.125 |
| 0.25 | 10.81e6 | 91.1% | 0.831 | 0.250 |
| 0.50 | 11.44e6 | **96.3%** | 0.946 | 0.312 |
| 0.75 | 11.75e6 | 98.9% | 0.981 | 0.500 |
| 1.00 | 11.88e6 | 100% | 1.000 | 1.000 |

β=0.5 回收 LOCAL→TRAJ 剩余增益的 0.848。**trajectory-aware credit signal 存在。** F 本身依赖完整后续 greedy，禁止作为在线 feature。

### 3E — 可观测信息能预测一部分 F

scenario GroupKFold，OOF ExtraTrees-C：

| ablation | OOF Spearman |
|---|---:|
| U_single only（Ridge） | 0.193 |
| state + candidate | 0.527 |
| full | **0.553** |

最佳预测选择：`U_single+0.25 F_hat` → 闭环 **9.375e6**，`R_pred≈0.13`。candidate-conditioned geometry 有真实 signal，但只能恢复很小一块 trajectory-aware gain。

### 3F — 经验上限约 0.5～0.6

| group | OOF Spearman |
|---|---:|
| ExtraTrees U_single | 0.295 |
| candidate geometry | **0.546** |
| state only | 0.071 |
| full | 0.553 |

不是“只用了 U_single”的问题。geometry 已经接近 full。组内 Spearman(`U_single`,F) median=0.259；(`F_hat`,F)=0.577；top-1 仍≈0.062。判断：**information-limited**。这不是数学上界，而是当前 first-hold 可观测特征的经验天花板。

### 3G — 连续结构 proxy 失败

`O_traj=leftover_queue·(1+idle)·access/(1+competition)`。

| score | 组内 Spearman vs F | 闭环 `U_traj` |
|---|---:|---:|
| U_single | 0.259 | 9.000e6 |
| O_traj | 0.002 | — |
| STRUCT_OC | 0.187 | 9.250e6 |
| FHAT | 0.353 | 9.375e6 |

STRUCT_OC 只改 2/16 个 state，不优于 FHAT。简单连续加性结构 **不能替代** `F_hat`，更不能替代 oracle F。

### 3H — 离散 feasible-set 失败

| score | 组内 Spearman vs F | 闭环 `U_traj` |
|---|---:|---:|
| degree | 0.036 | 4.500e6 |
| weighted_degree | 0.018 | 4.500e6 |
| Flex | 0.041 | 4.688e6 |
| S_disc | 0.187 | 9.250e6（1/16 state） |
| FHAT | 0.353 | 9.375e6 |

高 F candidate 的 leftover degree **更小**，不是更大。`argmax degree/Flex` 明显伤害闭环。正式 pool 内 conflict 全为 0。离散 feasible-set **不能解释** Phase-3F 的 geometry signal。

---

## 4. 三个层次

**层次 1：future-value signal 是否存在？**  
**是。** 证据：Phase-3D β sweep。oracle F 把 9.00e6 推到 11.44e6（β=0.5）直至 11.88e6（β=1）。

**层次 2：first-hold 可观测特征是否包含该 signal？**  
**是，但有限。** 证据：Phase-3E/3F OOF Spearman ≈ 0.53～0.55；candidate geometry ≈ 0.546。不是零，也不是接近 1。

**层次 3：是否已经找到足以在线恢复 trajectory value 的可靠 credit？**  
**否。** 证据：`R_pred≈0.13`；FHAT 闭环 9.375e6；TRAJ-ORACLE 11.88e6。Phase-3G/3H 的连续与离散 proxy 都不能补上这块 gap。

---

## 5. 最终机制结论（C 类）

当前 first-hold 可观测信息虽然包含与 future trajectory value 相关的 signal，但其信息量不足以可靠恢复 later-greedy / long-horizon trajectory utility。

连续结构 proxy 和离散 feasible-set proxy 均不能解释剩余 gap。

因此当前证据支持：这是 **observable-information / credit-assignment gap**，而不是已经找到一个简单 proxy 但尚未调好。

不支持写成：

- 模型不够复杂；
- 再训练一个 Transformer 就可以解决；
- 只要加入更多 feature 就可以解决；
- trajectory-aware credit 已经成功；
- candidate geometry 已经能够预测 future value。

---

## 6. 论文主线

```
High-Level planner
        ↓
first-hold candidate selection
        ↓
observable local utility U_single
        ↓
发现：local utility ≠ long-horizon trajectory value
        （3C：9.00e6 vs 11.88e6）
        ↓
构造 trajectory-aware counterfactual F
        ↓
证明 future-value credit 存在
        （3D：oracle β=0.5 → 11.44e6，约 TRAJ 的 96%）
        ↓
尝试从 first-hold observable features 恢复 F
        ↓
candidate geometry 提供部分 signal
        （3E/3F：OOF ≈ 0.53–0.55）
        ↓
但 OOF ceiling ≈ 0.55，闭环 R_pred ≈ 0.13
        ↓
尝试 continuous structural proxy
        ↓
失败（3G：O_traj=0.002，STRUCT_OC=0.187）
        ↓
尝试 discrete feasible-set proxy
        ↓
失败（3H：degree/Flex≤0.041；最大化有害）
        ↓
最终结论：
first-hold observable information
不足以可靠恢复 long-horizon trajectory credit。
```

---

## 7. 剩余 gap，以及为什么不再加 proxy

剩余闭环主要是：

`TRAJ 11.88e6 − FHAT 9.375e6 = 2.50e6`

相对 LOCAL→TRAJ 全缺口 2.875e6，可观测预测只回收约 13%。oracle F 在 β=0.5 已到 11.44e6，所以缺口不在“F 无用”，而在 **t=0 可观测特征还原不了 later-greedy 诱导的 F**。

不再继续加 proxy，因为：

1. 连续 leftover / access / competition 已经失败（3G）。
2. 离散 leftover 边 / Flex / pool 邻居已经失败（3H）。
3. ExtraTrees 全特征组合几乎不比 candidate geometry 更高（3F：0.553 vs 0.546）。
4. 再画公式或再调 α，只会重复同一 information ceiling，不会把 0.55 变成可靠 top-1，也不会把 9.375e6 拉到 11.88e6。
5. 相关升高 ≠ 算法有效；3G/3H 的选点几乎不变已经说明这一点。

Phase-3 正式结束。不启动新训练。不创建 Phase-4。
