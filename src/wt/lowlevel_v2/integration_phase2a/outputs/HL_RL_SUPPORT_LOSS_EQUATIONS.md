# HL-RL-SUPPORT：正式 utility loss 方程（来自磁盘代码，非推断）

本文件只记录 Phase-9.4.5 / Integration Phase-2 实际执行的公式。未改代码。

## 1. Candidate set log-likelihood \(\log P_\theta(\Gamma_j\mid s)\)

代码：

- `src/wt/htw_phase945/utility_loss.py` `candidate_logprobs`
- `src/wt/htw_phase943/set_likelihood_dp.py` `subset_dp_logprob`

对每个完整 assignment \(\Gamma\)（6 个 UAV，目标为 buoy 或 idle），在所有决定顺序上做 order-marginalized 子集 DP：

在每个已决定 UAV 子集 \(S\) 上，decoder 对尚未决定的 \((uav, token)\) 合法对做 **legal log-softmax**（`_log_softmax_legal`）。然后

\[
F(\emptyset)=0,\qquad
F(S\cup\{u\})=\mathrm{logsumexp}_{\text{orders}} \big(F(S)+\log \pi_\theta(token_u\mid S)\big).
\]

返回 \(F(\{0,\ldots,5\})=\log P_\theta(\Gamma\mid s)\)。

`set_nll = -subset_dp_logprob`（同一文件 `set_nll`）。

逐步 decoder（`decode_assignment` / `pick_candidate`）在每一步对同一 legal 集合取 argmax。该过程走完 6 步后得到一个完整 assignment。逐步 legal softmax 覆盖同一完整 assignment 空间，因此 \(\sum_{\Gamma} P_\theta(\Gamma\mid s)=1\) 在 **完整合法 assignment 空间** 上成立（不是只在 L=24 上）。本审计据此把 \(\sum_{j=1}^{24} P_\theta(\Gamma_j\mid s)\) 称为 **PoolMass**，并把 \(1-\mathrm{PoolMass}\) 记为 **OutsideMass**（对池内重复 hash 只计一次）。

## 2. Utility loss 之前是否对 candidate logP 做 softmax over L=24？

**是。** `utility_losses`：

```python
logp = candidate_logprobs(...)          # shape [L], L=24, 每项是 absolute log P_theta
log_p = torch.log_softmax(logp, dim=0)  # 仅在 24 个 candidate 上归一化
l_util = -(q * log_p).sum()
```

## 3. 实际优化的条件分布

\[
p_j=\frac{\exp(\log P_\theta(\Gamma_j\mid s))}{\sum_{l=1}^{L}\exp(\log P_\theta(\Gamma_l\mid s))}
=\mathrm{softmax}_{l\in\mathrm{pool}}(\log P_\theta(\Gamma_l\mid s))_j
\]

\[
q_j=\mathrm{softmax}\!\left(\frac{\mathrm{median\text{-}MAD}(U)_j}{\tau}\right),\quad \tau=1
\]

`soft_targets` / `normalize_state`：`src/wt/htw_phase945/utility_targets.py`。冻结：`phase945_utility_normalization.json` `method=median_mad`, `eps=1`。

\[
L_{\mathrm{utility}}=-\sum_{j=1}^{L} q_j\log p_j
\]

这是 **candidate-conditional** 交叉熵。对所有 \(\log P_j\) 加同一常数 \(c\) 不改变 \(p_j\)，因此 **不约束** candidate pool 的绝对概率质量 \(\sum_j P_\theta(\Gamma_j\mid s)\)。

## 4. Reference anchor

\[
L_{\mathrm{ref}}=\mathrm{set\_nll}(\Gamma_{\mathrm{ref}}\mid s)=-\log P_\theta(\Gamma_{\mathrm{ref}}\mid s)
\]

\(\Gamma_{\mathrm{ref}}\) 是数据集中的 oracle/reference assignment（`train["assignment"]`），**使用绝对 set probability**，不经过 pool softmax。权重 \(\lambda_{\mathrm{ref}}=0.10\)。

OLD-HL 与 NEW-HL 的 Phase-9.4.5 / Phase-2 训练都含此项；NEW-HL 未改 \(\lambda_{\mathrm{ref}}\)。

## 5. Ranking loss

`ranking_pairs`：仅当 \(|U_i-U_j|>\delta\)，\(\delta=\max(0.05\cdot\mathrm{range}(U),1)\)。

\[
L_{\mathrm{rank}}=\sum_{(i,j)} w_{ij}\,\mathrm{softplus}\big(-y_{ij}(\log P_i-\log P_j)\big)
\]

差 \(\log P_i-\log P_j\) **平移不变**，因此 ranking 也 **不约束** 绝对 pool mass。权重 \(\lambda_{\mathrm{rank}}=0.25\)。使用的是 absolute logP 的差，等价于 pool-conditional logits 的差。

## 6. Absolute vs pool-conditional

| 项 | 绝对 \(P_\theta(\Gamma\mid s)\) | 仅 pool-conditional \(p_j\) |
|---|---|---|
| \(L_{\mathrm{utility}}\) | 否（先 `log_softmax` over L=24） | 是 |
| \(L_{\mathrm{rank}}\) | 差，平移不变 | 等价于 conditional |
| \(L_{\mathrm{ref}}\) | 是 | 否 |

总损失（`phase945_loss_spec.json`）：

\[
L=L_{\mathrm{utility}}+0.25\,L_{\mathrm{rank}}+0.10\,L_{\mathrm{ref}}
\]

自由解码 **不** 在 L=24 上 argmax，而是逐步 `pick_candidate`。因此提高 \(p_j\) 不必提高 \(P_\theta(\Gamma_j)\) 相对 pool 外 assignment 的质量。
