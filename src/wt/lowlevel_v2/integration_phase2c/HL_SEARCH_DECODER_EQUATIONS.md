# Free decoder vs order-marginalized set likelihood

Source of truth: `wt.htw_phase943.evaluation.decode_assignment`, `pick_candidate`, `set_likelihood_dp.subset_dp_logprob`.

This document does **not** change inference. It records the frozen equations.

## 1. Candidate at each step

Partial assignment \(A\in\{-2\}\cup\{-1\}\cup\{0,\ldots,11\}^6\). Undecided UAV is \(-2\). Idle is \(-1\). Buoy index is \(0\ldots 11\).

A candidate is a pair \((m,k)\):

- \(m\in\{0,\ldots,5\}\) an undecided UAV
- \(k\in\{0,\ldots,11,12\}\) a buoy token or idle token `IDLE_INDEX=12`

Legal mask (`legal_candidate_mask`):

- already-decided UAV: no candidates
- remaining UAV: every unoccupied available buoy, plus idle always

Choosing \((m,k)\) writes \(A_m \leftarrow -1\) if \(k=12\), else \(A_m \leftarrow k\), and that buoy becomes occupied.

## 2. Score / logit

Encoder is frozen DeepSets. Decoder is permutation-equivariant `OrderFreeSetDecoder`.

Given partial \(A\), it builds set summaries of decided edges, remaining UAVs, remaining buoys, plus candidate physics, then a shared MLP scores every legal \((m,k)\). Illegal cells are \(-10^9\).

## 3. Free decoder is greedy joint argmax

`decode_assignment` loops \(M=6\) times:

1. score all legal \((m,k)\)
2. `pick_candidate`: \(\arg\max\) over legal logits
3. ties within `TIE_ATOL=1e-6` broken by `canonical_keys` (UAV xy, buoy xy, idle flag, \(-rate\)), not by sampling unless `shuffle_seed` is set

Official free decode does **not** sample and does **not** search a beam. It is **not** UAV-then-buoy sequential softmax. It is joint greedy over the same \((m,k)\) grid that the legal log-softmax uses.

## 4. Context update

After \((m,k)\): that UAV is decided; if \(k\) is a buoy it is occupied; remaining-UAV / remaining-buoy DeepSets summaries and physics are recomputed from the unordered partial set. Order of previous choices is not stored as a sequence in the state, only as the set of decided edges.

## 5. Idle

Idle is always a legal token for every undecided UAV. It competes in the same joint softmax as buoys.

## 6. Sequence / path log-probability

A generation path is an ordered list of legal edges \((m_t,k_t)_{t=1}^{6}\).

At partial \(A^{(t)}\):

\[
\log P_{\mathrm{path}}(A^{(t+1)}\mid A^{(t)})
=
\log
\frac{\exp z_{m_t k_t}(A^{(t)})}
{\sum_{(m',k')\in\mathrm{Legal}(A^{(t)})}\exp z_{m'k'}(A^{(t)})}
\]

\[
\log P_{\mathrm{path}}(\pi,A)
=
\sum_{t=1}^{6}
\log P_{\mathrm{path}}(A^{(t)}\mid A^{(t-1)})
\]

Beam search (Phase-2A diagnostic, unused at official inference) ranks incomplete hypotheses by this **path** sum.

## 7. Unordered set log-probability (training)

Training likelihood is order-marginalized set probability (`subset_dp_logprob`):

\[
\log P_{\mathrm{set}}(\Gamma\mid s)
=
\mathrm{logsumexp}_{\pi\in S_6}
\log P_{\mathrm{path}}(\pi,\Gamma\mid s)
\]

implemented as subset DP over UAV subsets: at mask of decided UAVs, legal log-softmax, then add the unique remaining UAV that matches \(\Gamma\).

This is **not** a score inside the L=24 pool. The pool softmax was Phase-2 COND loss; Phase-2B \(L_{\mathrm{abs}}\) uses these absolute \(\log P_{\mathrm{set}}\).

## 8. Distinctions that this audit tests

| Object | Maximized by official greedy? | Used in ABS training? |
|---|---|---|
| joint one-step logit / path | yes, myopically | no, except as DP factors |
| full path probability of one order | only implicitly | no |
| unordered \(P_{\mathrm{set}}\) | **not guaranteed** | **yes** |

`training_inference_objective_mismatch` is true if greedy/beam-path often misses the assignment with highest \(P_{\mathrm{set}}\).
