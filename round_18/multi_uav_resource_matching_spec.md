# Frozen deterministic multi-UAV resource matching

At each service slot, the current score is

`W[m,k] = (Q[k] / queue_capacity_bits) * R[m,k]`.

`R[m,k]` is the current static existing path-loss/Shannon rate from UAV `m` to
buoy `k`; only current state is read.  The exhaustive solver evaluates all
partial injective assignments `m -> k` (or unmatched `-1`) for the small
registered `M=2`, `K=3/5` setting.  It maximizes total score, then matched
cardinality, then the lexicographically smallest assignment tuple (where
unmatched sorts after all buoy indices).  This is deterministic, centralized,
and has no third-party dependency.

Constraints are exactly `sum_k x[m,k] <= 1`, `sum_m x[m,k] <= 1`, and
`x[m,k] in {0,1}`.  A matched buoy receives
`min(Q[k], R[m,k] * delta_t)`; no unmatched UAV transmits, and one shared
queue is decremented once before the unchanged arrival/capacity update.
