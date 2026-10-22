# Service-resource decoder specification (amendment)

## Diagnostic interface

The prospective sidecar form is `[delta_x, delta_y, pi_1, ..., pi_K]` for the
current, configuration-defined buoy count `K`.  The public `MaritimeEnv`
action remains only `[delta_x, delta_y]`; this file specifies a diagnostic
decoder and does not extend that action space.

Inputs are the current observable queue vector `Q`, current full-bandwidth
rate vector `R_full`, current valid-buoy mask `m`, and slot duration
`delta_t`.  No future queue, arrival, rate, payload, reward, position, or
action is accepted.

## Masked simplex and allocation

- `pi_k >= 0` for every buoy.
- If `m_k=false` (padding), `pi_k=fraction_k=0` exactly.
- If any buoy is valid, `sum_{m_k=true}(pi_k)=1`; otherwise the all-zero
  vector is legal.
- The diagnostic score is
  `score_k=(Q_k/max_valid(Q))*(R_full,k/max_valid(R_full))`, normalized over
  valid entries.  If every valid score is zero, the deterministic fallback is
  the current fixed-equal allocation `1/K`; there is no random assignment.
- `fraction_k=pi_k`, `0 <= sum(fraction_k) <= 1`, and
  `allocated_capacity_k=fraction_k*R_full,k*delta_t`.
- `actual_payload_k=min(Q_k, allocated_capacity_k)`, so actual payload can
  never exceed queue or allocated capacity.

The implementation rejects nonfinite or negative values, padding mass,
fractions whose total exceeds one, nonpositive slot durations, and every
capacity/queue violation.  It is dynamically sized by the supplied vectors
and mask; no fixed buoy count is assumed.
