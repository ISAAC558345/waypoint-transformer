# Task-Waypoint controllability specification (amendment)

For head response $Y_{s,a}$ at snapshot $s$ and legal first action $a$, the frozen formal score is:

`C_head = mean_s[var_a(Y_{s,a})] / (var_{s,a}(Y_{s,a}) + epsilon)`

The numerator is the mean action-conditioned variance at a fixed snapshot; the denominator is total response variance over every snapshot-action response in the reported group.  `epsilon=1e-12`.

| head | frozen Round-13 overall C |
|---|---:|
| C_service | 0.0037064542 |
| C_progress | 0.0096376949 |
| C_return | 0.0030245169 |

`H_step_return` is the minimum-controllability head.  All three values are low: this is evidence of broad indirect, movement-mediated controllability rather than a problem unique to one head.  These tables are deterministic splits of the immutable original raw formal result; no scenarios were rerun.
