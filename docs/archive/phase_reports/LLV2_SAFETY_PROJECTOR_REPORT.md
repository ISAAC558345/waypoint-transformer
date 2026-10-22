# SafetyProjector report (interface only)

Deterministic iterative projection on predicted next positions.
Constraints: unit-step, pairwise predicted distance >= d_safe (placeholder), ServiceSafeEnvelope inner disk.
If infeasible: `constraint_status=failure`. Not silently labeled safe.
No QP dependency in the current environment; iterative projection is the v1 implementation.

- design seed 8 status: `ok`
- intervention_norm: 0.0
- min predicted distance: 1.2433337643669597
- envelope_ok: True
- finite: True
- inner service radius: 15.5 (service_radius - placeholder margin 0.5)
- envelope active count on seed 8 after reset: 5

Not a scientific benchmark result.
