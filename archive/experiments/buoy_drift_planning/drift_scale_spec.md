# Drift-scale specification

The existing diagnostic configuration has `max_speed=10 m/s`, `delta_t=1 s`,
and `episode_horizon=64`; a UAV therefore moves at most 10 m per slot and 640
m over an unconstrained episode. Drift is parameterized as
`v_drift_max = r_v * max_speed`, with frozen ratios `r_v` of 0 (static), 0.05
(0.5 m/s), 0.10 (1.0 m/s), and 0.20 (2.0 m/s). These are respectively 0%, 5%,
10%, and 20% of UAV maximum speed.

For `slowly_varying_secondary`, the maximum is 1.0 m/s, rho is 0.95, and the
per-axis innovation standard deviation is 0.01 m/s. Constant drift is the
primary dynamic condition. Values are fixed before execution and are never
selected based on planner outcomes.
