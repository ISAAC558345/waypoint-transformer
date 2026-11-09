# Shared-spectrum buoy-uplink interference specification

## Signal direction and fixed parameters

The transmitter is always the assigned buoy; the receiver is its assigned UAV.
UAVs do **not** transmit in this model.  Let UAV `m` be assigned buoy `k_m`
and another UAV `j` be assigned distinct buoy `k_j`.  With the existing static
path gain `beta[m,k]`, fixed existing buoy power `P_b`, and fixed existing
receiver noise `N`, shared mode defines

`S_m = P_b beta[m,k_m]`

`I_m = eta_int * sum_{j != m} P_b beta[m,k_j]`

`SINR_m = S_m / (N + I_m)`

`R_m = B log2(1 + SINR_m)`.

Here `B` is the unchanged configured `total_bandwidth_hz`; no power,
bandwidth, noise, reference gain, or path-loss exponent is edited.  An
unmatched UAV has no desired uplink and no rate; a single matched UAV has
`I=0`.  Exclusive one-buoy-per-UAV and one-UAV-per-buoy matching remains in
force, and all matching inputs are present-state quantities only.

## Orthogonal control

`spectrum_mode="orthogonal"` delegates exactly to the locked Round-18 M=2
environment, including its `B/K` rate share.  It is therefore the required
same-scenario/same-action exact control, rather than a new analytical mode.
Shared mode uses the above active-uplink convention and introduces precisely
the specified cross-buoy signal term.  No UAV-to-UAV transmitter interference,
backhaul effect, power action, or beamforming is present.
