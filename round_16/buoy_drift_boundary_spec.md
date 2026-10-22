# Buoy-drift boundary specification

Buoys are constrained to the configured margin rectangle
`[buoy_margin, area_width-buoy_margin] x [buoy_margin, area_height-buoy_margin]`.
After each service slot, drift advances the current position by
`velocity * delta_t`. Each coordinate uses deterministic reflection: repeated
crossings are folded back into the legal interval and the corresponding
velocity component reverses for every odd number of boundary crossings.

This is applied identically in the dynamic environment and causal transition
model. Static mode has zero velocity and makes no buoy-state update. There is
no clipping with retained outward velocity, teleportation, or random reset.
