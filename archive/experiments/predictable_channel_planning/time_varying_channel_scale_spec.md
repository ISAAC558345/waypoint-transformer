# Frozen Round-17 predictable-channel specification

For buoy `k` at slot `t`, Round-17 preserves the existing geometric gain and
uses only `beta_k[t]=beta_geom,k[t]*c_k[t]`, with
`c_k[t]=exp(sigma*sin(theta_k[t]))` and
`theta_k[t+1]=theta_k[t]+omega (mod 2*pi)`.

`omega=2*pi/24` radians/slot. Ordinary v1 channel state exposes only current
`[sin(theta_k), cos(theta_k), omega/(2*pi/20)]`; its final frequency feature is
therefore 20/24 and lies in [0,1]. The known dynamics permit causal forward
simulation without exposing future samples.

The frozen tiers are static `sigma=0.00`, low `0.15`, medium `0.35`, and high
`0.55`. Static mode uses multiplier exactly one and must leave the preexisting
static channel bit-identical. Each scenario registers deterministic,
asynchronous per-buoy phases. A pre-registered `channel_phase_conflict` subset
sets buoy 0 near a current peak and buoy 1 near a current trough; its 24-slot
oscillator produces a ranking reversal within H=10 without outcome-based
selection.
