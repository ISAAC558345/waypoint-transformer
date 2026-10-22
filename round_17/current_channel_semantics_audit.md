# Current channel semantics audit (pre-Round-17)

This audit was read from `src/wt/maritime/channel.py`, `config.py`, `env.py`,
and the Round-15/16 planners before changing maritime environment code.

## Existing physical channel

For buoy horizontal coordinate `b_k`, UAV horizontal coordinate `u`, and fixed
altitude `h`, the current implementation uses

`d_xy,k = ||b_k-u||`, `d_3d,k = sqrt(d_xy,k^2+h^2)`, and
`beta_geom,k = beta0 / d_3d,k^alpha`.

The default constants are `beta0=1e-3`, `alpha=2.2`, UAV altitude `h=50 m`,
buoy transmit power `P=0.1 W`, total bandwidth `B=1e6 Hz`, and noise power
`N=1e-9 W`. Links are orthogonal: each of `K` buoys receives `B/K` bandwidth.
Thus the existing quantity named `snr_linear` is `P*beta_geom/N` (there is no
inter-user interference term), and rate is
`r_k=(B/K)*log2(1+snr_linear,k)`.

There is no fading, shadowing, or channel-randomness process in the current
channel implementation. The channel module is stateless and deterministic;
environment randomness concerns buoy placement and arrival processes instead.

## Existing visibility and planning use

The v1 observation exposes per-buoy relative coordinates, normalized 3-D
distance, allocated normalized current rate, queue, previous arrival, and
completion ratio (plus currently enabled drift velocity in the Round-16
dynamic-drift schema). Physical `info` exposes current distances, gains, SNR,
rates, and total rate. No observation exposes gain/SNR explicitly and neither
observation nor info exposes a future channel trace.

Round-15's frozen QRE and causal shadow transition call the deterministic
geometry channel calculation for the current simulated UAV/buoy positions.
Round-16 preserves that action lattice, continuation, objective, ranking, and
projection; it only additionally advances buoy position in its allowed scope.

Round-17 therefore preserves every base formula above and adds only the
registered slow, bounded multiplier to geometric gain.
