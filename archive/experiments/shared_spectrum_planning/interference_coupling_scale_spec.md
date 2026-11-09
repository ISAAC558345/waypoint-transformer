# Frozen interference-coupling scale

The following tiers are fixed before any Round-19 formal result is generated:

| tier | spectrum mode | eta_int |
| --- | --- | ---: |
| orthogonal control | orthogonal | 0.00 |
| low | shared | 0.25 |
| medium | shared | 0.50 |
| full shared | shared | 1.00 |

The tier values are protocol parameters only: `P_b`, `B`, `N`, reference gain,
path-loss exponent, movement actions, queue/reward semantics, and safety are
frozen from Round-18.  The ISR conflict classifier is also pre-registered as
`max(ISR_m) >= 0.25`, with `ISR_m=I_m/S_m`; it will not be retuned after
observing policy results.  A realism claim requires a stable result at low or
medium, never full shared alone.
