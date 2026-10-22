# Channel-variation scale audit (pre-registration)

The baseline five-buoy geometry envelope uses horizontal ranges 0, 25, 50, 75,
and 113.1 m at the existing 50 m altitude. Under the existing constants its
per-buoy static rate span is 360,949--853,989 bit/s. No result from a
long-horizon run informed this scale choice.

For the pre-registered multiplier `c=exp(sigma*sin(theta))`, the scale audit
gives the following endpoint envelope over that same geometry:

| tier | sigma | multiplier min/max | rate min/max (bit/s) | mean-rate ratio min/max |
| --- | ---: | ---: | ---: | ---: |
| static | 0.00 | 1.000 / 1.000 | 360,949 / 853,989 | 1.000 / 1.000 |
| low | 0.15 | 0.861 / 1.162 | 330,733 / 895,179 | 0.942 / 1.060 |
| medium | 0.35 | 0.705 / 1.419 | 292,647 / 950,527 | 0.866 / 1.141 |
| high | 0.55 | 0.577 / 1.733 | 257,225 / 1,006,288 | 0.792 / 1.223 |

All multipliers stay strictly positive. The primary period is fixed to 24
slots: H=5, H=10, and H=20 occupy 0.208, 0.417, and 0.833 of a cycle,
respectively. It is neither slot-independent noise nor episode-static drift.
