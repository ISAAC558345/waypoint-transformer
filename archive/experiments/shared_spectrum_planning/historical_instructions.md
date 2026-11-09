# Round-19 Working Rules

Round-19 is a deterministic, non-learning revalidation of M=2 maritime
coordination under shared-spectrum buoy-to-UAV uplink interference.  It starts
only after the verified, immutable Round-18 lock.

## Scope

- Primary: M=2, K=3/5, static buoy positions, static path-loss channel,
  fixed buoy transmit power, fixed movement action semantics, shared queues,
  exclusive matching, and the existing 15 m hard safety projection.
- The sole new physical coupling is co-channel cross-uplink interference from
  another *buoy's* simultaneous uplink at a receiving UAV.
- `spectrum_mode="orthogonal"` is a strict Round-18 M=2 control.  Shared mode
  uses the pre-registered overlap factors eta={0.25, 0.50, 1.00}.
- Use the frozen seven movement candidates, H={1,5,10}, causal current-state
  matching, and deterministic exhaustive assignment enumeration only.

## Prohibitions

- No neural model/training/checkpoint, optimizer/backward pass, HGNN, PPO,
  MAPPO, DDPG, multi-agent RL, dataset generation, or protected-data access.
- No reward/action/power/path-loss/noise/bandwidth configuration change;
  no power-control, beamforming, backhaul, buoy drift, channel variation, or
  primary M>2 experiments.
- Do not modify any Round-18 source, report, output, figure, or lock.  Do not
  create Round-20.

## Evidence and stopping rule

All generated material belongs under `round_19`.  The Round-18 lock must be
verified before a formal run; all relevant regression tests and compileall
must pass before Round-19 is locked.  Stop after the Round-19 acceptance
criteria and lock are complete.
