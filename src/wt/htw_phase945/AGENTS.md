# HTW Phase-9.4.5 Working Rules

Utility-aligned set supervision only.

- Do not modify HGNN/DeepSets encoders, order-free decoder, L3-B, SINR, hold,
  Task-Waypoint, or optimizer family/LR/epochs/batch except the new loss.
- No PPO. No confirmation leakage into candidate pools, utilities, tau, lambdas,
  or checkpoint selection.
- DeepSets and HGNN share one frozen candidate pool and one utility dataset.
- Checkpoint on validation candidate regret only. Confirmation payload is
  unused until after `phase945_prefinal_freeze.json`.
