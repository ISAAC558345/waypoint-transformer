# Round-24 Working Rules

Round-24 is a pre-registered initialization-seed extension of the locked
Round-22 multi-UAV Task-Waypoint policy protocol.  It measures the frozen
algorithm's distribution across seeds; it is not an algorithm-improvement
round.

## Immutable inputs

- `round_23/ROUND_LOCK.json` is immutable evidence.  Its approved lifecycle
  amendment may only permit this authorized successor; it does not alter the
  Round-23 lock.
- Round-22 checkpoints for seeds 0/1/2, the H=5 DeepSets predictor, the
  Round-20 dataset and its train/validation split, policy architecture,
  normalization, optimizer, sampler, loss, and closed-loop environment are
  frozen.
- Round-22 fresh-development outputs, the Round-20 development-test split,
  and Round-20 untouched evaluation must not be accessed for a rollout.

## Authorized work

- Pre-register exactly the 80-scenario `round24_seed_stability_dev_v1` fresh
  set before any new policy run, using only base seeds 24000--24009, K={3,5},
  and eta={0,.25,.5,1}.
- Train exactly seeds 3, 4, 5, 6, and 7 for each of the frozen predicted
  Task-Waypoint Transformer, matched State-only Transformer, and predicted
  Task-Waypoint MLP families: fifteen runs total.
- Perform validation diagnostics and exactly one post-freeze access to the new
  Round-24 fresh set.  It never authorizes Round-22 fresh reruns.

## Prohibited work

- Any predictor retraining/update, H-task or context change, architecture,
  loss, sampler, optimizer, learning-rate, environment, matching,
  interference, safety-projection, or reward change.
- Oracle seed extension, HGNN, PPO/MAPPO/MARL, power/beamforming/association
  actions, a seed 8 or later, reused development-test outcomes, or untouched
  evaluation.
- Round-25 creation.

## Completion

Record the exact 15/15 matrix, all frozen hashes, one new-fresh access, and
the 8-seed stability gates under `round_24`.  Stop after its lock.
