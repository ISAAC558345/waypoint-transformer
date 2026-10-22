# Round-7 Working Rules

- This round is limited to Gate-B root-cause auditing and versioned offline-data construction. It must not construct a learning model, optimizer, loss, backward pass, or checkpoint.
- Round-3 through Round-6 data, splits, manifests, checkpoints, figures, reports, and locks are read-only provenance inputs. New artifacts belong only under `round_07/`.
- Use only the existing single-UAV/static-buoy environment and its documented configuration fields. Do not add UAVs, buoy drift, channel variables, AoI, deadlines, power allocation, beamforming, HGNN, or PPO.
- New behaviours must use only current/past information and must use the same environment action processing. Noisy actions are radially projected before exactly one `env.step` call.
- Every scenario comparison must use a deterministic shared snapshot and materialized arrival trace. Quality tiers are outcome-derived within difficulty class, never policy-name labels.
- Scenario splits are group-level: all behaviours from one scenario seed remain in exactly one of train, validation, test, or untouched holdout.
- `training_performed=false`; `waypoint_training_performed=false`; `transformer_training_performed=false`; `closed_loop_evaluation_performed=false`; `dataset_generation_performed=true`; `hgnn_training_performed=false`; `ppo_training_performed=false`.
- Stop after Round-7 audit/data acceptance and lock. Do not create Round-8.
