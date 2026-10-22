# Round-9 Working Rules

- Start with diagnostics only.  No model, optimizer, backward pass, or new
  checkpoint may be created until `round09_mechanism_gate.json` selects the
  single permitted minimal branch.
- Round-7 and Round-8 sources, locks, data, outputs, figures, and checkpoints
  are immutable provenance inputs.  Do not use Git.
- The prior development test is permanently unavailable in this round.  Use
  only the locked v2 train and validation splits.  The Round-7 untouched
  holdout remains content-isolated: only manifest/count/hash metadata are
  allowed; all trajectory, trace, state, action, reward, and plot reads must
  fail through the access guard.
- No HGNN, PPO, multi-UAV, drift, resource actions, goal randomisation,
  hyperparameter search, task/reward/queue heads, or favorable-scene filtering.
- The only valid mechanism classifications are M1, M2, M3, M4.  M3 must stop
  without training.  New artifacts belong under `round_09/`, `report/`, and
  `docs/`; do not create Round-10.
