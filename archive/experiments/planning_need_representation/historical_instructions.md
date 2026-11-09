# Round-28 Working Rules

Round-28 is a diagnostic-only audit of current-observable planning-need
representations. It may analyse only the already stored Round-20 Joint-H1
train/validation trajectory states and frozen planner/policy inference. It
does not create a learned trigger, model, dataset, or scenario.

## Immutable inputs and sampling

- Verify the original Round-27 lock SHA-256
  `84B1F2993F2DA19C88874C92F33C287E174C3CD9830A7E43E78C73818B8A7267`
  and its narrow legal-successor amendment before decoding data.
- Decode only Round-20 `train` and `validation` Joint-H1 rows. Never decode
  development-test, fresh, or untouched rows.
- The frozen state sample is exactly `timestep in {0,2,4,6}` for every
  existing eligible Joint-H1 episode: 192 train states and 64 validation
  states. It is selected before utility computation and never by outcome.
- Every utility comparison uses the unchanged direct Round-27 definition:
  same reconstructed physical snapshot, H5 versus H1 first action, actual
  frozen environment transition, and identical causal H1 continuation. H10 is
  a secondary diagnostic only where ten registered steps remain (t=0).

## Allowed representation analysis

Current physical state only may supply task-preference overlap, assignment
competition, interference, queue, geometry, role/coverage, and static
capacity-pressure features. Normalization is fit on train states only.
Pre-registered equal-weight composite scores and train-derived P75/median
thresholds may be evaluated with nonparametric statistics and
scenario-cluster bootstrap. PCA is visualization only and fit on train only.

## Prohibited work

No optimizer, backward pass, neural/classifier/logistic/tree/forest/SVM,
policy/predictor/Task-Waypoint/distillation training, seed extension,
checkpoint write, environment/planner/reward/action/dataset/split change,
HGNN, PPO/MARL, fresh rollout, development-test reuse, untouched-evaluation
access, or Round-29 creation is permitted. Bridge remains diagnostic-only and
never a deployable trigger input.

## Completion

Record the expanded corpus, direct utility labels, current-only feature and
composite audits, frozen-policy relations, gates, figures, report, tests, and
an immutable lock under `round_28`; then stop. Any later trigger-model
training remains a recommendation only.
