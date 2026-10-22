# Round-8 training-closure audit

## Scope and provenance

Round-8 trains only the existing spatial two-dimensional Waypoint MLP and the
existing causal two-dimensional Transformer on Round-7 v2 development data.
The locked input is `round07_formal_20260809_1740`; no Round-7 file was
modified.  The protected holdout directory has an independent guard and its
final audit records `access_count=0`, `content_read=false`, and an unchanged
tree hash.

## Goal and data flow

`goal_schema_v2` is exactly `[target_task_completion_ratio,
maximum_allowed_residual_queue_ratio, minimum_terminal_energy_ratio]`.  It is
read from the pre-rollout desired-goal field.  Future/terminal positions,
actions, rewards, payloads, queues, hindsight goals, and outcome-derived
quality/policy/difficulty labels are excluded from model inputs.  States are
padded dynamically to `9 + 7 * max_num_buoys = 44`; node masks are width 5;
Waypoint outputs and actions are both two-dimensional.

The train/validation/development-test split is the locked R7 scenario split.
Sampling is scenario/trajectory balanced; it is not reward- or quality-
weighted.  All policy histories use context 10 and omit historical actions.

## Training and selection

The matrix contains 6 Waypoint runs (H=10,20 x 3 seeds) and 18 Transformer
runs (WT-H10, WT-H20, DirectGoal, StateOnly, Oracle-H10, Oracle-H20 x 3
seeds).  Waypoint selection is validation endpoint RMSE; policy selection is
validation action RMSE.  Training uses maximum 300 epochs, a 256-sample
scenario-balanced batch, Adam/AdamW, gradient clipping 1.0, and patience 30
validation points.  Full validation is performed every 10 training epochs.

Closed-loop selection uses rolling state -> frozen Waypoint -> causal policy
-> radial projection -> exactly one environment step.  It selected WT-H20 on
validation.  Only after that decision was frozen was development test run,
exactly once, for selected WT/Direct/State/Oracle models.

## Result and gate

Validation mean payloads (bits) were WT-H20 189,728,129; DirectGoal
174,638,781; StateOnly 188,782,407.  The paired WT-H20 minus DirectGoal
payload bootstrap CIs by model seed are respectively
[-32,762,708, 22,506,691], [-28,571,409, 43,742,780], and
[13,262,384, 65,724,826].  Hence only one of three lower bounds is positive.
The pre-registered result is **Gate B**: no reliable deployable long-horizon
communication gain.  `candidate_ready_for_next_phase=false`; no holdout
evaluation or Round-9 work is authorized.

## Files changed and verification

Added `src/wt/round8/`, `experiments/run_round08.py`,
`experiments/finalize_round08.py`, `tests/test_round08_isolation.py`,
`round_08/`, this audit, and the completion report.  Existing Round-3--7
sources and artifacts are unchanged.  Commands run:

```powershell
$env:PYTHONPATH = 'src'
& '..\decision-transformer-master\.venv-dt-gym\python.exe' -m unittest discover -s tests -p 'test_*.py' -q
& '..\decision-transformer-master\.venv-dt-gym\python.exe' -m compileall -q src experiments tests
& '..\decision-transformer-master\.venv-dt-gym\python.exe' -m wt.round8.round8_runner
& '..\decision-transformer-master\.venv-dt-gym\python.exe' -m wt.round8.finalize --root .
```

The final suite is 307/307 PASS; the Round-8 focused suite is 47/47 PASS.
