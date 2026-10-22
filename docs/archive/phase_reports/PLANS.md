# Project Round Plan

## Round-31 (completed and locked)

Round-31 is the expressly authorized inference-only diagnosis of the locked
Round-30 state-only DeepSets-PPO result. It re-audits the implemented PPO
semantics and analyzes frozen histories/checkpoints plus current-state
train/validation diagnostics for optimization health, advantage and credit
quality, observation sufficiency, DeepSets pooling aliasing, action geometry,
and K/eta-specific failure mechanisms. It does not train, tune, mutate the
environment/reward/matching/action contract, rerun fresh, or access untouched
evaluation. Frozen Round-30 fresh outputs are read only as existing evidence.
The locked result is R31-D: implementation, numerical instability,
advantage/credit attenuation, relational aliasing, and distribution shift do
not establish a single minimal fix. PPO is already below Independent on both
registered training and validation scenarios, while K5 task scale is only
moderate evidence. The recommendation is to stop the current direct PPO
baseline branch. It generated no fresh rollout or untouched access, passed
986/986 regression tests and compileall, and did not create Round-32.

## Round-30 (completed and locked)

Round-30 is the expressly authorized Direct Multi-UAV State-Only
DeepSets-PPO Learnability Baseline. It starts from a verified locked Round-29
and tests only whether direct, on-policy, current-observable RL can stably
outperform Random Valid, Independent Myopic, and the frozen state-only BC
policy in the existing Round-19 M=2 shared-spectrum environment. The physical
environment, reward, action interface, and all completed scientific artifacts
remain frozen. Training, validation, and one-access fresh scenario sets are
new and disjoint; historical protected and untouched sets remain inaccessible.
No Task-Waypoint, planner-trigger, planner target, graph/Transformer model,
offline BC warm start, reward shaping, or Round-31 was created. Five fixed
307,200-step PPO seeds completed, validation selected their frozen checkpoints,
and the pre-frozen fresh set was accessed once. The result is PPO-LC and PBC-B:
the paired validation and fresh PPO-minus-Independent hierarchical intervals
are below zero, so this round does not establish direct RL learnability. The
evidence is locked at `round_30/ROUND_LOCK.json`; no successor is authorized.

## Round-29 (completed and locked)

Round-29 is the expressly authorized diagnostic-only Cross-Scale Deferred-Service
Pressure Normalization and Independent Confirmation Audit. It tests whether the
Round-28 K=3/K=5 DSP reversal is a one-buoy-capacity scaling artifact or a
physical task-scale mechanism. Exactly three pre-registered current-observable
normalizations retain the frozen .5/.5 spatial combination and use one
train-P75 threshold each. DSP-V3 was selected using only the Round-28 train
replay rows, then assessed exactly once on the new, disjoint, K-balanced
40-scenario confirmation set (160 states, 320 H5-versus-H1 branches). The
locked CS-C / R29-C result does not confirm a cross-scale trigger and does not
authorize trigger-model training; its recommendation is to stop the current
planner-trigger representation route. No model, policy, trigger, checkpoint,
environment, reward, planner, protected dataset, or Round-30 was created or
modified. Evidence is locked at `round_29/ROUND_LOCK.json`.

## Round-28 (completed and locked)

Round-28 is the expressly authorized, diagnostic-only audit of whether direct
same-snapshot H5-versus-H1 planner utility has stable current-state physical
representations. It uses exactly the stored Round-20 Joint-H1 train/validation
states at timesteps 0, 2, 4, and 6; fixed current-only features, train-only
normalization, equal-weight composites, train-derived thresholds, and
scenario-cluster statistics. The locked REP-B / R28-B evidence identifies
some useful representation signal, but not causal planning-need separability
or readiness for trigger-model training. No policy or trigger model was
trained, no scientific artifact or protected data changed, no fresh rollout
ran, and Round-29 was not created. Evidence is locked at
`round_28/ROUND_LOCK.json`.

## Round-27 (completed and locked)

Round-27 is the explicitly authorized, diagnostic-only audit of direct
same-snapshot H=5 planner-action communication utility relative to frozen H=1,
and of pre-registered current-physical-state trigger rules. Its locked result
is CPA-B / TR-B / R27-B: useful H=5 first actions occur (13/64), and the
oracle has validation payload headroom, but the train-selected duplicate-intent
trigger is not validation-stable. It used the fixed 64-scenario Round-20
train/validation t=0 state pool, fixed deterministic H1 continuation, no
future-derived trigger input, and a train-only threshold. No training,
scientific-artifact change, protected-data access, fresh rollout, or Round-28
was made. Evidence is locked at `round_27/ROUND_LOCK.json`.

## Round-26 (completed and locked)

Round-26 is the explicitly authorized, audit-only study of mixed-behavior
executed joint-action ambiguity, whether true or frozen-predicted
Task-Waypoints reduce local action variance, and whether the frozen causal H5
planner supplies a coherent action-distillation target. It may use only
Round-20 train/validation data, frozen inference, and current-state causal
planner calls. No training, data/environment/planner modification, fresh
rollout, development-test/untouched access, or Round-27 creation is allowed.
The locked audit yields AS-B / TD-C / PAD-B and R26-D: executed actions are
locally heterogeneous and often multimodal, but frozen true or predicted
Task-Waypoints do not materially reduce that conditional variance, while the
causal H5 planner is deterministic and replays H10 exactly but lacks the
required payload-utility relation. This is a design-only conditional
planner-action distillation readiness result, not a training authorization.
Evidence is locked at `round_26/ROUND_LOCK.json`.

## Round-25 (completed and locked)

Round-25 is the explicitly authorized, inference-and-audit-only investigation
of whether mixed Round-20 behavior trajectories supply conflicting H=5
Task-Waypoint targets for the same observable state, and whether frozen
Round-19 causal planner targets are better aligned with high-quality behavior
than the unconditional mixed target. It reads only train/validation raw data,
runs frozen predictor/policy/planner inference, and must not train, alter data
or environment semantics, read development-test or untouched data, re-run any
fresh set, or create Round-26. The locked evidence yields SA-TW-C and PT-B:
the pre-registered evidence does not establish material overall supervision
ambiguity or a general planner-derived target candidate. The registered
recommendation is to stop the current overall Task-Waypoint policy-advantage
claim; no retraining was performed. Evidence is locked at
`round_25/ROUND_LOCK.json`.

## Round-24 (completed and locked)

Round-24 completed the explicitly authorized, pre-registered seed extension
of the locked Round-22 multi-UAV Task-Waypoint policy protocol. It trained
exactly seeds 3--7 for the predicted Task-Waypoint Transformer, matched
State-only Transformer, and predicted Task-Waypoint MLP (15/15 runs), while preserving
the H=5 DeepSets predictor, Hc=10 policy protocol, Round-20 train/validation
splits, environment, and all old seeds 0--2 byte-for-byte. The new
80-configuration `round24_seed_stability_dev_v1` set is pre-registered before
training and was evaluated exactly once only after checkpoints and analysis
were frozen. Round-22 fresh development was not rerun; Round-20
development-test reuse and untouched evaluation access remain zero. The
result is MS-C / R24-C: 4/8 fresh seeds were positive, but the hierarchical
interval crosses zero and tail sensitivity prevents an overall reproducible
Task-Waypoint advantage claim. No seed 8+, architecture/loss/optimizer
modification, HGNN, or PPO/MARL was made. At its lock time Round-25 had not
been created. Evidence is locked at `round_24/ROUND_LOCK.json`.

## Round-23 (completed and locked)

Round-23 is an inference-only audit of the already locked Round-22 frozen
policy checkpoints, Round-20 train/validation development data, and the
already saved Round-22 fresh-development raw outputs. It explains seed-level
closed-loop inconsistency, Task-Waypoint head causal utilization, and the
alignment of offline action error with payload. No model, optimizer, backward
pass, checkpoint, sampler, architecture, environment, reward, matching,
projection, fresh-development rollout, or untouched multi-UAV evaluation is
permitted. It may run new paired validation-only inference diagnostics with
the frozen policies and fixed validation snapshots. It must stop after its own
lock and must not create Round-24.

## Round-22 (completed and locked)

The explicitly authorized Round-22 study freezes the Round-21 validation-only
DeepSets H=5 Task-Waypoint predictor and trains exactly twelve offline
multi-UAV movement policies: predicted Task-Waypoint Transformer, matched
State-only Transformer, Task-Waypoint MLP, and offline-only Oracle Transformer,
over seeds 0/1/2.  It uses a shared-parameter edge-free DeepSets state encoder,
Hc=10 causal state histories without action tokens, validation-only action-RMSE
selection, and the unchanged Round-19/20 safety/matching/shared-spectrum
environment.  Round-20 train/validation remain the only model data; the former
development test is not reused and Round-20 untouched evaluation remains
absolutely prohibited.  A 60-scenario disjoint fresh policy-development set is
pre-registered before training and may be accessed once only after a V-A/V-B
validation gate.  No HGNN, graph Transformer, PPO/MAPPO/MARL, predictor
training, or resource-action policy is permitted.  The round stops after its
own lock and does not create Round-23.

Current-status correction: the completed Round-22 formal matrix contains
12/12 runs, passed 843/843 regression tests and compileall, and has final gate
MU-WP-B. The fresh development set was accessed exactly once after V-B; the
Round-20 development test was not reused, untouched multi-UAV evaluation
remains unaccessed, and `candidate_ready_for_untouched_eval=false`. Evidence
is locked at `round_22/ROUND_LOCK.json`; Round-23 has not been created.

## Round-21 (completed and locked)

The explicitly authorized supervised learning follow-on to locked Round-20
trained only a fixed two-layer heterogeneous spatial graph encoder and
Task-Waypoint predictor, comparing it with DeepSets at H=5/H=10 over three
seeds.  Validation-only selection is followed by the fixed selected-horizon
no-UU and no-interference-feature ablations.  The Round-20 datasets, labels,
environment and protected untouched evaluation set remain read-only; exactly
one guarded development-test access is reserved until after selection is
frozen.  No action policy, Transformer, PPO/MAPPO/MARL, resource policy, or
closed-loop neural evaluation occurred.  All 18 fixed runs, the one guarded
development-test access, 802 regression tests and compileall passed; Round-20
had 86/86 registered hashes before and after.  The H=5 selection yielded
TWG-A but GV-C/UU-C/IF-C, so policy-stage readiness is false. Evidence is
locked at `round_21/ROUND_LOCK.json`; no Round-22 exists.

## Round-20 (completed and locked)

The explicitly authorized follow-on to the locked Round-19 IPN-A result.  It
constructs a fixed 80-scenario, four-tier M=2 shared-spectrum development
dataset with deterministic mixed-quality behavior sources, scenario-level
60/20/20 isolation, graph-ready current-observable state, and H=5/H=10
multi-UAV Task-Waypoint supervision.  The Round-19 physical environment,
matching, safety projection, and frozen planners are reused without changes.
No neural training, HGNN, PPO/MARL, protected historical-set access, or
rollout from the newly pre-registered untouched-evaluation set is allowed.
Round-21 was subsequently opened by an explicit instruction; this locked
Round-20 description remains historical evidence.
The completed formal evidence contains 80 scenarios, 800 episodes, 9,600
transitions, 6,400 H5 and 2,400 H10 labels, 20 PNG + 20 SVG figures, and a
759/759 final regression. All three data-readiness gates are true. Evidence is
locked at `round_20/ROUND_LOCK.json`.

## Round-19 (completed and locked)

Shared-spectrum M=2 buoy-uplink interference coordination/planning
revalidation.  It begins from the verified immutable Round-18 lock and adds
only cross-buoy uplink interference under frozen eta={0.25,0.50,1.00}, with an
exact orthogonal control.  It is deterministic and non-learning: no HGNN,
PPO, MARL, power control, backhaul, protected data, or Round-20 work is
permitted until its registered gates and lock are complete. The exact
orthogonal and M=1 controls passed. ICN-B did not establish immediate
coordination, while the low shared tier established IPN-A (H10-H1 0.322 Mbit,
95% CI [0.075, 0.600]), realistic planning headroom, and
`multi_uav_shared_spectrum_offline_dataset_ready=true`. The coupling-v2 gate
remains MCI-C because immediate coordination is not stable. Evidence is locked
at `round_19/ROUND_LOCK.json`; Round-20 was later opened by explicit user
authorization.

## Round-17 (completed and locked)

Predictable time-varying channel planning revalidation completed with a
current-observable slow multiplier and static buoy drift in every primary
scenario. The 48-row static control was exact and the 576-episode formal audit
returned CPN-C: no established realistic causal long-horizon planning headroom.
The evidence is locked in `round_17/ROUND_LOCK.json`.

## Round-18 (completed and locked)

Minimal multi-UAV cooperative maritime-environment and long-horizon
coordination-necessity audit. This non-learning round adds only an M=2 primary
task coupling: a shared buoy queue, exclusive deterministic matching, and
hard UAV-UAV separation. Both buoy drift and channel variation are static in
primary experiments. The 42-scenario/168-episode audit returned CN-B, MPN-C,
and MC-C; it does not authorize training or Round-19. Evidence is locked in
`round_18/ROUND_LOCK.json`.

## Round-15 (completed and locked)

Long-Horizon Planning Necessity and Hierarchical-Control Headroom Audit.
This deterministic, non-learning round compares current-state myopic rules
against a pre-registered finite action-lattice lookahead at H={1,5,10,20} on
new Round-15 diagnostic scenarios.  It separates a causal current-arrival
persistence planner from a non-deployable oracle-trace upper bound. A
metadata-only pre-initialization holdout-report read is recorded as one access
and excluded from all computation. The paired actual-payload result is PN-C:
long-horizon planning was not required under this audit. The locked output is
`round_15/outputs/round15_planning_headroom_audit_20260825_1245`; no further
protected data may be opened and no Round-16 work is authorized.
The round stops at PN-A/PN-B/PN-C and must not create Round-16.

## Round-14 (completed and locked)

Task-Waypoint-to-Motion Mapping Redesign and Actionable Intermediate Guidance
Audit.  The scope is a deterministic, inference-only mapping from the frozen
H=10 task waypoint `[service, gross progress, H-step return]` plus current
valid geometry/rates to the existing two-dimensional movement action.  It
will measure teacher alignment on permitted train/validation data, assess
future-outcome relevance as an offline diagnostic, and run paired,
same-snapshot movement counterfactuals with a fixed `queue_rate_energy`
continuation.  No model/policy training, Spatial Waypoint work, service action
extension, environment/reward/resource modification, old-development-test,
or untouched-holdout access is allowed.  The round stopped at MAP-B: C had
the strongest validation teacher-action cosine (0.332), but its paired payload
interval against `queue_rate_energy` crossed zero.  It is therefore
diagnostic-only; no future policy round was created.  Evidence is locked at
`round_14/ROUND_LOCK.json`.

## Round-13 (completed and locked)

Evaluation-only Task-Waypoint action-controllability audit and minimal
service-intent action design. No neural model, policy, PPO, HGNN, or other
optimizer may be trained. The audit is limited to train/validation development
scenarios plus, if needed, explicitly marked diagnostic-only scenarios; old
development test and untouched holdout remain prohibited. Its goal is to
measure whether the three Task-Waypoint heads are controllable by the existing
two-dimensional movement action and whether a physically bounded explicit
service-intent action has diagnostic headroom. It must stop after its own
acceptance gate and must not create Round-14.

Current-status update: Round-13 is completed and locked at
`round_13/ROUND_LOCK.json`.  Its non-destructive completion amendment is
tracked separately by `round_13/ROUND_LOCK_AMENDMENT.json`; it does not alter
the original formal result or create Round-14.

## Round-7 (completed)

Gate-B diagnosis and `maritime_offline_dataset_v2` construction completed
without model training. The output has 60 scenario configurations, 720 legal
episodes, outcome-derived 40/40/20 high/medium/low quality tiers, seed-level
development/holdout isolation, explicit stitching/goal/horizon audits, and
H=10/H=20 retraining candidates. See
`report/round_07_report_20260809_1810.md`. No Round-8 was created.

## round_01 — completed and locked

Minimal single-UAV, static multi-buoy maritime communication environment:
configuration validation, deterministic large-scale channel, normalized state,
continuous motion, boundary handling, placeholder energy, reward decomposition,
smoke rollout, and 45 environment tests. Full pre-round-02 regression: 78/78.

## round_02 — completed and locked

Completed:

- four reproducible non-learning baselines;
- a shared evaluation and collection runner;
- semantically structured episode trajectory storage;
- episode-level train/validation/test splitting;
- K-step availability and leakage audit;
- baseline metrics, data-quality reports, and research figures;
- small, bounded smoke collection over multiple seeds and buoy counts.

Explicitly excluded: queue/environment redesign, offline RL training, Waypoint
Network, Transformer, Decision Transformer, behavior cloning, optimizers, and
all other learning models.

Acceptance evidence:

- 100/100 repository tests passed;
- canonical smoke run: 24 episodes and 192 transitions;
- train/validation/test each contain 8 episodes with no group leakage;
- exact reload, checksum, continuity, finite-value, and K-step audits passed;
- six figures were generated and visually checked;
- `training_performed=false`.

The smoke data validates the pipeline and `K=1,5` sample availability. It is
not sufficient for substantive Waypoint sample construction because eight-step
episodes yield no `K=10,20,30` samples and the current environment has no
desired goal, queue, urgency, task completion, or actual payload.

At its lock time no next round had been created. Round 03 was later opened by
an explicit user instruction.

## round_03 — completed and locked

Completed:

- opt-in task-semantics v1 with real per-buoy queues, reproducible arrivals,
  queue-bounded payload, task completion, desired/achieved goals, terminal
  reasons, and truthful reward components;
- four behaviorally distinct non-learning baselines:
  `random_valid`, `nearest_buoy`, `max_queue`, and `queue_rate_energy`;
- shared materialized arrival traces for fair cross-policy comparison;
- checksum-protected schema-v1 trajectories and deterministic
  train/validation/test grouping;
- padded and masked Waypoint samples for `K={1,5,10,20,30}`;
- 22-item data-quality/readiness audit and 12 visually checked figures.

Acceptance evidence:

- 146/146 repository tests and 46/46 focused Round-3 tests passed;
- canonical smoke: 24 episodes, 192 transitions;
- canonical pilot: 80 episodes, 6,604 transitions, 28,140 Waypoint samples;
- split episodes/transitions:
  train 48/3,910, validation 16/1,279, test 16/1,415;
- valid K samples: 6,604 / 6,284 / 5,884 / 5,084 / 4,284;
- exact reload/checksums/source-target recomputation passed;
- all 22 quality checks passed, with 55 high-quality and 25 mid-quality
  episodes, no duplicate trajectories, and no boundary or integrity errors;
- `training_performed=false` and no learning model was constructed.

Round-01 and Round-02 locks and generated artifacts remain unchanged. No next
round has been created; work stops pending explicit human authorization.

## round_04 — completed and locked

Completed:

- read-only, checksum-bound adaptation of the locked Round-3 pilot without
  regenerating or re-splitting it;
- two-dimensional physical relative-displacement targets for
  `H={1,5,10,20,30}`;
- 15 goal-conditioned and 15 matched state-only spatial MLP training runs for
  seeds `{0,1,2}`;
- zero, train-mean, and causal constant-velocity non-learning baselines;
- validation-only checkpoint and horizon selection;
- grouped raw/projected offline evaluation, strict provenance, resumable
  checkpoints, CSV metrics, and scientific figures.

Acceptance evidence:

- 56/56 focused and 202/202 repository tests passed; compileall passed;
- final canonical formal run:
  `round04_formal_canonical_20260802_173717`;
- 30/30 training runs completed and 60/60 best/last checkpoints strictly
  reloaded with current source/config/tensor provenance;
- validation selected H=30 without test data;
- selected-H test RMSE is 44.281 m for goal-conditioned, 49.006 m for
  state-only, and 52.564 m for zero displacement;
- 72 CSV files/10,415 rows contained no NaN/Inf;
- all 36 PNG/SVG figures passed visual and semantic QA;
- Round-3 lock, data, implementation, outputs, and figures remained unchanged.

Explicitly excluded and not performed: Transformer or Decision Transformer
training, action-policy training, behavior cloning, reinforcement learning,
closed-loop evaluation, HGNN, multiple UAVs, moving buoys, and
communication-resource actions.

The result demonstrates conditional offline supervised learnability, not
policy improvement. Hindsight-goal availability, policy-dependent errors,
split shift, correlated samples, and absence of low-quality trajectories
remain material limitations. No next round has been created; later work
requires explicit human authorization.

## round_05 — completed and locked

Completed:

- froze the validation-selected Round-4 H=30 spatial Waypoint Network and
  materialized provenance-bound predicted-waypoint caches for all locked splits;
- trained the exact 27-run causal two-dimensional offline policy matrix with
  no historical action tokens: WT, Direct-Goal, State-only, Oracle diagnostic,
  and two one-step MLP controls;
- selected WT `H_c=10`, seed 0, epoch 7 strictly on validation RMSE, then
  evaluated all 27 frozen selections on test exactly once;
- added policy-balanced sampling, raw/projected action metrics, grouped
  evaluation, recorded-state replay diagnostics, and a five-step disposable
  environment interface smoke;
- fixed float32 boundary-validity accounting and seeded model construction
  before parameter initialization; independent seed-reproducibility smoke runs
  produced bitwise-identical model state, history, and predictions.

Acceptance evidence:

- 247/247 repository tests and compileall passed after the final seed fix;
- formal canonical output: `round05_formal_canonical_20260803_1500`;
- 27/27 training runs and 27/27 test-once evaluations completed; 54/54
  best/last checkpoints carry optimizer, sampler, RNG, resume, and provenance;
- 27/27 best checkpoints reloaded with identical predictions; no CSV/JSON/NPZ
  NaN/Inf; target and projected action validity are 1.0;
- 16 PNG plus 16 SVG figures passed strict semantic and visual QA;
- Round-3 data/provenance and the Round-4 lock/checkpoint remained unchanged.

The selected WT test RMSE is 7.772 m. Direct-Goal, State-only, and Oracle
Hc=10 controls have closely competing test means (7.949, 7.975, and 7.905 m),
so this round does not establish a stable Waypoint advantage. It establishes
only offline hindsight-conditioned action prediction and a bounded interface.

Explicitly excluded and not performed: formal closed-loop return ranking, PPO
or online fine-tuning, Round-4 retraining, HGNN, multi-UAV, moving-buoy, and
communication-resource action extensions. No next round has been created.
## Round-6 (completed)

Frozen-model, paired reference-goal closed-loop evaluation completed on 60
new scenarios. Gate B: no reliable WT advantage over Direct-Goal; stop before
any HGNN/multi-UAV/PPO expansion. See `report/round_06_report_20260809_1600.md`.

Round-23 completed the frozen checkpoint audit with no training, no new fresh-development rollout, and no untouched multi-UAV evaluation. Evidence is locked at `round_23/ROUND_LOCK.json`; Round-24 has not been created.

## Round-32 (in progress)

Round-32 is an authorized read-only cross-round audit of episode, planner,
Task-Waypoint, and PPO effective-future semantics. It follows the byte-stable
Round-31 lock from `backup/snapshots/before_round_32_20260830_2231`. No model,
environment, planner, dataset, rollout, fresh evaluation, or protected
development-test record may be created or changed; work stops after Round-32.

Round-32 completed and locked with EH-C / R32-C. It corrects the conflicting
five-step premise to the locked twelve-step fact, preserves historical gates,
and recommends only a future benchmark-design interval of 11--20 slots. No
Round-33 was created.

## Round-33 (completed and locked)

Round-33 follows the immutable Round-32 lock after explicit authorization. It
created only a new T=20, M=2 physical benchmark and a non-learning H10-vs-H5
planner-separation validation. The sole planner exception was H=20 membership
in the existing joint finite-beam search; it remained descriptive only, while
H1/H5/H10 passed exact T=12 compatibility regression. The 64-scenario design
and 64-scenario confirmation populations produced HS-C and BENCH-C: H10-vs-H5
first-action and same-snapshot utility differences are both exactly zero, so
Task-Waypoint re-entry remains false. The governance compatibility repair
preserved the immutable Round-32 lock, passed 1040/1040 repository tests and
compileall, and Round-33 is locked. Its original lock remains immutable;
Round-34 proceeds only through a separately recorded successor amendment.

## Round-34 (completed and locked)

Round-34 is a no-training, no-environment-execution audit of benchmark
complexity, relational-decision ownership, and method suitability. A single
accidental `development_test` preflight decode is recorded in
`round_34/ROUND34_PROTECTED_ACCESS_EXCEPTION.json`; it is excluded from every
scientific output. Formal analysis restarts only from hash-locked clean
train/validation manifests and must stop on any second protected decode. The
audit returned BCX-B / R34-B and is locked after 1,087 regression tests and
compileall passed. It may recommend a future relational benchmark design but
cannot implement it, train any method, or create Round-35.

## Round-35 (in progress)

Round-35 is an explicitly authorized, append-only successor to the immutable
Round-34 lock. It implements only the Level-2 M=4/K=8/T=20 relational
shared-spectrum environment and an external association-preference action
under exact feasibility decoding. The historical Round-34 prohibition remains
its lock-time fact; the successor is authorized solely by
`round_34/ROUND_LOCK_AMENDMENT.json`. No model training or inference,
protected-data access, reward/physics redesign, planner work, Level-3, or
Round-36 is allowed.

Round-35 completed and locked with RAO-A / L2-A. It passed 38 dedicated tests,
1,125/1,125 full regression tests, and compileall. Its external association
preference causally controls exact feasible assignment, while Task-Waypoint
re-entry remains false and no HGNN performance claim is made. No Round-36 has
been created.

## Round-36 (in progress)

Round-36 is the explicitly authorized, append-only successor of the immutable
Round-35 lock, recorded in `round_35/ROUND_LOCK_AMENDMENT.json`. It is a
non-learning Level-2 M=4/K=8/T=20 exact first-action association audit with
64 frozen design and 64 frozen confirmation scenarios, anchors {0,8}, and all
3,393 legal assignments. Formal scientific execution uses exactly four workers;
candidate pruning, approximation, protected or untouched access, model
inference, training, and Round-37 are prohibited.
