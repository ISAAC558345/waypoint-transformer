# Round Versioning

This repository uses filesystem snapshots and manifests rather than Git for
round management.

## Round-31 (completed and locked)

Round-31 begins from `backup/snapshots/before_round_31_20260830_2116` after
the locked Round-30 source, report, output, figure, checkpoint, and lock
hashes were reverified. It is a frozen-artifact and inference-only failure
diagnosis of the direct state-only DeepSets-PPO baseline. It may read the
formal histories/checkpoints and run current-state diagnostics only on the
registered Round-30 train/validation populations; it may read, but never
rerun, the one-access Round-30 fresh outputs. No optimizer/backward/training,
new seed, hyperparameter search, reward/environment/matching/action change,
HGNN, Transformer actor, Task-Waypoint, or protected-data access is allowed.
It preserved the Round-30 lock byte-for-byte, recorded zero new fresh and
untouched rollouts, passed 986/986 regression tests and compileall, and is
locked at `round_31/ROUND_LOCK.json`. The R31-D conclusion stops the current
direct PPO baseline branch and does not create Round-32.

## Round-30 (completed and locked)

Round-30 begins from `backup/snapshots/before_round_30_20260830_1844` after
the locked Round-29 evidence and historical immutable state are verified. It
is a new, strictly state-only DeepSets-PPO learnability baseline in the
unchanged Round-19 M=2 shared-spectrum environment. Its new training,
validation, and fresh scenario sets must be manifest-frozen and mutually
disjoint before formal training; the fresh set has one deferred access and
historical untouched evaluation remains inaccessible. It may not reintroduce
Task-Waypoints, planner triggers/targets, graph or Transformer policies,
offline BC initialization, reward/action/environment changes, or Round-31.
The fixed five-seed 307,200-step study completed with one fresh confirmation
access and zero untouched-evaluation access. Its frozen result is PPO-LC and
PBC-B: current evidence does not establish direct RL learnability, and the
only recommendation is future PPO stability diagnosis under separately
authorized work. Evidence is locked at `round_30/ROUND_LOCK.json`.

## Round-29 (completed and locked)

Round-29 begins from `backup/snapshots/before_round_29_20260830_1800` after
the Round-28 immutable evidence is verified. It is limited to a current-state
cross-scale DSP normalization audit and a single new, manifest-frozen,
K-balanced confirmation diagnostic. Its locked CS-C result does not establish
a cross-scale trigger or authorize training; the final recommendation is
`R29-C_stop_planner_trigger_representation_route`. The run used 256 frozen
Round-28 replay states and one pre-registered 40-scenario confirmation access
(160 states; 320 H5-versus-H1 branches), with no training, physical-system or
protected-data modification, or Round-30 creation. Evidence is locked at
`round_29/ROUND_LOCK.json`.

## Round-28 (completed and locked)

Round-28 begins from `backup/snapshots/before_round_28_20260830_1624` after
the locked Round-27 evidence and its legal-successor lifecycle amendment are
verified. It is a pure current-state representation and causal-separability
audit on fixed stored Joint-H1 Round-20 train/validation states. Its locked
REP-B / R28-B result finds useful current-state signals but not the stable,
causally separable representation required for trigger-model training. No
training, scientific-artifact modification, protected-data access, or fresh
rollout occurred at lock time. Evidence is locked at
`round_28/ROUND_LOCK.json`.

## Round-27 (completed and locked)

Round-27 begins from `backup/snapshots/before_round_27_20260830_1447` after
the locked Round-26 evidence and its narrow legal-successor lifecycle
amendment were verified. Its pure same-snapshot diagnostic of direct planner
versus H1 communication utility and fixed causal trigger rules on Round-20
train/validation states is locked at `round_27/ROUND_LOCK.json`. The result
is CPA-B / TR-B / R27-B: the observed utility and oracle headroom do not
establish a stable deployable trigger, so it authorizes no training. No
scientific artifacts changed, protected data were accessed, fresh rollouts
were generated, or Round-28 was created.

## Round-26 (completed and locked)

Round-26 begins from `backup/snapshots/before_round_26_20260830_1322` after
the locked Round-25 evidence and its narrowly scoped legal-successor lifecycle
amendment were verified. It is a pure diagnostic of physically executed joint
actions, frozen Task-Waypoint/policy inference, and current-state causal H5
planner actions on frozen Round-20 train/validation data. It cannot train,
modify any scientific artifact, access protected data, generate fresh rollouts,
or create Round-27. Its registered outcome is AS-B / TD-C / PAD-B and R26-D:
the result permits only future design discussion of conditional
planner-action distillation; it does not permit training. Evidence is locked
at `round_26/ROUND_LOCK.json`.

## Round-25 (completed and locked)

Round-25 begins from `backup/snapshots/before_round_25_20260830_1203` after
the locked Round-24 evidence and its narrowly scoped legal-successor lifecycle
amendment were verified. It is an audit-only test of supervision ambiguity and
causal planner-target alignment on the frozen Round-20 train/validation data.
It cannot train or modify a model, policy, dataset, label, split, environment,
reward, matching, or interference rule; it cannot access development-test,
fresh outcomes, or untouched evaluation; and it cannot create Round-26. Its
locked SA-TW-C / PT-B diagnostic does not authorize retraining or a general
planner-derived supervision target; the formal recommendation is to stop the
current overall Task-Waypoint policy-advantage claim. Evidence is locked at
`round_25/ROUND_LOCK.json`.

## Round-24 (completed and locked)

Round-24 began from `backup/snapshots/before_round_24_20260829_1515` after
the locked Round-23 evidence and its narrowly scoped lifecycle-successor
amendment were verified. It is a pre-registered, pure initialization-seed
extension: exactly seeds 3--7 were trained for three frozen deployable
families, giving 15 new runs. The new 80-scenario fresh stability set was
registered before training and accessed exactly once only after the analysis
protocol was frozen. No old checkpoint was retrained, no protected Round-22
fresh or Round-20 development-test result was reused, and untouched evaluation
remained inaccessible. The locked MS-C / R24-C result does not support an
overall reproducible advantage; Round-25 had not been created at lock time.

## Lifecycle

1. Verify the current implementation and run its tests.
2. Create `backup/snapshots/before_round_XX_YYYYMMDD_HHMM/`.
3. Copy source, configuration, tests, round-control files, README, plans, and
   documentation. Exclude caches, figures, generated datasets, and temporary
   outputs.
4. Create `round_XX/AGENTS.md`, update `CURRENT_ROUND.txt`, and set
   `round_manifest.json` to `in_progress`.
5. Keep all generated artifacts inside the active round's directories.
6. After acceptance, write the round report, record checksums and outputs, set
   the manifest status to `completed`, and create `round_XX/ROUND_LOCK.json`.
7. A completed lock is append-only evidence. Do not silently rewrite it.

## Round-33 (completed and locked)

Round-33 begins from `backup/snapshots/before_round_33_20260830_2320` after
the locked Round-32 evidence was rehashed. It is a non-learning T=20
long-episode shared-spectrum M=2 benchmark. The only permitted physical degree
of freedom is `episode_horizon`; T=12 is retained for an exact compatibility
regression. An explicit human authorization adds H=20 to the existing joint
beam planner without changing its candidates, scoring, transitions, matching,
beam width, or tie breaks. H10-vs-H5 is primary, H5-vs-H1 secondary, and H20
descriptive only. New design and confirmation populations are manifest-frozen
and disjoint before outcome access. No model training, Task-Waypoint/HGNN use,
protected data, or Round-34 creation is permitted. The T=12 regression passed
24/24 rows, and the frozen 64-scenario design plus 64-scenario confirmation
benchmark returned HS-C / BENCH-C: H10-versus-H5 first-action and
same-snapshot utility differences are both exactly zero, so
`h10_specific_planning_value_confirmed=false` and Task-Waypoint re-entry is
false. Its narrowly scoped governance compatibility repair preserves the
byte-stable Round-32 original lock and lifecycle amendment, protects all
Round-32 scientific evidence, and rejects Round-34. The Round-33 lock records
1040/1040 full regression tests, 30/30 dedicated tests, compileall, and zero
development-test or untouched-evaluation access.

## Numbering

- `round_01`: completed minimal maritime environment (stage 1A).
- `round_02`: non-learning baselines, offline trajectories, and quality audit.
- `round_03`: completed queue/arrival/payload task semantics, corrected
  non-learning baselines, schema-v1 trajectories/goals, and masked Waypoint
  sample construction.
- `round_04`: authorized supervised two-dimensional spatial Waypoint Network
  training, validation-only fixed-horizon selection, and offline prediction
  evaluation using the locked Round-3 pilot; completed and locked with H=30
  as the validation-selected interface.
- `round_05`: in progress; freezes the validation-selected Round-4 H=30
  Waypoint Network and trains causal two-dimensional offline action predictors
  with context-length and ablation controls. Formal closed-loop policy ranking
  and reinforcement learning are outside this round.
- Later rounds must use the next integer only after explicit authorization.

## Snapshot restoration

Restoration is a deliberate manual operation. Never delete current work before
the snapshot has been inspected and the exact restoration scope approved.

## Round-19 (completed and locked)

Round-19 begins from `backup/snapshots/before_round_19_20260828_1252` after
the Round-18 lock and all registered source/output hashes passed verification.
It is an isolated, deterministic shared-spectrum buoy-uplink interference
revalidation with exact Round-18 orthogonal control.  No neural training,
optimizer, HGNN, PPO/MARL, protected-data access, or Round-20 creation is
permitted. Its active evidence is confined to `round_19`. The exact
orthogonal/M=1 controls passed and the frozen low shared tier met IPN-A, so
the locked result sets realistic planning headroom and shared-spectrum offline
dataset readiness true. ICN-B remains insufficient for an immediate
coordination claim and coupling-v2 is MCI-C.

## Round-20 (completed and locked)

Round-20 begins from `backup/snapshots/before_round_20_20260828_1849` after
the complete Round-19 lock, configuration, source, and formal-output hash
audit passed.  It may generate only new, pre-registered M=2/K={3,5}
shared-spectrum *development* trajectories and their offline graph and
Task-Waypoint labels.  It preserves static buoy/channel conditions, fixed
power/bandwidth, shared queues, exclusive matching, 15 m safety, and the
frozen deterministic Round-19 planners.  It must use equal scenario counts
for eta={0,.25,.5,1}, scenario-level split isolation, outcome-derived quality
tiers, no protected historical access, and zero policy rollouts from
`round20_multi_uav_untouched_eval_v1`.  No neural optimizer/training, HGNN,
PPO/MARL, resource/action extension, or Round-21 creation is permitted. The
formal raw trajectories contain 80 scenarios, 800 episodes, and 9,600
transitions; all outcome-quality, graph-ready, and Task-Waypoint readiness
gates passed, as did 759/759 final regression tests and compileall. A
reporting-only recovery deserialized the completed raw JSONL files after an
initial figure rendering field omission; it did not invoke an environment,
planner, or rollout. The unchanged 15 m separation was audited with a 1e-5 m
floating-point tolerance. The immutable evidence lock is
`round_20/ROUND_LOCK.json`; it remained immutable when Round-21 was later
opened by explicit authorization.

## Round-21 (completed and locked)

Round-21 begins from `backup/snapshots/before_round_21_20260828_2142` after
the Round-20 lock's 86 registered hashes passed exact re-verification.  Its
sole scope is supervised single-timestep HGNN Spatial Encoder and Multi-UAV
Task-Waypoint Predictor training, a parameter-fair DeepSets baseline, and the
pre-registered no-UU/no-interference ablations.  Round-20 data and physics are
read-only; raw graph identity arrays and all metadata/future fields are
adapter-excluded.  Development-test samples are guarded until validation-only
architecture/horizon/checkpoint selection is frozen, then evaluated once.
The untouched multi-UAV evaluation data was not opened or modeled.  The fixed
18-run study selected H=5 strictly on validation, then performed the one
registered development-test evaluation.  It returned TWG-A but GV-C, UU-C and
IF-C; policy-stage readiness is therefore false.  The immutable evidence lock
is `round_21/ROUND_LOCK.json`.  No Transformer/policy/RL/closed-loop work and
no Round-22 were created.

## Round-13 (completed and locked)

Round-13 begins from `backup/snapshots/before_round_13_20260811_0025` after
the completed Round-12 lock was rehashed. It is an audit/design-only round:
no model or policy training, old-development-test access, untouched-holdout
access, service-action implementation, or Round-14 creation is permitted.
The original formal evidence is locked at `round_13/ROUND_LOCK.json`.
Any completion amendment must be additive, preserve that lock and every
original formal artifact, and must not create Round-14.

## Round-14 (completed and locked)

Round-14 begins from `backup/snapshots/before_round_14_20260825_1149` after
the Round-13 original lock and its additive completion amendment were
verified.  It is an inference-and-audit-only redesign of the mapping from the
frozen H=10 Task-Waypoint heads and current observable geometry to the
existing two-dimensional movement action.  Its formal output is
`round_14/outputs/round14_task_to_motion_mapping_20260825_1205` and its
immutable evidence lock is `round_14/ROUND_LOCK.json`.  The mapping gate is
MAP-B: the selected C mapping's paired payload interval crosses zero, so it
remains diagnostic-only and does not authorize a future policy round.  No
neural or policy training, environment/resource/reward change,
old-development-test or untouched-holdout access occurred, and Round-15 was
not created.

## Round-15 (completed and locked)

Round-15 begins from `backup/snapshots/before_round_15_20260825_1228` after
the Round-14 lock and every protected output hash passed verification.  It is
a deterministic audit of whether the minimal environment has long-horizon
planning headroom.  It creates no neural model, checkpoint, or environment
change. Its causal planner uses only the arrival expectation implied by the
current environment configuration, while its separately marked oracle is a
non-deployable diagnostic upper bound. A metadata-only pre-initialization
holdout report read is recorded as count one and excluded from every Round-15
computation. The formal output is
`round_15/outputs/round15_planning_headroom_audit_20260825_1245`; its lock is
`round_15/ROUND_LOCK.json`. The PN-C result finds no evidence that long-horizon
planning is required in this environment under the registered diagnostic.
No further protected-data access or Round-16 creation is permitted.

## Round-16 (completed and locked)

Round-16 begins from `backup/snapshots/before_round_16_20260826_2047` after
the Round-15 formal-output hashes and prior lock chain passed verification. It
is an environment-extension and deterministic revalidation round limited to
observable, predictable two-dimensional buoy drift. No training, neural policy,
resource/reward/action change, multi-UAV, HGNN, or PPO is permitted. No new
holdout access of any kind is permitted; the new access count is zero. The
static same-seed/same-action control passed all 48 exact regression rows. The
formal 60-scenario diagnostic finds Dynamic Planning Necessity Gate DPN-C:
observable drift changes decisions but does not yield stable, causal H=10/H=20
payload headroom at realistic low or medium drift. No training, data-generation
for training, or Round-17 work is authorized. The immutable evidence lock is
`round_16/ROUND_LOCK.json`.

## Round-17 (completed and locked)

Round-17 begins from `backup/snapshots/before_round_17_20260827_1916` after
the Round-16 lock and every protected source/output/figure hash passed
verification. It is limited to a bounded, current-observable, predictable
slow channel multiplier. Primary experiments keep buoy drift static, preserve
the base path-loss and rate equations, and use the frozen Round-15/16 planner
except for a channel-state transition model. No training, protected-data
access, reward/action/resource change, HGNN, PPO, multi-UAV, or Round-18 is
permitted. The static same-seed/same-action control passed all 48 exact rows,
and the formal 576-episode audit found Channel Planning Necessity Gate CPN-C:
under the registered realistic slow-channel tiers, it does not establish stable
causal H=10/H=20 planning headroom. The immutable evidence lock is
`round_17/ROUND_LOCK.json`. No Round-18 has been created or authorized.

## Round-18 (in progress)

Round-18 begins from `backup/snapshots/before_round_18_20260827_2030` after
the Round-17 evidence lock was rehashed. It is a deterministic minimal
multi-UAV task-coupling and coordination-necessity audit. Primary scenarios
use two UAVs with static buoys and static channels; the only added couplings
are a shared buoy queue, exclusive deterministic matching, and hard UAV-UAV
safety separation. No training, protected-data access, reward/resource/action
extension, HGNN, PPO, or Round-19 is permitted.

## Round-18 (completed and locked)

Round-18 completed the minimal configurable multi-UAV task coupling with M=1
exact static regression and M=2 primary scenarios. The 42-scenario,
168-episode audit found CN-B, MPN-C, and MC-C: immediate coordination is not
stable, and long-horizon headroom is not stable across registered scenario
groups. The immutable evidence lock is `round_18/ROUND_LOCK.json`. No training
or protected-data access occurred and no Round-19 was created.

## Round-22 (completed and locked)

Round-22 completed the authorized fixed 12-run offline multi-UAV policy
matrix while preserving the frozen H=5 DeepSets Task-Waypoint predictor. Its
validation gate was V-B and its one permitted fresh-development access was
performed exactly once. The final result is MU-WP-B with
`candidate_ready_for_untouched_eval=false`; Round-20 development-test reuse
and untouched multi-UAV evaluation access both remain zero. The immutable
scientific evidence is `round_22/ROUND_LOCK.json`; Round-23 has not been
created. This completion entry corrects stale root-governance lifecycle text
without changing Round-22 scientific evidence.

## Round-23 (completed and locked)

Round-23 begins from `backup/snapshots/before_round_23_20260829_1436` after
the byte-stable Round-22 original lock and approved governance-state
synchronization amendment passed verification. It is limited to inference-only
seed-stability, head-causal-utilization, and offline-to-closed-loop alignment
diagnostics over frozen Round-22 policies and authorized Round-20
train/validation data. It reads existing fresh-development outputs only, never
re-runs fresh development, and never accesses untouched multi-UAV evaluation.
No training or Round-24 creation is authorized.

Round-23 completed the frozen checkpoint audit with no training, no new fresh-development rollout, and no untouched multi-UAV evaluation. Evidence is locked at `round_23/ROUND_LOCK.json`; Round-24 has not been created.

## Round-32 (in progress)

Round-32 is an explicitly authorized artifact-only audit following the
immutable Round-31 lock. It compares the actual finite episode horizon with
nominal planning, Task-Waypoint, and PPO credit horizons across locked
history. It creates neither a rollout nor a training artifact and screens
Round-20 protected JSONL records before decoding. No Round-33 is authorized.

Round-32 completed with EH-C / R32-C after 1003 regression tests and
compileall passed. Its historical interpretation overlay preserves all
scientific locks and reports a future-only 11--20 slot benchmark-design range.

## Round-34 (completed and locked)

Round-34 is the authorized successor of the immutable Round-33 lock. It is a
frozen Benchmark Complexity, Relational-Decision Opportunity, and Method
Suitability Audit, not a model, environment, or planner round. The only
exception is one accidental `development_test` record decode during preflight;
the exception record is immutable governance evidence, not scientific input.
All formal computation is restricted to the separately hash-locked clean
train/validation manifests. It completed with BCX-B / R34-B after 1,087
regression tests and compileall passed. No Round-35 may be created.

## Round-35 (in progress)

Round-35 begins from `backup/snapshots/before_round_35_20260902_0001` after
the immutable Round-34 lock was rehashed as
`59A472D367BEE8364C2D5CC4468759685795E6CB7E2176DFFD2B12F56C489AF0`.
It is restricted to a non-learning M=4/K=8/T=20 Level-2 shared-spectrum
benchmark and an external M×K association preference interface with exact
masked injective decoding. It must preserve legacy M=2/K={3,5} behavior,
fail closed before any protected decode, and stop without creating Round-36.

Round-35 completed and locked at `round_35/ROUND_LOCK.json` with RAO-A and
L2-A after 38 dedicated tests, 1,125 full regression tests, and compileall.
The next-stage label is non-learning planning/action-relevance validation only;
it is not an authorization to create Round-36 or train a method.

## Round-36 (in progress)

Round-36 is authorized only through the append-only
`round_35/ROUND_LOCK_AMENDMENT.json`; the Round-35 original lock and all
scientific artifacts remain immutable. It is a non-learning, exact Level-2
M=4/K=8/T=20 first-action association audit. The frozen scenario identities,
anchors {0,8}, full 3,393-assignment catalog, FAO horizons {1,5,10}, and
four-worker formal configuration are fixed before outcome access. No Round-37
may be created.
