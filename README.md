# 多无人机高低层轨迹规划

当前研究系统的权威说明在 [`docs/SYSTEM_OVERVIEW_CN.md`](docs/SYSTEM_OVERVIEW_CN.md)。在仓库根目录设置 `PYTHONPATH=src` 后，论文模块和实现包都可以导入。实现仍在 `src/wt`，下面的目录是按论文模块划分的入口。

```text
configs/        环境、高层、低层和场景划分
src/envs/       L3B 通信环境
src/models/     HGNN 评分器和 MAPPO Actor-Critic
src/algorithms/ 排序损失
src/policies/   正式高层规划器和闭环入口
src/utils/      SafetyProjector 与构图辅助
scripts/        train.py、evaluate.py、benchmark.py、plot_results.py
checkpoints/    冻结权重的路径索引
results/        评估表、统计和阶段输出
tests/
```

过程报告在 `docs/archive/phase_reports/`。`round_01` 到 `round_36` 与 `backup/` 仍在原处，因为轮次锁定和测试按这些目录名读取。

常用命令：

```powershell
$env:PYTHONPATH="src"
py -3 scripts/benchmark.py
py -3 scripts/evaluate.py --scenario 322002
```

下面保留上游 Waypoint Transformer / RvS 说明。

# Waypoint Transformer: Reinforcement Learning via Supervised Learning with Intermediate Targets

This repository is adapted based on the RvS repository, as provided by Emmons et al. (2021). The link to the original repository is: https://github.com/scottemmons/rvs.

# Installation

The code depends on MuJoCo 2.1.0 (for mujoco-py) and MuJoCo 2.1.1 (for dm-control). Here are [instructions for installing MuJoCo 2.1.0](https://github.com/openai/mujoco-py/tree/fb4babe73b1ef18b4bea4c6f36f6307e06335a2f#install-mujoco)
and [instructions for installing MuJoCo 2.1.1](https://github.com/deepmind/dm_control/tree/84fc2faa95ca2b354f3274bb3f3e0d29df7fb337#requirements-and-installation).

# Reproducing Experiments

The `experiments` directory contains a launch script for each task, consistent with the RvS repository.

To train a waypoint network on a particular environment, run:
```bash
bash experiments/launch_waypoint_nets.sh
```

Based on the trained waypoint network, choose a particular checkpoint to use for WT. For example, suppose we wanted to run the experiment for Kitchen Complete across 5 seeds:

```bash
GOAL_NETWORK_CKPT=kitchen_complete_goalwaypt_net/files/checkpoints/gcsl-kitchen-complete-v0-epoch=028-val_loss=3.9324e-04.ckpt bash experiments/kitchen/launch_kitchen_rvs_g_complete.sh
```

## Maritime offline-data validation (round 02)

The added maritime path currently covers a minimal single-UAV/static-buoy
environment, four parameter-free baselines, complete-episode collection,
episode-level splitting, K-step availability auditing, metrics, and scientific
figures. Round 02 performs no model training.

From this repository root on Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
& "..\decision-transformer-master\.venv-dt-gym\python.exe" `
  -m unittest discover -s tests -v
& "..\decision-transformer-master\.venv-dt-gym\python.exe" `
  -m wt.maritime.runner --smoke
```

The accepted smoke output is under
`round_02/outputs/round02_smoke_20260730_212701_283649/`, with figures under
`round_02/figures/round02_smoke_20260730_212701_283649/`.

See
[`docs/round02_non_learning_offline_data.md`](docs/round02_non_learning_offline_data.md)
for policy fallbacks, trajectory shapes, split rules, K-step results, metric
semantics, and current data-coverage limits. The accepted Round-2/v0
environment does not provide queues, urgency, task completion, desired goals,
or actual delivered payload; those values are explicitly unavailable rather
than approximated. Round-3 adds the required task semantics behind an explicit
v1 compatibility boundary.

## Maritime task and Waypoint-sample validation (round 03)

Round 03 adds opt-in `task_semantics_version="v1"` while preserving the
Round-2/v0 path. The v1 environment has per-buoy queues, reproducible
constant/Poisson/shared-trace arrivals, queue-bounded actual payload,
task-completion and terminal-goal fields, four meaningful non-learning
baselines, schema-v1 episode storage, and masked Waypoint targets for
`K={1,5,10,20,30}`. No learning model is constructed or trained.

From this repository root on Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
& "..\decision-transformer-master\.venv-dt-gym\python.exe" `
  -m unittest discover -s tests -p "test_*.py" -q
& "..\decision-transformer-master\.venv-dt-gym\python.exe" `
  -m compileall -q src tests
& "..\decision-transformer-master\.venv-dt-gym\python.exe" `
  -m wt.maritime.runner --round3-smoke
& "..\decision-transformer-master\.venv-dt-gym\python.exe" `
  -m wt.maritime.runner --round3-pilot
```

Accepted evidence:

- 146/146 repository tests and 46/46 Round-3 focused tests passed;
- canonical smoke: 24 episodes and 192 transitions;
- canonical pilot: 80 episodes, 6,604 transitions, and 28,140 Waypoint
  samples;
- valid sample counts for `K=1,5,10,20,30` are
  `6,604 / 6,284 / 5,884 / 5,084 / 4,284`;
- exact checksums/reload, source-target recomputation, masks, split isolation,
  shared arrival traces, and all 22 quality checks passed;
- 12 scientific figures were generated and visually checked;
- `training_performed=false`.

Canonical artifacts are under
`round_03/outputs/round03_pilot_canonical_20260730_230000/`, with figures under
`round_03/figures/round03_pilot_canonical_20260730_230000/`.

See
[`docs/waypoint_sample_schema_v1.md`](docs/waypoint_sample_schema_v1.md) and
[`report/round_03_report_20260730_230203.md`](report/round_03_report_20260730_230203.md)
for the slot order, goal/Waypoint layouts, masks, baseline metrics, data
quality, and current modeling limitations.

## Maritime spatial Waypoint Network (round 04)

Round 04 trains only a supervised two-dimensional spatial Waypoint MLP on the
locked Round-3 pilot. The target is the raw relative UAV displacement
`q[t+H]-q[t]`, normalized by the maximum reachable distance. Independent
goal-conditioned and matched state-only models use `H={1,5,10,20,30}` and
seeds `{0,1,2}`. Validation alone selects checkpoints and the recommended H;
test data never trigger training or selection.

From this repository root on Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$python = "..\decision-transformer-master\.venv-dt-gym\python.exe"

& $python -m unittest tests.test_round04_spatial_waypoint -q
& $python -m unittest discover -s tests -p "test_*.py" -q
& $python -m compileall -q src tests

& $python -m wt.round4.round4_runner --smoke --device cuda
& $python -m wt.round4.round4_runner --formal --device cuda
```

Accepted evidence:

- 56/56 focused and 202/202 repository tests pass;
- 30/30 formal training runs complete: 15 goal-conditioned and 15
  state-only;
- goal-conditioned input/output shapes are `(batch,80) -> (batch,2)`;
  state-only uses `(batch,67) -> (batch,2)`;
- validation recommends H=30 with RMSE `29.252 ± 1.189 m`, normalized error
  `0.09751`, and direction cosine `0.4578`;
- H=30 test goal-conditioned RMSE is `44.281 m`, versus `49.006 m` for
  state-only and `52.564 m` for zero displacement;
- all 60 best/last checkpoints strictly reload with exact source,
  implementation/config, and tensor provenance;
- 36 PNG/SVG scientific figure files pass visual QA;
- no Transformer, action policy, reinforcement learning, or closed-loop
  evaluation is performed.

The accepted formal output is
`round_04/outputs/round04_formal_canonical_20260802_173717/`, with figures in
`round_04/figures/round04_formal_canonical_20260802_173717/`.

The goal is `achieved_terminal_goal`, an explicitly declared hindsight
channel. Results are heterogeneous across collection policies, validation and
test are distribution-shifted, and the pilot contains no low-quality
trajectories. Therefore these results establish only bounded offline
supervised learnability and an engineering interface for a later authorized
stage—not policy or return improvement.

See
[`report/round_04_input_leakage_audit_20260802_1558.md`](report/round_04_input_leakage_audit_20260802_1558.md)
and
[`report/round_04_report_20260802_174159.md`](report/round_04_report_20260802_174159.md)
for the exact schema, all H/seed runs, selection rule, baselines, grouped
metrics, projection audit, provenance, and limitations.

## Maritime causal offline action policy (round 05)

Round 05 freezes the validation-selected Round-4 H=30 spatial Waypoint Network
and trains a causal two-dimensional offline action policy. It uses only
current/past state, node mask, predicted waypoint (WT only), and remaining
time; historical action labels and future state/waypoint labels are excluded.
It compares `H_c={1,10}` for WT, Direct-Goal, and State-only Transformers,
plus Oracle-Waypoint diagnostics and one-step MLP controls.

On Windows PowerShell from this repository root:

```powershell
$env:PYTHONPATH = "src"
$python = "..\decision-transformer-master\.venv-dt-gym\python.exe"

& $python -m unittest discover -s tests -p "test_*.py" -q
& $python -m compileall -q src experiments tests
& $python -m wt.round5.round5_runner --smoke --device cuda
& $python -m wt.round5.round5_runner --formal --device cuda
```

Accepted evidence:

- 247/247 tests and compileall pass;
- canonical formal output is
  `round_05/outputs/round05_formal_canonical_20260803_1500/`;
- all 27 configured runs train and evaluate test exactly once, after validation
  selection is frozen; 27/27 checkpoint reload predictions are identical;
- WT validation selects `H_c=10`, seed 0, epoch 7; selected test action RMSE
  is 7.772 m (raw), with 48/1,415 predictions radially projected for deployment;
- 16 PNG and 16 SVG figures pass strict visual/semantic QA.

This is an offline hindsight-conditioned action-prediction result only. The
goal source is `achieved_terminal_goal`, no deployable online goal mechanism is
established, and no formal closed-loop evaluation, return comparison, PPO,
online fine-tuning, HGNN, multi-UAV, drift, or communication-resource action
was performed. See `report/round_05_report_20260803_151500.md` for all seeds,
groups, limitations, and exact artifact paths.
## Round-6 closed-loop evaluation

Round-6 is evaluation-only and freezes all Round-4/5 checkpoints. Its 60 new
shared-trace scenarios and paired reference-goal protocol are documented in
`report/round_06_report_20260809_1600.md`. The result is Gate B: no reliable
closed-loop Waypoint advantage; no later complex-environment round is opened.

## Round-7 Gate-B data diagnosis

Round-7 is a non-learning audit and dataset-construction round. It preserves
Rounds 3--6, creates `maritime_offline_dataset_v2`, and records outcome-derived
quality tiers, difficulty tiers, seed-level splits, stitching coverage, goal
information, and H=5/10/20/30 target diagnostics. The canonical formal output
is `round_07/outputs/round07_formal_20260809_1740/`; see
`report/round_07_report_20260809_1810.md`. No model was trained and no
Round-8 was created.
