# Round-30 Completion Report

## Scope

Direct Multi-UAV State-Only DeepSets-PPO Learnability Baseline in the frozen Round-19 M=2 shared-spectrum environment. No Task-Waypoint input/loss, planner trigger/target, graph/HGNN, PPO Transformer, offline BC initialization, reward, physics, or action-space modification was used.

## Required answers

| # | Question | Result |
|---:|---|---|
| 1 | PPO actor architecture | Parameter-shared UAV DeepSets encoder + masked pooled UAV/buoy summaries + MLP heads for 2-D Gaussian mean/log-std. |
| 2 | Critic architecture | Centralized current-state DeepSets critic using masked pooled UAV and buoy encodings plus current global configuration. |
| 3 | Input features | ID-free current UAV (position, velocity, energy, path, payload, binary assignment), buoy (position, queue, arrival, cumulative payload, binary match, K), masks, K/eta/time globals. |
| 4 | Action distribution | Independent per-UAV 2-D Normal, tanh squashing; PPO likelihood is the raw pre-safety sample and executed projected action is separately saved. |
| 5 | Training steps / seed | 307,200 environment steps (150 synchronous 2,048-step updates). |
| 6 | All five seeds completed | yes; seeds 0, 1, 2, 3, and 4 are frozen checkpoints. |
| 7 | Early versus late training payload | s0=26,788,863.864, s1=26,866,285.666, s2=26,836,033.568, s3=26,783,123.376, s4=26,852,187.316 → s0=26,850,349.835, s1=26,750,639.598, s2=27,056,301.412, s3=26,798,752.911, s4=26,814,600.865 |
| 8 | Validation PPO payload | s0=26,956,236.378, s1=26,962,723.934, s2=26,924,312.817, s3=26,882,020.086, s4=26,956,342.319 |
| 9 | Fresh PPO payload | s0=26,229,628.080, s1=26,247,704.390, s2=26,240,614.872, s3=26,183,287.522, s4=26,241,654.456 |
| 10 | PPO vs Random | validation 301,178.798; fresh 132,030.284 bits. |
| 11 | PPO vs Independent | validation -465,014.617; fresh -323,552.597 bits. |
| 12 | PPO vs State-only BC | validation -28,401.686; fresh 53,973.423 bits. |
| 13 | PPO vs H1 | validation -345,244.177; fresh -93,650.185 bits. |
| 14 | PPO vs H5 | validation -510,624.436; fresh -139,405.739 bits. |
| 15 | PPO vs H10 | validation -510,624.436; fresh -139,405.739 bits. |
| 16 | Positive seed fraction | PPO−Independent: validation 0.000, fresh 0.000; seed-cluster and leave-one-out statistics are in ppo_uncertainty_audit.csv. |
| 17 | Hierarchical CI: PPO−Independent | validation [-605,109.347, -337,453.357], fresh [-444,039.848, -220,115.749] bits. |
| 18 | Hierarchical CI: PPO−BC | validation [-152,953.742, 90,127.629], fresh [-52,924.336, 162,709.918] bits. |
| 19 | K3/K5 | K3=14,962.470; K5=-662,067.665 (fresh PPO−Independent bits). |
| 20 | eta=.25/.5 | eta_025=-41,224.011; eta_05=-300,426.376 (fresh PPO−Independent bits). |
| 21 | Conflict-state result | See ppo_conflict_analysis.csv; it stratifies only the frozen current-state interference_conflict outcome and never feeds deployment policy input. |
| 22 | Completion | validation 0.881739; fresh 0.883150. |
| 23 | Residual queue | validation 3,890,994.239; fresh 3,710,474.793 bits. |
| 24 | Safety correction | PPO projection rate validation 0.000000; fresh 0.000000; raw/executed per-step records are retained. |
| 25 | Entropy/KL health | s0: entropy=-1.8240, KL=0.02145; s1: entropy=-1.7693, KL=0.02098; s2: entropy=-1.4730, KL=0.02193; s3: entropy=-1.7195, KL=0.02117; s4: entropy=-1.7877, KL=0.02130 |
| 26 | Critic explained variance | s0=0.94532; s1=0.94197; s2=0.93652; s3=0.94615; s4=0.94071 |
| 27 | PPO Learnability Gate | PPO-LC |
| 28 | PPO-vs-BC Gate | PBC-B |
| 29 | Direct RL learnability | Current evidence does not establish direct RL learnability |
| 30 | Next step | PPO_stability_diagnosis before adding architecture complexity in a separately authorized round; Round-30 itself stops and does not authorize a successor. |
| 31 | Fresh access count | 1 |
| 32 | Untouched eval access count | 0 |
| 33 | Round-31 created | no |

## Reproducibility

The five fixed seeds, new train/validation/fresh manifests, single fixed PPO configuration, validation-only checkpoint choice, checkpoint hashes, paired baseline rows, all three bootstrap types, raw/executed action logs, and figure manifest are in the formal output directory. Fresh scenarios were accessed once only after the pre-fresh freeze; historical protected and untouched evaluation access remains zero.
