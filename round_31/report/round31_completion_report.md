# Round-31 completion report

## Result

Round-31 is an inference-only diagnosis of locked Round-30. The Round-30
lock, source tree, report, checkpoints, and every registered output hash were
unchanged before and after the audit. No model was trained or changed.

The Round-31 failure-diagnosis gate is **R31-D**: no single supported,
minimal fixable bottleneck was established. The recommendation is to stop the
current direct state-only PPO baseline branch. This is a recommendation only;
it neither authorizes a new round nor creates Round-32.

## Scope and access

| Item | Result |
| --- | --- |
| PPO training / optimizer updates / backward passes | 0 / 0 / 0 |
| Checkpoint writes | 0 |
| Current-state diagnostic transitions | 12,672, only Round-30 train/validation |
| New fresh rollouts | 0 |
| Untouched evaluation access | 0 |
| Round-30 fresh evidence | frozen manifests/raw outputs read only |
| Round-30 lock SHA-256 | `7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125` |
| Round-32 | not created |

## PPO implementation audit

`implementation_bug_found=false`. The real Round-30 source was checked for
the clipped-minimum surrogate; stored old log probability and recomputed new
log probability at the same pre-tanh sample; tanh Jacobian correction; GAE;
advantage normalization; the registered four epochs/four minibatches;
gradient clipping; and deterministic `tanh(mu)` evaluation. There is no value
clipping: the implementation deliberately uses ordinary squared value loss.
The entropy term is the analytic base-Normal entropy surrogate, rather than a
false claim that it is the exact tanh-distribution entropy.

The finite twelve-step episode horizon is treated as an endpoint for both
`terminated` and `truncated`: GAE has no bootstrap past it. Raw sampled policy
actions, rather than safety-projected executed displacements, are used in the
PPO likelihood as required by the frozen Round-30 contract.

## Optimization and actor health

Late-phase averages across the five frozen histories were: KL 0.03163, clip
fraction 0.29444, gradient norm before clipping 14.11, explained variance
0.98382, and action standard deviation 0.08611. All diagnostics were finite;
there is no NaN/Inf or divergent-loss evidence. The complete early/middle/late
matrix is in `ppo_optimization_health.csv`.

The actor did change. Early-to-selected-checkpoint action RMSEs were 0.06193,
0.08974, 0.00000, 0.13636, and 0.04752 for seeds 0--4 (seed 2 selected its
early checkpoint). Thus the result is not merely a critic learning while the
actor is numerically frozen. However, selected PPO actions are conservative:
validation mean per-UAV movement norm is 0.09012, versus 1.0 for Independent
and H1 and 0.44903 for frozen state-only BC. This is observable outcome
behavior, but by itself does not identify a uniquely supported PPO correction.

The critic/actor asymmetry is clear: per-seed mean critic explained variance
is 0.9365--0.9462 while selected actor action displacement from early is only
0--0.1364. Value prediction can become easy even when the resulting policy
improvement points toward low-movement behavior that remains worse than the
myopic baseline.

## Advantage, reward, and credit

The inference-only reconstruction gives a mean positive/negative advantage
fraction of 0.2402/0.7598, advantage standard deviation 0.2981, absolute
magnitude 0.3187, effective-sample proxy 8.08 per twelve-step episode, and
episode advantage variance 0.1245. Crucially, advantages are aligned with
communication return: five-seed mean Pearson/Spearman correlations are
0.7589/0.7824 for future payload and 0.7478/0.7671 for environment
return-to-go. The corresponding queue-reduction correlations are
0.5294/0.4827. Therefore the evidence does not support an almost-unrelated or
numerically absent payload credit signal.

Reward scale is coherent in the reconstructed trajectories: payload reward is
roughly an order of magnitude above the mean energy penalty (illustrative
seed-0 training means 0.16586 versus -0.02754), while terminal unfinished
queue penalty is small on a per-step basis. This is separate from the earlier
reward-semantics alignment result: both semantics and numerical scale are
audited here, and neither supports a reward-change claim.

With frozen `gamma=0.99` and `lambda=0.95`, gamma powers are 0.95099 (H5),
0.90438 (H10), and 0.81791 (H20); GAE trace weights are 0.73586, 0.54149,
and 0.29321. There is expected attenuation, but not a severe disappearance of
H5/H10 signal in this twelve-step task.

## Generalization and distribution

PPO is already below Independent on its registered training scenarios:
mean PPO-minus-Independent payload is -434,955 bits, compared with -465,015
bits on validation. Every seed is negative on both. Hence the failure is not
primarily a train-to-validation generalization failure. Manifest comparison
shows no structural distribution shift in K, eta, queue pressure, geometry,
initial UAV configuration, or episode horizon; the maximum normalized
total-variation difference is 0.125, attributable to finite balanced
population counts and not interpreted as a causal shift.

## Representation and task scale

The actor feature extractor does not explicitly supply UAV-to-buoy relative
dx/dy, distance, current rate, queue-weighted rate, or cross-interference
potential. `pairwise_relation_not_explicit=true`. Its absolute UAV/buoy
coordinates still make these quantities theoretically derivable, so this
absence alone is not proof of insufficiency.

The required current-only kNN diagnostic uses the actual raw actor-input
vector (canonicalized ID-free sets), not a learned hidden vector; all same
episode and temporal neighbors are excluded. Full relational features add
only current UAV-buoy dx/dy, distance, rate, queue-weighted rate,
interference potential, and UAV-UAV relation. The resulting RILS values are:

| Group | RILS |
| --- | ---: |
| overall | -0.02633 |
| K3 | -0.05574 |
| K5 | -0.00044 |
| eta=.25 | 0.05165 |
| eta=.5 | 0.08496 |

Thus full relational neighborhoods do not materially lower H1 action
ambiguity overall or at K5. The positive eta=.25/.5 values are modest and do
not meet the pre-registered strong relational-bottleneck criterion.

The frozen fresh evidence remains sharply K-dependent: PPO-minus-Independent
is +14,962 bits at K3 and -662,068 bits at K5. K5 has higher reconstructed
validation advantage standard deviation (0.41167 versus 0.18536), higher
PPO ISR (0.29566 versus 0.23063), and slightly greater conservative movement
(0.10090 versus 0.07934), but not higher RILS. K5 task scale is therefore
moderate failure evidence, not evidence that a relational encoder is the
specific repair.

## Action and matching outcomes

Across five PPO seeds on 48 frozen validation reset states, mean PPO action
cosines are 0.2355 to Independent, 0.1288 to H1, and 0.2140 to H5; mean RMSEs
are 0.6942, 0.7015, and 0.6920. PPO is not a close directional copy of any
reference. Its validation aggregate rate is 4.29 Mbps and ISR 0.263, versus
4.57 Mbps/0.251 for Independent. Service diversity remains 2.0 distinct
served buoys because legacy matching is exclusive; this does not reveal a
PPO-specific duplicate-service defect.

Deterministic evaluation is exact: maximum `tanh(mu)` discrepancy is below
7.3e-9 for every checkpoint. The 64-sample stochastic tanh expectation
diagnostic differs from deterministic action by RMSE 0.0199--0.0466; it does
not indicate that Round-30 accidentally evaluated random samples or an
incorrect standard deviation.

## Evidence classification

| Mechanism | Rating | Main evidence |
| --- | --- | --- |
| PPO implementation bug | unsupported | source semantics audit |
| Numerical optimization instability | weak | finite losses/KL/clipping/EV |
| Weak or noisy advantage | weak | strong payload/return alignment |
| Long-horizon attenuation | weak | frozen gamma/GAE weights |
| Relational-state insufficiency | weak | RILS not material at K5 |
| Train-validation distribution shift | unsupported | PPO already loses on train |
| Exploration/action-scale issue | weak | conservative policy outcome, but no unique correction identified |
| K5 task-scale difficulty | moderate | frozen K5 gap, ISR, advantage variance |

The mandatory next-step answer is therefore **D: stop the present direct PPO
baseline branch**. No modified PPO, relational actor, curriculum, or other
candidate is trained in this round.
