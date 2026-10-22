# LL-V2.2C MAPPO mask reduction audit

Source: `src/wt/lowlevel_v2/phase1b/mappo.py` `ICARMAPPOLocked.update` (no code change in this phase).

## Policy surrogate

Actual implementation:

```
per_uav = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
denom = mask.sum().clamp_min(1)
policy_loss = (per_uav * mask).sum() / denom
```

This is **form B**:

`loss = sum(mask * loss_i) / max(sum(mask), 1)`

`actor_mask_denominator = ACTIVE_SAMPLES`

(UAV-steps with `policy_active_mask = 1 = NOT(service_lock OR base_zero_lock)`).

Not form A `mean(mask * loss_i)`, which would divide by total UAV-steps and shrink the actor gradient by `active_fraction`.

Lines in `mappo.py`:

- `denom = mask.sum().clamp_min(1.0)`
- `policy_loss = (per_uav * mask).sum() / denom`

## Entropy

```
entropy_term = (entropy * mask).sum() / denom
```

Same **ACTIVE_SAMPLES** denominator.

`entropy_mask_denominator = ACTIVE_SAMPLES`

KL and clip fraction in the logger also use the same `denom`.

## Value / GAE

`value_loss = 0.5 * (target - value).mean()` over **all env-steps**, unmasked.

GAE advantages are env-level `(T, n_env)`, then:

```
adv = (advantages.reshape(-1) - advantages.mean()) / advantages.std()
```

Advantage **normalization includes all env-steps**, including locked UAV majorities. Then `advantage` is broadcast to 6 UAVs.

So:

- Actor loss denom: ACTIVE UAV-samples
- Entropy denom: ACTIVE UAV-samples
- Advantage scale: ALL env-samples
- Value loss denom: ALL env-samples

## Frozen gradient check

On frozen minibatches, `G_current / G_active_normalized = 1.0` for all 10+10 batches, confirming the live code is already active-normalized.
