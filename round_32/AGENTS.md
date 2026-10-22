# Round-32 Working Rules

Round-32 is the **Cross-Round Episode-Horizon, Planning-Horizon,
Task-Waypoint-Horizon and Effective-Future Semantics Audit**.

## Scope

- This is a read-only audit of locked source and historical artifacts.  It
  creates no environment transition, rollout, dataset, model, checkpoint,
  optimizer, backward pass, or training update.
- It may read the frozen planning traces from Rounds 15--19 and the locked
  Round-20 train/validation JSONL data.  Development-test JSONL lines are
  screened by their split token before decoding and are never inspected.
- It must distinguish a planner's nominal shadow horizon from the future that
  remains in the real finite episode.  It must not reinterpret a batch size as
  an episode horizon.

## Frozen technical contract

- Do not modify the maritime environment, planner, reward, graph schema,
  dataset, model, PPO configuration, or any completed-round artifact.
- Do not access or regenerate any fresh rollout, untouched evaluation, or
  Round-20 development-test record.  Do not add HGNN, PPO, MARL, BC, or
  Task-Waypoint training work.

## Completion

- New artifacts are restricted to `round_32/`, except the append-only root
  lifecycle state required to register this authorized audit and a narrowly
  scoped predecessor lifecycle amendment if regression verification requires
  one.
- Stop after the Round-32 lock.  Do not create Round-33.
