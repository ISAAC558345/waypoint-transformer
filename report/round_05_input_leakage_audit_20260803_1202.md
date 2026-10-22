# Round-5 input and leakage audit (opening)

Status: **in progress**. This opening audit defines the boundary that the
implementation and final tests must enforce.

## Immutable evidence

- Round-3 lock SHA-256:
  `601C6679D17EE25534C9A415C75BBF0DC28A167E2ED5EFE11BA0FB3A5C984375`.
- Round-4 lock SHA-256:
  `A47895852DD03365DE168DD552961E24D33CAD3C13C3EF5E646507F19C4E3679`.
- Frozen H=30 goal-conditioned seed-1 checkpoint SHA-256:
  `F92133AACA1FF6F0839EE7C817F40B696F7FD8FB15973A6118D222F83F01368D`.
- The seed was chosen by H=30 validation endpoint RMSE: seed 1
  `28.204474065 m`, seed 2 `28.636965121 m`, seed 0 `30.915033653 m`.
  Test data did not participate.

Strict reload was repeated twice before Round-5. Predictions were bitwise
identical (`max_abs_difference=0`) and the source Round-3 hashes were 177/177
valid.

## Allowed information paths

```text
masked current/past state -----------------------> state encoder
achieved_terminal_goal -> frozen W_phi ----------> predicted H30 delta
achieved_terminal_goal --------------------------> Direct-Goal control only
predicted H30 delta ------------------------------> WT / waypoint MLP
remaining fraction ------------------------------> Transformer variants
executed current displacement -------------------> loss target only
```

The achieved terminal goal makes this a hindsight-conditioned offline study.
It must not be described as a causally available online goal.

## Disallowed paths

The policy feature dictionaries and model signatures must reject or omit:

- true future positions or true H=30 displacement for deployable variants;
- current and historical actions;
- rewards, episode returns, RTG, advantage, or quality weights;
- policy ID, quality class, collection seed, arrival-trace ID, or terminal
  reason;
- test metrics in training, early stopping, seed choice, or context selection.

The oracle diagnostic is the sole consumer of true H=30 spatial displacement.
It is non-deployable, excluded from selection, and never used by the interface
smoke.

## Structural risks and controls

1. The canonical 62-dimensional state already contains a task-waypoint summary
   and a node-mask copy. Round-5 preserves the locked representation and also
   passes the explicit five-slot node mask used by Round-4; it does not silently
   reinterpret dimensions.
2. Feature masks and timestep attention masks have different ranks and are
   named separately to prevent broadcasting.
3. The official legacy GPT can interleave true action tokens. Round-5 reuses
   its audited causal Block only and has no action-token argument or embedding.
4. The official D4RL data loader can re-split and randomly choose goals.
   Round-5 instead adapts the locked Round-3 episode split directly.
5. H=30 predictions for the final 29 transitions are outside the checkpoint's
   full-horizon training support. They are retained for a complete action-label
   sequence but explicitly flagged with the effective remaining horizon;
   conclusions must disclose this support limitation.

This file will be closed with test evidence and output hashes at completion.

## Closing evidence — 2026-08-03

Status: **passed for the authorized offline scope**.

- The canonical run is
  `round_05/outputs/round05_formal_canonical_20260803_1500`.
- The exact 27-run matrix completed before model selection was frozen. Every
  best epoch, WT context choice, and recommended seed used validation raw
  action RMSE only. Test access begins after the recorded freeze time;
  `test_used_for_selection=false` and every model has one persisted test
  evaluation.
- The selected deployable-interface model is WT `H_c=10`, seed 0, epoch 7.
  Its checkpoint is
  `A01A1C2905FF61C0A0C889D6D3E9B77359F470BC2A05DE92594E6D6F884949C0`.
- Frozen Waypoint provenance is still the original H=30 checkpoint
  `F92133AACA1FF6F0839EE7C817F40B696F7FD8FB15973A6118D222F83F01368D`.
  It remained in eval/no-grad mode, has no optimizer, and no new Waypoint
  checkpoint was written.
- The sequence manifest records no split or episode crossing and no historical
  actions. Padding is excluded from attention and loss; 45 targeted tests and
  the 247-test repository regression cover these paths.
- Replay uses recorded states only and never advances the environment. The
  interface smoke uses a disposable environment copy for five steps, recomputes
  state and waypoint at every step, and records
  `closed_loop_interface_smoke_performed=true` together with
  `closed_loop_evaluation_performed=false`.
- Round-3 provenance is 178/178 valid; the Round-3 lock, Round-4 lock, and
  frozen checkpoint hashes remain respectively
  `601C6679...984375`, `A4789585...3679`, and `F92133AA...1368D`.

Remaining scientific limitations are intentional, not leakage exceptions:
`achieved_terminal_goal` makes this hindsight-conditioned offline prediction;
no online goal mechanism is established; the fixed pilot lacks low-quality
trajectories; transition samples are correlated; and no result supports a
closed-loop return, communication-performance, or deployment claim.
