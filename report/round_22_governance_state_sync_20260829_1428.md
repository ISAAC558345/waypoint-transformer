# Round-22 governance-state synchronization

## Scope and authority

This governance-only correction resolved a stale root lifecycle state. The
authoritative final scientific record is the immutable
`round_22/ROUND_LOCK.json`, corroborated by the Round-22 completion report,
formal training manifest, final gate, access audit, and policy checkpoint
manifest. No Round-22 experiment was rerun.

## Stale state and correction

The stale file was `CURRENT_ROUND.txt`. Before synchronization it stated
`status=in_progress`, `closed_loop_evaluation=false`, and
`fresh_dev_access_count=0`. It now states `status=locked`,
`closed_loop_evaluation=true`, and `fresh_dev_access_count=1`, while retaining
the existing schema and the zero Round-20 reuse and untouched-evaluation
counts. `PLANS.md` and `ROUND_VERSIONING.md` now explicitly record the locked
Round-22 result without deleting historical descriptions.

## Integrity result

- Original Round-22 lock SHA before/after:
  `1D464A696ED8DE11E7AEDB14479F55ABB4F9F3E2621837E89CD854CD850190E4`.
- Scientific artifacts, datasets, metrics, predictor, frozen raw fresh-dev
  outputs, and model checkpoints were not modified.
- The predictor SHA remains
  `645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221`.
- All 12 formal best checkpoints and their matching last checkpoints verify
  against the locked manifest; all locked output hashes verify.
- No fresh-dev rerun occurred; final fresh-dev access count is exactly 1.
- Round-20 development-test reuse remains 0. Untouched-evaluation model and
  rollout access remain 0.
- The final MU-WP gate remains `MU-WP-B` and
  `candidate_ready_for_untouched_eval=false`.

## Amendment and verification

The approved additive record is `round_22/ROUND_LOCK_AMENDMENT.json`. Its
strict verifier permits only the three synchronized root-governance files and
the explicitly documented historical test semantic correction: the test now
asserts that Round-23 was absent at Round-22 lock time, without forbidding a
later authorized Round-23.

The governance-specific/Round-21/Round-22 targeted suite passed 88/88. The
complete repository regression passed 848/848, and `py -3.14 -m compileall -q
src tests` passed. Round-23 remains absent. Since the original lock, all
authoritative outputs, root governance, and verification checks are now
consistent, `resume_round23_creation_allowed=true`; this report does not
create Round-23.
