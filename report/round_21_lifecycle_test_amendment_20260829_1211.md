# Round-21 lifecycle-test amendment audit

## Result

**PASS — `resume_round22_allowed=true`.**  This is a governance-only
amendment.  It was completed before any Round-22 policy training, fresh-dev
outcome access, or untouched-evaluation access.

## Scope

1. The erroneous test was
   `tests/test_round21_hgnn_task_waypoint.py::Round21HGNNTests.test_round22_is_absent`.
2. Its original assertion was `assertFalse((ROOT / "round_22").exists())`.
3. It permanently blocked every legitimate later round because it treated a
   lock-time fact as an eternal filesystem invariant.
4. The amended test verifies the actual historical meaning: Round-21 was
   completed with `round22_created=false`; if Round-22 now exists, its recorded
   initialization time is after Round-21's actual `completed_at` timestamp.

## Immutable evidence

- Original Round-21 lock SHA before and after:
  `F754D441B0A8A3DF77EDAD7EAF2E2CA3062A32025444524270795897E9958BD6`.
- The original lock is byte-for-byte unchanged.
- Original lifecycle-test SHA:
  `D3F5593CE5E053994F84A37BEFC248DFAF712FA99C22081C38BA4A41B2EB5A36`.
- Amended lifecycle-test SHA:
  `A1D28501A3C031A914ACEA35B7C586895D776D60EFCEC200912B70EEFDAB712C`.
- The strict one-path overlay is
  `round_21/ROUND_LOCK_AMENDMENT.json` (SHA
  `8A20C43599E0DD970FD8DDC267F0D44953025E3C57BE3BD0BD2B031B33A52789`).
- All Round-21 scientific artifacts remain unchanged: dataset, checkpoints,
  metrics, formal outputs, scientific reports, HGNN/DeepSets source, and the
  original `ROUND_LOCK.json`.  The only changed historical file is the
  approved lifecycle test; this governance report is new evidence rather than
  a replacement for a scientific report.

## Verification

- The original Round-21 final result remains historical evidence:
  **802/802 PASS** at lock time; it was not rewritten.
- Amended lifecycle test: **1/1 PASS**.
- Current Round-21 test module: **43/43 PASS**.
- Current Round-22 governance module: **6/6 PASS**.
- Current repository regression, using the writable test temporary directory:
  **808/808 PASS**.
- `py -3.14 -m compileall -q src tests`: **PASS**.

## Round-22 initialization preservation

- The selected frozen DeepSets H=5 seed-1 predictor remains
  `645F235032DE9639A6D42318A41CC8F0E2175E86E38545DA98E0BE5CAEB80221`.
- The fresh-dev manifest remains
  `C8570659A88FE062CB7D199EA68468B91A8062C3E320E01F367BDB98CA2988B1`.
- Fresh-dev access count is `0`, with no environment construction or outcome
  read.  Untouched-evaluation model and rollout access counts are both `0`.
- Round-20 development-test reuse count is `0`; no Round-23 exists.

The overlay verifier confirms every original Round-21 registered hash except
the explicitly approved test path, whose original and amended hashes are both
recorded above.  Its amendment audit reports zero unapproved mismatches.
