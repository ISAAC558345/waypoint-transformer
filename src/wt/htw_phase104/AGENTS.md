# HTW Phase-10.4 Working Rules

Multi-seed independent validation of gated low-level PPO. No method edits.

- Freeze DeepSets-util high-level seed 0 and InRangeZeroMotionGate from Phase-10.3.
- Train only low-level seeds 1-4 with the Phase-10.1 budget; reuse seed-0 final checkpoint.
- Independent ordinary scenarios only; do not open confirmation / protected / untouched.
- Stop after LLEXEC A/B/C. Do not start Phase-10.5.
