# HTW Phase-10.2 Working Rules

Frozen-model in-range vs completion diagnosis.

- No training. No reward, network, L3-B, SINR, or hold edits.
- No confirmation / protected / untouched access. No new Round.
- Replay at most 16 design episodes (seeds 8-15, B0 and B1).
- At most 16 snapshots, 3 branches, each at most remaining hold (cap 15).
- Do not pick snapshots by which intervention wins.
