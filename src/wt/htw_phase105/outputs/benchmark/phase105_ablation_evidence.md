# Phase-10.5 ablation evidence map

| ID | Claim | Source | Status on current L3-B |
| --- | --- | --- | --- |
| A | Utility-aligned supervision vs old single-reference | Phase-9.4.5 UOBJ-A | Formal L3-B confirmation (historical confirmation set, not this pre-final set) |
| B | HGNN vs DeepSets | Phase-9.4.5 GRAPH-U-C | Formal: HGNN not stably better; encoder frozen to DeepSets |
| C | Order-free decoder | Phase-9.4.3 ORDERFIX-A | Formal architecture freeze; permutation dependence removed |
| D | Task Option temporal hold | Phase-6.4 hold=5/10/15 | Historical mechanism on L2-A, not a current L3-B primary table |
| E | Low-level PPO vs gated fly-toward | Phase-10.4 LLEXEC-C | Formal L3-B independent validation: no stable PPO advantage |
| F | InRangeZeroMotionGate | Phase-10.3 design + Phase-10.4 B2-B1 5/5 | Formal: keep as deterministic execution rule, not a PPO credit |

This pre-final 64-scenario table is a new ordinary benchmark of the frozen FINAL-HIER stack. It is not a final untouched test.
