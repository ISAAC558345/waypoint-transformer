# Round-32 completion report

## Result

Round-32 is a read-only cross-round horizon-semantics audit. It created no
environment transition, rollout, dataset, model, optimizer update, backward
pass, checkpoint, fresh access, development-test decode, or untouched
evaluation access. The pre-Round-32 snapshot is
`backup/snapshots/before_round_32_20260830_2231`.

The prompt's “five-step” premise conflicts with the locked sources and is
corrected here, rather than propagated: Rounds 18--31's multi-UAV task,
including Round-30 PPO and Round-31's report, uses a **twelve-step** episode
horizon. `256 × 8 = 2048` is the number of transitions in a PPO batch, not an
episode length.

## Four separate meanings of horizon

- `T_episode`: real environment decisions before normal time-limit truncation.
- `H_plan`: nominal planner-shadow depth.
- `H_task`: future Task-Waypoint label window.
- `H_c`: policy context length (Round-22/24: 10).

At state `t`, the physically executable future is
`min(H_plan, T_episode - t)`. A source shadow may or may not obey that bound;
the audit reports both instead of silently treating them as equivalent.

## Direct answers

1. Round-15 episode horizon: **64**.
2. Round-16: **64**.
3. Round-17: **64**.
4. Round-18: **12**.
5. Round-19: **12**.
6. Round-20 dataset: **12**.
7. Round-22 policy environment: **12**; `H_task=5`, `H_c=10`.
8. Round-30 PPO: **12**, not five.
9. It is not a five-step Round-30 task. Locked `round30/scenarios.py` sets
   `EPISODE_HORIZON=12`; it is the inherited Round-19/20 multi-UAV convention,
   not a smoke leftover or a runtime reduction.
10. H5 mean effective future is **4.84375** in the 64-step R15--17 traces and
    **4.166667** in the 12-step R18/19 traces.
11. H10 is **9.296875** in R15--17 and **6.25** in R18/19.
12. H20 is **17.03125** in R15--17. R18/19 did not register H20; in a
    twelve-step task a hypothetical H20 can never have a complete 20-step
    future (mean executable future would be 6.5 over all decision times).
13. R18/19 H5/H10 equality is primarily **EQ-B** before the final five
    decisions: distinct nominal shadows choose the same first action. At
    `t>=7`, the post-hoc real remaining window also agrees (**EQ-D**), but the
    source shadow itself is still not clock-aware. **EQ-C is false**:
    `range(horizon)` is executed and H>5 is not ignored.
14. H10/H5 first-action equality is **100%** in both locked traces (504/504
    R18 state pairs; 288/288 R19 state pairs). Planning scores are not stored
    in those trajectory traces, so their equality rate is not assessable.
15. In R18/19, H10 has a complete real ten-step future at **3/12 = 25%** of
    decision times (126/504 R18 trace states and 72/288 R19 trace states).
16. H20 has no complete real future in the twelve-step task; in the original
    64-step R15--17 traces it is complete at **45/64 = 70.3125%** of times.
17. H5 Task-Waypoint labels are full-window labels at **100% of emitted
    samples**; eligible anchors are `t=0..7`, **8/12 = 66.67%** of possible
    decision times.
18. H10 labels are also full-window at **100% of emitted samples**; anchors
    are only `t=0..2`, **3/12 = 25%** of decision times. In the permitted
    train/validation data that is 1,920 full H10 and 0 truncated H10 samples;
    the historical all-split count remains 2,400.
19. Near-terminal Task-Waypoint targets are not shortened. The builder only
    emits `start in range(0, len(transitions)-H+1)`, so late anchors are
    omitted rather than labeled with a partial terminal window.
20. A 2,048-transition PPO batch is 256 rollout slots in each of eight
    environments. It contains many separately terminated/truncated 12-step
    episodes and does not grant 2,048-step credit.
21. PPO mean maximum future-lag depth is **5.5** steps (mean remaining reward
    terms 6.5); terminated and truncated both set the GAE nonterminal factor
    to zero, so there is no terminal bootstrap.
22. The actual twelve-step UAV path bound is **120 m** (`12 × 10 m`), not a
    hypothetical five-step 50 m bound.
23. 120 m is **42.43%** of the 282.84 m default map diagonal, **211.82%** of
    the 56.65 m median buoy-pair spacing, and **83.76%** of p90 pair spacing
    (143.27 m).
24. Instantaneous observed queue/service time is **1.00** slot median,
    **1.51** p75, and **3.69** p90. These are Q/current-service ratios, not
    guaranteed clearance times because arrivals continue.
25. Inter-region buoy-pair relocation is **10.84** slots p75 and **14.33**
    slots p90. Thus a 12-step episode covers p75 but not p90 relocation.
26. The 12-step task covers short action/service processes and some medium
    relocation, but not a complete H20 or p90 relocation; it cannot by itself
    substantiate a general terminal-aware long-horizon-control claim.
27. Round-19's locked **IPN-A** remains an immutable historical result, but
    it does not support a distinct terminal-aware H10 long-horizon claim.
28. The more accurate claim is: “under the registered nominal-shadow
    protocol, multi-step lookahead outperformed H1”; its H10-specific and
    terminal-aware interpretation is effective-horizon-limited.
29. H5/H10 Task-Waypoints are genuine complete intermediate windows where
    emitted, but not uniformly available across an episode: H5 has early
    8/12-anchor coverage and H10 only early 3/12-anchor coverage. They are
    not truncated episode summaries, but claims require that coverage limit.
30. Episode-Horizon Gate: **EH-C**.
31. `episode_horizon_redesign_required=true` for a future benchmark intended
    to test terminal-aware H20/general long-horizon coordination.
32. Round-32 Gate: **R32-C**.
33. Recommended future range: **11--20 slots**. The lower endpoint is
    `ceil(max(H10, p75 relocation, p75 Q/service))=11`; the primary endpoint
    is `ceil(max(H20, p90 relocation, p90 Q/service))=20`.
34. Yes, a genuinely long-episode benchmark should be designed before making
    a new long-horizon claim; this round does not implement it.
35. No historical round should be re-run automatically. Any re-run needs a
    subsequent explicit human authorization.
36. Fresh new rollout count: **0**.
37. Untouched-evaluation access count: **0**.
38. Round-33 was **not created**.

## Historical interpretation protection

No historical lock, gate, metric, figure, report, model, or dataset was
altered. `historical_gate_horizon_reassessment.{json,csv}` adds an
interpretation boundary only: Round-19 IPN-A is preserved as a historical
nominal-shadow result, while a terminal-aware H10-specific causal claim is
not identified by the locked traces.

## Audit evidence

The outputs include the requested four-horizon specification, cross-round
matrix, per-state effective planner horizon, HUR statistics, R18/R19 equality
audit, dynamic-round subset, Task-Waypoint reconstruction, PPO endpoint
audit, timescale tables, validity tables, gates, and 12 PNG/SVG figure pairs.
The raw Round-20 JSONL parser screened protected split tokens before JSON
decode; `development_test_decoded_count=0`.

## Final verification

The dedicated Round-32 test suite passed 17 tests. The complete repository
suite passed **1003 tests** (`py -3.14 -m unittest discover -s tests -q`) and
`py -3.14 -m compileall -q src/wt tests/test_round32_horizon_semantics.py`
passed before the Round-32 lock.
