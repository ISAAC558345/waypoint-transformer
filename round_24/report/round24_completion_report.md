# Round-24 Completion Report

Round-24 was a pre-registered pure seed extension. No algorithm, predictor, data split, environment, sampler, loss, architecture, optimizer, or learning-rate field changed.

## Required answers

1. New training seeds: 3, 4, 5, 6, 7 for each of the three frozen deployable families.
2. New formal runs: 15/15 completed.
3. Eight validation seed payload deltas (bits): 0: -89003.28, 1: 73965.71, 2: 112667.61, 3: 149848.39, 4: -4388.94, 5: 91023.58, 6: -193.65, 7: -55381.85.
4. Validation positive-seed count: 4.
5. Validation median seed delta: 36886.03 bits.
6. Validation hierarchical 95% CI: [-61750.94, 132450.48] bits.
7. Eight fresh seed payload deltas (bits): 0: -29633.66, 1: -122301.34, 2: 162930.44, 3: 185504.81, 4: -51115.19, 5: 41743.09, 6: 127030.05, 7: -23413.50.
8. Fresh positive-seed count: 4.
9. Fresh median seed delta: 9164.79 bits.
10. Fresh mean seed delta: 36343.09 bits.
11. Fresh hierarchical 95% CI: [-45090.54, 120349.28] bits.
12. Fresh seed-cluster 95% CI: [-35872.30, 107962.18] bits.
13. Leave-one-seed-out stable: true (remove 0: 45768.34; remove 1: 59006.58; remove 2: 18259.18; remove 3: 15034.27; remove 4: 48837.13; remove 5: 35571.66; remove 6: 23387.81; remove 7: 44879.74).
14. Single-seed/tail domination: true.
15. Conflict eight-seed validation result: s0=-161025.81; s1=94595.90; s2=170923.49; s3=221039.07; s4=-96399.59; s5=172314.51; s6=-5070.98; s7=-97260.38.
16. Bridge eight-seed validation result: s0=-89003.28; s1=73965.71; s2=112667.61; s3=149848.39; s4=-4388.94; s5=91023.58; s6=-193.65; s7=-55381.85; fresh bridge is unavailable without future labels.
17. Eta=.25 fresh result: s0=76346.03; s1=-14585.63; s2=26031.69; s3=103306.84; s4=86662.05; s5=-48278.93; s6=109409.00; s7=-73902.74.
18. Eta=.5 fresh result: s0=79715.69; s1=-129797.52; s2=291368.27; s3=90860.01; s4=22864.21; s5=54751.37; s6=69222.99; s7=-40206.37.
19. K=3 fresh result: s0=-65937.53; s1=30230.18; s2=141918.05; s3=146374.20; s4=-96774.56; s5=50760.08; s6=49109.07; s7=-95760.51.
20. K=5 fresh result: s0=6670.21; s1=-274832.85; s2=183942.83; s3=224635.43; s4=-5455.81; s5=32726.10; s6=204951.04; s7=48933.51.
21. Task-WP Transformer fresh seed variance: 7190753008.58.
22. State-only Transformer fresh seed variance: 6241018043.16.
23. Task-WP MLP fresh seed variance: 1362016498.90.
24. Transformer vs MLP: TV8-B; per-seed deltas: 0: -102116.86, 1: 43745.19, 2: -4063.58, 3: 134153.30, 4: -85543.19, 5: -72934.20, 6: 107543.90, 7: 18808.77.
25. Return dominance persists: true (dominant head by seed: {0: 'return', 1: 'return', 2: 'return', 3: 'return', 4: 'return', 5: 'return', 6: 'return', 7: 'return'}).
26. Offline RMSE-to-fresh-payload correlation (descriptive only): Pearson=0.524, Spearman=0.571.
27. MS Gate: MS-C.
28. conditional_waypoint_value_confirmed: false.
29. candidate_ready_for_untouched_eval: false.
30. More training seeds recommended: false; no seed 8+ was trained.
31. Round-22 fresh new rollout count: 0.
32. Round-20 development-test reuse count: 0.
33. Untouched-evaluation access count: 0.
34. Round-24 fresh access count: 1.
35. Round-25 created: false.

Fresh bridge conditional analysis is explicitly marked unavailable because deriving that label would require future labels; validation bridge uses the frozen Round-20 definition.
