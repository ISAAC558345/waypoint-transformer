# Round-29 Completion Report

1. Original DSP: 0.090000--2.463855 train-minmax unserved mass and frozen train-minmax distance, equal .5/.5.
2. unserved_high_queue_mass is a raw sum over unserved buoy queues divided by one-buoy capacity; it has implicit K dependence.
3. Train-only confirmation primary variant: DSP-V3.
4. Confirmation scenarios / states: 40 / 160.
5. Overall active utility + CI: 165986.577849 [25030.007612, 314343.602791].
6. Activation / precision / recall / harmful: 0.250000 / 0.350000 / 0.666667 / 0.150000.
7. K3 utility + CI: 89064.357831 [0.000000, 385945.550602].
8. K5 utility + CI: 203023.202303 [27494.996188, 401543.571140].
9. Cross-K interaction: 113958.844471 [-223096.152603, 343582.609625].
10. Eta=.25/.5 active utility: 27097.984363 / 240772.743573.
11. Cross-Scale Gate: CS-C.
12. cross_scale_trigger_confirmed: False.
13. task_scale_conditional_trigger: False.
14. minimal_trigger_model_training_ready: False.
15. Recommendation: R29-C_stop_planner_trigger_representation_route.
16. No neural/model/policy training, protected-data access, or Round-30 creation occurred.
