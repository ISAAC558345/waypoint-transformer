# Phase-5ZA Correspondence-Preserving Pooling

Replace only the assignment mean-pool. Do not train the encoder.

- Train phi and a fresh Phase-5Y-shaped value head on 310100-310109.
- Keep the Phase-5L encoder, REP, TEMP, candidate pool, and MAPPO frozen.
- Pass the structural permutation and swap-collision gate before any epoch.
- Do not use 310110-310117 for gradients, early stopping, or hyperparameter search.
- Do not open 310118-310125 or 230000-230127.
- Do not modify Residual MAPPO, the reward, B2, tau, Gate-1, Candidate-C, or the planner.
- Do not call deployment decide().
- planner_ready and deployment_verified stay false.
