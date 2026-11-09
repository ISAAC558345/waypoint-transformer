# Round-11 Working Rules

- Scope is supervised training and validation of the three Round-10 task-aware
  communication waypoint labels only: service distribution, gross transmitted
  payload, and discounted H-step return.
- Use only the locked Round-10 train/validation samples.  The old development
  test and untouched holdout are content-isolated and never loaded, tuned on,
  evaluated, or plotted.
- The model input is exactly observable state plus buoy padding mask.  Do not
  introduce a constant goal, task-goal embedding, spatial head, policy,
  Transformer, reinforcement learning, or closed-loop evaluation.
- Preserve earlier locks and stop after Round-11; do not create Round-12.
