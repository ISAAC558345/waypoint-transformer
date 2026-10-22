# Phase-5 HGNN High-Level + Shared Task Urgency

Replace the experimental high-level encoder with an HGNN and pass one shared queue urgency into the urgency-aware low-level.

- Do not edit the deployment decide() path, reward, safety projector, hold, or L3-B physics.
- Do not add an intermediate goal-evaluation layer, HAC/HIRO, or joint high/low updates.
- Do not read 220000–223063, 104000–106063, or 230000–230127.
- planner_ready and deployment_verified stay false.
