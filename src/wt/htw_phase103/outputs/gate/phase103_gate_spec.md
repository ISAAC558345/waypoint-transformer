# In-range zero-motion gate

Pre-action only. `gate = non-idle AND valid target AND queue>0 AND env._in_range`.
If true, env movement input is (0,0); else executor output is unchanged.
Idle / empty queue: gate off, original executor motion kept. No extra radius. No hold change.
Same wrapper for B2 and B3. Not PPO training.
