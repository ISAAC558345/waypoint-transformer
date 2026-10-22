# Frozen multi-UAV safety specification

`multi_uav_min_separation_m = 15.0` is frozen for Round-18.  The existing
maximum one-slot motion is 10 m in a 200 m by 200 m map; 15 m is therefore a
fixed 1.5-step separation rather than a new task reward scale.

After every per-UAV boundary projection, every pair closer than 15 m is
separated by an equal-and-opposite displacement along its current pair axis.
For zero separation, the axis is deterministically derived from the unordered
pair index but its two corrections remain exact opposites.  Corrections are
then clipped to the existing map bounds and repeated over a fixed number of
pair passes.  The operation is permutation-equivariant: relabelling UAV rows
and unlabelling the result gives the same physical positions.  It reports raw
collision-risk pairs, corrected pairs, minimum distance, and correction norms.

Safety is a hard feasibility projection.  No collision reward, role bonus, or
planning bonus is introduced.
