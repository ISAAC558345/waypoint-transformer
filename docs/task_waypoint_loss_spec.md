# Task-Waypoint future training loss specification

No loss is optimized in Round-10.  A later authorized round may use masked
spatial Huber/MSE, masked service cross entropy, masked per-buoy gross-service
regression, and an H-step-return scalar regression.  Padding buoy weights are
zero; `service_valid=0` excludes no-service windows.  Each head has an
independent physical normalization; return standardization must use train only.
Loss weights are deliberately not chosen before a training-free scale audit.
