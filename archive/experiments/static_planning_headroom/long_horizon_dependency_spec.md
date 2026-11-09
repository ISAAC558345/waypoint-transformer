# Round-15 long-horizon dependency specification

Round-15 is a deterministic, single-UAV diagnostic audit. It evaluates whether
the existing queue, arrival, rate, geometry, movement, and energy state creates
measurable value for a finite action-lattice lookahead. It does not add a
controller, train a model, alter rewards, or change maritime environment
physics.

The primary comparison is paired actual cumulative transmitted payload for the
same new diagnostic scenario: causal H=5, H=10, or H=20 minus causal H=1.
Paired bootstrap units are scenario IDs. Secondary outcomes are final residual
queue, return, movement distance, energy, completion step, queue conservation,
and boundary violations.

The causal planner sees the current observation, static configuration, and the
pre-registered expected-arrival assumption only. It never reads a realized
future arrival trace. The oracle planner is a non-deployable upper-bound
diagnostic and is explicitly labelled `oracle_exogenous_knowledge=true`.

`Delayed benefit` is reported only when H=20 sacrifices immediate causal
one-step payload relative to H=1 and its labelled oracle continuation produces
strictly more future payload. The `current_queue_tie_bridge_proxy` is not a
Round-7 bridge result: it is merely whether the largest two present queues are
within 10 percent. No Round-7 development, holdout, trajectory, or result
artifact is used for this relation.

PN-A, PN-B, and PN-C are descriptive classifications derived from the paired
audit. There is no artificial PASS threshold and the audit does not authorize a
future round. The one historical metadata-only holdout access recorded before
Round-15 initialization is disclosed in manifests and excluded from all audit
inputs and analyses.
