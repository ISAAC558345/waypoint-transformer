# Round-34 Working Rules

Round-34 is a **frozen Benchmark Complexity, Relational-Decision Opportunity,
and Method Suitability Audit**. It has no model fitting, no model inference,
no environment execution, no planner execution, no new rollout, and no
modification of environment physics, rewards, the planner, historic datasets,
or completed-round artifacts.

Permitted scientific evidence is limited to source/configuration inspection,
the clean-manifest-listed frozen Round-20 train/validation records,
combinatorial calculations, and byte-stable Round-33 reports. Development,
untouched, unknown, and alias-unresolved records must be rejected by lexical
metadata allowlisting before JSON decoding. The audit may write only Round-34
artifacts.

One accidental `development_test` decode occurred during the preflight and is
recorded in `ROUND34_PROTECTED_ACCESS_EXCEPTION.json`. It is a governance
exception, not usable evidence. All formal scientific artifacts must derive
only from the hash-locked clean manifests; a second protected decode is a hard
stop.

The audit must distinguish neural-policy outputs from deterministic matching,
fixed resource rules, and environment safety correction. It must state the
actual M=2, K in {3,5} benchmark constraints rather than substitute K^M for
the legal injective-assignment count. It may propose a future benchmark design
but must not implement it.

Round-34 stops after its own lock. It cannot create Round-35.
