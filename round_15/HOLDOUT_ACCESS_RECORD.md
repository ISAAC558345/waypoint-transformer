# Round-15 historical holdout access record

Before the Round-15 protocol and outputs were initialized, one command read the
first ten lines of `round_07/outputs/.../trajectory_quality_audit.csv`. This
was an accidental metadata-only inspection; no analysis, statistic, scenario,
policy choice, threshold, or figure uses that material. The user explicitly
authorized recording this access and continuing on 2026-08-25.

The manifest therefore records `untouched_holdout_accessed=true` and
`untouched_holdout_access_count=1`. No subsequent Round-15 command reads any
old development-test, fresh-development, holdout, or untouched-holdout input.
All Round-15 data are newly materialized under the
`round15_planning_diagnostic` namespace.
