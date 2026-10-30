"""Opportunity labels for the matched-mask sample. They are not a performance pass."""
from __future__ import annotations

from .protocol import (
    EXPECTED_SNAPSHOTS,
    OPPORTUNITY_ABSENT,
    OPPORTUNITY_LOCAL,
    OPPORTUNITY_REPLICATED,
    OPPORTUNITY_THIN,
)


def opportunity_evidence(rows: list[dict], snapshots: list[dict]) -> str:
    """Judge replicated, local, absent, then insufficient. Missing candidates are not absence."""
    alternatives = [row for row in rows if row.get("interpretable") and not row.get("is_baseline")]
    replicated = {int(row["scenario_id"]) for row in alternatives if row.get("s2")}
    if len(replicated) >= 2:
        return OPPORTUNITY_REPLICATED
    if any(row.get("s0") for row in alternatives):
        return OPPORTUNITY_LOCAL
    covered = len(snapshots) == int(EXPECTED_SNAPSHOTS) and all(int(item.get("interpretable_alternatives", 0)) >= 1 for item in snapshots)
    if covered:
        return OPPORTUNITY_ABSENT
    return OPPORTUNITY_THIN
