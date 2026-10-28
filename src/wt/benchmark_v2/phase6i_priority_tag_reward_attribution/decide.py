"""Map attribution measurements to one next-design letter.

The rule is fixed before the replay table is interpreted:
C when saved rows, summary intervals, or the age/tag decomposition disagree.
B when at least two L2 seeds have a positive age-service paired mean that is
larger than the tag-increment paired mean.
A in every other valid case.
This module does not select a formula and does not start training.
"""
from __future__ import annotations


def next_design(
    reproduced: bool,
    decomposition_ok: bool,
    summary_matches: bool,
    seed_rows: list[dict],
) -> str:
    if not reproduced or not decomposition_ok or not summary_matches:
        return "C"
    if len(seed_rows) < 1:
        return "C"
    age_larger = [
        float(row["age_delta_mean"]) > float(row["tag_increment_delta_mean"]) and float(row["age_delta_mean"]) > 0.0
        for row in seed_rows
    ]
    if sum(bool(item) for item in age_larger) >= 2:
        return "B"
    return "A"
