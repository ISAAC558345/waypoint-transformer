"""Confirm this diagnosis still sits on the frozen L0 and the published E17 result."""
from __future__ import annotations

import json

from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.identity import (
    contract_problems as e17_contract_problems,
    history_problems as e17_history_problems,
)
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.session import watched_problems

from .artifacts import git_bytes, text_matches
from .protocol import (
    ALTERNATIVE_CAP,
    CANDIDATE_LIMIT,
    E17_GATE,
    E17_OPPORTUNITY,
    E17_OUTPUT,
    ROOT,
    SOURCE_COMMIT,
)


FROZEN_PATHS = (
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/__init__.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/attribution.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/candidates.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/criteria.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/identity.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/protocol.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/report.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/run.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_feasibility/session.py",
    "tests/test_priority_tag_assignment_feasibility.py",
    "docs/research/HTW_UrgencyAware_Assignment_Feasibility_Report_CN.md",
    "results/phase6i_priority_tag_assignment_feasibility/summary.json",
    "results/phase6i_priority_tag_assignment_feasibility/baseline_rows.csv",
    "results/phase6i_priority_tag_assignment_feasibility/candidate_rows.csv",
    "results/phase6i_priority_tag_assignment_feasibility/snapshot_manifest.json",
)


def contract_problems() -> list[str]:
    problems = list(e17_contract_problems())
    if int(ALTERNATIVE_CAP) != 4 or int(CANDIDATE_LIMIT) != 5:
        problems.append("matched-mask cap is not baseline plus four alternatives")
    return problems


def history_problems() -> list[str]:
    problems = list(e17_history_problems())
    summary = json.loads((E17_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    if summary.get("scientific_gate") != E17_GATE:
        problems.append("E17 gate was rewritten")
    if summary.get("opportunity_evidence") != E17_OPPORTUNITY:
        problems.append("E17 opportunity label was rewritten")
    if summary.get("release_eligible") or summary.get("merged"):
        problems.append("E17 release or merge flag was rewritten")
    if int(summary.get("training_updates", -1)) != 0:
        problems.append("E17 training count was rewritten")
    return problems


def frozen_source_problems() -> list[str]:
    problems = list(watched_problems())
    for relative in FROZEN_PATHS:
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"frozen E17 file is missing: {relative}")
            continue
        try:
            blob = git_bytes(SOURCE_COMMIT, relative)
        except RuntimeError as exc:
            problems.append(str(exc))
            continue
        if not text_matches(path, blob):
            problems.append(f"frozen E17 file changed after the source commit: {relative}")
    return problems
