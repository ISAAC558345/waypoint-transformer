"""Keep E15 through E18 results and the frozen low-level sources unchanged."""
from __future__ import annotations

import json

from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.artifacts import git_bytes, text_matches
from wt.benchmark_v2.phase6i_priority_tag_assignment_coverage.identity import (
    frozen_source_problems as e18_frozen_source_problems,
    history_problems as e18_history_problems,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256

from .protocol import (
    E17_GATE,
    E17_OPPORTUNITY,
    E17_OUTPUT,
    E18_GATE,
    E18_OPPORTUNITY,
    E18_OUTPUT,
    FORMAL_ACTOR_SHA256,
    ROOT,
    SOURCE_COMMIT,
    scenario_contract,
)


E18_PATHS = (
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/__init__.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/artifacts.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/codec.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/coverage.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/criteria.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/detail.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/identity.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/prepare.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/protocol.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/replay.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/report.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/run.py",
    "src/wt/benchmark_v2/phase6i_priority_tag_assignment_coverage/selection.py",
    "tests/test_priority_tag_assignment_coverage.py",
    "docs/research/HTW_UrgencyAware_Assignment_Coverage_Diagnosis_Report_CN.md",
    "results/phase6i_priority_tag_assignment_coverage/summary.json",
    "src/wt/benchmark_v2/phase6e/generators.py",
    "src/wt/benchmark_v2/phase6c/channel.py",
    "src/wt/l3b/config.py",
    "src/wt/l3b/env.py",
)


def contract_problems() -> list[str]:
    return list(scenario_contract())


def history_problems() -> list[str]:
    problems = list(e18_history_problems())
    e17 = json.loads((E17_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    e18 = json.loads((E18_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    if e17.get("scientific_gate") != E17_GATE or e17.get("opportunity_evidence") != E17_OPPORTUNITY:
        problems.append("E17 published gate or opportunity label changed")
    if e17.get("release_eligible") or e17.get("merged") or int(e17.get("training_updates", -1)) != 0:
        problems.append("E17 release, merge, or training flag changed")
    if e18.get("scientific_gate") != E18_GATE or e18.get("opportunity_evidence") != E18_OPPORTUNITY:
        problems.append("E18 published gate or opportunity label changed")
    if e18.get("release_eligible") or e18.get("merged") or int(e18.get("training_updates", -1)) != 0:
        problems.append("E18 release, merge, or training flag changed")
    if not e18.get("opportunity_published"):
        problems.append("E18 opportunity publication flag changed")
    return problems


def frozen_source_problems() -> list[str]:
    problems = list(e18_frozen_source_problems())
    if file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
        problems.append("formal low-level checkpoint hash is not the registered SHA-256")
    for relative in E18_PATHS:
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"frozen source is missing: {relative}")
            continue
        try:
            blob = git_bytes(SOURCE_COMMIT, relative)
        except RuntimeError as exc:
            problems.append(str(exc))
            continue
        if relative.endswith(".pt"):
            if path.read_bytes() != blob:
                problems.append(f"binary source changed: {relative}")
            continue
        if not text_matches(path, blob):
            problems.append(f"frozen source changed after the E18 result: {relative}")
    return problems
