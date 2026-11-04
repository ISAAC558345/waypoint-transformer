"""Registered R1 budget. These values are the run contract, not a search space."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "legacy_formal_repro_r1"
CONFIG_DIR = OUTPUT / "configs"
EPISODE_DIR = OUTPUT / "episodes"
ARCHIVE_DIR = OUTPUT / "evidence_archive"
SCENARIOS = (322000, 322001)
ARMS = ("scheme_c", "g1_rate")
HORIZON = 48
STEP_BUDGET = 192
PROCESS_SEED_OFFSET = 1_000_000
DECISION_TIMES = (0, 15, 30, 45)
INSTRUCTION_COMMIT = "d2b214f35c5c35133f8412c57b86cd3605ef8ed6"
EVIDENCE_COMMIT = "8a34efc718665a7cfe4375b662e8f5249263f03d"
REMOVAL_COMMIT = "adb0f6fa59cdb488c50f8b49d2c42a489dd8fb80"
FAILURE_LOG_COMMIT = "7e9e372e6ab603342f59c334301cdcc42ccc2e61"
FAILURE_LOG_PATH = "results/ocean_observation_m0/baseline_tests.log"
HISTORICAL_REQUIREMENTS = ROOT / "requirements.txt"

SUMMARY_SOURCES = (
    "results/phase6iab_planner_stability/summary.json",
    "results/phase6iac_independent_holdout/summary.json",
    "results/phase6if_ranking_loss/summary.json",
)


def episode_sources() -> tuple[str, ...]:
    """Episode JSON that belongs to the two saved summaries. This does not open a new episode."""
    rows = [f"results/phase6iab_planner_stability/episodes/scenario_{scenario}.json" for scenario in range(322048, 322064)]
    rows.extend(
        f"results/phase6iac_independent_holdout/episodes/scenario_{scenario}.json" for scenario in range(322064, 322080)
    )
    return tuple(rows)


def training_sources() -> tuple[str, ...]:
    """Ranking-loss text records. Checkpoints stay referenced by hash."""
    return (
        "results/phase6if_ranking_loss/label_census.json",
        "results/phase6if_ranking_loss/weight_audit.json",
        "results/phase6if_ranking_loss/validation_loss_backfill.json",
        "results/phase6if_ranking_loss/runs/a/result.json",
        "results/phase6if_ranking_loss/runs/b/result.json",
        "results/phase6if_ranking_loss/runs/c/result.json",
    )


def checkpoint_sources() -> tuple[str, ...]:
    names = []
    for run_name, epochs in (("a", (1, 2)), ("b", (1,)), ("c", (1, 2, 3, 4, 5))):
        for epoch in epochs:
            names.append(f"results/phase6if_ranking_loss/runs/{run_name}/epoch_{epoch:03d}.pt")
    return tuple(names)
