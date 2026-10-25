"""Stage A paths and the fixed episode-age scale. No training settings."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DECISION_TIMES, HOLD
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_stage_a"
SCHEMA_VERSION = "episode_admission_age_v1"
AGE_SCALE_STEPS = 48
AGE_ORIGIN_RESET = "reset_convention"
AGE_ORIGIN_ARRIVAL = "environment_arrival"
INSTRUCTION_COMMIT = "cd383b0b7e42bd24e5d18968a475489d18953a70"
DUST_BITS = 1.0e-8


def smoke_scenario_ids() -> tuple[int, ...]:
    """First two registry train ids. Reserved ids are refused."""
    chosen = tuple(int(item) for item in tuple(TRAIN_POOL)[:2])
    if len(chosen) != 2 or any(item in RESERVED_TEST for item in chosen):
        raise RuntimeError("smoke split is not two non-reserved train scenarios")
    return chosen
