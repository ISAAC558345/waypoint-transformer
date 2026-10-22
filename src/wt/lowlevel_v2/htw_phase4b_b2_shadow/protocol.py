"""B2 shadow-integration protocol. Frozen ExtraTrees + Phase-4A labels only."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.htw_phase4b_b2_validation.protocol import ET, N_PERM, N_SPLITS, PERM_SEED
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
REPORT_NAME = "HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md"
PRIMARY = "EXEC"

# Phase-3E candidate-conditioned geometry only. No leftover / state / future keys.
P3E_CAND_KEYS = (
    "idle_count",
    "idle_frac",
    "n_unique_buoys",
    "competition_count",
    "mean_uav_target_dist",
    "max_uav_target_dist",
    "n_in_range",
    "assigned_queue_mean",
    "cand_mean_sinr",
    "cand_mean_rate",
    "cand_mean_interference",
    "assign_0",
    "assign_1",
    "assign_2",
    "assign_3",
    "assign_4",
    "assign_5",
)
B2_FEATURE_KEYS = ("U_single",) + P3E_CAND_KEYS

# Locked LSO numbers from HTW_Phase4B_B2_IndependentValidation_Audit_CN.md
LOCKED_R_B2_STATE = 0.9164176101676103
LOCKED_R_LOCAL_STATE = 0.7582407176157175
LOCKED_R_B2_FOLD = 0.9146000296000296
LOCKED_R_LOCAL_FOLD = 0.7535336885336885
LOCKED_LSO_CSV = (
    ROOT
    / "src"
    / "wt"
    / "lowlevel_v2"
    / "htw_phase4b_b2_validation"
    / "outputs"
    / "phase4b_b2_ranking.csv"
)
PLANNER_PATHS = (
    ROOT / "src" / "wt" / "htw_phase100" / "high_level.py",
    ROOT / "src" / "wt" / "htw_phase943" / "evaluation.py",
    ROOT / "src" / "wt" / "htw_phase945" / "candidate_pool.py",
)
CONS_ATOL_R = 1e-6
PERM_DROP_EXISTING = 0.03

__all__ = [
    "B2_FEATURE_KEYS",
    "CONS_ATOL_R",
    "ET",
    "FIG",
    "LOCKED_LSO_CSV",
    "LOCKED_R_B2_FOLD",
    "LOCKED_R_B2_STATE",
    "LOCKED_R_LOCAL_FOLD",
    "LOCKED_R_LOCAL_STATE",
    "N_PERM",
    "N_SPLITS",
    "OUTPUT",
    "P3E_CAND_KEYS",
    "PERM_DROP_EXISTING",
    "PERM_SEED",
    "PHASE",
    "PLANNER_PATHS",
    "PRIMARY",
    "REPORT_NAME",
    "ROOT",
]
