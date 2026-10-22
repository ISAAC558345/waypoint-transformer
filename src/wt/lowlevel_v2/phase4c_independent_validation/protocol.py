"""Phase-4C independent-validation protocol. Frozen B2, new eval labels only."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.htw_phase4b_b2_shadow.protocol import B2_FEATURE_KEYS, P3E_CAND_KEYS
from wt.lowlevel_v2.htw_phase4b_b2_validation.protocol import ET, N_PERM, PERM_SEED
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
REPORT_NAME = "HTW_Phase4C_IndependentValidation_Audit_CN.md"
PRIMARY = "EXEC"
ALIGN_ATOL = 1.0
PERM_DROP_EXISTING = 0.03

P4B_DESIGN = tuple(range(221000, 221016))
P4B_FOLDS = (
    {
        "fold": 0,
        "train": (221001, 221002, 221003, 221004, 221006, 221007, 221008, 221009, 221011, 221012, 221013, 221014),
        "validation": (),
        "test": (221000, 221005, 221010, 221015),
    },
    {
        "fold": 1,
        "train": (221000, 221001, 221002, 221003, 221005, 221006, 221007, 221008, 221010, 221011, 221012, 221013, 221015),
        "validation": (),
        "test": (221004, 221009, 221014),
    },
    {
        "fold": 2,
        "train": (221000, 221001, 221002, 221004, 221005, 221006, 221007, 221009, 221010, 221011, 221012, 221014, 221015),
        "validation": (),
        "test": (221003, 221008, 221013),
    },
    {
        "fold": 3,
        "train": (221000, 221001, 221003, 221004, 221005, 221006, 221008, 221009, 221010, 221011, 221013, 221014, 221015),
        "validation": (),
        "test": (221002, 221007, 221012),
    },
    {
        "fold": 4,
        "train": (221000, 221002, 221003, 221004, 221005, 221007, 221008, 221009, 221010, 221012, 221013, 221014, 221015),
        "validation": (),
        "test": (221001, 221006, 221011),
    },
)

__all__ = [
    "ALIGN_ATOL",
    "B2_FEATURE_KEYS",
    "ET",
    "FIG",
    "N_PERM",
    "OUTPUT",
    "P3E_CAND_KEYS",
    "P4B_DESIGN",
    "P4B_FOLDS",
    "PERM_DROP_EXISTING",
    "PERM_SEED",
    "PHASE",
    "PRIMARY",
    "REPORT_NAME",
    "ROOT",
]
