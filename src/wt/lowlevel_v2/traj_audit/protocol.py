"""Audit protocol. Frozen models only."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.integration_phase2.protocol import LOW_SEEDS, OLD_HL_CKPT, UTILITY_EPS, ckpt_path
from wt.lowlevel_v2.integration_phase2b.protocol import abs_ckpt
from wt.lowlevel_v2.integration_phase3.protocol import exec_ckpt, prefix_ckpt
from wt.lowlevel_v2.phase1.protocol import load_pools
from wt.lowlevel_v2.protocol import ROOT

AUDIT = Path(__file__).resolve().parent
OUTPUT = AUDIT / "outputs"

HL_SPECS = (
    {"family": "OLD", "seed": "old0", "path": OLD_HL_CKPT},
    {"family": "ABS", "seed": 0, "path": abs_ckpt(0)},
    {"family": "PREFIX", "seed": 0, "path": prefix_ckpt(0)},
    {"family": "EXEC", "seed": 0, "path": exec_ckpt(0)},
)

ORACLE_FAMILIES = ("PREFIX", "EXEC")
LL_RUN = 0


__all__ = [
    "AUDIT",
    "HL_SPECS",
    "LL_RUN",
    "LOW_SEEDS",
    "OLD_HL_CKPT",
    "ORACLE_FAMILIES",
    "OUTPUT",
    "ROOT",
    "UTILITY_EPS",
    "abs_ckpt",
    "ckpt_path",
    "exec_ckpt",
    "load_pools",
    "prefix_ckpt",
]
