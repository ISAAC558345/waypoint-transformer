"""Development-set audit paths. Does not allocate scenarios or fit models."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
REPORT_NAME = "HTW_Phase4E_RemediationDevelopmentAudit_CN.md"
TAU = 1534933.5944128654

__all__ = ["OUTPUT", "REPORT_NAME", "ROOT", "TAU"]
