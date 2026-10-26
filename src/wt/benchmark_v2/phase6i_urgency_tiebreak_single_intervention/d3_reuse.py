"""Read the D3 t=0 choice. Later states are not reused from that pilot."""
from __future__ import annotations

import csv

import numpy as np

from .protocol import D3_OUTPUT, EPS_BITS


def _assignment(text: str) -> np.ndarray:
    return np.asarray([int(item) for item in str(text).split("|")], dtype=np.int64)


def load_t0(scenario_id: int) -> dict:
    path = D3_OUTPUT / "decision_rows.csv"
    if not path.is_file():
        raise RuntimeError("D3 decision log is missing")
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["scenario_id"]) == int(scenario_id) and int(row["time_step"]) == 0:
                rows.append(row)
    tie = next((row for row in rows if row["arm"] == "payload_first_urgency_tiebreak"), None)
    if tie is None:
        raise RuntimeError(f"D3 has no t=0 tie-break row for {scenario_id}")
    if int(tie["payload_gap_within_one_bit"]) != 1 or abs(float(tie["payload_gap_bits"])) > float(EPS_BITS):
        raise RuntimeError("D3 t=0 row already violates the 1-bit payload tie")
    if int(tie["label_horizon"]) != 15 or tie["sampling_source"] != "stored_j15":
        raise RuntimeError("D3 t=0 row is not the stored 15-step sample")
    return {
        "state_hash": str(tie["state_hash"]),
        "pool_hash": str(tie["pool_hash"]),
        "subset_hash": str(tie["subset_hash"]),
        "subset_count": int(tie["subset_count"]),
        "pool_count": int(tie["pool_count"]),
        "sampling_source": str(tie["sampling_source"]),
        "payload_first_assignment": str(tie["payload_first_assignment"]),
        "tiebreak_assignment": str(tie["tiebreak_assignment"]),
        "tiebreak_vector": _assignment(tie["tiebreak_assignment"]),
        "payload_first_vector": _assignment(tie["payload_first_assignment"]),
        "label_payload_gap_bits": float(tie["payload_gap_bits"]),
        "label_urgent_gain_bits": float(tie["urgent_gain_bits"]),
        "payload_tie_count": int(tie["payload_tie_count"]),
        "assignment_changed": int(tie["assignment_changed"]),
        "max_label_ledger_gap_bits": float(tie["max_label_ledger_gap_bits"]),
        "max_label_accounting_error_bits": float(tie["max_label_accounting_error_bits"]),
    }
