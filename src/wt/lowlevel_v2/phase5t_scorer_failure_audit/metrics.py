"""Gate for the failure-mechanism audit. Flags are computed from protocol thresholds."""
from __future__ import annotations


def choose_gate(*, data: bool, objective: bool, representation: bool, optimization: bool, temporal: bool) -> dict:
    """At least two strong mechanisms become T-E. The next phase may change only the first priority."""
    present = {
        "DATA": bool(data),
        "OBJ": bool(objective),
        "REP": bool(representation),
        "OPT": bool(optimization),
        "TEMP": bool(temporal),
    }
    priority = [name for name in ("REP", "TEMP", "OBJ", "DATA", "OPT") if present[name]]
    count = len(priority)
    if count >= 2:
        gate = "T-E"
    elif priority == ["DATA"]:
        gate = "T-A"
    elif priority == ["OBJ"]:
        gate = "T-B"
    elif priority in (["REP"], ["TEMP"]):
        gate = "T-C"
    elif priority == ["OPT"]:
        gate = "T-D"
    else:
        gate = "T-E"
        priority = ["NONE"]
    return {"gate": gate, "priority": priority, "gate_condition_met": count >= 1}


__all__ = ["choose_gate"]
