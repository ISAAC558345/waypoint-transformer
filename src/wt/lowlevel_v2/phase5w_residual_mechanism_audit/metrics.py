"""Gate for the residual-mechanism audit. No model is trained."""
from __future__ import annotations


def choose_gate(*, shortcut: bool, synergy: bool) -> str:
    if bool(shortcut) and bool(synergy):
        return "W-C"
    if bool(synergy):
        return "W-B"
    if bool(shortcut):
        return "W-A"
    return "W-D"


def synergy_value(delta_all: float, single_gains: list[float]) -> float:
    return float(delta_all) - float(sum(single_gains))


def synergy_ratio(delta_all: float, single_gains: list[float]) -> float:
    positive = sum(value for value in single_gains if value > 0.0)
    return float(delta_all - positive) / (abs(float(delta_all)) + 1.0e-6)


__all__ = ["choose_gate", "synergy_ratio", "synergy_value"]
