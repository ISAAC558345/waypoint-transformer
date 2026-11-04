"""Shared-slot order frozen before the derived B2 comparison.

The order is a total order. A later transmission waits only when its
proposed interval would occupy the shore channel before the earlier one
has finished, including the existing MAC guard.
"""
from __future__ import annotations

PHASE_ORDER = ("command", "ack", "payload", "backhaul")
EQUIVALENCE_LABEL = "策略等价、复用已算结果"
TELEMETRY_ON_DELIVERY_CLOCK = False
MAX_DERIVED_METHOD_ROWS = 12
SOURCE_RESULT_COMMIT = "12ff4ae7fdf3863ac56c3a81bd2168983453043b"


def sort_key(event: dict) -> tuple:
    """generation, UAV id, buoy index, then command / ack / payload / backhaul."""
    return (
        float(event["generation_s"]),
        int(event["uav"]),
        int(event["buoy_index"]),
        PHASE_ORDER.index(str(event["phase"])),
    )
