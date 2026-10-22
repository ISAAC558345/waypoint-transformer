"""Deterministic arrival-trace construction for task semantics v1.

The environment consumes a complete private trace but exposes only the arrival
that has already occurred.  Formal comparisons materialize one trace per
scenario case and pass identical immutable copies to every policy.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Optional, Tuple

import numpy as np

from .config import MaritimeEnvConfig


ARRIVAL_TRACE_SCHEMA_VERSION = "arrival_trace_v1"


def derive_scenario_trace_seed(
    scenario_id: str,
    scenario_seed: int,
    num_buoys: int,
    horizon: int,
) -> int:
    """Derive a stable private seed without using Python's randomized hash."""
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")
    for name, value, positive in (
        ("scenario_seed", scenario_seed, False),
        ("num_buoys", num_buoys, True),
        ("horizon", horizon, True),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        if int(value) < (1 if positive else 0):
            raise ValueError(f"{name} is outside its valid range")
    payload = (
        f"{ARRIVAL_TRACE_SCHEMA_VERSION}\x1f{scenario_id}\x1f"
        f"{int(scenario_seed)}\x1f{int(num_buoys)}\x1f{int(horizon)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def arrival_trace_id(
    trace_bits: np.ndarray,
    *,
    mode: str,
    trace_seed: Optional[int],
) -> str:
    """Return a content-addressed SHA-256 identifier for one trace."""
    trace = _validated_trace(trace_bits)
    header = {
        "schema_version": ARRIVAL_TRACE_SCHEMA_VERSION,
        "mode": str(mode),
        "trace_seed": None if trace_seed is None else int(trace_seed),
        "shape": list(trace.shape),
        "dtype": "float64-little-endian",
        "unit": "bit",
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    digest.update(np.asarray(trace, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


def materialize_arrival_trace(
    config: MaritimeEnvConfig,
    *,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """Materialize constant, Poisson-packet, or supplied trace arrivals.

    Poisson arrivals use a private ``numpy.random.Generator``.  The random
    variable is a packet count and is converted to bits using the configured
    packet size, avoiding a dimensionally ambiguous Poisson draw over bits.
    """
    if not isinstance(config, MaritimeEnvConfig):
        raise TypeError("config must be MaritimeEnvConfig")
    trace_seed = int(config.seed if seed is None else seed)
    shape = (int(config.episode_horizon), int(config.num_buoys))
    if config.arrival_mode == "constant":
        trace = np.full(
            shape,
            float(config.constant_arrival_bits_per_step),
            dtype=np.float64,
        )
    elif config.arrival_mode == "poisson":
        rng = np.random.default_rng(trace_seed)
        mean_packets = (
            float(config.poisson_arrival_packets_per_second)
            * float(config.delta_t)
        )
        trace = (
            rng.poisson(mean_packets, size=shape).astype(np.float64)
            * float(config.arrival_packet_size_bits)
        )
    elif config.arrival_mode == "trace":
        trace = np.asarray(config.arrival_trace_bits, dtype=np.float64)
        trace_seed = None  # supplied traces are identified by content
    else:  # guarded by config validation
        raise RuntimeError(f"unsupported arrival mode {config.arrival_mode!r}")

    trace = _validated_trace(trace, expected_shape=shape)
    identifier = arrival_trace_id(
        trace, mode=config.arrival_mode, trace_seed=trace_seed
    )
    if (
        config.arrival_trace_id is not None
        and str(config.arrival_trace_id) != identifier
    ):
        raise ValueError(
            "configured arrival_trace_id does not match arrival_trace_bits"
        )
    return trace.copy(), identifier


def _validated_trace(
    trace_bits: Any,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    try:
        trace = np.asarray(trace_bits, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("arrival trace must be a finite numeric matrix") from error
    if trace.ndim != 2:
        raise ValueError("arrival trace must have rank 2 [time, buoy]")
    if expected_shape is not None and trace.shape != expected_shape:
        raise ValueError(
            f"arrival trace must have shape {expected_shape}, got {trace.shape}"
        )
    if not np.all(np.isfinite(trace)) or np.any(trace < 0.0):
        raise ValueError("arrival trace must be finite and nonnegative")
    return trace.copy()


__all__ = [
    "ARRIVAL_TRACE_SCHEMA_VERSION",
    "arrival_trace_id",
    "derive_scenario_trace_seed",
    "materialize_arrival_trace",
]
