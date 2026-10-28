"""Bounded bonus on bits served from tag=1 FIFO buckets.

The numerator is served_tagged_bits. Age-weighted urgency is not an input.
"""
from __future__ import annotations

from wt.benchmark_v2.phase6i_priority_tag_reward.reward import capped_bonus

from .protocol import LAMBDA_TAG, RHO_TAG


def validate_config(lambda_tag: float, rho_tag: float) -> None:
    if float(lambda_tag) != float(LAMBDA_TAG) or float(rho_tag) != float(RHO_TAG):
        raise RuntimeError("L3 reward config is not the preregistered candidate")
    if not (0.0 < float(RHO_TAG) < float(LAMBDA_TAG)):
        raise RuntimeError("L3 cap must stay below the tagged coefficient")


def direct_tag_bonus(served_tagged_bits: float, served_bits: float) -> float:
    """Return min(lambda_tag * tagged/1e6, rho_tag * served/1e6), or 0 when nothing is served."""
    validate_config(LAMBDA_TAG, RHO_TAG)
    tagged = float(served_tagged_bits)
    served = float(served_bits)
    if tagged < 0.0 or served < 0.0 or tagged - served > 1.0:
        raise RuntimeError("tagged service is outside the bits served on this step")
    return capped_bonus(tagged, served, float(LAMBDA_TAG), float(RHO_TAG))
