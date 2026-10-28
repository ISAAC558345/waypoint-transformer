"""Two capped bonuses. Age urgency and tagged bits stay separate inputs."""
from __future__ import annotations

from .protocol import LAMBDA_AGE, LAMBDA_TAG, RHO_AGE, RHO_TAG, TOTAL_BONUS_CAP


def validate_config() -> None:
    if (float(LAMBDA_AGE), float(RHO_AGE), float(LAMBDA_TAG), float(RHO_TAG)) != (0.0005, 0.00025, 0.0010, 0.00075):
        raise RuntimeError("L3 coefficients are not the preregistered candidate")
    if abs(float(RHO_AGE) + float(RHO_TAG) - float(TOTAL_BONUS_CAP)) > 1.0e-15:
        raise RuntimeError("age and tag caps do not add up to the payload cap")


def split_bonus(age_service_bits: float, served_tagged_bits: float, served_bits: float) -> tuple[float, float]:
    """Return bonus_age, bonus_tag. A zero-service step returns zeros."""
    validate_config()
    age = float(age_service_bits)
    tagged = float(served_tagged_bits)
    served = float(served_bits)
    if age < 0.0 or tagged < 0.0 or served < 0.0:
        raise RuntimeError("service inputs to the L3 bonus are negative")
    if tagged - served > 1.0 or age - served > 1.0:
        raise RuntimeError("age or tagged service exceeds the bits served on this step")
    if served == 0.0:
        return 0.0, 0.0
    payload_term = served / 1.0e6
    age_term = age / 1.0e6
    tag_term = tagged / 1.0e6
    bonus_age = float(min(float(LAMBDA_AGE) * age_term, float(RHO_AGE) * payload_term))
    bonus_tag = float(min(float(LAMBDA_TAG) * tag_term, float(RHO_TAG) * payload_term))
    if bonus_age < 0.0 or bonus_tag < 0.0:
        raise RuntimeError("L3 bonus became negative")
    if bonus_age - float(RHO_AGE) * payload_term > 1.0e-12 or bonus_tag - float(RHO_TAG) * payload_term > 1.0e-12:
        raise RuntimeError("L3 bonus exceeded its component cap")
    if bonus_age + bonus_tag - float(TOTAL_BONUS_CAP) * payload_term > 1.0e-12:
        raise RuntimeError("L3 bonus exceeded 0.1 percent of payload")
    return bonus_age, bonus_tag
