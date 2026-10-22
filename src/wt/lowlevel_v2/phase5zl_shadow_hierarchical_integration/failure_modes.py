"""Pending failure-mode interface. It does not install a deployment fallback."""
from __future__ import annotations


def audit() -> dict:
    pending = "pending_human_approval"
    stop = {
        "action": "stop",
        "may_use_existing_legal_assignment": False,
        "verified_fallback": False,
        "human_rule_required": True,
        "status": pending,
    }
    return {
        "installs_new_fallback": False,
        "deepsets_not_a_new_fallback": True,
        "b2_not_a_new_fallback": True,
        "oracle_not_a_new_fallback": True,
        "modes": {
            "empty_candidate_pool": stop,
            "all_uavs_idle": stop,
            "illegal_association": stop,
            "true_h1_non_finite": stop,
            "model_r1_non_finite": stop,
            "safety_projector_failure": {
                **stop,
                "note": "The frozen executor already maps a failed projection to a zero action. That is frozen executor behavior, not a newly approved deployment fallback.",
            },
            "inference_timeout": {**stop, "note": "No timeout is specified."},
            "missing_checkpoint": stop,
            "counterfactual_step_exception": stop,
            "exact_value_tie": {
                "action": "stable_argsort_descending",
                "may_use_existing_legal_assignment": True,
                "verified_fallback": False,
                "human_rule_required": True,
                "status": pending,
                "note": "Stable descending argsort is the replay convention. It is not an approved deployment tie rule.",
            },
        },
    }


__all__ = ["audit"]
