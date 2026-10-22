"""Shadow first-hold scorer: U_hat = U_single + ExtraTrees(U_single + Phase-3E geometry).

This module is a side-channel policy. It does not patch high_level.decide,
decode_assignment, generate_pool, reward, utility, or Low-Level.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from wt.lowlevel_v2.htw_phase4b_b2_shadow.protocol import B2_FEATURE_KEYS, ET, P3E_CAND_KEYS
from wt.lowlevel_v2.phase3e_future_value_audit.run_phase3e import candidate_vector


def extract_candidate_geometry(env, gamma) -> dict[str, float]:
    """Phase-3E candidate-conditioned geometry only. No leftover, state, or future keys."""
    vec = candidate_vector(env, np.asarray(gamma, dtype=np.int64))
    return {k: float(vec[k]) for k in P3E_CAND_KEYS}


def feature_matrix(rows: list[dict], keys: tuple[str, ...] = B2_FEATURE_KEYS) -> np.ndarray:
    return np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)


def assert_no_future_keys(rows: list[dict], keys: tuple[str, ...]) -> None:
    banned = {
        "U_traj",
        "F",
        "C_abs",
        "C_ratio",
        "remaining_arrivals",
        "leftover_queue",
        "leftover_access",
        "in_range_frac",
        "uav_y_mean",
        "uav_y_std",
        "uav_pairwise_mean",
        "min_pairwise_dist",
        "queue_mean",
        "queue_hot_mass",
    }
    leak = [k for k in keys if k in banned]
    if leak:
        raise RuntimeError(f"B2 shadow features leak forbidden keys: {leak}")
    for row in rows[:1]:
        if any(k not in row for k in keys):
            missing = [k for k in keys if k not in row]
            raise RuntimeError(f"B2 shadow missing features: {missing}")


@dataclass
class B2ShadowScorer:
    """Official planner is not used. Selection is argmax U_hat on a frozen 24-pool."""

    model: ExtraTreesRegressor | None = field(default=None)
    feature_keys: tuple[str, ...] = B2_FEATURE_KEYS

    def fit(self, x: np.ndarray, y: np.ndarray) -> "B2ShadowScorer":
        self.model = ExtraTreesRegressor(**ET)
        self.model.fit(x, y)
        return self

    def predict_credit(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("B2ShadowScorer.fit must run before predict")
        return np.asarray(self.model.predict(x), dtype=np.float64)

    def score_matrix(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_single = np.asarray(x[:, self.feature_keys.index("U_single")], dtype=np.float64)
        c_hat = self.predict_credit(x)
        return c_hat, u_single + c_hat

    def select_first_hold(self, pool_rows: list[dict]) -> dict:
        """Shadow first-hold decision. Does not call high_level.decide."""
        assert_no_future_keys(pool_rows, self.feature_keys)
        x = feature_matrix(pool_rows, self.feature_keys)
        c_hat, u_hat = self.score_matrix(x)
        idx = int(np.argmax(u_hat))
        chosen = pool_rows[idx]
        return {
            "idx": idx,
            "candidate_id": int(chosen["candidate_id"]),
            "assignment_hash": chosen.get("assignment_hash"),
            "gamma": chosen.get("gamma"),
            "U_single": float(chosen["U_single"]),
            "C_hat": float(c_hat[idx]),
            "U_hat": float(u_hat[idx]),
            "C_hat_all": c_hat,
            "U_hat_all": u_hat,
            "source": "B2-SHADOW",
            "official_decide_called": False,
        }


__all__ = [
    "B2ShadowScorer",
    "assert_no_future_keys",
    "extract_candidate_geometry",
    "feature_matrix",
]
