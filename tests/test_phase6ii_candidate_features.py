"""Candidate-conditioned features stay row-aligned and match the SINR formula."""
from __future__ import annotations

import unittest

import numpy as np

from wt.benchmark_v2.phase6c.channel import link_rates
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6ii_candidate_features.audit import PROPOSED, assign_gate, separated
from wt.benchmark_v2.phase6ii_candidate_features.features import candidate_features
from wt.l3b.config import L3BConfig


def _state() -> dict[str, np.ndarray]:
    config = L3BConfig()
    uav = np.asarray([[10.0, 10.0], [40.0, 10.0], [10.0, 40.0], [40.0, 40.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float64)
    buoy = np.asarray([[12.0, 12.0], [8.0, 30.0], [50.0, 12.0], [48.0, 40.0], [20.0, 50.0], [60.0, 20.0], [100.0, 10.0], [110.0, 20.0], [90.0, 40.0], [120.0, 50.0], [80.0, 60.0], [140.0, 30.0]], dtype=np.float64)
    delta = uav[:, None, :] - buoy[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    gain = np.maximum(distance, config.min_link_m) ** (-config.path_loss_exp)
    queue = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0e6, 4.0e6, 4.0e6, 4.0e6, 4.0e6, 4.0e6], dtype=np.float64)
    return {
        "uav_buoy_distance_m": distance,
        "uav_buoy_gain": gain,
        "uav_buoy_in_range": (distance <= config.service_radius_m).astype(np.uint8),
        "uav_buoy_hypothetical_rate": np.zeros_like(distance),
        "q_bits": queue,
        "association": np.asarray([-1, -1, -1, -1, -1, -1], dtype=np.int64),
        "service_progress": np.linspace(0.0, 0.5, config.num_uavs),
        "buoy_position_m": buoy,
        "uav_position_m": uav,
    }


class Phase6IIFeatureTests(unittest.TestCase):
    def test_open_pool_excludes_reserved_scenarios(self) -> None:
        self.assertTrue(set(VALIDATION_POOL).isdisjoint(RESERVED_TEST))

    def test_instant_rate_matches_the_environment_formula(self) -> None:
        config = L3BConfig()
        arrays = _state()
        assignment = np.asarray([[6, 7, 8, 9, 10, 11], [-1, -1, -1, -1, -1, -1], [6, 6, 8, 9, 10, 11]], dtype=np.int64)
        features = candidate_features(arrays, assignment, config)
        for row in assignment:
            _sinr, rate, _interference = link_rates(
                arrays["uav_buoy_gain"],
                row,
                arrays["q_bits"],
                arrays["uav_buoy_in_range"] > 0,
                bandwidth_hz=config.bandwidth_hz,
                tx_power=config.tx_power,
                noise_power=config.noise_power,
                kappa_same=config.kappa_same_cluster,
                kappa_cross=config.kappa_cross_cluster,
            )
            got = candidate_features(arrays, row, config)["instant_rate_bps"][0]
            self.assertTrue(np.allclose(got, rate))
        self.assertTrue(np.all(features["instant_rate_bps"][1] == 0.0))
        self.assertTrue(np.all(features["instant_rate_bps"][2, :2] == 0.0))
        self.assertEqual(float(features["kinematic_bits"][1]), 0.0)
        self.assertTrue(np.isfinite(features["kinematic_bits"]).all())

    def test_uav_relabeling_preserves_sums(self) -> None:
        config = L3BConfig()
        arrays = _state()
        assignment = np.asarray([[6, 7, 8, 9, 10, 11]], dtype=np.int64)
        perm = np.asarray([1, 0, 2, 3, 4, 5], dtype=np.int64)
        moved = {
            "uav_buoy_distance_m": arrays["uav_buoy_distance_m"][perm],
            "uav_buoy_gain": arrays["uav_buoy_gain"][perm],
            "uav_buoy_in_range": arrays["uav_buoy_in_range"][perm],
            "uav_buoy_hypothetical_rate": arrays["uav_buoy_hypothetical_rate"][perm],
            "q_bits": arrays["q_bits"],
            "association": arrays["association"][perm],
            "service_progress": arrays["service_progress"][perm],
            "buoy_position_m": arrays["buoy_position_m"],
            "uav_position_m": arrays["uav_position_m"][perm],
        }
        before = candidate_features(arrays, assignment, config)
        after = candidate_features(moved, assignment[:, perm], config)
        self.assertTrue(np.allclose(np.sum(before["geometric_rate_bps"]), np.sum(after["geometric_rate_bps"])))
        self.assertTrue(np.allclose(before["geometric_rate_bps"][0, perm], after["geometric_rate_bps"][0]))

    def test_same_buoy_set_has_identical_hover_interference(self) -> None:
        config = L3BConfig()
        arrays = _state()
        assignment = np.asarray([[6, 7, 8, 9, 10, 11], [11, 10, 9, 8, 7, 6]], dtype=np.int64)
        features = candidate_features(arrays, assignment, config)
        self.assertAlmostEqual(float(features["hover_interference_w"][0]), float(features["hover_interference_w"][1]))

    def test_gate_requires_a_stable_control(self) -> None:
        self.assertFalse(separated(1.0, 1.01, "distance_m"))
        row = {
            "hamming": {"scheme_c": 2},
            "j15_gap": {"scheme_c": 100.0},
            "separated": {name: 0 for name in PROPOSED},
            "favors_oracle": {name: 0 for name in PROPOSED},
            "same_set_relative_range": {name: 0.0 for name in PROPOSED},
        }
        self.assertEqual(assign_gate([row] * 12, {0: [row], 30: [row]}), "CANDIDATE_FEATURE_DESIGN_BLOCKED")


if __name__ == "__main__":
    unittest.main()
