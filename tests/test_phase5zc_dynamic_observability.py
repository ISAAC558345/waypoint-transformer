"""Phase-5ZC stays an audit: no probe and no fresh scenario."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase5zc_dynamic_observability.actions import radial_components
from wt.lowlevel_v2.phase5zc_dynamic_observability.metrics import choose_gate, nearest_neighbor
from wt.lowlevel_v2.phase5zc_dynamic_observability.protocol import assert_consumed


class Phase5ZCDynamicObservabilityTest(unittest.TestCase):
    def test_radial_component_points_toward_the_assigned_buoy(self) -> None:
        action = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        uav = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)
        buoy = np.array([[10.0, 0.0]], dtype=np.float64)
        radial, tangential, _heading = radial_components(action, uav, buoy, np.array([0, -1]))
        self.assertAlmostEqual(float(radial[0]), 1.0)
        self.assertAlmostEqual(float(tangential[0]), 0.0)
        self.assertAlmostEqual(float(radial[1]), 0.0)

    def test_scalar_neighbor_uses_normalized_gap(self) -> None:
        labels = np.array([0.0, 1.0e6, 5.0e6], dtype=np.float64)
        scores = labels.copy()
        result = nearest_neighbor(scores[:, None], labels)
        self.assertEqual(result["contradiction_1"], 0.0)

    def test_gate_priority_is_locked(self) -> None:
        self.assertEqual(choose_gate(l0_rank=True, phy_rank=False, act0_informative=True, act0_gain=1.0, h_dyn="1"), "DYN-A")
        self.assertEqual(choose_gate(l0_rank=False, phy_rank=False, act0_informative=True, act0_gain=0.10, h_dyn="1"), "DYN-B")
        self.assertEqual(choose_gate(l0_rank=False, phy_rank=False, act0_informative=False, act0_gain=0.50, h_dyn="3"), "DYN-C")
        self.assertEqual(choose_gate(l0_rank=False, phy_rank=False, act0_informative=False, act0_gain=0.50, h_dyn=""), "DYN-D")

    def test_fresh_scenarios_stay_closed(self) -> None:
        assert_consumed(310114)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230050)

    def test_package_does_not_fit_a_probe(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in Path(__file__).resolve().parents[1].joinpath("src/wt/lowlevel_v2/phase5zc_dynamic_observability").glob("*.py"))
        for banned in ("torch.optim", ".backward(", "RandomForest", "ExtraTrees", "XGBoost", "LogisticRegression"):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
