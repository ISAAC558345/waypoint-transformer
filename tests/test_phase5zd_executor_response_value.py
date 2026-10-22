"""Matched H=1 response value stays frozen outside its own scorer."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase5zd_executor_response_value.metrics import choose_gate
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES, assert_consumed
from wt.lowlevel_v2.phase5zd_executor_response_value.response import assemble_response
from wt.lowlevel_v2.phase5zd_executor_response_value.scorer import ExecutorConditionedJointScorer


class Phase5ZDExecutorResponseValueTest(unittest.TestCase):
    def test_radial_motion_is_the_realized_displacement(self) -> None:
        response = assemble_response(
            sinr_0=np.zeros(1),
            sinr_1=np.ones(1),
            interference_0=np.zeros(1),
            interference_1=np.zeros(1),
            progress_0=np.zeros(1),
            progress_1=np.array([0.2]),
            rate_1=np.zeros(1),
            pos_0=np.array([[0.0, 0.0]]),
            pos_1=np.array([[2.0, 0.0]]),
            buoy_pos=np.array([[10.0, 0.0]]),
            assignment_0=np.array([-1]),
            assignment_1=np.array([0]),
            in_range=np.ones(1),
            service_active=np.ones(1),
        )
        self.assertAlmostEqual(float(response[0, 12]), 2.0)
        self.assertAlmostEqual(float(response[0, 13]), 1.0)
        self.assertEqual(float(response[0, 11]), 0.0)
        self.assertNotIn("payload", RESPONSE_FEATURES)

    def test_matched_scorers_have_one_parameter_count(self) -> None:
        torch.manual_seed(0)
        first = ExecutorConditionedJointScorer()
        second = ExecutorConditionedJointScorer()
        self.assertEqual(sum(parameter.numel() for parameter in first.parameters()), sum(parameter.numel() for parameter in second.parameters()))
        count, width = 3, 128
        features = {
            "h_u": torch.randn(1, 6, width),
            "h_b": torch.randn(1, 12, width),
            "h_edge": torch.randn(1, 6, 12, width),
            "h_uu": torch.randn(1, 6, 6, width),
            "z_global": torch.randn(1, width),
            "z_rel": torch.randn(count, 119),
            "response": torch.zeros(count, 6, len(RESPONSE_FEATURES)),
        }
        assignments = torch.tensor([[0, 1, -1, 2, -1, -1], [1, 0, -1, 2, -1, -1], [-1, 2, 0, 1, -1, -1]])
        baseline = first(features, assignments)
        features["response"] = torch.ones(count, 6, len(RESPONSE_FEATURES))
        self.assertFalse(torch.allclose(baseline, first(features, assignments)))

    def test_gate_priority(self) -> None:
        self.assertEqual(choose_gate(integrity_ok=False, ref1_reproduced=True, confirmed=True, mean_improved=True), "RESP-STATE-FAIL")
        self.assertEqual(choose_gate(integrity_ok=True, ref1_reproduced=False, confirmed=True, mean_improved=True), "RESP-D")
        self.assertEqual(choose_gate(integrity_ok=True, ref1_reproduced=True, confirmed=True, mean_improved=True), "RESP-A")
        self.assertEqual(choose_gate(integrity_ok=True, ref1_reproduced=True, confirmed=False, mean_improved=True), "RESP-B")
        self.assertEqual(choose_gate(integrity_ok=True, ref1_reproduced=True, confirmed=False, mean_improved=False), "RESP-C")

    def test_fresh_scenarios_stay_closed(self) -> None:
        assert_consumed(310100)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230000)

    def test_package_does_not_call_deployment_decide(self) -> None:
        text = "\n".join(path.read_text(encoding="utf-8") for path in Path(__file__).resolve().parents[1].joinpath("src/wt/lowlevel_v2/phase5zd_executor_response_value").glob("*.py"))
        self.assertNotIn("decide(", text)
        self.assertIn("Adam", text)


if __name__ == "__main__":
    unittest.main()
