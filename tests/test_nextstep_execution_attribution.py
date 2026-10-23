"""Attribution labels and the reserved-test boundary. No episode replay."""
from __future__ import annotations

import unittest

from wt.benchmark_v2.phase6i_execution_attribution.run_attribution import (
    assert_validation_scenario,
    observation_flags,
)


def _row(**overrides):
    base = {
        "idle": 0,
        "window_steps": 15,
        "served_bits": 1000.0,
        "tol_bits": 1.0,
        "entered_service_radius": 1,
        "first_entry_step": 1,
        "effective_service_steps": 10,
        "projection_count": 0,
    }
    base.update(overrides)
    return base


class ExecutionAttributionRulesTest(unittest.TestCase):
    def test_reserved_scenario_is_refused(self):
        with self.assertRaises(RuntimeError):
            assert_validation_scenario(322064)
        assert_validation_scenario(322048)

    def test_never_entered_is_reachability(self):
        flags = observation_flags(_row(entered_service_radius=0, first_entry_step=None, served_bits=0.0, effective_service_steps=0))
        self.assertEqual(flags["reachability"], 1)
        self.assertEqual(flags["delivered"], 0)

    def test_entry_without_bits_is_a_communication_observation(self):
        flags = observation_flags(_row(served_bits=0.0, effective_service_steps=0, projection_count=12))
        self.assertEqual(flags["communication_observation"], 1)
        self.assertEqual(flags["projection"], 1)

    def test_late_short_service_is_utilization(self):
        flags = observation_flags(_row(first_entry_step=10, effective_service_steps=2, served_bits=50.0))
        self.assertEqual(flags["utilization"], 1)
        self.assertEqual(flags["delivered"], 0)


if __name__ == "__main__":
    unittest.main()
