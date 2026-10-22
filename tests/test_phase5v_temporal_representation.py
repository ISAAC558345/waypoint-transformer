"""Phase-5V checks. They do not train or score a confirmation scenario."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5v_temporal_representation.metrics import choose_gate
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import TEMP_INPUT_DIM, assert_consumed
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import observability_rows, temporal_matrix


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5v_temporal_representation"


def _snapshot() -> dict:
    config = L3BConfig()
    return {
        "uav_pos": np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]], dtype=np.float64),
        "buoy_pos": np.array([[i * 3.0, 8.0] for i in range(config.num_buoys)], dtype=np.float64),
        "progress": np.array([0.4, 0.2, 0, 0, 0, 0], dtype=np.float64),
        "current_assignment": np.array([0, 1, -1, -1, -1, -1], dtype=np.int64),
        "velocity": np.array([[0.2, 0.0], [-0.1, 0.0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float64),
        "last_sinr": np.array([2.0, 0.5, 0, 0, 0, 0], dtype=np.float64),
        "time_step": 15,
    }


class Phase5VTests(unittest.TestCase):
    def test_missing_history_is_not_allowed(self) -> None:
        rows = {row["feature"]: row for row in observability_rows()}
        self.assertEqual(rows["hold_remaining"]["allowed"], 0)
        self.assertEqual(rows["option_age"]["allowed"], 0)
        self.assertEqual(rows["consecutive_steps_on_target"]["allowed"], 0)
        self.assertEqual(rows["waiting_time_aoi_deadline"]["allowed"], 0)
        self.assertEqual(rows["radial_closing_velocity"]["allowed"], 1)
        self.assertEqual(rows["episode_fraction"]["candidate_specific"], 0)
        assert_consumed(310109)
        assert_consumed(310117)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230000)

    def test_gate_letters(self) -> None:
        self.assertEqual(
            choose_gate(
                regression=False,
                j_up=True,
                j_ci=True,
                t15_down=True,
                majority_scenario=True,
                majority_seed=True,
                ablation_used=True,
                case_clear=True,
                promising=True,
            ),
            "V-A",
        )
        self.assertEqual(
            choose_gate(
                regression=False,
                j_up=True,
                j_ci=False,
                t15_down=True,
                majority_scenario=False,
                majority_seed=True,
                ablation_used=True,
                case_clear=False,
                promising=True,
            ),
            "V-B",
        )
        self.assertEqual(
            choose_gate(
                regression=False,
                j_up=False,
                j_ci=False,
                t15_down=False,
                majority_scenario=False,
                majority_seed=False,
                ablation_used=False,
                case_clear=False,
                promising=False,
            ),
            "V-C",
        )
        self.assertEqual(
            choose_gate(
                regression=True,
                j_up=False,
                j_ci=False,
                t15_down=False,
                majority_scenario=False,
                majority_seed=False,
                ablation_used=False,
                case_clear=False,
                promising=False,
            ),
            "V-D",
        )

    def test_temporal_features_follow_the_candidate_and_the_uav_order(self) -> None:
        config = L3BConfig()
        snapshot = _snapshot()
        assignment = np.array([[0, 1, 2, -1, -1, -1]], dtype=np.int64)
        original = temporal_matrix(snapshot, assignment, config)
        self.assertEqual(original.shape[1], TEMP_INPUT_DIM)
        self.assertTrue(np.all(np.isfinite(original)))
        swapped = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in snapshot.items()}
        for key in ("uav_pos", "progress", "current_assignment", "velocity", "last_sinr"):
            block = np.array(swapped[key][[1, 0]], copy=True)
            swapped[key][[0, 1]] = block
        swapped_assignment = assignment.copy()
        swapped_assignment[0, [0, 1]] = swapped_assignment[0, [1, 0]]
        changed = temporal_matrix(swapped, swapped_assignment, config)
        self.assertTrue(np.allclose(original, changed, atol=1.0e-5))
        stopped = dict(snapshot)
        stopped["velocity"] = np.zeros_like(snapshot["velocity"])
        stopped_values = temporal_matrix(stopped, assignment, config)
        self.assertEqual(float(stopped_values[0, 20 + 4 * 2]), 0.0)

    def test_package_does_not_call_decide(self) -> None:
        called = []
        for path in PACKAGE.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.append(node.func.attr)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called.append(node.func.id)
        self.assertNotIn("decide", called)
        for relative in (
            "src/wt/htw_phase100/high_level.py",
            "src/wt/lowlevel_v2/residual_policy.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/reward.py",
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/planner.py",
        ):
            self.assertNotIn("phase5v", (ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
