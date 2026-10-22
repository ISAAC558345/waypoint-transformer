"""Phase-5U checks. They do not train or open a confirmation scenario."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.metrics import choose_gate
from wt.lowlevel_v2.phase5u_representation_repair.protocol import REL_DIM, assert_consumed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5u_representation_repair"


def _snapshot() -> dict:
    config = L3BConfig()
    return {
        "uav_pos": np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]], dtype=np.float64),
        "buoy_pos": np.array([[i * 3.0, 8.0 + (i % 3)] for i in range(config.num_buoys)], dtype=np.float64),
        "queue": np.linspace(1, 4, config.num_buoys),
        "urgency": np.linspace(0.2, 1.0, config.num_buoys),
        "progress": np.linspace(0.0, 0.5, config.num_uavs),
        "current_assignment": np.array([0, 1, -1, -1, -1, -1], dtype=np.int64),
        "distance": np.linalg.norm(
            np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]], dtype=np.float64)[:, None, :]
            - np.array([[i * 3.0, 8.0 + (i % 3)] for i in range(config.num_buoys)], dtype=np.float64)[None, :, :],
            axis=-1,
        ),
        "eta": np.ones((config.num_uavs, config.num_buoys)),
        "rate": np.full((config.num_uavs, config.num_buoys), 1.0e6),
        "reference_rate": 2.0e6,
    }


class Phase5UTests(unittest.TestCase):
    def test_consumed_seeds_and_gate(self) -> None:
        assert_consumed(310100)
        assert_consumed(310117)
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230000)
        flags = dict(
            collision_down=True,
            collision_worse=False,
            regret_down=True,
            j_up=True,
            j_stable=True,
            no_reversal=True,
            rank_clear=True,
            rank_improved=True,
            t15_still_high=False,
            payload_regression=False,
        )
        self.assertEqual(choose_gate(**flags), "U-A")
        flags["t15_still_high"] = True
        self.assertEqual(choose_gate(**flags), "U-B")
        self.assertEqual(
            choose_gate(
                collision_down=False,
                collision_worse=False,
                regret_down=False,
                j_up=False,
                j_stable=False,
                no_reversal=True,
                rank_clear=False,
                rank_improved=False,
                t15_still_high=True,
                payload_regression=False,
            ),
            "U-C",
        )
        self.assertEqual(
            choose_gate(
                collision_down=True,
                collision_worse=False,
                regret_down=True,
                j_up=False,
                j_stable=False,
                no_reversal=False,
                rank_clear=True,
                rank_improved=True,
                t15_still_high=False,
                payload_regression=True,
            ),
            "U-D",
        )

    def test_relational_features_are_permutation_invariant(self) -> None:
        snapshot = _snapshot()
        config = L3BConfig()
        assignment = np.array([[0, 1, 2, -1, -1, -1]], dtype=np.int64)
        original = relational_matrix(snapshot, assignment, config)
        self.assertEqual(original.shape, (1, REL_DIM))
        self.assertTrue(np.all(np.isfinite(original)))
        swapped = {key: np.array(value, copy=True) for key, value in snapshot.items() if key != "reference_rate"}
        swapped["reference_rate"] = snapshot["reference_rate"]
        for key in ("uav_pos", "progress", "current_assignment", "distance", "eta", "rate"):
            block = np.array(swapped[key][[1, 0]], copy=True)
            swapped[key][[0, 1]] = block
        swapped_assignment = assignment.copy()
        swapped_assignment[0, [0, 1]] = swapped_assignment[0, [1, 0]]
        changed = relational_matrix(swapped, swapped_assignment, config)
        self.assertTrue(np.allclose(original, changed, atol=1.0e-5))

    def test_package_does_not_call_decide_or_edit_frozen_files(self) -> None:
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
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("phase5u", text)


if __name__ == "__main__":
    unittest.main()
