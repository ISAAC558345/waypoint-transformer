"""Phase-5L freeze checks. They do not roll out scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase9.const import BUOY_DIM, EDGE_UB_DIM
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN, append_queue_edge
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import assert_phase5l_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5l_joint_encoder"


class Phase5LTests(unittest.TestCase):
    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5l_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5l_seed(310102)
        assert_phase5l_seed(310100)
        assert_phase5l_seed(310101)

    def test_queue_edge_uses_current_buoy_queue(self) -> None:
        batch = {
            "edge": np.zeros((1, 2, 3, EDGE_UB_DIM), dtype=np.float32),
            "buoy": np.zeros((1, 3, int(BUOY_DIM) + 1), dtype=np.float32),
        }
        batch["buoy"][0, 1, 2] = 0.25
        updated = append_queue_edge(batch)
        self.assertEqual(int(updated["edge"].shape[-1]), int(EDGE_UB_DIM) + 1)
        self.assertAlmostEqual(float(updated["edge"][0, 0, 1, -1]), 0.25)
        self.assertEqual(int(batch["edge"].shape[-1]), int(EDGE_UB_DIM))

    def test_readout_starts_at_zero_and_decoder_is_not_inside_the_encoder(self) -> None:
        model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
        self.assertTrue(torch.count_nonzero(model.mix.weight) == 0)
        self.assertTrue(torch.count_nonzero(model.u_proj.weight) == 0)
        names = [name for name, _parameter in model.named_parameters()]
        self.assertFalse(any("decoder" in name or name.startswith("pair.") for name in names))

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
            "src/wt/lowlevel_v2/phase5_hgnn_urgency/encoder.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("phase5l", text)


if __name__ == "__main__":
    unittest.main()
