"""Phase-5N checks. They do not roll out scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5n_joint_planner.head import JointAssignmentHead, assignment_scores, hungarian_from_plan, masked_plan
from wt.lowlevel_v2.phase5n_joint_planner.matchings import maximal_assignments
from wt.lowlevel_v2.phase5n_joint_planner.protocol import assert_phase5n_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5n_joint_planner"


class Phase5NTests(unittest.TestCase):
    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5n_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5n_seed(310102)
        assert_phase5n_seed(310100)
        assert_phase5n_seed(310101)

    def test_sinkhorn_blocks_non_candidates_and_scores_whole_assignments(self) -> None:
        logits = torch.zeros(2, 3)
        mask = torch.tensor([[True, True, False], [True, True, False]])
        plan = masked_plan(logits, mask)
        self.assertLess(float(plan[:, 2].sum()), 1.0e-4)
        self.assertTrue(torch.allclose(plan.sum(dim=1), torch.ones(2), atol=5.0e-2))
        assignments = torch.tensor([[0, 1], [1, 0]])
        scores = assignment_scores(plan, assignments)
        self.assertEqual(tuple(scores.shape), (2,))
        joints = torch.tensor([2.0, 0.0])
        loss = joint_rank_loss(scores, joints)
        self.assertTrue(torch.isfinite(loss))
        head = JointAssignmentHead(width=4)
        h_u = torch.zeros(1, 2, 4)
        h_b = torch.zeros(1, 3, 4)
        z_global = torch.zeros(1, 4)
        self.assertEqual(tuple(head.pair_scores(h_u, h_b, z_global).shape), (2, 3))

    def test_maximal_pool_can_assign_a_later_uav(self) -> None:
        options = [[0, 1], [0, 1], [0, 1]]
        pool = maximal_assignments(options)
        self.assertEqual(len(pool), 6)
        self.assertTrue(any(int(item[0]) < 0 and int(item[2]) >= 0 for item in pool))
        for item in pool:
            chosen = [int(buoy) for buoy in item if int(buoy) >= 0]
            self.assertEqual(len(chosen), len(set(chosen)))
            self.assertEqual(len(chosen), 2)

    def test_hungarian_idles_a_uav_whose_column_is_outside_the_mask(self) -> None:
        plan = torch.tensor([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
        mask = torch.tensor([[True, False, False], [True, False, False]])
        assignment = hungarian_from_plan(plan, mask)
        self.assertEqual(int(np.sum(assignment >= 0)), 1)
        self.assertEqual(int(assignment[int(np.flatnonzero(assignment >= 0)[0])]), 0)

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
            self.assertNotIn("phase5n", text)


if __name__ == "__main__":
    unittest.main()
