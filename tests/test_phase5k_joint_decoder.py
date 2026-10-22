"""Phase-5K freeze and decoder-coupling checks. They do not roll out scenarios."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase5k_joint_decoder.decoders import capacity_plan, gather_assignment_score, hungarian_assign, joint_pick
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5k_joint_decoder.protocol import assert_phase5k_seed, split_name


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "wt" / "lowlevel_v2" / "phase5k_joint_decoder"


class Phase5KTests(unittest.TestCase):
    def test_holdout_and_other_scenarios_are_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_phase5k_seed(230000)
        with self.assertRaises(RuntimeError):
            assert_phase5k_seed(310102)
        assert_phase5k_seed(310100)
        assert_phase5k_seed(310101)
        self.assertEqual(split_name(310101, 15), "test")
        self.assertEqual(split_name(310100, 0), "train")
        self.assertEqual(split_name(310100, 30), "val")

    def test_capacity_plan_and_joint_pick_remove_duplicate_targets(self) -> None:
        plan = capacity_plan(torch.tensor([[5.0, 0.0, 0.0], [4.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        self.assertEqual(tuple(plan.shape), (3, 3))
        self.assertTrue(torch.allclose(plan.sum(dim=1), torch.ones(3), atol=1.0e-4))
        self.assertLessEqual(float(plan.sum(dim=0).max()), 1.0 + 1.0e-4)
        coupled = np.asarray([[0.55, 0.45], [0.50, 0.10]], dtype=np.float64)
        assignment = joint_pick(coupled)
        self.assertTrue(np.array_equal(assignment, np.asarray([0, 1])))
        self.assertEqual(len(set(int(item) for item in assignment)), 2)
        matched = hungarian_assign(np.asarray([[10.0, 0.0], [9.0, 1.0]], dtype=np.float64))
        self.assertTrue(np.array_equal(matched, np.asarray([0, 1])))
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assignments = torch.tensor([[0, 1], [-1, 0]])
        scores = gather_assignment_score(matrix, assignments, logarithmic=False)
        self.assertTrue(torch.allclose(scores, torch.tensor([1.0 + 4.0, 3.0])))

    def test_rank_loss_follows_joint_order_rather_than_label_difference(self) -> None:
        scores = torch.tensor([0.0, 1.0], requires_grad=True)
        joints = torch.tensor([2.0, 0.0])
        loss = joint_rank_loss(scores, joints)
        loss.backward()
        self.assertLess(float(scores.grad[0]), 0.0)
        self.assertGreater(float(scores.grad[1]), 0.0)
        aligned = joint_rank_loss(torch.tensor([1.0, 0.0]), joints)
        self.assertLess(float(aligned), float(loss.detach()))

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
            self.assertNotIn("phase5k", text)


if __name__ == "__main__":
    unittest.main()
