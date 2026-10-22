"""Pair-before-pool keeps UAV-target binding and the Phase-5Y head width."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from wt.lowlevel_v2.phase5za_correspondence_pooling.metrics import choose_gate
from wt.lowlevel_v2.phase5za_correspondence_pooling.pool import (
    CorrespondencePreservingValue,
    permute_buoys,
    permute_uavs,
)
from wt.lowlevel_v2.phase5za_correspondence_pooling.protocol import HEAD_INPUT_DIM, assert_consumed


def _features(count: int = 4) -> tuple[dict, torch.Tensor]:
    torch.manual_seed(0)
    features = {
        "h_u": torch.randn(1, 6, 128),
        "h_b": torch.randn(1, 12, 128),
        "h_edge": torch.randn(1, 6, 12, 128),
        "h_uu": torch.randn(1, 6, 6, 128),
        "z_global": torch.randn(128),
        "z_rel": torch.randn(count, 119),
    }
    assignments = torch.tensor(
        [
            [2, 5, -1, 4, -1, -1],
            [5, 2, -1, 4, -1, -1],
            [-1, 4, 2, 5, -1, -1],
            [0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    return features, assignments


class Phase5ZACorrespondencePoolingTest(unittest.TestCase):
    def test_permutations_and_swap_sensitivity(self) -> None:
        features, assignments = _features()
        module = CorrespondencePreservingValue()
        module.eval()
        with torch.no_grad():
            base = module.assignment_embedding(features, assignments)
            value = module(features, assignments)
            uav_features, uav_assignments = permute_uavs(features, assignments, torch.tensor([1, 0, 2, 3, 4, 5]))
            buoy_features, buoy_assignments = permute_buoys(features, assignments, torch.arange(12).flip(0))
            self.assertLessEqual(float(torch.max(torch.abs(value - module(uav_features, uav_assignments)))), 1.0e-5)
            self.assertLessEqual(float(torch.max(torch.abs(value - module(buoy_features, buoy_assignments)))), 1.0e-5)
            order = torch.tensor([2, 0, 3, 1])
            shuffled = {key: value.detach().clone() for key, value in features.items()}
            shuffled["z_rel"] = features["z_rel"][order]
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel())
            moved = module.assignment_embedding(shuffled, assignments[order])[inverse]
            self.assertLessEqual(float(torch.max(torch.abs(base - moved))), 1.0e-5)
            self.assertGreater(float(torch.linalg.vector_norm(base[0] - base[1])), 1.0e-3)

    def test_head_width_matches_phase5y(self) -> None:
        module = CorrespondencePreservingValue()
        self.assertEqual(int(module.mlp[0].in_features), int(HEAD_INPUT_DIM))
        self.assertEqual(int(module.mlp[0].out_features), 128)
        names = [name for name, _ in module.named_parameters()]
        self.assertTrue(all(name.startswith("pool.phi") or name.startswith("mlp") for name in names))

    def test_gate_priority(self) -> None:
        flags = dict(
            structural_ok=True,
            collision_fixed=True,
            improved_v=True,
            ci_above_zero=True,
            improved_y=True,
            scenarios_ok=True,
            seeds_ok=True,
            top3_ok=True,
            regret_ok=True,
            case_ok=True,
        )
        self.assertEqual(choose_gate(**flags), "POOL-A")
        self.assertEqual(choose_gate(**{**flags, "structural_ok": False}), "POOL-D")
        self.assertEqual(choose_gate(**{**flags, "collision_fixed": False}), "POOL-D")
        self.assertEqual(choose_gate(**{**flags, "ci_above_zero": False}), "POOL-B")
        self.assertEqual(choose_gate(**{**flags, "improved_v": False, "ci_above_zero": False}), "POOL-C")

    def test_holdout_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_consumed(310118)
        with self.assertRaises(RuntimeError):
            assert_consumed(230100)

    def test_package_does_not_call_decide(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "wt" / "lowlevel_v2" / "phase5za_correspondence_pooling"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("decide(", text)
        self.assertNotIn("payload_h", text)
        self.assertNotIn(".backward(", Path(root / "pool.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
