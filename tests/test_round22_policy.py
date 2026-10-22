"""Structural contracts for the frozen-predictor Round-22 policy stage."""
from __future__ import annotations

import compileall
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.round22.data import CONTEXT_LENGTH, H_TASK, R22_OUTPUT, FrozenDeepSetsPredictor
from wt.round22.models import DEPLOYABLE_FAMILIES, ORACLE_TRANSFORMER, POLICY_FAMILIES, PREDICTED_MLP, PREDICTED_TRANSFORMER, STATE_ONLY_TRANSFORMER, PolicySchema, build_policy, rollout_action
from wt.round22.training import _phi, action_mse


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def fake_batch(k_count: int = 5) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(2202 + k_count); b, h, m, k = 3, 10, 2, 5
    attention = np.ones((b, h), dtype=bool); attention[0, :4] = False
    uav = rng.normal(size=(b, h, m, 8)).astype("f"); buoy = rng.normal(size=(b, h, k, 7)).astype("f")
    phi = rng.normal(size=(b, h, m, 11)).astype("f")
    uav[~attention] = 0; buoy[~attention] = 0; phi[~attention] = 0
    buoy_mask = np.zeros((b, h, k), dtype=bool); buoy_mask[..., :k_count] = True; buoy_mask[~attention] = False
    uav_mask = np.ones((b, h, m), dtype=bool); uav_mask[~attention] = False
    return {"uav_nodes": torch.from_numpy(uav), "buoy_nodes": torch.from_numpy(buoy), "uav_mask": torch.from_numpy(uav_mask), "buoy_mask": torch.from_numpy(buoy_mask), "global_features": torch.from_numpy(rng.normal(size=(b, h, 6)).astype("f")), "attention_mask": torch.from_numpy(attention), "predicted_phi": torch.from_numpy(phi), "oracle_phi": torch.from_numpy(phi + .25), "actions": torch.from_numpy(rng.uniform(-1, 1, size=(b, m, 2)).astype("f"))}


class Round22PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(22); cls.batch = fake_batch(); cls.predictor_record = json.loads((R22_OUTPUT / "selected_deepsets_task_waypoint_predictor.json").read_text(encoding="utf-8"))

    def test_01_round21_original_lock_unchanged(self): self.assertEqual(digest(ROOT / "round_21" / "ROUND_LOCK.json"), "F754D441B0A8A3DF77EDAD7EAF2E2CA3062A32025444524270795897E9958BD6")
    def test_02_lifecycle_amendment_present(self): self.assertTrue((ROOT / "round_21" / "ROUND_LOCK_AMENDMENT.json").is_file())
    def test_03_predictor_selection_is_frozen(self): self.assertEqual(self.predictor_record["model_mode"], "eval")
    def test_04_predictor_has_no_grad(self): self.assertTrue(all(not p.requires_grad for p in FrozenDeepSetsPredictor().model.parameters()))
    def test_05_predictor_sha_unchanged(self): self.assertEqual(digest(ROOT / self.predictor_record["checkpoint"]), self.predictor_record["checkpoint_sha256_before_policy_training"])
    def test_06_h_task_is_five(self): self.assertEqual(H_TASK, 5)
    def test_07_no_hgnn(self): self.assertNotIn("GraphLayer", (ROOT / "src/wt/round22/models.py").read_text(encoding="utf-8"))
    def test_08_no_graph_message_passing(self): self.assertNotIn("relation_messages", (ROOT / "src/wt/round22/models.py").read_text(encoding="utf-8"))
    def test_09_shared_uav_encoder(self): self.assertEqual(build_policy(PREDICTED_TRANSFORMER).state_encoder.uav_encoder[0].in_features, 8)
    def test_10_shared_buoy_encoder(self): self.assertEqual(build_policy(PREDICTED_TRANSFORMER).state_encoder.buoy_encoder[0].in_features, 7)
    def test_11_k3_and_k5_use_same_model(self):
        model = build_policy(PREDICTED_TRANSFORMER).eval(); a, b = fake_batch(3), fake_batch(5)
        self.assertEqual(tuple(model(uav_nodes=a["uav_nodes"], buoy_nodes=a["buoy_nodes"], uav_mask=a["uav_mask"], buoy_mask=a["buoy_mask"], global_features=a["global_features"], attention_mask=a["attention_mask"], phi=a["predicted_phi"]).shape), tuple(model(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"]).shape))
    def test_12_uav_permutation_equivariance(self):
        model = build_policy(PREDICTED_TRANSFORMER).eval(); b = self.batch; p = torch.tensor([1, 0]); first = model(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"]); second = model(uav_nodes=b["uav_nodes"][:, :, p], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"][:, :, p], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"][:, :, p]); self.assertTrue(torch.allclose(first[:, p], second, atol=1e-5))
    def test_13_buoy_permutation_invariance(self):
        model = build_policy(PREDICTED_TRANSFORMER).eval(); b = self.batch; p = torch.tensor([2, 0, 4, 1, 3]); first = model(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"]); phi = torch.cat((b["predicted_phi"][..., p], b["predicted_phi"][..., 5:10][..., p], b["predicted_phi"][..., 10:]), dim=-1); second = model(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"][:, :, p], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"][:, :, p], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=phi); self.assertTrue(torch.allclose(first, second, atol=1e-5))
    def test_14_raw_ids_excluded(self): self.assertNotIn("episode_id", (ROOT / "src/wt/round22/models.py").read_text(encoding="utf-8"))
    def test_15_deployable_uses_predicted_phi(self): self.assertIs(_phi(PREDICTED_TRANSFORMER, self.batch), self.batch["predicted_phi"])
    def test_16_deployable_cannot_select_true_phi(self): self.assertIsNot(_phi(PREDICTED_TRANSFORMER, self.batch), self.batch["oracle_phi"])
    def test_17_state_only_rejects_phi(self):
        b = self.batch; model = build_policy(STATE_ONLY_TRANSFORMER)
        with self.assertRaises(ValueError): model(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"])
    def test_18_oracle_is_non_deployable(self): self.assertFalse(build_policy(ORACLE_TRANSFORMER).deployable)
    def test_19_oracle_rollout_is_forbidden(self):
        with self.assertRaises(PermissionError): rollout_action(ORACLE_TRANSFORMER, build_policy(ORACLE_TRANSFORMER), **{})
    def test_20_mlp_has_no_temporal_dependency(self):
        b = self.batch; model = build_policy(PREDICTED_MLP).eval(); a = model(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"]); changed = {key: value.clone() for key, value in b.items()}; changed["uav_nodes"][:, :-1] += 100; self.assertTrue(torch.allclose(a, model(uav_nodes=changed["uav_nodes"], buoy_nodes=changed["buoy_nodes"], uav_mask=changed["uav_mask"], buoy_mask=changed["buoy_mask"], global_features=changed["global_features"], attention_mask=changed["attention_mask"], phi=changed["predicted_phi"]), atol=1e-6))
    def test_21_context_is_ten(self): self.assertEqual(CONTEXT_LENGTH, 10)
    def test_22_historical_actions_absent(self): self.assertFalse(build_policy(PREDICTED_TRANSFORMER).uses_historical_actions)
    def test_23_causal_mask_is_implemented(self): self.assertIn("causal_allowed", (ROOT / "src/wt/round22/models.py").read_text(encoding="utf-8"))
    def test_24_padding_is_excluded_from_loss(self):
        p, t = torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2)); t[:, 1] = 1e6; self.assertEqual(float(action_mse(p, t, torch.tensor([[True, False]]))), 0.0)
    def test_25_joint_action_shape(self):
        b = self.batch; result = build_policy(PREDICTED_TRANSFORMER)(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"]); self.assertEqual(tuple(result.shape[1:]), (2, 2))
    def test_26_safety_projection_is_reused(self): self.assertIn("multi_uav_min_separation_m=15.0", (ROOT / "src/wt/round22/scenarios.py").read_text(encoding="utf-8"))
    def test_27_matching_is_unchanged(self): self.assertIn('"legacy"', (ROOT / "src/wt/round22/scenarios.py").read_text(encoding="utf-8"))
    def test_28_shared_spectrum_is_unchanged(self): self.assertIn('"shared"', (ROOT / "src/wt/round22/scenarios.py").read_text(encoding="utf-8"))
    def test_29_fresh_manifest_is_unchanged(self):
        audit = json.loads((R22_OUTPUT / "round22_initialization_governance_audit.json").read_text(encoding="utf-8")); self.assertEqual(digest(R22_OUTPUT / "fresh_multi_uav_policy_dev_manifest.json"), audit["fresh_dev_manifest_sha256"])
    def test_30_fresh_access_obeys_validation_gate(self):
        audit = json.loads((R22_OUTPUT / "access_audit.json").read_text(encoding="utf-8")); self.assertIn(audit["fresh_dev_access_count"], (0, 1))
        if audit["fresh_dev_access_count"] == 1:
            gate = json.loads((R22_OUTPUT / "round22_validation_policy_gate.json").read_text(encoding="utf-8")); self.assertIn(gate["gate"], ("V-A", "V-B"))
    def test_31_round20_development_test_is_not_reused(self): self.assertEqual(json.loads((R22_OUTPUT / "access_audit.json").read_text(encoding="utf-8"))["round20_development_test_reuse_count"], 0)
    def test_32_untouched_eval_is_zero(self): self.assertEqual(json.loads((R22_OUTPUT / "access_audit.json").read_text(encoding="utf-8"))["untouched_eval_model_access_count"], 0)
    def test_33_no_ppo_or_marl(self):
        text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/wt/round22").glob("*.py")); self.assertNotIn("PPO", text); self.assertNotIn("MAPPO", text); self.assertNotIn("MARL", text)
    def test_34_forward_and_loss_are_finite(self):
        b = self.batch; pred = build_policy(PREDICTED_TRANSFORMER)(uav_nodes=b["uav_nodes"], buoy_nodes=b["buoy_nodes"], uav_mask=b["uav_mask"], buoy_mask=b["buoy_mask"], global_features=b["global_features"], attention_mask=b["attention_mask"], phi=b["predicted_phi"]); self.assertTrue(torch.isfinite(pred).all()); self.assertTrue(torch.isfinite(action_mse(pred, b["actions"], b["uav_mask"][:, -1])) )
    def test_35_compileall_passes(self): self.assertTrue(compileall.compile_dir(str(ROOT / "src/wt/round22"), quiet=1))


if __name__ == "__main__":
    unittest.main()
