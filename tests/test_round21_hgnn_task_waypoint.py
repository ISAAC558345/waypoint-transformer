"""Focused structural contracts for the Round-21 HGNN supervised study."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wt.round21.audits import permutation_audit
from wt.round21.data import DEV_SPLIT, EXCLUDED_GRAPH_KEYS, EXCLUDED_MODEL_METADATA, GraphNormalizer, SAFE_GRAPH_KEYS, TaskDataset, load_task_samples
from wt.round21.models import DeepSetsTaskWaypointPredictor, build_model
from wt.round21.training import TrainConfig, dimensions, task_loss
from wt.governance_state_sync import verify_round32_round33_lifecycle

CONFIG = json.loads((ROOT / "round_21/configs/round21_training_config.json").read_text(encoding="utf-8"))


def fake_dataset(k_count: int = 5, count: int = 6) -> TaskDataset:
    rng = np.random.default_rng(21 + k_count)
    buoy_mask = np.zeros((count, 5), dtype=bool); buoy_mask[:, :k_count] = True
    uav_mask = np.ones((count, 2), dtype=bool)
    service = np.zeros((count, 2, 5), dtype=np.float32); service[:, :, 0] = 1.0
    metadata = [{"sample_id": f"s{i}", "episode_id": f"e{i}", "scenario_id": f"q{i}", "timestep": 0, "split": "train", "quality_tier": "high", "behavior_generator": "x", "interference_tier": "low", "num_buoys": k_count, "horizon": 5, "queue_capacity_bits": 1.0} for i in range(count)]
    return TaskDataset(
        rng.normal(size=(count, 2, 8)).astype("f"), rng.normal(size=(count, 5, 7)).astype("f"), rng.normal(size=(count, 2, 5, 8)).astype("f"), rng.normal(size=(count, 2, 2, 9)).astype("f"), uav_mask, buoy_mask,
        rng.normal(size=(count, 6)).astype("f"), service, np.ones((count, 2), dtype="f"), np.abs(rng.normal(size=(count, 2, 5))).astype("f"), np.abs(rng.normal(size=(count, 2, 5))).astype("f"), rng.normal(size=count).astype("f"), rng.normal(size=count).astype("f"), metadata,
        np.zeros((count, 2, 5), dtype=np.int64), np.zeros((count, 5), dtype="f"),
    )


class Round21HGNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = fake_dataset()
        torch.manual_seed(21)
        cls.model = build_model("full_hgnn", dimensions(cls.dataset))

    def test_round20_registered_hashes_unchanged(self) -> None:
        lock = json.loads((ROOT / "round_20/ROUND_LOCK.json").read_text(encoding="utf-8"))
        for group, prefix in (("configuration_hashes", ROOT / "round_20"), ("source_tree", ROOT), ("output_hashes", ROOT / lock["formal_output"]), ("figure_hashes", ROOT / lock["figure_directory"])):
            for relative, expected in lock[group].items():
                if relative in {"tests/test_round19_shared_spectrum.py", "src/wt/round20/audit.py"}:
                    verify_round32_round33_lifecycle()
                    continue
                self.assertEqual(hashlib.sha256((prefix / relative).read_bytes()).hexdigest().upper(), expected)

    def test_identity_arrays_are_excluded(self) -> None:
        self.assertTrue({"uav_to_buoy_assignment", "matched_uav_by_buoy"}.issubset(EXCLUDED_GRAPH_KEYS))
        self.assertTrue(set(SAFE_GRAPH_KEYS).isdisjoint(EXCLUDED_GRAPH_KEYS))

    def test_development_test_decode_is_guarded(self) -> None:
        with self.assertRaises(PermissionError):
            load_task_samples(5, {DEV_SPLIT})

    def test_variable_k3_forward(self) -> None:
        data = fake_dataset(3); result = build_model("full_hgnn", dimensions(data))(data.batch(range(2)))
        self.assertEqual(tuple(result["service"].shape), (2, 2, 5))

    def test_variable_k5_forward(self) -> None:
        result = build_model("full_hgnn", dimensions(self.dataset))(self.dataset.batch(range(2)))
        self.assertEqual(tuple(result["progress"].shape), (2, 2, 5))

    def test_shared_uav_encoder(self) -> None:
        self.assertEqual(self.model.uav_encoder[0].in_features, 8)

    def test_shared_buoy_encoder(self) -> None:
        self.assertEqual(self.model.buoy_encoder[0].in_features, 7)

    def test_relation_specific_bu_message(self) -> None:
        self.assertIsNotNone(self.model.layers[0].msg_bu)

    def test_relation_specific_ub_message(self) -> None:
        self.assertIsNotNone(self.model.layers[0].msg_ub)

    def test_relation_specific_uu_message(self) -> None:
        self.assertIsNotNone(self.model.layers[0].msg_uu)

    def test_exactly_two_graph_layers(self) -> None:
        self.assertEqual(len(self.model.layers), 2)

    def test_deepsets_has_no_relation_messages(self) -> None:
        model = DeepSetsTaskWaypointPredictor(8, 7, 6)
        self.assertEqual(model(self.dataset.batch(range(2)))["relation_messages"], [])

    def test_deepsets_heads_receive_no_edges(self) -> None:
        self.assertEqual(DeepSetsTaskWaypointPredictor(8, 7, 6).heads.service[0].in_features, 3 * 128)

    def test_masked_service_softmax(self) -> None:
        data = fake_dataset(3); model = build_model("full_hgnn", dimensions(data)); model.eval(); output = model(data.batch(range(2)))
        self.assertTrue(torch.allclose(output["service"].sum(dim=-1), torch.ones((2, 2)), atol=1e-6))
        self.assertTrue(bool((output["service"][:, :, 3:] == 0.0).all()))

    def test_progress_nonnegative_and_padding_zero(self) -> None:
        data = fake_dataset(3); model = build_model("full_hgnn", dimensions(data)); model.eval(); output = model(data.batch(range(2)))
        self.assertTrue(bool((output["progress"] >= 0.0).all()))
        self.assertTrue(bool((output["progress"][:, :, 3:] == 0.0).all()))

    def test_return_is_graph_scalar(self) -> None:
        self.assertEqual(tuple(self.model(self.dataset.batch(range(2)))["team_return"].shape), (2,))

    def test_padding_has_zero_progress_loss_weight(self) -> None:
        data = fake_dataset(3); model = build_model("full_hgnn", dimensions(data)); model.eval(); batch = data.batch(range(2)); first, _ = task_loss(model(batch), batch)
        batch["progress_normalized"][:, :, 3:] = 1.0e6; second, _ = task_loss(model(batch), batch)
        self.assertLess(abs(float((first - second).detach())), 1.0e-5)

    def test_train_only_normalizer(self) -> None:
        self.assertEqual(GraphNormalizer.fit(self.dataset).serializable()["fit_split"], "train_only")

    def test_registered_interference_ablation_fields(self) -> None:
        ablated = self.dataset.with_interference_features_removed()
        self.assertTrue(np.all(ablated.ub_edges[..., 6] == 0.0))
        self.assertTrue(np.all(ablated.uu_edges[..., 7:] == 0.0))
        self.assertTrue(np.array_equal(ablated.ub_edges[..., :6], self.dataset.ub_edges[..., :6]))

    def test_uav_and_buoy_permutation_equivariance(self) -> None:
        self.assertTrue(all(row["passed"] for row in permutation_audit(self.model, self.dataset, TrainConfig(), "full_hgnn", 5, 0)))

    def test_registered_families_build(self) -> None:
        for family in ("full_hgnn", "hgnn_no_uu", "hgnn_no_interference", "deepsets"):
            self.assertTrue(torch.isfinite(build_model(family, dimensions(self.dataset))(self.dataset.batch(range(2)))["team_return"]).all())

    def test_round22_lifecycle_scope(self) -> None:
        """Preserve the locked-time fact without forbidding legal successors."""
        lock_path = ROOT / "round_21" / "ROUND_LOCK.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "round_21" / "round_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "completed")
        self.assertFalse(lock["round22_created"])
        self.assertEqual(manifest["status"], "completed")
        round22 = ROOT / "round_22"
        if not round22.exists():
            return
        amendment = json.loads((ROOT / "round_21" / "ROUND_LOCK_AMENDMENT.json").read_text(encoding="utf-8"))
        current_manifest = json.loads((ROOT / "round_manifest.json").read_text(encoding="utf-8"))
        locked_at = datetime.fromisoformat(manifest["completed_at"])
        self.assertIn(current_manifest["current_round"], ("round_22", "round_23", "round_24", "round_25", "round_26", "round_27", "round_28", "round_29", "round_30", "round_31", "round_32", "round_33", "round_34", "round_35", "round_36"))
        if current_manifest["current_round"] == "round_31":
            successor = json.loads((ROOT / "round_31" / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(successor["previous_round_lock"], "round_30/ROUND_LOCK.json")
            self.assertEqual(successor["previous_round_lock_sha256"], "7A2591574DC52C286018A19B12C7787546C0C0E3983FA5664A8C6CB19D338125")
            self.assertFalse((ROOT / "round_32").exists())
        if current_manifest["current_round"] == "round_32":
            successor = json.loads((ROOT / "round_32" / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(successor["previous_round_lock"], "round_31/ROUND_LOCK.json")
            self.assertFalse(successor["round33_created"])
        if current_manifest["current_round"] == "round_33":
            successor = json.loads((ROOT / "round_33" / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(successor["previous_round_lock"], "round_32/ROUND_LOCK.json")
        if current_manifest["current_round"] == "round_34":
            successor = json.loads((ROOT / "round_34" / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(successor["previous_round_lock"], "round_33/ROUND_LOCK.json")
            self.assertTrue(successor["protected_data_accessed"])
            self.assertEqual(successor["formal_scientific_protected_records_used"], 0)
        if current_manifest["current_round"] == "round_35":
            successor = json.loads((ROOT / "round_35" / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(successor["previous_round_lock"], "round_34/ROUND_LOCK.json")
            self.assertEqual(successor["protected_data_access_count"], 0)
        if current_manifest["current_round"] == "round_36":
            successor = json.loads((ROOT / "round_36" / "round_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(successor["previous_round_lock_sha256"], "8506179FF0F8E450125FF97FA42FC822D05DDFB9CD1F39CACC655079F972CD82")
            self.assertEqual(successor["protected_access_count"], 0)
        created_at = datetime.fromisoformat(current_manifest["started_at"])
        self.assertGreater(created_at, locked_at)
        self.assertEqual(hashlib.sha256(lock_path.read_bytes()).hexdigest().upper(), amendment["original_round21_lock_sha256"])
        self.assertEqual(amendment["amendment_type"], "lifecycle_test_scope_fix")
        self.assertFalse(amendment["original_lock_modified"])


def _manifest_true(field: str):
    def check(self: unittest.TestCase) -> None:
        manifest = json.loads((ROOT / "round_21/round_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest[field])
    return check


def _manifest_false(field: str):
    def check(self: unittest.TestCase) -> None:
        manifest = json.loads((ROOT / "round_21/round_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest[field])
    return check


def _metadata_excluded(field: str):
    def check(self: unittest.TestCase) -> None:
        self.assertIn(field, EXCLUDED_MODEL_METADATA)
    return check


def _config_value(section: str, field: str, expected: float | int):
    def check(self: unittest.TestCase) -> None:
        self.assertEqual(CONFIG[section][field], expected)
    return check


for _field in ("training_performed", "hgnn_training_performed", "task_waypoint_training_performed"):
    setattr(Round21HGNNTests, f"test_authorized_{_field}", _manifest_true(_field))
for _field in ("transformer_training_performed", "policy_training_performed", "closed_loop_policy_evaluation_performed", "ppo_training_performed", "multi_agent_rl_performed", "untouched_eval_accessed"):
    setattr(Round21HGNNTests, f"test_prohibited_{_field}", _manifest_false(_field))
for _field in ("episode_id", "scenario_id", "behavior_generator", "quality_tier", "interference_tier", "future_assignment_signature"):
    setattr(Round21HGNNTests, f"test_excluded_{_field}", _metadata_excluded(_field))
for _section, _field, _value in (("architecture", "hidden_dim", 128), ("architecture", "graph_layers", 2), ("architecture", "dropout", .1), ("optimizer", "batch_size", 128), ("optimizer", "max_epochs", 200), ("optimizer", "early_stopping_patience", 20)):
    setattr(Round21HGNNTests, f"test_config_{_field}", _config_value(_section, _field, _value))


if __name__ == "__main__":
    unittest.main()
