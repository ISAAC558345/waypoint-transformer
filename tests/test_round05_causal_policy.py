"""Round-5 frozen-waypoint causal-policy acceptance tests.

The numbered tests intentionally mirror the 45-item Round-5 checklist.  The
locked Round-3 dataset and selected Round-4 checkpoint are loaded exactly once
for the class.  Trainer state-machine checks use a tiny synthetic dataset and
write only to a disposable Round-5 test directory.
"""

from __future__ import annotations

import compileall
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wt.round5.causal_policy import (
    CausalPolicyConfig,
    CausalTrajectoryPolicy,
    count_parameters,
    masked_action_mse,
)
from wt.round5.frozen_waypoint import (
    SELECTED_WAYPOINT_CHECKPOINT_SHA256,
    build_frozen_waypoint_cache,
    load_frozen_waypoint_predictor,
)
from wt.round5.round5_data import (
    CANONICAL_COLLECTION_POLICIES,
    build_round5_sequence_datasets,
    build_round5_sequence_split,
    load_round5_transition_dataset,
)
from wt.round5.round5_diagnostics import (
    run_interface_smoke,
    run_recorded_replay,
)
from wt.round5.round5_evaluation import (
    compute_action_metrics,
    radial_action_projection,
)
from wt.round5.round5_runner import build_seeded_policy
from wt.round5.round5_training import (
    PolicyBalancedBatchSampler,
    Round5Trainer,
    TrainingConfig,
    load_policy_checkpoint,
    make_policy_balanced_loader,
    prepare_batch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = (
    REPOSITORY_ROOT
    / "round_03"
    / "outputs"
    / "round03_pilot_canonical_20260730_230000"
)
ROUND3_LOCK = REPOSITORY_ROOT / "round_03" / "ROUND_LOCK.json"
ROUND4_LOCK = REPOSITORY_ROOT / "round_04" / "ROUND_LOCK.json"
ROUND4_FORMAL = (
    REPOSITORY_ROOT
    / "round_04"
    / "outputs"
    / "round04_formal_canonical_20260802_173717"
)
SELECTED_CHECKPOINT = (
    ROUND4_FORMAL
    / "horizon_030"
    / "goal_conditioned"
    / "seed_001"
    / "checkpoints"
    / "best.pt"
)
EXPECTED_ROUND3_LOCK_SHA256 = (
    "601C6679D17EE25534C9A415C75BBF0DC28A167E2ED5EFE11BA0FB3A5C984375"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _snapshot(root: Path):
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class _TinySequenceDataset(Dataset):
    """Four-policy deterministic sequence dataset for trainer-only tests."""

    def __init__(self, *, context_length: int = 2, count: int = 8):
        generator = torch.Generator().manual_seed(20260803)
        self.states = torch.randn(count, context_length, 3, generator=generator)
        self.node_masks = torch.ones(count, context_length, 2)
        self.conditions = torch.randn(count, context_length, 2, generator=generator)
        self.remaining_time = torch.linspace(1.0, 0.2, context_length).view(1, -1, 1).repeat(count, 1, 1)
        self.attention_mask = torch.ones(count, context_length, dtype=torch.bool)
        self.actions = torch.tanh(self.states[..., :2] * 0.2)
        self.loss_mask = torch.zeros(count, context_length, dtype=torch.bool)
        self.loss_mask[:, -1] = True
        repeated = list(CANONICAL_COLLECTION_POLICIES) * (count // 4)
        self.policy_names = np.asarray(repeated, dtype="<U32")

    def __len__(self):
        return int(self.states.shape[0])

    def __getitem__(self, index):
        return {
            "states": self.states[index],
            "node_masks": self.node_masks[index],
            "conditions": self.conditions[index],
            "remaining_time": self.remaining_time[index],
            "attention_mask": self.attention_mask[index],
            "actions": self.actions[index],
            "loss_mask": self.loss_mask[index],
        }


class _SmokeEnvironment:
    """Disposable integration double; it records only bounded smoke steps."""

    def __init__(self):
        self.step_calls = 0
        self.actions = []

    def reset(self):
        self.step_calls = 0
        self.actions = []
        return np.asarray([0.0, 0.0]), {"remaining_time": 1.0}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.step_calls += 1
        observation = np.asarray([self.step_calls, -self.step_calls], dtype=np.float64)
        return observation, {"remaining_time": 1.0 - 0.1 * self.step_calls}, False, False


class TestRound05CausalPolicy(unittest.TestCase):
    """Exactly numbered acceptance checks matching the Round-5 specification."""

    @classmethod
    def setUpClass(cls):
        if not CANONICAL_ROOT.is_dir() or not SELECTED_CHECKPOINT.is_file():
            raise unittest.SkipTest("locked Round-3/Round-4 canonical inputs unavailable")
        cls.round3_before = _snapshot(CANONICAL_ROOT)
        cls.round4_checkpoint_before = _snapshot(SELECTED_CHECKPOINT.parent)
        cls.round4_lock = json.loads(ROUND4_LOCK.read_text(encoding="utf-8"))
        cls.dataset = load_round5_transition_dataset(
            CANONICAL_ROOT, waypoint_horizon=30, verify_round3_lock=True
        )
        cls.predictor = load_frozen_waypoint_predictor(
            SELECTED_CHECKPOINT, cls.dataset, device="cpu"
        )
        cls.cache = build_frozen_waypoint_cache(
            cls.dataset, cls.predictor, batch_size=2048
        )
        cls.predicted_h10 = build_round5_sequence_datasets(
            cls.dataset,
            context_length=10,
            condition_source="predicted_waypoint",
            waypoint_cache=cls.cache,
        )
        cls.predicted_h1 = build_round5_sequence_datasets(
            cls.dataset,
            context_length=1,
            condition_source="predicted_waypoint",
            waypoint_cache=cls.cache,
        )
        # Test checkpoints are disposable and intentionally live in the OS
        # temporary area.  This keeps a read-only source checkout testable and
        # prevents test-only files from being mistaken for Round-5 results.
        requested_temp = os.environ.get("ROUND05_TEST_TMPDIR")
        if requested_temp:
            temporary_parent = Path(requested_temp).expanduser().resolve()
            temporary_parent.mkdir(parents=True, exist_ok=True)
            cls.temporary_root = Path(
                tempfile.mkdtemp(prefix="_round05_tests_", dir=temporary_parent)
            )
        else:
            cls.temporary_root = Path(tempfile.mkdtemp(prefix="_round05_tests_"))

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "temporary_root", None) is not None:
            shutil.rmtree(cls.temporary_root, ignore_errors=True)

    @classmethod
    def _tiny_model(cls, *, context_length=2, variant="wt_transformer"):
        condition_dim = 0 if variant == "state_only_transformer" else 2
        return CausalTrajectoryPolicy(
            CausalPolicyConfig(
                variant=variant,
                state_dim=3,
                node_mask_dim=2,
                condition_dim=condition_dim,
                action_dim=2,
                context_length=context_length,
                n_embd=8,
                n_layer=1,
                n_head=2,
                dropout=0.0,
            )
        )

    @classmethod
    def _tiny_training_components(cls, output_name: str, *, max_epochs=1, patience=2):
        dataset = _TinySequenceDataset()
        config = TrainingConfig(
            seed=7,
            action_scale=10.0,
            batch_size=8,
            max_epochs=max_epochs,
            patience=patience,
            samples_per_epoch=8,
            expected_policy_count=4,
            deterministic_algorithms=True,
            device="cpu",
        )
        model = cls._tiny_model()
        train_loader = make_policy_balanced_loader(dataset, config)
        validation_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        trainer = Round5Trainer(
            model,
            config,
            cls.temporary_root / output_name,
            provenance={"test_fixture": "tiny", "selection_split": "validation"},
        )
        return dataset, config, model, train_loader, validation_loader, trainer

    def test_001_round3_data_hash_is_unchanged(self):
        self.assertEqual(_sha256(ROUND3_LOCK), EXPECTED_ROUND3_LOCK_SHA256)
        locked = self.round4_lock["canonical_input"]
        self.assertEqual(_sha256(CANONICAL_ROOT / "dataset_manifest.json"), locked["dataset_manifest_sha256"])
        self.assertEqual(_sha256(CANONICAL_ROOT / "waypoint_sample_manifest.json"), locked["waypoint_manifest_sha256"])
        self.assertEqual(_snapshot(CANONICAL_ROOT), self.round3_before)

    def test_002_round4_selected_checkpoint_hash_is_unchanged(self):
        self.assertEqual(_sha256(SELECTED_CHECKPOINT), SELECTED_WAYPOINT_CHECKPOINT_SHA256)
        self.assertEqual(self.predictor.checkpoint_sha256, SELECTED_WAYPOINT_CHECKPOINT_SHA256)

    def test_003_selected_horizon_and_seed_are_validation_derived(self):
        selected = json.loads((ROUND4_FORMAL / "selected_waypoint_config.json").read_text(encoding="utf-8"))
        self.assertEqual(selected["selection_split"], "validation")
        self.assertFalse(selected["test_metrics_used_for_selection"])
        self.assertEqual(selected["recommended_horizon"], 30)
        with (ROUND4_FORMAL / "waypoint_training_summary.csv").open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if int(row["horizon"]) == 30 and row["model_kind"] == "goal_conditioned"]
        best = min(rows, key=lambda row: float(row["best_validation_endpoint_rmse_m"]))
        self.assertEqual(int(best["seed"]), self.predictor.selected_seed)
        self.assertEqual(self.predictor.selected_seed, 1)

    def test_004_waypoint_parameters_are_frozen(self):
        self.assertTrue(all(not parameter.requires_grad for parameter in self.predictor.model.parameters()))
        policy = self._tiny_model()
        policy_ids = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
        waypoint_ids = {id(parameter) for parameter in self.predictor.model.parameters()}
        self.assertFalse(policy_ids & waypoint_ids)

    def test_005_waypoint_inference_never_builds_gradients(self):
        before = self.predictor.parameter_sha256
        prediction = self.predictor.predict(self.dataset.validation, batch_size=512)
        self.assertFalse(torch.is_grad_enabled() and prediction.dtype == object)
        self.assertTrue(all(parameter.grad is None for parameter in self.predictor.model.parameters()))
        self.assertEqual(self.predictor.parameter_sha256, before)

    def test_006_waypoint_network_is_in_eval_mode(self):
        self.assertFalse(self.predictor.model.training)
        self.predictor.predict(self.dataset.train, batch_size=2048)
        self.assertFalse(self.predictor.model.training)

    def test_007_wt_sequences_use_cached_predicted_waypoints(self):
        sequence = self.predicted_h1["train"]
        self.assertEqual(sequence.condition_source, "predicted_waypoint")
        np.testing.assert_array_equal(sequence.conditions[:, 0], self.cache.train.predicted_waypoints)

    def test_008_main_model_does_not_read_true_future_waypoints(self):
        sequence = self.predicted_h1["validation"]
        model = CausalTrajectoryPolicy(CausalPolicyConfig(
            variant="wt_transformer", state_dim=sequence.state_dim,
            node_mask_dim=sequence.node_mask_dim, condition_dim=sequence.condition_dim,
            action_dim=sequence.action_dim, context_length=1, n_embd=8,
            n_layer=1, n_head=2, dropout=0.0,
        ))
        batch = {name: torch.as_tensor(value[:2]) for name, value in sequence.as_mapping().items()}
        batch["oracle_waypoints"] = torch.full((2, 1, 2), 1.0e6)
        inputs, _, _ = prepare_batch(batch, model, torch.device("cpu"))
        self.assertNotIn("oracle_waypoints", inputs)
        self.assertFalse(np.array_equal(sequence.conditions[:, 0], self.dataset.validation.oracle_waypoints))

    def test_009_oracle_waypoint_transformer_is_non_deployable(self):
        config = CausalPolicyConfig(
            variant="oracle_waypoint_transformer", state_dim=self.dataset.state_dim,
            node_mask_dim=self.dataset.node_mask_dim, condition_dim=self.dataset.action_dim,
            action_dim=self.dataset.action_dim, context_length=10, n_embd=8,
            n_layer=1, n_head=2, dropout=0.0,
        )
        oracle = build_round5_sequence_split(
            self.dataset.validation, context_length=10, condition_source="oracle_waypoint"
        )
        self.assertFalse(config.deployable)
        self.assertEqual(oracle.condition_source, "oracle_waypoint")

    def test_010_hindsight_goal_is_confined_to_goal_or_waypoint_chains(self):
        direct = build_round5_sequence_split(self.dataset.train, context_length=1, condition_source="direct_goal")
        state_only = build_round5_sequence_split(self.dataset.train, context_length=1, condition_source="state_only")
        np.testing.assert_array_equal(direct.conditions[:, 0], self.dataset.train.goals)
        self.assertEqual(state_only.condition_dim, 0)
        self.assertNotIn("goals", self.predicted_h1["train"].as_mapping())

    def test_011_causal_policy_outputs_are_invariant_to_future_state_changes(self):
        model = self._tiny_model(context_length=4).eval()
        states = torch.randn(1, 4, 3)
        nodes = torch.ones(1, 4, 2)
        conditions = torch.randn(1, 4, 2)
        remaining = torch.ones(1, 4, 1)
        mask = torch.ones(1, 4, dtype=torch.bool)
        changed = states.clone(); changed[:, 3] += 1000.0
        with torch.no_grad():
            left = model(states, nodes, conditions, remaining, mask)
            right = model(changed, nodes, conditions, remaining, mask)
        torch.testing.assert_close(left[:, :3], right[:, :3], rtol=0.0, atol=1e-6)

    def test_012_policy_id_is_metadata_only(self):
        sequence = self.predicted_h1["train"]
        self.assertIn("policy_name", sequence.metadata)
        self.assertNotIn("policy_name", sequence.as_mapping())
        self.assertNotIn("policy_id", sequence.as_mapping())

    def test_013_quality_tier_is_metadata_only(self):
        sequence = self.predicted_h1["train"]
        self.assertIn("quality_class", sequence.metadata)
        self.assertNotIn("quality_class", sequence.as_mapping())
        self.assertNotIn("quality_tier", sequence.as_mapping())

    def test_014_action_labels_are_separated_from_model_inputs(self):
        model = self._tiny_model()
        batch = next(iter(DataLoader(_TinySequenceDataset(), batch_size=2)))
        inputs, targets, loss_mask = prepare_batch(batch, model, torch.device("cpu"))
        self.assertNotIn("actions", inputs)
        self.assertNotIn("targets", inputs)
        self.assertEqual(targets.shape, (2, 2, 2))
        self.assertEqual(loss_mask.shape, (2, 2))

    def test_015_historical_actions_are_not_policy_tokens(self):
        model = self._tiny_model()
        self.assertFalse(model.uses_action_tokens)
        self.assertFalse(model.gpt_config.model_actions)
        self.assertNotIn("actions", inspect.signature(model.forward).parameters)
        self.assertFalse(any("action_encoder" in name for name, _ in model.named_modules()))

    def test_016_sequences_never_cross_episode_boundaries(self):
        sequence = self.predicted_h10["train"]
        transition = self.dataset.train
        lookup = {(str(ep), int(t)): index for index, (ep, t) in enumerate(zip(transition.episode_id, transition.t))}
        for anchor in range(len(sequence)):
            episode = str(transition.episode_id[anchor]); timestep = int(transition.t[anchor])
            expected_t = list(range(max(0, timestep - 9), timestep + 1))
            source = [lookup[(episode, value)] for value in expected_t]
            width = len(source)
            np.testing.assert_array_equal(sequence.states[anchor, -width:], transition.states[source])

    def test_017_sequence_datasets_preserve_disjoint_splits(self):
        episode_sets = []
        for split_name in ("train", "validation", "test"):
            sequence = self.predicted_h10[split_name]
            self.assertEqual(sequence.split_name, split_name)
            self.assertEqual(set(sequence.metadata["episode_id"]), set(self.dataset.split(split_name).episode_id))
            episode_sets.append(set(sequence.metadata["episode_id"]))
        self.assertFalse(episode_sets[0] & episode_sets[1])
        self.assertFalse(episode_sets[0] & episode_sets[2])
        self.assertFalse(episode_sets[1] & episode_sets[2])

    def test_018_no_sequence_is_created_after_termination(self):
        found = 0
        for split_name in ("train", "validation", "test"):
            split = self.dataset.split(split_name)
            rows = np.flatnonzero(split.terminated); found += len(rows)
            self.assertTrue(np.all(split.t[rows] == split.episode_length[rows] - 1))
            for row in rows:
                self.assertFalse(np.any((split.episode_id == split.episode_id[row]) & (split.t > split.t[row])))
        self.assertGreater(found, 0)

    def test_019_no_sequence_is_created_after_truncation(self):
        found = 0
        for split_name in ("train", "validation", "test"):
            split = self.dataset.split(split_name)
            rows = np.flatnonzero(split.truncated); found += len(rows)
            self.assertTrue(np.all(split.t[rows] == split.episode_length[rows] - 1))
            for row in rows:
                self.assertFalse(np.any((split.episode_id == split.episode_id[row]) & (split.t > split.t[row])))
        self.assertGreater(found, 0)

    def test_020_causal_attention_blocks_future_tokens(self):
        model = self._tiny_model(context_length=4).eval()
        base = {
            "states": torch.randn(2, 4, 3), "node_masks": torch.ones(2, 4, 2),
            "conditions": torch.randn(2, 4, 2), "remaining_time": torch.ones(2, 4, 1),
            "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        }
        changed = {name: value.clone() for name, value in base.items()}
        changed["conditions"][:, 3] *= -1000.0
        with torch.no_grad():
            first = model(**base); second = model(**changed)
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0.0, atol=1e-6)

    def test_021_padding_is_excluded_from_attention_and_loss(self):
        prediction = torch.tensor([[[99.0, 99.0], [0.2, -0.1]]])
        target_a = torch.tensor([[[-99.0, -99.0], [0.0, 0.0]]])
        target_b = target_a.clone(); target_b[:, 0] = 1.0e9
        mask = torch.tensor([[False, True]])
        torch.testing.assert_close(masked_action_mse(prediction, target_a, mask), masked_action_mse(prediction, target_b, mask))
        model = self._tiny_model().eval()
        batch = {"states": torch.zeros(1, 2, 3), "node_masks": torch.zeros(1, 2, 2), "conditions": torch.zeros(1, 2, 2), "remaining_time": torch.zeros(1, 2, 1), "attention_mask": mask}
        altered = {name: value.clone() for name, value in batch.items()}; altered["states"][:, 0] = 1.0e8
        with torch.no_grad():
            left = model(**batch); right = model(**altered)
        torch.testing.assert_close(left[:, -1], right[:, -1], rtol=0.0, atol=1e-6)

    def test_022_context_lengths_one_and_ten_have_exact_shapes(self):
        for context, mapping in ((1, self.predicted_h1), (10, self.predicted_h10)):
            for split_name in ("train", "validation", "test"):
                sequence = mapping[split_name]
                self.assertEqual(sequence.context_length, context)
                self.assertEqual(sequence.states.shape[:2], (len(sequence), context))
                self.assertEqual(sequence.attention_mask.shape, (len(sequence), context))

    def test_023_dimensions_are_derived_from_the_locked_schema(self):
        dimensions = self.dataset.schema["dimensions"]
        self.assertEqual(self.dataset.state_dim, dimensions["state_dim"])
        self.assertEqual(self.dataset.goal_dim, dimensions["goal_dim"])
        self.assertEqual(self.dataset.node_mask_dim, dimensions["node_mask_dim"])
        self.assertEqual(self.dataset.action_dim, dimensions["action_dim"])
        self.assertEqual(self.predictor.input_dim, self.dataset.state_dim + self.dataset.goal_dim + self.dataset.node_mask_dim)

    def test_024_models_do_not_assume_current_62_13_5_dimensions(self):
        config = CausalPolicyConfig(
            variant="direct_goal_transformer", state_dim=7, node_mask_dim=2,
            condition_dim=3, action_dim=2, context_length=3,
            n_embd=12, n_layer=1, n_head=3, dropout=0.0,
        )
        model = CausalTrajectoryPolicy(config)
        output = model(torch.zeros(4, 3, 7), torch.ones(4, 3, 2), torch.zeros(4, 3, 3), torch.ones(4, 3, 1), torch.ones(4, 3, dtype=torch.bool))
        self.assertEqual(output.shape, (4, 3, 2))

    def test_025_action_output_dimension_is_strictly_two(self):
        sequence = self.predicted_h1["train"]
        config = CausalPolicyConfig(
            variant="wt_transformer", state_dim=sequence.state_dim,
            node_mask_dim=sequence.node_mask_dim, condition_dim=sequence.condition_dim,
            action_dim=self.dataset.action_dim, context_length=1,
            n_embd=8, n_layer=1, n_head=2, dropout=0.0,
        )
        model = CausalTrajectoryPolicy(config)
        batch = {name: torch.as_tensor(value[:3]) for name, value in sequence.as_mapping().items() if name not in {"actions", "loss_mask"}}
        self.assertEqual(model(**batch).shape, (3, 1, 2))
        with self.assertRaises(ValueError):
            radial_action_projection(np.zeros((1, 3)), self.dataset.maximum_step_distance_m)

    def test_026_action_normalization_and_denormalization_are_inverse(self):
        for split_name in ("train", "validation", "test"):
            split = self.dataset.split(split_name)
            np.testing.assert_allclose(split.actions * self.dataset.maximum_step_distance_m, split.actions_m, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(split.actions_m, split.next_positions_m - split.current_positions_m, rtol=0.0, atol=1e-10)

    def test_027_radial_projection_obeys_maximum_step_distance(self):
        values = np.asarray([[2.0, 2.0], [0.3, -0.4], [0.0, 0.0]])
        projection = radial_action_projection(values, self.dataset.maximum_step_distance_m)
        self.assertTrue(np.all(np.linalg.norm(projection.projected_m, axis=1) <= self.dataset.maximum_step_distance_m + 1e-9))
        self.assertTrue(projection.activation_mask[0])
        np.testing.assert_allclose(projection.projected_normalized[1:], values[1:])

    def test_028_near_zero_direction_metrics_are_finite_and_division_safe(self):
        metrics = compute_action_metrics(np.zeros((3, 2)), np.zeros((3, 2)), 10.0)
        self.assertEqual(metrics["direction_metric_eligible_count"], 0)
        self.assertFalse(metrics["direction_metric_available"])
        self.assertTrue(all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float))))
        # A nominal 10 m float32 displacement can reconstruct a few 1e-7 m
        # beyond the boundary.  The configured 1e-6 m numerical threshold must
        # keep that executed action from being falsely reported invalid.
        just_above = np.nextafter(np.float32(10.0), np.float32(np.inf))
        boundary = np.asarray([[just_above, 0.0]], dtype=np.float64)
        boundary_metrics = compute_action_metrics(boundary / 10.0, boundary / 10.0, 10.0)
        self.assertEqual(boundary_metrics["target_action_validity_rate"], 1.0)
        self.assertEqual(boundary_metrics["action_validity_rate"], 1.0)

    def test_029_policy_balanced_sampler_is_equal_per_batch(self):
        labels = np.asarray(["max_queue"] * 11 + ["nearest_buoy"] * 3 + ["queue_rate_energy"] * 5 + ["random_valid"] * 2)
        sampler = PolicyBalancedBatchSampler(labels, 8, seed=19, samples_per_epoch=24, expected_policy_count=4)
        for batch in sampler:
            observed = labels[np.asarray(batch)]
            self.assertEqual({name: int(np.sum(observed == name)) for name in CANONICAL_COLLECTION_POLICIES}, {name: 2 for name in CANONICAL_COLLECTION_POLICIES})
        self.assertEqual(set(sampler.last_counts.values()), {6})

    def test_030_direct_goal_control_rejects_waypoint_cache(self):
        with self.assertRaises(ValueError):
            build_round5_sequence_split(self.dataset.train, context_length=1, condition_source="direct_goal", waypoint_cache_split=self.cache.train)
        direct = build_round5_sequence_split(self.dataset.train, context_length=1, condition_source="direct_goal")
        self.assertEqual(direct.condition_dim, self.dataset.goal_dim)

    def test_031_wt_control_does_not_directly_receive_global_goal(self):
        sequence = self.predicted_h1["train"]
        self.assertEqual(sequence.condition_dim, self.dataset.action_dim)
        self.assertNotEqual(sequence.condition_dim, self.dataset.goal_dim)
        self.assertNotIn("goals", sequence.as_mapping())

    def test_032_state_only_control_receives_neither_goal_nor_waypoint(self):
        with self.assertRaises(ValueError):
            build_round5_sequence_split(self.dataset.train, context_length=1, condition_source="state_only", waypoint_cache_split=self.cache.train)
        state_only = build_round5_sequence_split(self.dataset.train, context_length=1, condition_source="state_only")
        self.assertEqual(state_only.condition_dim, 0)
        self.assertEqual(state_only.conditions.shape[-1], 0)

    def test_033_three_transformers_have_comparable_identical_backbones(self):
        common = dict(state_dim=7, node_mask_dim=2, action_dim=2, context_length=10, n_embd=16, n_layer=2, n_head=4, dropout=0.1)
        models = [
            CausalTrajectoryPolicy(CausalPolicyConfig(variant="wt_transformer", condition_dim=2, **common)),
            CausalTrajectoryPolicy(CausalPolicyConfig(variant="direct_goal_transformer", condition_dim=3, **common)),
            CausalTrajectoryPolicy(CausalPolicyConfig(variant="state_only_transformer", condition_dim=0, **common)),
        ]
        signatures = [(m.gpt_config.n_embd, m.gpt_config.n_layer, m.gpt_config.n_head, m.gpt_config.resid_pdrop, sum(p.numel() for p in m.blocks.parameters()), sum(p.numel() for p in m.action_head.parameters())) for m in models]
        self.assertEqual(len(set(signatures)), 1)
        self.assertLess(max(count_parameters(m) for m in models) - min(count_parameters(m) for m in models), 100)

    def test_034_checkpoint_reload_preserves_predictions(self):
        _, config, model, train_loader, validation_loader, trainer = self._tiny_training_components("reload", max_epochs=1)
        seeded_a = build_seeded_policy(model.config, config)
        _ = torch.rand(31)
        seeded_b = build_seeded_policy(model.config, config)
        for left, right in zip(seeded_a.state_dict().values(), seeded_b.state_dict().values()):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        result = trainer.fit(train_loader, validation_loader)
        batch = next(iter(validation_loader)); inputs, _, _ = prepare_batch(batch, model, torch.device("cpu"))
        model.eval()
        with torch.no_grad(): before = model(**inputs).clone()
        reloaded = self._tiny_model(); payload = load_policy_checkpoint(Path(result.best_checkpoint), reloaded)
        reloaded.eval()
        with torch.no_grad(): after = reloaded(**inputs)
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
        self.assertEqual(payload["selection_split"], "validation")

    def test_035_resume_continues_epoch_and_optimizer_state(self):
        _, config, model, train_loader, validation_loader, trainer = self._tiny_training_components("resume", max_epochs=2, patience=3)
        original = trainer._train_epoch; calls = {"count": 0}
        def interrupt_after_one(loader):
            if calls["count"]:
                raise RuntimeError("simulated interruption")
            calls["count"] += 1
            return original(loader)
        with patch.object(trainer, "_train_epoch", side_effect=interrupt_after_one):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                trainer.fit(train_loader, validation_loader)
        interrupted = _torch_load(trainer.last_path)
        self.assertEqual(interrupted["epoch"], 0)
        resumed_model = self._tiny_model()
        resumed_train = make_policy_balanced_loader(_TinySequenceDataset(), config)
        resumed_validation = DataLoader(_TinySequenceDataset(), batch_size=8)
        resumed = Round5Trainer(resumed_model, config, trainer.output_dir, provenance=trainer.provenance)
        result = resumed.fit(resumed_train, resumed_validation, resume_from=trainer.last_path)
        final = _torch_load(Path(result.last_checkpoint))
        self.assertEqual(final["epoch"], 1)
        self.assertEqual([row["epoch"] for row in final["history"]], [0, 1])
        steps = [float(state["step"]) for state in final["optimizer_state_dict"]["state"].values() if "step" in state]
        self.assertTrue(steps and min(steps) >= 2.0)

    def test_036_best_checkpoint_is_selected_only_by_validation_metric(self):
        _, _, _, train_loader, validation_loader, trainer = self._tiny_training_components("validation_selection", max_epochs=3, patience=3)
        validation_values = iter([3.0, 1.0, 2.0])
        def validation_only(_loader):
            value = next(validation_values)
            return {"normalized_action_mse": value / 10.0, "raw_action_rmse_m": value, "raw_action_mae_l2_m": value, "valid_timestep_count": 8.0}
        with patch.object(trainer, "_train_epoch", return_value={"normalized_action_mse": 0.5}), patch.object(trainer, "evaluate_loader", side_effect=validation_only):
            result = trainer.fit(train_loader, validation_loader)
        best = _torch_load(Path(result.best_checkpoint))
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["best_validation_raw_action_rmse_m"], 1.0)
        self.assertFalse(best["test_used_for_selection"])

    def test_037_test_data_cannot_participate_in_context_selection(self):
        parameters = inspect.signature(Round5Trainer.fit).parameters
        self.assertNotIn("test_loader", parameters)
        config = json.loads((REPOSITORY_ROOT / "round_05" / "configs" / "causal_transformer_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(config["training"]["test_used_for_selection"])
        self.assertEqual(config["context_lengths"], [1, 10])

    def test_038_replay_mode_never_advances_an_environment(self):
        records = [
            {"episode_id": "e0", "t": index, "policy_id": "max_queue", "num_buoys": 3, "state": np.asarray([index]), "goal": np.asarray([0.0]), "node_mask": np.ones(1), "remaining_time": 1.0, "true_action_normalized": np.zeros(2)}
            for index in range(3)
        ]
        seen_tokens = []
        result = run_recorded_replay(
            records,
            waypoint_callback=lambda state, goal, mask, row: np.asarray([float(row["t"]), 0.0]),
            action_callback=lambda history: (seen_tokens.append(history) or np.zeros(2)),
            max_step_distance_m=10.0,
            context_length=2,
            selected_episode_ids=["e0"],
        )
        self.assertFalse(result["environment_advanced"])
        self.assertFalse(result["formal_closed_loop_evaluation"])
        self.assertTrue(all("true_action_normalized" not in token for history in seen_tokens for token in history))

    def test_039_interface_smoke_recomputes_waypoint_each_step(self):
        environment = _SmokeEnvironment(); calls = []
        result = run_interface_smoke(
            environment,
            state_callback=lambda observation, info, step: np.asarray(observation),
            waypoint_callback=lambda state, info, step: (calls.append(step) or np.asarray([step + 0.25, -step])),
            action_callback=lambda history, state, waypoint, step: np.asarray([0.2, -0.1]),
            max_step_distance_m=10.0, steps=4, context_length=10,
        )
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertTrue(result["waypoint_recomputed_each_step"])
        self.assertEqual(result["waypoint_callback_count"], result["steps_executed"])

    def test_040_interface_smoke_does_not_precompute_a_future_trajectory(self):
        environment = _SmokeEnvironment(); history_lengths = []
        result = run_interface_smoke(
            environment,
            state_callback=lambda observation, info, step: np.asarray(observation),
            waypoint_callback=lambda state, info, step: np.asarray([step, step]),
            action_callback=lambda history, state, waypoint, step: (history_lengths.append(len(history)) or np.zeros(2)),
            max_step_distance_m=10.0, steps=3, context_length=10,
        )
        self.assertEqual(history_lengths, [1, 2, 3])
        self.assertEqual(result["steps_executed"], 3)
        self.assertFalse(result["closed_loop_evaluation_performed"])
        with self.assertRaises(ValueError):
            run_interface_smoke(environment, lambda *args: np.zeros(1), lambda *args: np.zeros(2), lambda *args: np.zeros(2), 10.0, steps=6)

    def test_041_all_frozen_and_policy_predictions_are_finite(self):
        for split_name in ("train", "validation", "test"):
            self.assertTrue(np.all(np.isfinite(self.cache.split(split_name).predicted_waypoints)))
        sequence = self.predicted_h10["validation"]
        model = CausalTrajectoryPolicy(CausalPolicyConfig(
            variant="wt_transformer", state_dim=sequence.state_dim,
            node_mask_dim=sequence.node_mask_dim, condition_dim=sequence.condition_dim,
            action_dim=sequence.action_dim, context_length=10, n_embd=8,
            n_layer=1, n_head=2, dropout=0.0,
        )).eval()
        batch = {name: torch.as_tensor(value[:4]) for name, value in sequence.as_mapping().items() if name not in {"actions", "loss_mask"}}
        with torch.no_grad(): prediction = model(**batch)
        self.assertTrue(torch.isfinite(prediction).all().item())

    def test_042_every_action_is_legal_or_radially_projectable(self):
        raw = np.asarray([[1.0, 1.0], [-1.0, 1.0], [0.2, -0.3], [0.0, 0.0]])
        projection = radial_action_projection(raw, self.dataset.maximum_step_distance_m)
        self.assertTrue(np.all(projection.raw_valid_mask | projection.activation_mask))
        self.assertTrue(np.all(projection.projected_valid_mask))

    def test_043_training_status_fields_are_logically_consistent(self):
        manifest = json.loads((REPOSITORY_ROOT / "round_manifest.json").read_text(encoding="utf-8"))
        flags = manifest["status_flags"]
        required = {"training_performed", "waypoint_training_performed", "waypoint_checkpoint_frozen", "transformer_training_performed", "policy_training_performed", "closed_loop_interface_smoke_performed", "closed_loop_evaluation_performed", "ppo_training_performed", "hgnn_training_performed"}
        self.assertTrue(required <= set(flags))
        self.assertTrue(all(isinstance(flags[name], bool) for name in required))
        self.assertFalse(flags["waypoint_training_performed"])
        self.assertTrue(flags["waypoint_checkpoint_frozen"])
        self.assertFalse(flags["ppo_training_performed"])
        self.assertFalse(flags["hgnn_training_performed"])
        if flags["training_performed"]:
            self.assertTrue(flags["transformer_training_performed"] and flags["policy_training_performed"])
        # Round-6 is allowed to record a frozen-model closed-loop evaluation in
        # the root manifest.  Round-5's own immutable lock remains the source
        # of truth for the historical assertion that it did not perform one.
        round5_lock = json.loads((REPOSITORY_ROOT / "round_05" / "ROUND_LOCK.json").read_text(encoding="utf-8"))
        self.assertFalse(round5_lock["training_scope"]["closed_loop_evaluation_performed"])

    def test_044_no_ppo_hgnn_or_new_waypoint_checkpoint_is_created(self):
        manifest = json.loads((REPOSITORY_ROOT / "round_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["training_allowed"]["reinforcement_learning"])
        self.assertEqual(_snapshot(SELECTED_CHECKPOINT.parent), self.round4_checkpoint_before)
        self.assertEqual(_sha256(SELECTED_CHECKPOINT), SELECTED_WAYPOINT_CHECKPOINT_SHA256)
        forbidden = {"ppo", "hgnn"}
        names = {type(module).__name__.lower() for module in self._tiny_model().modules()}
        self.assertFalse(any(any(term in name for term in forbidden) for name in names))

    def test_045_compileall_passes_for_round5_and_its_test(self):
        previous_prefix = getattr(__import__("sys"), "pycache_prefix", None)
        __import__("sys").pycache_prefix = str(self.temporary_root / "pycache")
        try:
            self.assertTrue(compileall.compile_dir(str(REPOSITORY_ROOT / "src" / "wt" / "round5"), quiet=1))
            self.assertTrue(compileall.compile_file(str(Path(__file__).resolve()), quiet=1))
        finally:
            __import__("sys").pycache_prefix = previous_prefix


if __name__ == "__main__":
    unittest.main(verbosity=2)
