"""Round-4 spatial Waypoint data, model, training, and evaluation tests.

The locked Round-3 canonical dataset is loaded once for the whole class.  All
write-producing trainer tests use tiny synthetic arrays under a temporary
directory; no test writes to Round-3.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from wt.maritime.dataset_v1 import load_dataset_v1
from wt.round4.round4_data import (
    CANONICAL_GOAL_SOURCE,
    GOAL_CONDITIONED_SOURCE_FIELDS,
    PREDICTOR_LEAKAGE_DENYLIST,
    STATE_ONLY_SOURCE_FIELDS,
    assert_predictor_fields_safe,
    load_round4_spatial_dataset,
)
from wt.round4.round4_evaluation import (
    assign_episode_length_bands,
    compute_spatial_waypoint_metrics,
    constant_velocity_baseline,
    evaluate_spatial_waypoint_predictions,
    fit_train_mean_baseline,
    predict_train_mean_baseline,
    radial_project_unit_disk,
    summarize_spatial_waypoint_metrics,
    zero_baseline,
)
from wt.round4.round4_runner import _run_manifest
from wt.round4.round4_training import (
    SpatialTrainingConfig,
    load_spatial_waypoint_checkpoint,
    predict_spatial_waypoints,
    train_spatial_waypoint,
)
from wt.round4.spatial_waypoint import (
    SpatialWaypointMLP,
    rebuild_spatial_waypoint_model,
    set_deterministic_seed,
    spatial_input_dim,
    spatial_waypoint_loss,
)
from wt.maritime.waypoint_samples import load_waypoint_samples


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = (
    REPOSITORY_ROOT
    / "round_03"
    / "outputs"
    / "round03_pilot_canonical_20260730_230000"
)
ROUND3_LOCK = REPOSITORY_ROOT / "round_03" / "ROUND_LOCK.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_snapshot(root: Path):
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class _TinyAdam:
    """Minimal optimizer double for fast trainer state-machine tests."""

    def __init__(self, parameters, lr, weight_decay=0.0):
        self.parameters = list(parameters)
        self.learning_rate = float(lr)
        self.weight_decay = float(weight_decay)
        self.param_groups = [{"lr": self.learning_rate}]
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if self.weight_decay:
                    update = update + self.weight_decay * parameter
                parameter.add_(update, alpha=-self.learning_rate)

    def state_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, state):
        self.learning_rate = float(state["learning_rate"])
        self.weight_decay = float(state["weight_decay"])
        self.param_groups[0]["lr"] = self.learning_rate


def _train_tiny(**kwargs):
    with patch(
        "wt.round4.round4_training.torch.optim.Adam", _TinyAdam
    ):
        return train_spatial_waypoint(**kwargs)


class TestRound04SpatialWaypoint(unittest.TestCase):
    """Numbered acceptance tests matching the Round-4 audit checklist."""

    @classmethod
    def setUpClass(cls):
        if not CANONICAL_ROOT.is_dir():
            raise unittest.SkipTest("Round-3 canonical pilot is unavailable")
        cls.source_snapshot_before = _file_snapshot(CANONICAL_ROOT)
        cls.lock = json.loads(ROUND3_LOCK.read_text(encoding="utf-8"))
        cls.dataset = load_round4_spatial_dataset(
            CANONICAL_ROOT,
            horizon=5,
            goal_source=CANONICAL_GOAL_SOURCE,
        )
        cls.episodes = load_dataset_v1(
            CANONICAL_ROOT, verify_checksums=True
        )
        cls.episodes_by_id = {
            episode.episode_id: episode for episode in cls.episodes
        }
        cls.waypoint_batches = load_waypoint_samples(
            CANONICAL_ROOT, verify_checksums=True
        )
        cls.source_snapshot_after = _file_snapshot(CANONICAL_ROOT)
        cls.waypoint_schema = json.loads(
            (CANONICAL_ROOT / "waypoint_sample_schema.json").read_text(
                encoding="utf-8"
            )
        )

        temporary_parent = Path(
            os.environ.get(
                "ROUND4_TEST_TMPDIR",
                str(REPOSITORY_ROOT.parent / "decision-transformer-master"),
            )
        )
        cls.temporary_root = temporary_parent / "_t{}".format(os.getpid())
        cls.temporary_root.mkdir(parents=False, exist_ok=False)
        rng = np.random.default_rng(20260802)
        cls.train_x = rng.normal(size=(24, 7)).astype(np.float32)
        cls.validation_x = rng.normal(size=(10, 7)).astype(np.float32)
        weights = np.asarray(
            [
                [0.08, -0.03],
                [0.02, 0.04],
                [-0.05, 0.01],
                [0.03, 0.02],
                [0.01, -0.02],
                [-0.02, 0.05],
                [0.04, 0.01],
            ],
            dtype=np.float32,
        )
        cls.train_y = cls.train_x @ weights
        cls.validation_y = cls.validation_x @ weights
        cls.training_config = SpatialTrainingConfig(
            hidden_dim=8,
            num_hidden_layers=1,
            dropout=0.0,
            learning_rate=1.0e-3,
            batch_size=8,
            maximum_epochs=2,
            early_stopping_patience=0,
            gradient_clipping=1.0,
            device="cpu",
            deterministic=True,
            num_workers=0,
        )
        cls.training_dir = cls.temporary_root / "ordinary"
        cls.training_result = _train_tiny(
            train_inputs=cls.train_x,
            train_targets_normalized=cls.train_y,
            validation_inputs=cls.validation_x,
            validation_targets_normalized=cls.validation_y,
            horizon=5,
            maximum_step_distance_m=10.0,
            seed=17,
            model_kind="state_only",
            output_dir=cls.training_dir,
            config=cls.training_config,
            provenance={"test_fixture": "tiny_synthetic_train_only"},
            resume=False,
        )
        cls.reloaded_model, cls.checkpoint_payload = (
            load_spatial_waypoint_checkpoint(
                cls.training_result.best_checkpoint,
                device="cpu",
                expected_horizon=5,
                expected_model_kind="state_only",
                expected_input_dim=7,
            )
        )
        cls.resumed_result = _train_tiny(
            train_inputs=cls.train_x,
            train_targets_normalized=cls.train_y,
            validation_inputs=cls.validation_x,
            validation_targets_normalized=cls.validation_y,
            horizon=5,
            maximum_step_distance_m=10.0,
            seed=17,
            model_kind="state_only",
            output_dir=cls.training_dir,
            config=cls.training_config,
            provenance={"test_fixture": "tiny_synthetic_train_only"},
            resume=True,
        )

        early_config = SpatialTrainingConfig(
            hidden_dim=4,
            num_hidden_layers=1,
            dropout=0.0,
            learning_rate=1.0e-30,
            batch_size=8,
            maximum_epochs=4,
            early_stopping_patience=1,
            gradient_clipping=1.0,
            device="cpu",
            deterministic=True,
            num_workers=0,
        )
        zero_train = np.zeros((16, 4), dtype=np.float32)
        zero_validation = np.zeros((8, 4), dtype=np.float32)
        cls.early_result = _train_tiny(
            train_inputs=zero_train,
            train_targets_normalized=np.zeros((16, 2), dtype=np.float32),
            validation_inputs=zero_validation,
            validation_targets_normalized=np.zeros(
                (8, 2), dtype=np.float32
            ),
            horizon=1,
            maximum_step_distance_m=10.0,
            seed=3,
            model_kind="state_only",
            output_dir=cls.temporary_root / "early_stop",
            config=early_config,
            provenance={"test_fixture": "early_stop"},
            resume=False,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.temporary_root.is_dir():
            shutil.rmtree(str(cls.temporary_root))

    def test_001_round3_lock_is_completed(self):
        self.assertEqual(self.lock["round"], "round_03")
        self.assertEqual(self.lock["status"], "completed")
        self.assertFalse(self.lock["training_performed"])

    def test_002_round3_locked_pilot_hashes_match(self):
        prefix = "round_03/outputs/{}/".format(
            CANONICAL_ROOT.name
        )
        entries = {
            path: digest
            for path, digest in self.lock[
                "canonical_output_sha256"
            ].items()
            if path.startswith(prefix)
        }
        self.assertGreaterEqual(len(entries), 17)
        for relative, expected in entries.items():
            self.assertEqual(
                _sha256(REPOSITORY_ROOT / relative).lower(),
                expected.lower(),
                relative,
            )

    def test_003_round3_source_remains_byte_layout_read_only(self):
        self.assertEqual(
            self.source_snapshot_before, self.source_snapshot_after
        )

    def test_004_adapter_exposes_verified_source_hashes(self):
        hashes = self.dataset.source_hashes
        self.assertGreaterEqual(len(hashes), 178)
        self.assertTrue(
            any(name.endswith("dataset_manifest.json") for name in hashes)
        )
        self.assertTrue(
            any(name.endswith("ROUND_LOCK.json") for name in hashes)
        )

    def test_005_schema_and_provenance_versions_are_exposed(self):
        self.assertEqual(
            self.dataset.schema_version,
            "round04_spatial_waypoint_data_v1",
        )
        self.assertEqual(
            self.dataset.source_run_id,
            "round03_pilot_canonical_20260730_230000",
        )
        self.assertTrue(self.dataset.provenance["source_read_only"])
        self.assertTrue(
            self.dataset.provenance["round3_lock_verified"]
        )

    def test_006_canonical_schema_drives_input_dimensions(self):
        dimensions = self.waypoint_schema["dimensions"]
        self.assertEqual(self.dataset.state_dim, dimensions["state_dim"])
        self.assertEqual(self.dataset.goal_dim, dimensions["waypoint_dim"])
        self.assertEqual(
            self.dataset.max_num_buoys,
            dimensions["max_num_buoys"],
        )
        self.assertEqual(self.dataset.state_only_input_dim, 67)
        self.assertEqual(self.dataset.goal_conditioned_input_dim, 80)

    def test_007_spatial_input_dim_is_dynamic_not_hard_coded(self):
        schema = {
            "dimensions": {
                "state_dim": 7,
                "waypoint_dim": 3,
                "max_num_buoys": 2,
            }
        }
        self.assertEqual(spatial_input_dim(schema), 9)
        self.assertEqual(
            spatial_input_dim(schema, goal_conditioned=True), 12
        )

    def test_008_locked_episode_split_counts_are_preserved(self):
        expected = {"train": 48, "validation": 16, "test": 16}
        observed = {
            name: len(set(self.dataset.split(name).episode_id.tolist()))
            for name in expected
        }
        self.assertEqual(observed, expected)

    def test_009_episode_sets_are_disjoint_across_splits(self):
        sets = [
            set(self.dataset.split(name).episode_id.tolist())
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 80)

    def test_010_fixed_horizon_filter_has_locked_sample_counts(self):
        self.assertEqual(self.dataset.horizon, 5)
        self.assertEqual(len(self.dataset.train), 3718)
        self.assertEqual(len(self.dataset.validation), 1215)
        self.assertEqual(len(self.dataset.test), 1351)

    def test_011_predictor_and_target_shapes_are_exact(self):
        for split in self.dataset.splits.values():
            count = len(split)
            self.assertEqual(
                split.goal_conditioned_inputs.shape, (count, 80)
            )
            self.assertEqual(split.state_only_inputs.shape, (count, 67))
            self.assertEqual(split.normalized_delta_xy.shape, (count, 2))

    def test_012_input_concatenation_order_is_exact(self):
        split = self.dataset.train
        np.testing.assert_array_equal(
            split.state_only_inputs[:, :62], split.masked_state_t
        )
        np.testing.assert_array_equal(
            split.state_only_inputs[:, 62:],
            split.valid_node_mask_t.astype(np.float32),
        )
        np.testing.assert_array_equal(
            split.goal_conditioned_inputs[:, :62], split.masked_state_t
        )
        np.testing.assert_array_equal(
            split.goal_conditioned_inputs[:, 62:75], split.masked_goal
        )
        np.testing.assert_array_equal(
            split.goal_conditioned_inputs[:, 75:],
            split.valid_node_mask_t.astype(np.float32),
        )

    def test_013_state_and_goal_padding_is_zero_after_mask(self):
        for split in self.dataset.splits.values():
            self.assertEqual(
                np.count_nonzero(
                    split.masked_state_t[~split.state_mask_t]
                ),
                0,
            )
            self.assertEqual(
                np.count_nonzero(split.masked_goal[~split.goal_mask]), 0
            )

    def test_014_raw_delta_is_reconstructed_from_physical_positions(self):
        split = self.dataset.validation
        indices = np.linspace(0, len(split) - 1, 17, dtype=int)
        for index in indices:
            episode = self.episodes_by_id[str(split.episode_id[index])]
            start = int(split.t[index])
            target_index = int(split.target_transition_index[index])
            np.testing.assert_allclose(
                split.q_current_m[index],
                episode.uav_positions_m[start],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                split.q_future_m[index],
                episode.next_uav_positions_m[target_index],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                split.raw_delta_xy_m[index],
                split.q_future_m[index] - split.q_current_m[index],
                rtol=0.0,
                atol=0.0,
            )

    def test_015_temporal_target_indices_follow_t_plus_h(self):
        for split in self.dataset.splits.values():
            np.testing.assert_array_equal(
                split.target_t, split.t + self.dataset.horizon
            )
            np.testing.assert_array_equal(
                split.target_transition_index,
                split.t + self.dataset.horizon - 1,
            )

    def test_016_targets_do_not_cross_episode_boundaries(self):
        for split in self.dataset.splits.values():
            self.assertTrue(
                np.all(split.target_t <= split.episode_length)
            )
            self.assertTrue(np.all(split.t >= 0))

    def test_017_endpoint_done_is_allowed_as_the_horizon_endpoint(self):
        endpoint_count = sum(
            int(np.count_nonzero(split.endpoint_terminated))
            + int(np.count_nonzero(split.endpoint_truncated))
            for split in self.dataset.splits.values()
        )
        self.assertGreater(endpoint_count, 0)
        for split in self.dataset.splits.values():
            done = split.endpoint_terminated | split.endpoint_truncated
            self.assertTrue(
                np.all(split.target_t[done] == split.episode_length[done])
            )

    def test_018_internal_done_flags_are_excluded(self):
        for name, full_batch in self.waypoint_batches.items():
            selected = full_batch.k == self.dataset.horizon
            self.assertFalse(
                np.any(
                    full_batch.terminated_between_t_and_t_plus_k[
                        selected
                    ]
                )
            )
            self.assertFalse(
                np.any(
                    full_batch.truncated_between_t_and_t_plus_k[selected]
                )
            )

    def test_019_spatial_targets_obey_horizon_reachability(self):
        scale = self.dataset.horizon_normalization_distance_m
        for split in self.dataset.splits.values():
            distances = np.linalg.norm(split.raw_delta_xy_m, axis=1)
            self.assertLessEqual(float(np.max(distances)), scale + 1.0e-6)
            self.assertLessEqual(
                float(np.max(np.linalg.norm(split.normalized_delta_xy, axis=1))),
                1.0 + 1.0e-6,
            )

    def test_020_normalization_uses_h_times_max_speed_times_dt(self):
        self.assertEqual(self.dataset.max_step_distance_m, 10.0)
        self.assertEqual(
            self.dataset.horizon_normalization_distance_m, 50.0
        )
        for split in self.dataset.splits.values():
            reconstructed = (
                split.normalized_delta_xy.astype(np.float64) * 50.0
            )
            np.testing.assert_allclose(
                reconstructed,
                split.raw_delta_xy_m,
                rtol=0.0,
                atol=1.0e-5,
            )

    def test_021_current_velocity_is_causal_position_history(self):
        split = self.dataset.test
        indices = np.linspace(0, len(split) - 1, 19, dtype=int)
        for index in indices:
            episode = self.episodes_by_id[str(split.episode_id[index])]
            start = int(split.t[index])
            expected = np.zeros(2, dtype=np.float64)
            if start > 0:
                expected = (
                    episode.uav_positions_m[start]
                    - episode.uav_positions_m[start - 1]
                ) / self.dataset.delta_t_seconds
            np.testing.assert_allclose(
                split.current_velocity_mps[index],
                expected,
                rtol=1.0e-5,
                atol=2.0e-6,
            )

    def test_022_canonical_goal_channel_is_explicit_hindsight(self):
        self.assertEqual(
            self.dataset.goal_source, "achieved_terminal_goal"
        )
        audit = self.dataset.goal_source_audits[
            "achieved_terminal_goal"
        ]
        self.assertTrue(audit.is_hindsight_goal)
        self.assertIn("Hindsight", audit.limitation)
        for split in self.dataset.splits.values():
            self.assertNotEqual(
                split.goal_conditioned_inputs.shape[1],
                split.state_only_inputs.shape[1],
            )

    def test_023_target_fields_are_on_the_leakage_denylist(self):
        required = {
            "waypoint_target_t_plus_k",
            "valid_node_mask_t_plus_k",
            "raw_delta_xy_m",
            "q_future_m",
            "normalized_delta_xy",
        }
        self.assertTrue(required.issubset(PREDICTOR_LEAKAGE_DENYLIST))
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_predictor_fields_safe(
                ("state_t", "waypoint_target_t_plus_k"),
                goal_conditioned=False,
            )

    def test_024_evaluation_metadata_is_denied_as_predictor_input(self):
        for field in (
            "episode_id",
            "policy_id",
            "quality_class",
            "num_buoys",
            "endpoint_terminated",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    assert_predictor_fields_safe(
                        STATE_ONLY_SOURCE_FIELDS + (field,),
                        goal_conditioned=False,
                    )

    def test_025_only_exact_allowlisted_recipes_are_accepted(self):
        self.assertEqual(
            assert_predictor_fields_safe(
                STATE_ONLY_SOURCE_FIELDS, goal_conditioned=False
            ),
            STATE_ONLY_SOURCE_FIELDS,
        )
        self.assertEqual(
            assert_predictor_fields_safe(
                GOAL_CONDITIONED_SOURCE_FIELDS,
                goal_conditioned=True,
            ),
            GOAL_CONDITIONED_SOURCE_FIELDS,
        )
        with self.assertRaises(ValueError):
            assert_predictor_fields_safe(
                ("state_t",), goal_conditioned=False
            )

    def test_026_desired_goal_audit_has_only_observed_patterns(self):
        audit = self.dataset.goal_source_audits["desired_goal"]
        self.assertEqual(audit.unique_goal_and_mask_pairs, 2)
        self.assertEqual(audit.unique_episode_goal_and_mask_pairs, 2)
        self.assertEqual(audit.position_mask_true_rows, 0)
        self.assertFalse(audit.primary_experiment_eligible)

    def test_027_desired_goal_variation_is_not_fabricated(self):
        audit = self.dataset.goal_source_audits["desired_goal"]
        self.assertEqual(
            audit.changed_from_canonical_rows, audit.sample_count
        )
        self.assertIn("no synthetic variation", audit.limitation)
        self.assertEqual(
            audit.materialization,
            "in_memory_from_authoritative_episode_desired_goal",
        )

    def test_028_zero_baseline_is_target_independent(self):
        prediction = zero_baseline(7)
        self.assertEqual(prediction.shape, (7, 2))
        self.assertEqual(prediction.dtype, np.float64)
        self.assertEqual(np.count_nonzero(prediction), 0)

    def test_029_train_mean_baseline_fits_only_train_values(self):
        targets = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        baseline = fit_train_mean_baseline(
            targets,
            train_metadata={"split": np.asarray(["train", "train"])},
        )
        np.testing.assert_allclose(
            baseline["mean_delta_normalized"], [2.0, 3.0]
        )
        prediction = predict_train_mean_baseline(
            baseline, num_samples=3
        )
        np.testing.assert_allclose(
            prediction, np.asarray([[2.0, 3.0]] * 3)
        )

    def test_030_train_mean_rejects_validation_or_test_rows(self):
        targets = np.zeros((2, 2), dtype=np.float64)
        for label in ("validation", "test"):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "train rows only"):
                    fit_train_mean_baseline(
                        targets,
                        train_metadata={
                            "split": np.asarray(["train", label])
                        },
                    )

    def test_031_train_mean_supports_independent_horizons(self):
        targets = np.asarray(
            [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]
        )
        horizons = np.asarray([1, 1, 5, 5])
        baseline = fit_train_mean_baseline(
            targets,
            train_metadata={"split": np.full(4, "train")},
            horizon_steps=horizons,
        )
        prediction = predict_train_mean_baseline(
            baseline, horizon_steps=np.asarray([5, 1])
        )
        np.testing.assert_allclose(prediction, [[0.0, 3.0], [2.0, 0.0]])

    def test_032_constant_velocity_uses_current_velocity_only(self):
        result = constant_velocity_baseline(
            np.asarray([[50.0, 50.0]]),
            np.asarray([[2.0, -1.0]]),
            horizon_steps=5,
            delta_t_s=1.0,
            maximum_step_distance_m=10.0,
            area_lower_bound_m=[0.0, 0.0],
            area_upper_bound_m=[200.0, 200.0],
        )
        np.testing.assert_allclose(result["predicted_delta_m"], [[10.0, -5.0]])
        np.testing.assert_allclose(
            result["predictions_normalized"], [[0.2, -0.1]]
        )
        self.assertFalse(result["boundary_clipping_activated"][0])

    def test_033_constant_velocity_clips_physical_endpoint(self):
        result = constant_velocity_baseline(
            np.asarray([[198.0, 1.0]]),
            np.asarray([[10.0, -10.0]]),
            horizon_steps=5,
            delta_t_s=1.0,
            maximum_step_distance_m=10.0,
            area_lower_bound_m=[0.0, 0.0],
            area_upper_bound_m=[200.0, 200.0],
        )
        np.testing.assert_allclose(
            result["predicted_endpoint_m"], [[200.0, 0.0]]
        )
        self.assertTrue(result["boundary_clipping_activated"][0])

    def test_034_schema_constructed_models_have_expected_input_dims(self):
        state_model = SpatialWaypointMLP.from_schema(
            self.waypoint_schema,
            goal_conditioned=False,
            hidden_dim=8,
            num_hidden_layers=1,
        )
        goal_model = SpatialWaypointMLP.from_schema(
            self.waypoint_schema,
            goal_conditioned=True,
            hidden_dim=8,
            num_hidden_layers=1,
        )
        self.assertEqual(state_model.input_dim, 67)
        self.assertEqual(goal_model.input_dim, 80)
        self.assertEqual(state_model.output_dim, 2)
        self.assertEqual(goal_model.output_dim, 2)

    def test_035_model_forward_shape_is_batch_by_two(self):
        model = SpatialWaypointMLP(
            input_dim=7, hidden_dim=8, num_hidden_layers=1
        )
        output = model(torch.zeros((6, 7), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (6, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_036_loss_is_mean_of_squared_vector_norm(self):
        prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        observed = spatial_waypoint_loss(prediction, target)
        expected = ((1.0**2 + 2.0**2) + (2.0**2 + 3.0**2)) / 2.0
        self.assertAlmostEqual(float(observed), expected)

    def test_037_tiny_forward_loss_backward_has_finite_gradients(self):
        model = SpatialWaypointMLP(
            input_dim=7, hidden_dim=8, num_hidden_layers=1
        )
        prediction = model(torch.from_numpy(self.train_x[:5]))
        loss = spatial_waypoint_loss(
            prediction, torch.from_numpy(self.train_y[:5])
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_038_seed_reproduces_first_batch_and_distinguishes_initialization(self):
        def one_step(seed):
            set_deterministic_seed(seed)
            model = SpatialWaypointMLP(
                input_dim=7, hidden_dim=8, num_hidden_layers=1
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
            values = torch.from_numpy(self.train_x[:5])
            targets = torch.from_numpy(self.train_y[:5])
            loss = spatial_waypoint_loss(model(values), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return (
                float(loss.detach()),
                [parameter.detach().clone() for parameter in model.parameters()],
            )

        first_loss, first_parameters = one_step(41)
        second_loss, second_parameters = one_step(41)
        self.assertEqual(first_loss, second_loss)
        for left, right in zip(first_parameters, second_parameters):
            self.assertTrue(torch.equal(left, right))

        _, different_parameters = one_step(42)
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(first_parameters, different_parameters)
            )
        )

    def test_039_in_memory_checkpoint_roundtrip_preserves_predictions(self):
        set_deterministic_seed(12)
        model = SpatialWaypointMLP(
            input_dim=7, hidden_dim=8, num_hidden_layers=1
        )
        model.eval()
        payload = model.checkpoint_payload(metadata={"horizon": 5})
        rebuilt = rebuild_spatial_waypoint_model(
            payload, expected_input_dim=7
        )
        rebuilt.eval()
        values = torch.from_numpy(self.validation_x)
        with torch.no_grad():
            np.testing.assert_array_equal(
                model(values).numpy(), rebuilt(values).numpy()
            )

    def test_040_tiny_training_writes_required_local_artifacts(self):
        result = self.training_result
        for path in (
            result.best_checkpoint,
            result.last_checkpoint,
            result.history_file,
            result.config_file,
            self.training_dir / "training_result.json",
        ):
            self.assertTrue(Path(path).is_file(), path)
        self.assertEqual(result.output_dim, 2)
        self.assertEqual(result.model_kind, "state_only")

    def test_041_checkpoint_reload_preserves_saved_validation_metric(self):
        prediction = predict_spatial_waypoints(
            self.reloaded_model, self.validation_x, device="cpu"
        )
        normalized_mse = float(
            np.mean(
                np.sum(
                    np.square(
                        prediction.astype(np.float64)
                        - self.validation_y.astype(np.float64)
                    ),
                    axis=1,
                )
            )
        )
        self.assertAlmostEqual(
            normalized_mse,
            self.training_result.best_validation_normalized_mse,
            places=6,
        )

    def test_042_resume_restores_last_checkpoint_state(self):
        self.assertTrue(self.resumed_result.resumed)
        self.assertEqual(
            self.resumed_result.completed_epochs,
            self.training_result.completed_epochs,
        )
        self.assertEqual(self.checkpoint_payload["horizon"], 5)
        self.assertIn("optimizer_state_dict", self.checkpoint_payload)

    def test_043_early_stopping_uses_validation_without_full_epoch_budget(self):
        self.assertTrue(self.early_result.stopped_early)
        self.assertLess(self.early_result.completed_epochs, 4)
        self.assertGreaterEqual(self.early_result.best_epoch, 0)

    def test_044_trainer_interface_has_no_test_tensor_arguments(self):
        parameters = set(inspect.signature(train_spatial_waypoint).parameters)
        self.assertNotIn("test_inputs", parameters)
        self.assertNotIn("test_targets", parameters)
        self.assertNotIn("test_metadata", parameters)
        self.assertIn("validation_inputs", parameters)

    def test_045_resolved_training_config_records_test_data_unused(self):
        config = json.loads(
            Path(self.training_result.config_file).read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(config["test_data_used"])
        self.assertEqual(
            config["selection_rule"],
            "minimum validation future endpoint RMSE",
        )

    def test_046_radial_projection_is_evaluation_only_and_bounded(self):
        model = SpatialWaypointMLP(
            input_dim=2, hidden_dim=2, num_hidden_layers=1
        )
        self.assertEqual(
            model.architecture_metadata()["output_activation"], "identity"
        )
        values = np.asarray([[2.0, 0.0], [0.3, 0.4]])
        projected = radial_project_unit_disk(values)
        np.testing.assert_allclose(
            projected["predictions_normalized"],
            [[1.0, 0.0], [0.3, 0.4]],
        )
        self.assertEqual(
            projected["projection_activated"].tolist(), [True, False]
        )

    def test_047_perfect_predictions_have_zero_spatial_errors(self):
        target = np.asarray([[0.2, 0.1], [-0.3, 0.4]])
        positions = np.asarray([[10.0, 20.0], [30.0, 40.0]])
        metrics = compute_spatial_waypoint_metrics(
            target,
            target,
            horizon_steps=5,
            maximum_step_distance_m=10.0,
            current_positions_m=positions,
            target_deltas_m=target * 50.0,
            target_endpoints_m=positions + target * 50.0,
        )
        self.assertEqual(
            np.count_nonzero(metrics["normalized_squared_norm_error"]), 0
        )
        self.assertEqual(np.count_nonzero(metrics["endpoint_l2_error_m"]), 0)
        self.assertTrue(np.all(metrics["target_reachable"]))

    def test_048_near_zero_direction_is_marked_undefined_but_finite(self):
        metrics = compute_spatial_waypoint_metrics(
            np.asarray([[0.0, 0.0], [0.2, 0.0]]),
            np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            horizon_steps=1,
            maximum_step_distance_m=10.0,
            direction_near_zero_threshold=1.0e-3,
        )
        self.assertFalse(np.any(metrics["direction_valid"]))
        self.assertTrue(
            np.all(np.isfinite(metrics["direction_cosine_similarity"]))
        )
        self.assertTrue(np.all(np.isfinite(metrics["angular_error_deg"])))

    def test_049_inconsistent_raw_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            compute_spatial_waypoint_metrics(
                np.asarray([[0.0, 0.0]]),
                np.asarray([[0.1, 0.0]]),
                horizon_steps=1,
                maximum_step_distance_m=10.0,
                target_deltas_m=np.asarray([[9.0, 0.0]]),
            )

    def test_050_metrics_group_by_policy_and_buoy_scale(self):
        target = np.asarray([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        metrics = compute_spatial_waypoint_metrics(
            np.zeros_like(target),
            target,
            horizon_steps=1,
            maximum_step_distance_m=10.0,
        )
        rows = summarize_spatial_waypoint_metrics(
            metrics,
            metadata={
                "policy_id": np.asarray(["a", "a", "b"]),
                "num_buoys": np.asarray([3, 5, 3]),
            },
            group_fields=("policy_id", "num_buoys"),
        )
        self.assertEqual(len(rows), 1 + 2 + 2)
        self.assertEqual(rows[0]["group_kind"], "overall")

    def test_051_quality_class_aliases_quality_tier_for_grouping(self):
        target = np.asarray([[0.1, 0.0], [0.2, 0.0]])
        metrics = compute_spatial_waypoint_metrics(
            target,
            target,
            horizon_steps=1,
            maximum_step_distance_m=10.0,
        )
        rows = summarize_spatial_waypoint_metrics(
            metrics,
            metadata={
                "quality_class": np.asarray(["high", "mid"]),
                "q_current_m": np.zeros((2, 2)),
            },
            group_fields=("quality_tier",),
        )
        groups = {row["group_value"] for row in rows[1:]}
        self.assertEqual(groups, {"high", "mid"})

    def test_052_episode_length_bands_require_fixed_external_edges(self):
        bands = assign_episode_length_bands(
            np.asarray([10, 50, 100]),
            edges=np.asarray([40, 80]),
            labels=("short", "medium", "long"),
        )
        self.assertEqual(bands.tolist(), ["short", "medium", "long"])
        with self.assertRaises(ValueError):
            assign_episode_length_bands(
                [10, 20], edges=[20, 10]
            )

    def test_053_evaluation_reports_projection_activation_and_groups(self):
        result = evaluate_spatial_waypoint_predictions(
            np.asarray([[2.0, 0.0], [0.1, 0.0], [1.0 + 5.0e-10, 0.0]]),
            np.asarray([[0.5, 0.0], [0.1, 0.0], [1.0, 0.0]]),
            horizon_steps=1,
            maximum_step_distance_m=10.0,
            metadata={"policy_id": np.asarray(["a", "b", "b"])},
            group_fields=("policy_id",),
            apply_radial_projection=True,
            reachability_tolerance=1.0e-9,
        )
        self.assertEqual(
            set(result), {"unprojected", "projected", "projection"}
        )
        activation = result["projection"]["per_sample"][
            "projection_activated"
        ]
        self.assertEqual(activation.tolist(), [True, False, False])
        self.assertEqual(len(result["projected"]["groups"]), 3)

    def test_054_checkpoint_status_authorizes_only_waypoint_training(self):
        status = self.checkpoint_payload["status"]
        self.assertTrue(status["training_performed"])
        self.assertTrue(status["waypoint_training_performed"])
        self.assertFalse(status["transformer_training_performed"])
        self.assertFalse(status["policy_training_performed"])
        self.assertFalse(status["closed_loop_evaluation_performed"])

    def test_055_checkpoint_contains_no_transformer_or_policy_state(self):
        self.assertEqual(
            self.checkpoint_payload["model_class"], "SpatialWaypointMLP"
        )
        forbidden = [
            key
            for key in self.checkpoint_payload
            if "transformer" in key.lower() or "policy" in key.lower()
        ]
        self.assertEqual(forbidden, [])
        checkpoint_names = sorted(
            path.name
            for path in (self.training_dir / "checkpoints").iterdir()
        )
        self.assertEqual(checkpoint_names, ["best.pt", "last.pt"])

    def test_056_run_manifest_status_never_claims_policy_or_transformer(self):
        config_path = self.temporary_root / "manifest_config.json"
        config_path.write_text("{}\n", encoding="utf-8")
        in_progress = _run_manifest(
            run_id="unit_test",
            mode="smoke",
            config_path=config_path,
            source_dir=CANONICAL_ROOT,
            horizons=[5],
            seeds=[0],
            status="in_progress",
        )
        completed = _run_manifest(
            run_id="unit_test",
            mode="smoke",
            config_path=config_path,
            source_dir=CANONICAL_ROOT,
            horizons=[5],
            seeds=[0],
            status="completed",
        )
        self.assertFalse(in_progress["training_performed"])
        self.assertTrue(completed["waypoint_training_performed"])
        self.assertFalse(completed["transformer_training_performed"])
        self.assertFalse(completed["policy_training_performed"])
        self.assertFalse(completed["test_used_for_model_selection"])


if __name__ == "__main__":
    unittest.main()
