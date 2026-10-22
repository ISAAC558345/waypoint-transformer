"""Focused contracts for the in-progress Round-12 policy implementation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from wt.round12.contracts import (
    PREDICTOR_SHA256,
    assert_predictor_frozen,
    load_frozen_predictor,
    predicted_phi,
    sha256,
)
from wt.round12.data import ROUND12_EPISODE_HORIZON, Round12TransitionSplit
from wt.round12.formal import _paired_statistics, _parse_scenario
from wt.round12.policy_models import (
    ORACLE_TASK_WAYPOINT_TRANSFORMER,
    PREDICTED_TASK_WAYPOINT_TRANSFORMER,
    STATE_ONLY_TRANSFORMER,
    TASK_WAYPOINT_MLP,
    Round12PolicySchema,
    build_family,
    mask_phi,
    rollout_action,
)
from wt.round12.sampler import ScenarioBehaviorBalancedSampler
from wt.round12.sequence import build_causal_sequence_split, causal_attention_mask
from wt.round12.trainer import Round12PolicyTrainer, Round12TrainingConfig


ROOT = Path(__file__).resolve().parents[1]


def _transitions(split: str = "train") -> Round12TransitionSplit:
    # Two scenario/behaviour strata plus a synthetic terminal boundary inside
    # ep-a, which makes history-reset behaviour directly observable.
    count, state_dim, buoys = 8, 3, 2
    states = np.arange(count * state_dim, dtype=np.float32).reshape(count, state_dim)
    mask = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (count, 1))
    phi = np.zeros((count, 2 * buoys + 1), dtype=np.float32)
    phi[:, 0] = np.linspace(0.1, 0.8, count)
    phi[:, buoys] = np.linspace(0.2, 0.9, count)
    phi[:, -1] = np.linspace(-1.0, 1.0, count)
    episode = np.asarray(["ep-a", "ep-a", "ep-a", "ep-b", "ep-b", "ep-c", "ep-c", "ep-d"])
    timestep = np.asarray([0, 1, 2, 0, 1, 0, 1, 0], dtype=np.int64)
    scenario = np.asarray(["s0", "s0", "s0", "s1", "s1", "s0", "s0", "s1"])
    behaviour = np.asarray(["b0", "b0", "b0", "b1", "b1", "b1", "b1", "b0"])
    return Round12TransitionSplit(
        split_name=split,
        states=states,
        node_masks=mask,
        actions=np.tanh(states[:, :2] / 10.0).astype(np.float32),
        predicted_phi=phi,
        oracle_phi=phi + np.array([0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32),
        remaining_time=np.linspace(1.0, 0.5, count, dtype=np.float32)[:, None],
        episode_id=episode,
        scenario_id=scenario,
        behavior_generator=behaviour,
        quality=np.asarray(["q0"] * count),
        difficulty=np.asarray(["d0"] * count),
        num_buoys=np.ones(count, dtype=np.int64),
        timestep=timestep,
        terminated=np.asarray([False, True, False, False, False, False, False, False]),
        truncated=np.zeros(count, dtype=bool),
    )


class Round12Contracts(unittest.TestCase):
    def test_01_sequence_does_not_cross_episode_or_terminal(self) -> None:
        sequence = build_causal_sequence_split(_transitions())
        # ep-a/t=2 occurs after the explicit terminal at t=1, so it starts
        # cleanly rather than inheriting either pre-terminal action/state.
        self.assertEqual(int(sequence.attention_mask[2].sum()), 1)
        self.assertTrue(np.array_equal(sequence.states[2, -1], _transitions().states[2]))

    def test_02_sequence_is_split_local(self) -> None:
        train = build_causal_sequence_split(_transitions("train"))
        validation = build_causal_sequence_split(_transitions("validation"))
        self.assertTrue(np.all(train.metadata["split"] == "train"))
        self.assertTrue(np.all(validation.metadata["split"] == "validation"))

    def test_03_causal_mask_blocks_future(self) -> None:
        schema = Round12PolicySchema(state_dim=3, max_buoys=2)
        model = build_family(STATE_ONLY_TRANSFORMER, schema).eval()
        states = torch.zeros(1, 10, 3)
        masks = torch.tensor([[[1.0, 0.0]] * 10])
        remaining = torch.ones(1, 10, 1)
        attention = torch.ones(1, 10, dtype=torch.bool)
        changed = states.clone()
        changed[:, 7:] = 100.0
        with torch.no_grad():
            first = model(states, masks, remaining_time=remaining, attention_mask=attention)
            second = model(changed, masks, remaining_time=remaining, attention_mask=attention)
        self.assertTrue(torch.equal(first[:, :7], second[:, :7]))
        mask = causal_attention_mask(np.array([[False] * 3 + [True] * 7], dtype=bool))
        self.assertFalse(mask[0, 3, 4])
        self.assertTrue(mask[0, 9, 3])

    def test_04_padding_is_zero_and_not_in_action_loss(self) -> None:
        sequence = build_causal_sequence_split(_transitions())
        self.assertTrue(np.all(sequence.loss_mask <= sequence.attention_mask))
        self.assertTrue(np.all(sequence.loss_mask[:, -1]))
        self.assertTrue(np.all(sequence.states[~sequence.attention_mask] == 0.0))

    def test_05_policy_has_no_historical_action_input(self) -> None:
        schema = Round12PolicySchema(state_dim=3, max_buoys=2)
        model = build_family(PREDICTED_TASK_WAYPOINT_TRANSFORMER, schema).eval()
        self.assertFalse(model.uses_action_tokens)
        sequence = build_causal_sequence_split(_transitions())
        with torch.no_grad():
            predicted = model(
                torch.from_numpy(sequence.states[:1]),
                torch.from_numpy(sequence.node_masks[:1]),
                torch.from_numpy(sequence.predicted_phi[:1]),
                torch.from_numpy(sequence.remaining_time[:1]),
                torch.from_numpy(sequence.attention_mask[:1]),
            )
        self.assertEqual(tuple(predicted.shape), (1, 10, 2))

    def test_06_state_only_rejects_phi(self) -> None:
        schema = Round12PolicySchema(state_dim=3, max_buoys=2)
        model = build_family(STATE_ONLY_TRANSFORMER, schema)
        with self.assertRaises(ValueError):
            model(
                torch.zeros(1, 3), torch.ones(1, 2), torch.zeros(1, 5), torch.ones(1, 1)
            )

    def test_06b_deployable_input_uses_predicted_phi_only(self) -> None:
        batch = {
            "states": torch.zeros(1, 10, 3),
            "node_masks": torch.ones(1, 10, 2),
            "predicted_phi": torch.full((1, 10, 5), 2.0),
            "oracle_phi": torch.full((1, 10, 5), 9.0),
            "remaining_time": torch.ones(1, 10, 1),
            "attention_mask": torch.ones(1, 10, dtype=torch.bool),
        }
        trainer = object.__new__(Round12PolicyTrainer)
        trainer.family = PREDICTED_TASK_WAYPOINT_TRANSFORMER
        self.assertTrue(torch.equal(trainer._model_inputs(batch)["conditions"], batch["predicted_phi"]))
        trainer.family = STATE_ONLY_TRANSFORMER
        self.assertNotIn("conditions", trainer._model_inputs(batch))
        trainer.family = ORACLE_TASK_WAYPOINT_TRANSFORMER
        self.assertTrue(torch.equal(trainer._model_inputs(batch)["conditions"], batch["oracle_phi"]))

    def test_07_oracle_is_not_deployable(self) -> None:
        schema = Round12PolicySchema(state_dim=3, max_buoys=2)
        oracle = build_family(ORACLE_TASK_WAYPOINT_TRANSFORMER, schema)
        self.assertFalse(oracle.deployable)
        with self.assertRaises(PermissionError):
            rollout_action(
                ORACLE_TASK_WAYPOINT_TRANSFORMER,
                oracle,
                states=torch.zeros(1, 1, 3),
                node_masks=torch.ones(1, 1, 2),
                conditions=torch.zeros(1, 1, 5),
                remaining_time=torch.ones(1, 1, 1),
            )

    def test_08_mlp_is_non_temporal(self) -> None:
        schema = Round12PolicySchema(state_dim=3, max_buoys=2)
        mlp = build_family(TASK_WAYPOINT_MLP, schema).eval()
        with self.assertRaises(ValueError):
            mlp(torch.zeros(1, 10, 3), torch.ones(1, 10, 2), torch.zeros(1, 10, 5))
        with torch.no_grad():
            output = mlp(torch.zeros(1, 3), torch.ones(1, 2), torch.zeros(1, 5))
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_09_predictor_is_frozen_and_hash_pinned(self) -> None:
        path = ROOT / "round_11/checkpoints/H10/seed_2/best.pt"
        before = sha256(path)
        predictor = load_frozen_predictor(ROOT, 44, 5)
        output = predicted_phi(predictor, torch.zeros(2, 44), torch.tensor([[1, 1, 1, 0, 0]] * 2))
        assert_predictor_frozen(predictor)
        self.assertFalse(output.requires_grad)
        self.assertEqual(before, PREDICTOR_SHA256)
        self.assertEqual(sha256(path), before)

    def test_10_phi_padding_and_action_dimension(self) -> None:
        phi = torch.ones(1, 5)
        masked = mask_phi(phi, torch.tensor([[True, False]]), 2)
        self.assertEqual(float(masked[0, 1]), 0.0)
        self.assertEqual(float(masked[0, 3]), 0.0)
        self.assertEqual(_transitions().actions.shape[1], 2)

    def test_11_sampler_balances_and_audits_metadata(self) -> None:
        sampler = ScenarioBehaviorBalancedSampler(build_causal_sequence_split(_transitions()), seed=7)
        audit = sampler.audit(sampler.sample_indices(16))
        self.assertEqual(set(audit["sampled_shares"]), {"quality", "difficulty", "behavior_generator", "num_buoys"})
        self.assertAlmostEqual(sum(audit["scenario_share"].values()), 1.0)
        self.assertAlmostEqual(sum(audit["behavior_generator_share"].values()), 1.0)
        self.assertAlmostEqual(audit["scenario_share"]["s0"], 0.5)
        self.assertAlmostEqual(audit["behavior_generator_share"]["b0"], 0.5)

    def test_12_protected_accesses_remain_zero(self) -> None:
        fresh = json.loads((ROOT / "round_12/fresh_policy_dev_manifest.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "round_12/round_manifest.json").read_text(encoding="utf-8"))
        # The frozen protocol permits exactly one fresh-development evaluation
        # after V-A/V-B, never a second access or training use.
        self.assertIn(fresh["access_count"], (0, 1))
        if fresh["access_count"] == 1:
            self.assertFalse(fresh["selection_or_training_use"])
        self.assertFalse(manifest["old_development_test_accessed"])
        self.assertFalse(manifest["untouched_holdout_accessed"])
        # A later audit-only round may exist, but it cannot retrospectively
        # turn Round-12's protected data controls into an access pathway.
        round13_manifest = ROOT / "round_13/round_manifest.json"
        if round13_manifest.exists():
            round13 = json.loads(round13_manifest.read_text(encoding="utf-8"))
            self.assertFalse(round13["old_development_test_accessed"])
            self.assertFalse(round13["untouched_holdout_accessed"])
            self.assertFalse(round13["policy_training_performed"])

    def test_12b_formal_statistics_are_scenario_paired(self) -> None:
        rows = []
        for scenario in ("s0", "s1"):
            for seed in (0, 1, 2):
                rows.extend((
                    {"scenario_id": scenario, "model_seed": seed, "family": PREDICTED_TASK_WAYPOINT_TRANSFORMER, "cumulative_actual_payload": 2.0, "episode_return": 2.0, "completion": 1.0, "residual_queue_bits": 1.0},
                    {"scenario_id": scenario, "model_seed": seed, "family": STATE_ONLY_TRANSFORMER, "cumulative_actual_payload": 1.0, "episode_return": 1.0, "completion": .5, "residual_queue_bits": 2.0},
                ))
        protocol = {"bootstrap_seed": 12, "bootstrap_replicates": 32, "tie_tolerance": 1e-12}
        statistics = _paired_statistics(rows, protocol, label="unit")
        payload = [row for row in statistics if row["metric"] == "Delta_payload"]
        self.assertEqual({row["model_seed"] for row in payload}, {0, 1, 2, "pooled_scenario_mean"})
        self.assertTrue(all(row["scenario_count"] == 2 for row in payload))
        self.assertEqual(_parse_scenario("maritime_offline_dataset_v2__seed-02003__buoys-5"), (2003, 5))

    def test_12c_remaining_time_uses_configured_horizon(self) -> None:
        self.assertEqual(ROUND12_EPISODE_HORIZON, 128)

    def test_13_trainer_checkpoint_resume_contract(self) -> None:
        schema = Round12PolicySchema(state_dim=3, max_buoys=2)
        train = build_causal_sequence_split(_transitions("train"))
        validation = build_causal_sequence_split(_transitions("validation"))
        config = Round12TrainingConfig(seed=3, batch_size=4, max_epochs=1, patience=1, batches_per_epoch=1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "checkpoints"
            initial = Round12PolicyTrainer(
                TASK_WAYPOINT_MLP, schema, config, root=ROOT, output_dir=output
            ).fit(train, validation)
            resumed = Round12PolicyTrainer(
                TASK_WAYPOINT_MLP, schema, config, root=ROOT, output_dir=output
            ).fit(train, validation, resume_from=Path(initial.last_checkpoint))
        self.assertEqual(resumed.best_epoch, initial.best_epoch)
        self.assertTrue(Path(initial.best_checkpoint).exists() is False)  # TemporaryDirectory cleaned up.


if __name__ == "__main__":
    unittest.main()
