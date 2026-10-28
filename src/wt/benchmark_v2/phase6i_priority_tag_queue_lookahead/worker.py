"""L4 worker. The parent reward is unchanged; only the actor feature width grows."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.worker import DirectServiceWorker
from wt.benchmark_v2.phase6i_priority_tag_reward import rollout as rollout_module
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FEATURE_NAMES

from .features import actor_features
from .protocol import ACTOR_FEATURE_COUNT, HEAD_FEATURE_COUNT


class QueueLookaheadWorker(DirectServiceWorker):
    def __init__(self, model_seed: int, env_index: int, actor, critic, *, deterministic: bool, scenario_id: int | None, use_lookahead: bool) -> None:
        super().__init__(
            int(model_seed),
            int(env_index),
            actor,
            critic,
            deterministic=bool(deterministic),
            scenario_id=scenario_id,
            use_bonus=True,
        )
        self.use_lookahead = bool(use_lookahead)

    def _priority(self) -> np.ndarray:
        width = int(ACTOR_FEATURE_COUNT)
        if self.ledger is None or self.option is None:
            return np.zeros((int(self.env.config.num_uavs), width), dtype=np.float64)
        features = actor_features(
            self.ledger,
            int(self.env.t),
            self.option.association,
            int(self.env.config.num_buoys),
            use_lookahead=self.use_lookahead,
        )
        if not self.use_lookahead and float(np.max(np.abs(features[:, HEAD_FEATURE_COUNT:]))) != 0.0:
            raise RuntimeError("control group received a nonzero lookahead column")
        return features

    def step(self) -> dict:
        original = rollout_module.critic_vector
        seen: dict = {}

        def _critic(env, option, last_movement, priority):
            columns = np.asarray(priority, dtype=np.float64)
            if int(columns.shape[-1]) != int(ACTOR_FEATURE_COUNT):
                raise RuntimeError("critic bridge expected the five-column actor features")
            return original(env, option, last_movement, columns[:, : int(HEAD_FEATURE_COUNT)])

        def _priority():
            fed = QueueLookaheadWorker._priority(self)
            if self.ledger is None or self.option is None:
                seen["true"] = np.zeros_like(fed[:, HEAD_FEATURE_COUNT:])
                seen["association"] = np.full(int(self.env.config.num_uavs), -1, dtype=np.int64)
            else:
                seen["association"] = np.asarray(self.option.association, dtype=np.int64).copy()
                true = actor_features(
                    self.ledger,
                    int(self.env.t),
                    self.option.association,
                    int(self.env.config.num_buoys),
                    use_lookahead=True,
                )
                seen["true"] = true[:, HEAD_FEATURE_COUNT:].copy()
            return fed

        rollout_module.critic_vector = _critic
        self._priority = _priority
        try:
            record = super().step()
        finally:
            rollout_module.critic_vector = original
            del self._priority
        priority = np.asarray(record["priority"], dtype=np.float64)
        if int(priority.shape[-1]) != int(ACTOR_FEATURE_COUNT):
            raise RuntimeError("stored actor features do not have five columns")
        if tuple(FEATURE_NAMES) != ("assigned_head_age_urgency", "assigned_head_synthetic_tag", "assigned_head_valid"):
            raise RuntimeError("head feature names drifted")
        record["association"] = np.asarray(seen["association"], dtype=np.int64).copy()
        record["lookahead_input"] = priority[:, HEAD_FEATURE_COUNT:].copy()
        record["lookahead_true"] = np.asarray(seen["true"], dtype=np.float64).copy()
        record["motion_penalty"] = float(record["served_bits"]) / 1.0e6 - float(record["base_reward"])
        return record
