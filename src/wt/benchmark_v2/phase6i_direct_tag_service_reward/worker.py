"""L3 worker. The parent step still applies the formal base reward with a zero urgency coefficient."""
from __future__ import annotations

from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import ExperimentWorker

from .protocol import RHO_TAG
from .reward import direct_tag_bonus


class DirectTagWorker(ExperimentWorker):
    def __init__(self, model_seed: int, env_index: int, actor, critic, *, deterministic: bool, scenario_id: int | None = None) -> None:
        super().__init__(
            int(model_seed),
            int(env_index),
            actor,
            critic,
            lambda_u=0.0,
            rho=float(RHO_TAG),
            use_ledger=True,
            deterministic=bool(deterministic),
            scenario_id=scenario_id,
        )

    def step(self) -> dict:
        record = super().step()
        bonus = direct_tag_bonus(float(record["served_tagged_bits"]), float(record["served_bits"]))
        record["bonus"] = float(bonus)
        record["reward"] = float(record["base_reward"]) + float(bonus)
        return record
