"""L3 worker. The parent applies the formal base reward; this step adds the two bonuses."""
from __future__ import annotations

from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import ExperimentWorker

from .protocol import RHO_TAG
from .reward import split_bonus


class DirectServiceWorker(ExperimentWorker):
    def __init__(
        self,
        model_seed: int,
        env_index: int,
        actor,
        critic,
        *,
        deterministic: bool,
        scenario_id: int | None = None,
        use_bonus: bool = True,
    ) -> None:
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
        self.use_bonus = bool(use_bonus)

    def step(self) -> dict:
        carried: dict = {}
        original_reset = self.reset

        def _capture_reset() -> None:
            attribution = getattr(self.ledger, "step_attribution", None)
            if isinstance(attribution, dict):
                carried["attribution"] = attribution
            original_reset()

        self.reset = _capture_reset
        try:
            record = super().step()
        finally:
            del self.reset
        attribution = getattr(self.ledger, "step_attribution", None)
        if not isinstance(attribution, dict) or "age_service_bits" not in attribution:
            attribution = carried.get("attribution")
        if not isinstance(attribution, dict) or "age_service_bits" not in attribution:
            raise RuntimeError("L3 reward requires the pre-service FIFO attribution")
        age_bits = float(attribution["age_service_bits"])
        tagged_bits = float(attribution["served_tagged_bits"])
        if abs(tagged_bits - float(record["served_tagged_bits"])) > 1.0e-6:
            raise RuntimeError("pre-service tagged allocation does not match the ledger")
        if abs(float(attribution["served_ordinary_bits"]) - float(record["served_ordinary_bits"])) > 1.0e-6:
            raise RuntimeError("pre-service ordinary allocation does not match the ledger")
        if self.use_bonus:
            bonus_age, bonus_tag = split_bonus(age_bits, tagged_bits, float(record["served_bits"]))
        else:
            bonus_age, bonus_tag = 0.0, 0.0
        record["age_service_bits"] = age_bits
        record["tag_increment_service_bits"] = float(attribution["tag_increment_service_bits"])
        record["combined_urgency_served_bits"] = float(attribution["combined_urgency_served_bits"])
        record["bonus_age"] = float(bonus_age)
        record["bonus_tag"] = float(bonus_tag)
        record["bonus"] = float(bonus_age + bonus_tag)
        record["reward"] = float(record["base_reward"]) + float(record["bonus"])
        return record
