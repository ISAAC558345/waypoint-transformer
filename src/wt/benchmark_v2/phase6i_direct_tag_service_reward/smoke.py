"""Train-pool smoke for the L3 bonus. It does not read the development pool."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import PriorityTagActor, load_formal_actor, transplant_legacy_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import (
    FORMAL_ACTOR,
    LEGACY_ACTOR,
    SMOKE_SCENARIOS,
    SMOKE_STEPS,
    assert_train_scenario,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import ExperimentWorker, run_fixed_episode
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import watched_hashes

from .protocol import FORMAL_ACTOR_SHA256, LAMBDA_TAG, RHO_TAG
from .reward import direct_tag_bonus
from .worker import DirectTagWorker


def _reward_problems(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        served = float(row["served_bits"])
        tagged = float(row["served_tagged_bits"])
        bonus = float(row["bonus"])
        if tagged - served > 1.0:
            problems.append("tagged service exceeded served bits")
        if served == 0.0 and bonus != 0.0:
            problems.append("zero service produced a nonzero bonus")
        expected = direct_tag_bonus(tagged, served)
        if abs(bonus - expected) > 1.0e-12:
            problems.append("logged bonus does not match the direct tagged-service formula")
        cap = float(RHO_TAG) * served / 1.0e6
        if bonus < 0.0 or bonus - cap > 1.0e-12:
            problems.append("bonus left the payload-relative cap")
        if abs(float(row["reward"]) - float(row["base_reward"]) - bonus) > 1.0e-12:
            problems.append("L3 reward is not the formal base plus the tagged bonus")
        if float(row["ledger_gap_bits"]) > 1.0:
            problems.append("ledger gap exceeded one bit")
        if abs(float(row["served_ordinary_bits"]) + tagged - served) > 1.0:
            problems.append("FIFO tag split does not match served bits")
    return problems


def _same_trace(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right):
        if not np.array_equal(first["action"], second["action"]):
            return False
        if float(first["payload"]) != float(second["payload"]):
            return False
        if not np.array_equal(first["fingerprint"]["q_bits"], second["fingerprint"]["q_bits"]):
            return False
    return True


def run_smoke() -> dict:
    before = watched_hashes()
    problems: list[str] = []
    if before.get("src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt") != FORMAL_ACTOR_SHA256:
        problems.append("formal checkpoint hash changed before smoke")
    actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
    actor.eval()
    observed = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
    transplant_legacy_actor(load_formal_actor(LEGACY_ACTOR, LOG_STD_INIT), observed)
    observed.eval()
    scenarios = []
    for scenario in SMOKE_SCENARIOS:
        assert_train_scenario(int(scenario))
        enabled = DirectTagWorker(0, 0, actor, None, deterministic=True, scenario_id=int(scenario))
        plain = ExperimentWorker(
            0, 0, actor, None, lambda_u=0.0, rho=float(RHO_TAG), use_ledger=False, deterministic=True, scenario_id=int(scenario)
        )
        enabled_rows = run_fixed_episode(enabled, SMOKE_STEPS)
        plain_rows = run_fixed_episode(plain, SMOKE_STEPS)
        problems.extend(_reward_problems(enabled_rows))
        if not _same_trace(enabled_rows, plain_rows):
            problems.append(f"ledger follower changed the transition on scenario {int(scenario)}")
        repeat = DirectTagWorker(0, 0, actor, None, deterministic=True, scenario_id=int(scenario))
        repeated = run_fixed_episode(repeat, SMOKE_STEPS)
        if not _same_trace(enabled_rows, repeated):
            problems.append(f"same state and action were not reproduced on scenario {int(scenario)}")
        priority_rows = run_fixed_episode(
            DirectTagWorker(0, 1, observed, None, deterministic=True, scenario_id=int(scenario)),
            SMOKE_STEPS,
        )
        problems.extend(_reward_problems(priority_rows))
        scenarios.append(
            {
                "scenario_id": int(scenario),
                "steps": len(enabled_rows),
                "served_bits": float(sum(row["served_bits"] for row in enabled_rows)),
                "served_tagged_bits": float(sum(row["served_tagged_bits"] for row in enabled_rows)),
                "max_bonus": float(max(row["bonus"] for row in enabled_rows)),
                "max_ledger_gap_bits": float(max(row["ledger_gap_bits"] for row in enabled_rows)),
            }
        )
    after = watched_hashes()
    if before != after:
        problems.append("formal files changed during smoke")
    if float(LAMBDA_TAG) != 0.002 or float(RHO_TAG) != 0.001:
        problems.append("smoke saw a drifted L3 config")
    return {"passed": bool(not problems), "problems": problems, "scenarios": scenarios, "watched_sha256": before}
