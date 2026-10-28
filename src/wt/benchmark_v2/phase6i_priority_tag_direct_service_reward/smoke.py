"""Train-pool smoke. Development scenarios are not read here."""
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
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import ExperimentWorker, capture_bundle, restore_bundle, run_fixed_episode
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import watched_hashes
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import _install, _restore

from .protocol import FORMAL_ACTOR_SHA256
from .reward import split_bonus
from .worker import DirectServiceWorker


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
        if not np.array_equal(first["fingerprint"]["uav_positions"], second["fingerprint"]["uav_positions"]):
            return False
    return True


def _reward_problems(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        served = float(row["served_bits"])
        tagged = float(row["served_tagged_bits"])
        age = float(row["age_service_bits"])
        if abs(float(row["served_ordinary_bits"]) + tagged - served) > 1.0:
            problems.append("FIFO tag split does not match served bits")
        increment = float(row["tag_increment_service_bits"])
        combined = float(row["combined_urgency_served_bits"])
        if abs(age + increment - combined) > 1.0e-6 or abs(combined - float(row["urgency_served_bits"])) > 1.0e-6:
            problems.append("age and tag increment do not rebuild combined urgency")
        bonus_age, bonus_tag = split_bonus(age, tagged, served)
        if abs(float(row["bonus_age"]) - bonus_age) > 1.0e-12 or abs(float(row["bonus_tag"]) - bonus_tag) > 1.0e-12:
            problems.append("logged bonuses do not match the preregistered formula")
        if served == 0.0 and (float(row["bonus_age"]) != 0.0 or float(row["bonus_tag"]) != 0.0):
            problems.append("zero service produced a nonzero bonus")
        if float(row["ledger_gap_bits"]) > 1.0:
            problems.append("ledger gap exceeded one bit")
    return problems


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
    rollout, original = _install()
    scenarios = []
    try:
        for scenario in SMOKE_SCENARIOS:
            assert_train_scenario(int(scenario))
            enabled = DirectServiceWorker(0, 0, actor, None, deterministic=True, scenario_id=int(scenario))
            plain = ExperimentWorker(
                0, 0, actor, None, lambda_u=0.0, rho=0.001, use_ledger=False, deterministic=True, scenario_id=int(scenario)
            )
            disabled = DirectServiceWorker(0, 0, actor, None, deterministic=True, scenario_id=int(scenario), use_bonus=False)
            base = ExperimentWorker(
                0, 0, actor, None, lambda_u=0.0, rho=0.001, use_ledger=True, deterministic=True, scenario_id=int(scenario)
            )
            enabled_rows = run_fixed_episode(enabled, SMOKE_STEPS)
            plain_rows = run_fixed_episode(plain, SMOKE_STEPS)
            disabled_rows = run_fixed_episode(disabled, SMOKE_STEPS)
            base_rows = run_fixed_episode(base, SMOKE_STEPS)
            problems.extend(_reward_problems(enabled_rows))
            if not _same_trace(enabled_rows, plain_rows):
                problems.append(f"ledger follower changed the transition on scenario {int(scenario)}")
            probe = DirectServiceWorker(0, 0, actor, None, deterministic=True, scenario_id=int(scenario))
            probe.reset()
            probe.step()
            bundle = capture_bundle(probe)
            rng_state = probe.rng.bit_generator.state
            restored_left = probe.step()
            restore_bundle(probe, bundle)
            probe.rng.bit_generator.state = rng_state
            restored_right = probe.step()
            if not _same_trace([restored_left], [restored_right]):
                problems.append(f"capture/restore did not repeat the transition on scenario {int(scenario)}")
            repeat = DirectServiceWorker(0, 0, actor, None, deterministic=True, scenario_id=int(scenario))
            if not _same_trace(enabled_rows, run_fixed_episode(repeat, SMOKE_STEPS)):
                problems.append(f"same state and action were not reproduced on scenario {int(scenario)}")
            for left, right in zip(disabled_rows, base_rows):
                if abs(float(left["reward"]) - float(right["reward"])) > 1.0e-12:
                    problems.append("bonus-disabled reward does not match the L1 base reward")
                    break
                if float(left["bonus"]) != 0.0:
                    problems.append("bonus-disabled path produced a bonus")
                    break
            priority_rows = run_fixed_episode(
                DirectServiceWorker(0, 1, observed, None, deterministic=True, scenario_id=int(scenario)),
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
                }
            )
    finally:
        _restore(rollout, original)
    after = watched_hashes()
    if before != after:
        problems.append("formal files changed during smoke")
    return {"passed": bool(not problems), "problems": problems, "scenarios": scenarios}
