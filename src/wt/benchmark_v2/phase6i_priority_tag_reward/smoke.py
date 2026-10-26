"""Short train-pool smoke. It does not train and it does not read the development pool."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
from wt.lowlevel_v2.residual_policy import ResidualActor

from .actor import (
    PriorityTagActor,
    load_formal_actor,
    refuse_formal_state_on_experiment_actor,
    transplant_legacy_actor,
)
from .protocol import (
    EXPECTED_FORMAL_ACTOR_SHA256,
    FORMAL_ACTOR,
    GATE_SMOKE_BLOCKED,
    GATE_SMOKE_PASS,
    LAMBDA_U,
    LEGACY_ACTOR,
    RHO,
    ROOT,
    SMOKE_SCENARIOS,
    SMOKE_STEPS,
    WATCHED_PATHS,
)
from .rollout import ExperimentWorker, capture_bundle, restore_bundle, run_fixed_episode


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def watched_hashes() -> dict[str, str]:
    return {relative: file_sha256(ROOT / relative) for relative in WATCHED_PATHS}


def _same_trace(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False
    for first, second in zip(left, right):
        if not np.array_equal(first["action"], second["action"]):
            return False
        if abs(float(first["payload"]) - float(second["payload"])) > 0.0:
            return False
        if not np.array_equal(first["fingerprint"]["q_bits"], second["fingerprint"]["q_bits"]):
            return False
        if not np.array_equal(first["fingerprint"]["uav_positions"], second["fingerprint"]["uav_positions"]):
            return False
    return True


def _reward_checks(rows: list[dict], *, lambda_u: float) -> list[str]:
    problems = []
    for row in rows:
        served = float(row["served_bits"])
        bonus = float(row["bonus"])
        cap = float(RHO) * served / 1.0e6
        if bonus < 0.0 or (served == 0.0 and bonus != 0.0) or bonus - cap > 1.0e-12:
            problems.append("bonus left the predeclared cap")
        if float(lambda_u) == 0.0 and bonus != 0.0:
            problems.append("lambda 0 produced a nonzero bonus")
        if float(lambda_u) == 0.0 and float(row["reward"]) != float(row["base_reward"]):
            problems.append("lambda 0 reward does not match the formal reward")
        if float(row["ledger_gap_bits"]) > 1.0:
            problems.append("ledger gap exceeded one bit")
    return problems


def _capture_check(actor, scenario: int) -> list[str]:
    worker = ExperimentWorker(0, 0, actor, None, lambda_u=0.0, rho=RHO, use_ledger=True, deterministic=True, scenario_id=scenario)
    from .protocol import assert_train_scenario

    assert_train_scenario(scenario)
    run_fixed_episode(worker, 4)
    bundle = capture_bundle(worker)
    first = [worker.step()["fingerprint"] for _ in range(2)]
    restore_bundle(worker, bundle)
    second = [worker.step()["fingerprint"] for _ in range(2)]
    problems = []
    for left, right in zip(first, second):
        if left["t"] != right["t"] or left["payload_bits"] != right["payload_bits"]:
            problems.append("capture/restore changed the formal payload or time")
        if not np.array_equal(left["q_bits"], right["q_bits"]) or not np.array_equal(left["uav_positions"], right["uav_positions"]):
            problems.append("capture/restore changed the formal queue or positions")
    return problems


def _transplant_check(formal: ResidualActor) -> list[str]:
    experiment = PriorityTagActor(log_std_init=float(LOG_STD_INIT))
    refuse_formal_state_on_experiment_actor(experiment, formal)
    transplant_legacy_actor(formal, experiment)
    generator = torch.Generator().manual_seed(7)
    self_f = torch.randn(2, 6, 5, generator=generator)
    task_f = torch.randn(2, 6, 8, generator=generator)
    comm_f = torch.randn(2, 6, 4, generator=generator)
    neigh = torch.randn(2, 6, 5, 11, generator=generator)
    near = torch.rand(2, 6, generator=generator)
    priority = torch.rand(2, 6, 3, generator=generator)
    with torch.no_grad():
        formal_mu, _formal_std = formal(self_f, task_f, comm_f, neigh, near)
        experiment_mu, _experiment_std = experiment(self_f, task_f, comm_f, neigh, near, priority)
    if float(torch.max(torch.abs(formal_mu - experiment_mu))) > 1.0e-6:
        return ["zero-column transplant changed the formal mean action"]
    return []


def run_smoke() -> dict:
    before = watched_hashes()
    problems: list[str] = []
    if before["src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"] != EXPECTED_FORMAL_ACTOR_SHA256:
        problems.append("formal L0 checkpoint hash changed")
    formal = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
    legacy = load_formal_actor(LEGACY_ACTOR, LOG_STD_INIT)
    problems.extend(_transplant_check(legacy))
    scenarios = []
    started = time.perf_counter()
    first_elapsed = None
    for index, scenario in enumerate(SMOKE_SCENARIOS):
        mark = time.perf_counter()
        enabled = ExperimentWorker(
            0, 0, formal, None, lambda_u=0.0, rho=RHO, use_ledger=True, deterministic=True, scenario_id=int(scenario)
        )
        disabled = ExperimentWorker(
            0, 0, formal, None, lambda_u=0.0, rho=RHO, use_ledger=False, deterministic=True, scenario_id=int(scenario)
        )
        enabled_rows = run_fixed_episode(enabled, SMOKE_STEPS)
        disabled_rows = run_fixed_episode(disabled, SMOKE_STEPS)
        if index == 0:
            first_elapsed = time.perf_counter() - mark
        problems.extend(_reward_checks(enabled_rows, lambda_u=0.0))
        if not _same_trace(enabled_rows, disabled_rows):
            problems.append(f"ledger follower changed scenario {scenario}")
        positive = ExperimentWorker(
            0, 1, formal, None, lambda_u=LAMBDA_U, rho=RHO, use_ledger=True, deterministic=True, scenario_id=int(scenario)
        )
        positive_rows = run_fixed_episode(positive, SMOKE_STEPS)
        problems.extend(_reward_checks(positive_rows, lambda_u=LAMBDA_U))
        problems.extend(_capture_check(formal, int(scenario)))
        scenarios.append(
            {
                "scenario_id": int(scenario),
                "steps": len(enabled_rows),
                "payload_bits": float(enabled_rows[-1]["payload"]) if enabled_rows else 0.0,
                "urgency_served_bits": float(sum(row["urgency_served_bits"] for row in enabled_rows)),
                "served_tagged_bits": float(sum(row["served_tagged_bits"] for row in enabled_rows)),
                "max_ledger_gap_bits": float(max(row["ledger_gap_bits"] for row in enabled_rows)),
                "max_bonus": float(max(row["bonus"] for row in positive_rows)),
            }
        )
    after = watched_hashes()
    if before != after:
        problems.append("watched formal files changed during smoke")
    elapsed = time.perf_counter() - started
    return {
        "gate": GATE_SMOKE_PASS if not problems else GATE_SMOKE_BLOCKED,
        "problems": problems,
        "scenarios": scenarios,
        "first_scenario_seconds": first_elapsed,
        "elapsed_seconds": elapsed,
        "watched_sha256": before,
        "formal_actor_sha256": before["src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"],
        "legacy_actor_sha256": before["results/phase5_hgnn_urgency/checkpoints/mappo_original.pt"],
        "smoke_steps": SMOKE_STEPS,
        "lambda_u": LAMBDA_U,
        "rho": RHO,
    }
