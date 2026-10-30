"""Record which high-level interface this diagnosis is allowed to touch."""
from __future__ import annotations

import json

from wt.benchmark_v2.phase6e.generators import build_pools
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment
from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger
from wt.benchmark_v2.phase6i_priority_tag.tags import draw_tag
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import WATCHED_PATHS
from wt.benchmark_v2.phase6i_priority_tag_reward.rollout import ExperimentWorker
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    ACTIVE_CANDIDATE_K,
    GENERATOR,
    HOLD,
    POLICY_FILE,
)
from wt.l3b.config import L3BConfig
from wt.l3b.env import L3BEnv

from .protocol import (
    CANDIDATE_LIMIT,
    DECISION_TIMES,
    E15_GATE,
    E16_GATE,
    EXCLUDED_TAIL_TIME,
    FORMAL_POOL_K,
    HOLD_STEPS,
    HORIZON,
    MAX_SIMULATOR_STEPS,
    POOL_K,
    ROOT,
    SCENARIOS,
)


def _policy() -> dict:
    if not POLICY_FILE.is_file():
        return {"present": False}
    payload = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    return {
        "present": True,
        "high_level_policy": payload.get("high_level_policy"),
        "generator": payload.get("generator"),
        "k": payload.get("k"),
        "hold": payload.get("hold"),
        "historical_formal_k": payload.get("historical_formal_k"),
        "path": str(POLICY_FILE.relative_to(ROOT)).replace("\\", "/"),
    }


def contract_problems() -> list[str]:
    problems = []
    if int(HOLD_STEPS) != 15 or int(HOLD) != 15:
        problems.append("hold is not the registered 15 steps")
    if int(HORIZON) != 48 or int(L3BConfig.episode_horizon) != 48:
        problems.append("episode horizon is not the registered 48 steps")
    if DECISION_TIMES != (0, 15, 30) or int(EXCLUDED_TAIL_TIME) != 45:
        problems.append("decision boundaries do not match 0, 15, 30 with the t=45 tail excluded")
    if SCENARIOS != tuple(range(321000, 321008)):
        problems.append("scenario list is not the first eight registered training ids")
    if int(POOL_K) != 5 or GENERATOR != "G1_RATE":
        problems.append("experiment pool is not G1_RATE k=5")
    if int(ACTIVE_CANDIDATE_K) != int(FORMAL_POOL_K) or int(FORMAL_POOL_K) != 6:
        problems.append("formal candidate contract is not k=6")
    if int(MAX_SIMULATOR_STEPS) != 2544 or int(CANDIDATE_LIMIT) != 5:
        problems.append("registered simulator budget is not 2544 steps")
    policy = _policy()
    if not policy.get("present") or policy.get("high_level_policy") != "scheme_c" or int(policy.get("k", 0)) != 6:
        problems.append("active formal policy is not scheme C on k=6")
    worker = (ROOT / "src/wt/benchmark_v2/phase6g_lowlevel_mappo/worker.py").read_text(encoding="utf-8")
    if "(\"G1_RATE\", 5)" not in worker or "def sample_assignment" not in worker:
        problems.append("frozen sampler is not the G1_RATE k=5 pool draw")
    rollout = (ROOT / "src/wt/benchmark_v2/phase6i_priority_tag_reward/rollout.py").read_text(encoding="utf-8")
    if "class ExperimentWorker" not in rollout or "sample_assignment" not in rollout:
        problems.append("experiment worker no longer exposes the sampled assignment hook")
    return problems


def history_problems() -> list[str]:
    problems = []
    e15 = json.loads((ROOT / "results/phase6i_priority_tag_clip_benchmark/summary.json").read_text(encoding="utf-8"))
    e16 = json.loads((ROOT / "results/phase6i_priority_tag_clip_closure/summary.json").read_text(encoding="utf-8"))
    if e15.get("scientific_gate") != E15_GATE or int(e15.get("score", {}).get("passing_seeds", -1)) != 0:
        problems.append("E15 summary no longer matches the registered inconclusive result")
    if e16.get("scientific_gate") != E16_GATE or e16.get("e15_gate") != E15_GATE:
        problems.append("E16 summary no longer matches the registered closure")
    return problems


def interface_identity() -> dict:
    policy = _policy()
    return {
        "diagnosis": "frozen L0 counterfactual inside the experiment G1_RATE k=5 pool",
        "not_a_formal_scheme_c_evaluation": True,
        "candidate_generation": {
            "function": "build_pools",
            "module": build_pools.__module__,
            "rule": "G1_RATE",
            "k": 5,
            "features": "current rate, distance, queue, and geometry only",
            "future_arrivals_read": False,
        },
        "assignment_draw": {
            "function": "sample_assignment",
            "module": sample_assignment.__module__,
            "rule": "one uniform index from the current pool",
            "advance": "one integers draw, then the same post-draw state on every branch",
        },
        "option": {
            "function": "make_option",
            "module": make_option.__module__,
            "hold": 15,
        },
        "experiment_worker": {
            "class": ExperimentWorker.__name__,
            "module": ExperimentWorker.__module__,
            "samples_when_pending_reaches_zero": True,
        },
        "formal_scheme_c": {
            "policy_file": policy,
            "generator": GENERATOR,
            "k": int(ACTIVE_CANDIDATE_K),
            "hold": int(HOLD),
            "entry": "phase6iaa FormalPlanner",
            "called_by_this_diagnosis": False,
            "scenario_permission": "separate from the experiment train-pool sampler",
        },
        "low_level": {
            "actor": "ResidualActor deterministic mean",
            "critic_used": False,
            "checkpoint": "phase6g_seed0.pt",
        },
        "fifo": {
            "ledger": PriorityTagLedger.__name__,
            "tag_draw": draw_tag.__name__,
            "tag_depends_on_served_bits": False,
            "service_order": "admission_step then insert_index",
        },
        "arrivals": {
            "function": L3BEnv.arrivals_at.__qualname__,
            "depends_on_queue_state": False,
            "clock": "deterministic in time_step",
        },
        "split_guard": "assert_allowed train split for 321000-321007 only",
        "safety_projector_modified": False,
        "formal_reward_modified": False,
        "watched_paths": list(WATCHED_PATHS),
    }
