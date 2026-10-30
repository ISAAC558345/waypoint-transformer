"""Frozen L0 trajectories and same-snapshot assignment branches. No parameter update."""
from __future__ import annotations

import copy
import random
import subprocess

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6e.generators import build_pools, is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    HOLD,
    LOG_STD_INIT,
    SERVICE_SAFETY_MARGIN_M,
)
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option
from wt.benchmark_v2.phase6i_priority_tag.follow import follow_step
from wt.benchmark_v2.phase6i_priority_tag.ledger import PriorityTagLedger, TagBucket
from wt.benchmark_v2.phase6i_priority_tag.protocol import DUST_BITS
from wt.benchmark_v2.phase6i_priority_tag_reward.features import queue_age
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR, WATCHED_PATHS, assert_train_scenario
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256
from wt.htw_phase100.contract import TaskOption
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step, minimum_pairwise_distance
from wt.lowlevel_v2.observation import FORBIDDEN_KEYS, build_actor_observation, communication_features
from wt.lowlevel_v2.residual_policy import ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .attribution import attribute_served, copy_buckets
from .candidates import assignment_key, select_registered
from .criteria import opportunity_evidence, pareto_slots, posthoc_best, service_levels
from .identity import contract_problems, history_problems, interface_identity
from .protocol import (
    ASSIGNMENT_SEED_BASE,
    BIT_TOL,
    BUCKET_TOL,
    CANDIDATE_LIMIT,
    DECISION_TIMES,
    E16_RESULT_COMMIT,
    EXCLUDED_TAIL_TIME,
    FORMAL_ACTOR_SHA256,
    HOLD_STEPS,
    HORIZON,
    MAX_SIMULATOR_STEPS,
    PATH_TOL_M,
    PROCESS_SEED,
    PROCESS_SEED_ROLE,
    ROOT,
    SCENARIOS,
)


class StepBudget:
    def __init__(self) -> None:
        self.reference = 0
        self.candidate = 0
        self.repeat = 0
        self.fixture = 0

    def add(self, kind: str, count: int = 1) -> None:
        if kind == "reference":
            self.reference += int(count)
        elif kind == "candidate":
            self.candidate += int(count)
        elif kind == "repeat":
            self.repeat += int(count)
        elif kind == "fixture":
            self.fixture += int(count)
        else:
            raise RuntimeError(f"unknown step kind {kind}")
        if self.total > int(MAX_SIMULATOR_STEPS):
            raise RuntimeError("registered simulator step budget was exceeded")

    @property
    def total(self) -> int:
        return int(self.reference + self.candidate + self.repeat + self.fixture)

    def as_dict(self) -> dict:
        return {
            "reference_steps": int(self.reference),
            "candidate_steps": int(self.candidate),
            "repeat_steps": int(self.repeat),
            "fixture_steps": int(self.fixture),
            "total_steps": int(self.total),
            "max_steps": int(MAX_SIMULATOR_STEPS),
            "training_updates": 0,
            "parameter_updates": 0,
        }


def _same(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _process_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def _restore_process(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].clone())


def _option_payload(option: TaskOption | None) -> dict | None:
    if option is None:
        return None
    urgency = None if option.urgency is None else np.asarray(option.urgency, dtype=np.float64).copy()
    return {
        "targets": np.asarray(option.targets, dtype=np.int64).copy(),
        "decision_time": int(option.decision_time),
        "remaining_hold": int(option.remaining_hold),
        "uav_mask": np.asarray(option.uav_mask, dtype=bool).copy(),
        "buoy_mask": np.asarray(option.buoy_mask, dtype=bool).copy(),
        "association": np.asarray(option.association, dtype=np.int64).copy(),
        "urgency": urgency,
    }


def _option_from(payload: dict | None) -> TaskOption | None:
    if payload is None:
        return None
    urgency = None if payload["urgency"] is None else np.asarray(payload["urgency"], dtype=np.float64).copy()
    return TaskOption(
        targets=np.asarray(payload["targets"], dtype=np.int64).copy(),
        decision_time=int(payload["decision_time"]),
        remaining_hold=int(payload["remaining_hold"]),
        uav_mask=np.asarray(payload["uav_mask"], dtype=bool).copy(),
        buoy_mask=np.asarray(payload["buoy_mask"], dtype=bool).copy(),
        association=np.asarray(payload["association"], dtype=np.int64).copy(),
        urgency=urgency,
    )


def _ledger_gap(left: dict, right: dict) -> float:
    def index(payload: dict) -> dict:
        found = {}
        for buoy, buckets in enumerate(payload["buckets"]):
            for item in buckets:
                found[(int(buoy), int(item["insert_index"]))] = item
        return found

    a = index(left)
    b = index(right)
    if a.keys() != b.keys():
        return 1.0e9
    if not a:
        return 0.0
    gaps = []
    for key, item in a.items():
        other = b[key]
        if int(item["priority_tag"]) != int(other["priority_tag"]) or int(item["admission_step"]) != int(other["admission_step"]):
            return 1.0e9
        gaps.append(abs(float(item["bits_remaining"]) - float(other["bits_remaining"])))
    return float(max(gaps))


def _inventory(ledger: PriorityTagLedger, time_step: int) -> dict:
    known_ids = []
    heads = []
    tagged = 0.0
    ordinary = 0.0
    blocked = 0.0
    for buoy, buckets in enumerate(ledger.buckets):
        alive = [
            bucket
            for bucket in buckets
            if float(bucket.bits_remaining) > float(DUST_BITS) and int(bucket.admission_step) <= int(time_step)
        ]
        alive.sort(key=lambda bucket: (int(bucket.admission_step), int(bucket.insert_index)))
        if not alive:
            continue
        head = alive[0]
        behind = float(sum(float(bucket.bits_remaining) for bucket in alive[1:]))
        blocked += behind
        heads.append(
            {
                "buoy": int(buoy),
                "insert_index": int(head.insert_index),
                "priority_tag": int(head.priority_tag),
                "bits_remaining": float(head.bits_remaining),
                "admission_step": int(head.admission_step),
                "fifo_blocked_bits": behind,
                "fifo_blocked_count": len(alive) - 1,
            }
        )
        for bucket in alive:
            known_ids.append(int(bucket.insert_index))
            if int(bucket.priority_tag) == 1:
                tagged += float(bucket.bits_remaining)
            else:
                ordinary += float(bucket.bits_remaining)
    return {
        "known_ids": known_ids,
        "known_tagged_bits": float(tagged),
        "known_ordinary_bits": float(ordinary),
        "fifo_blocked_bits": float(blocked),
        "heads": heads,
    }


def _window_match(left: list[dict], right: list[dict]) -> dict:
    if len(left) != len(right) or not left:
        return {"ok": False, "reason": "window length differs", "action_gap": 1.0e9, "position_gap_m": 1.0e9, "payload_gap_bits": 1.0e9, "queue_gap_bits": 1.0e9, "bucket_gap_bits": 1.0e9, "generated_gap_bits": 1.0e9, "tags_equal": False, "rng_equal": False}
    action_gap = 0.0
    position_gap = 0.0
    payload_gap = 0.0
    queue_gap = 0.0
    bucket_gap = 0.0
    generated_gap = 0.0
    tags_equal = True
    rng_equal = True
    for first, second in zip(left, right):
        action_gap = max(action_gap, float(np.max(np.abs(first["action"] - second["action"]))))
        position_gap = max(position_gap, float(np.max(np.abs(first["positions"] - second["positions"]))))
        payload_gap = max(payload_gap, abs(float(first["payload"]) - float(second["payload"])))
        queue_gap = max(queue_gap, float(np.max(np.abs(first["q_bits"] - second["q_bits"]))))
        bucket_gap = max(bucket_gap, _ledger_gap(first["ledger"], second["ledger"]))
        generated_gap = max(generated_gap, float(np.max(np.abs(first["generated"] - second["generated"]))))
        tags_equal = tags_equal and first["tags"] == second["tags"]
        rng_equal = rng_equal and _same(first["assignment_rng"], second["assignment_rng"]) and _same(first["env_rng"], second["env_rng"])
    ok = (
        action_gap <= float(PATH_TOL_M)
        and position_gap <= float(PATH_TOL_M)
        and payload_gap <= float(BIT_TOL)
        and queue_gap <= float(BIT_TOL)
        and bucket_gap <= float(BUCKET_TOL)
        and generated_gap <= float(BUCKET_TOL)
        and tags_equal
        and rng_equal
    )
    return {
        "ok": bool(ok),
        "reason": "" if ok else "reference window mismatch",
        "action_gap": float(action_gap),
        "position_gap_m": float(position_gap),
        "payload_gap_bits": float(payload_gap),
        "queue_gap_bits": float(queue_gap),
        "bucket_gap_bits": float(bucket_gap),
        "generated_gap_bits": float(generated_gap),
        "tags_equal": bool(tags_equal),
        "rng_equal": bool(rng_equal),
    }


def _exogenous_match(left: list[dict], right: list[dict]) -> dict:
    matched = _window_match(left, right)
    ok = bool(matched["generated_gap_bits"] <= float(BUCKET_TOL) and matched["tags_equal"] and matched["rng_equal"] and len(left) == len(right))
    return {"ok": ok, "generated_gap_bits": matched["generated_gap_bits"], "tags_equal": matched["tags_equal"], "rng_equal": matched["rng_equal"]}


def _blob(relative: str) -> bytes:
    proc = subprocess.run(
        ["git", "-c", "core.longpaths=true", "show", f"{E16_RESULT_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot read {relative} at the E16 result commit")
    return proc.stdout


def watched_problems() -> list[str]:
    problems = []
    for relative in WATCHED_PATHS:
        current = (ROOT / relative).read_bytes()
        blob = _blob(relative)
        same = current == blob
        if not same and relative.endswith(".py"):
            same = current.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
        if not same:
            problems.append(f"watched path changed after E16: {relative}")
    formal = "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"
    if file_sha256(ROOT / formal) != FORMAL_ACTOR_SHA256:
        problems.append("formal low-level checkpoint hash is not the registered SHA-256")
    return problems


def load_actor() -> ResidualActor:
    if file_sha256(FORMAL_ACTOR) != FORMAL_ACTOR_SHA256:
        raise RuntimeError("formal low-level checkpoint hash is not the registered SHA-256")
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(FORMAL_ACTOR, map_location="cpu", weights_only=False)
    state = payload["actor"] if isinstance(payload, dict) and "actor" in payload else payload
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


class FrozenAssignmentSession:
    """One scenario. Branches restore this object instead of calling env.reset."""

    def __init__(self, actor: ResidualActor, scenario_id: int, budget: StepBudget) -> None:
        self.scenario_id = int(scenario_id)
        assert_train_scenario(self.scenario_id)
        self.actor = actor
        self.budget = budget
        self.env = L3BBitQueueEnvV2()
        self.projector = SafetyProjector(
            CollisionConfig(0.4, 0.2, placeholder=False),
            safety_margin=float(SERVICE_SAFETY_MARGIN_M),
        )
        self.ledger = PriorityTagLedger(int(self.env.config.num_buoys))
        self.assignment_rng = np.random.default_rng(ASSIGNMENT_SEED_BASE + self.scenario_id)
        self.option: TaskOption | None = None
        self.pending = 0
        self.hold_known_ids: set[int] | None = None
        self.last_movement = np.zeros((int(self.env.config.num_uavs), 2), dtype=np.float64)

    def reset_episode(self) -> None:
        self.env.reset(seed=self.scenario_id)
        self.ledger = PriorityTagLedger(int(self.env.config.num_buoys))
        self.ledger.reset_from_queue(self.env.q_bits)
        self.assignment_rng = np.random.default_rng(ASSIGNMENT_SEED_BASE + self.scenario_id)
        self.option = None
        self.pending = 0
        self.hold_known_ids = None
        self.last_movement = np.zeros((int(self.env.config.num_uavs), 2), dtype=np.float64)

    def capture(self) -> dict:
        return {
            "env": capture_state(self.env),
            "env_rng": copy.deepcopy(self.env.rng.bit_generator.state),
            "ledger": self.ledger.capture(),
            "option": _option_payload(self.option),
            "pending": int(self.pending),
            "last_movement": self.last_movement.copy(),
            "assignment_rng": copy.deepcopy(self.assignment_rng.bit_generator.state),
            "process": _process_state(),
            "t": int(self.env.t),
            "known": _inventory(self.ledger, int(self.env.t)),
        }

    def restore(self, snap: dict) -> None:
        restore_state(self.env, snap["env"])
        self.env.rng.bit_generator.state = copy.deepcopy(snap["env_rng"])
        self.ledger.restore(copy.deepcopy(snap["ledger"]))
        self.option = _option_from(snap["option"])
        self.pending = int(snap["pending"])
        self.last_movement = np.asarray(snap["last_movement"], dtype=np.float64).copy()
        self.assignment_rng.bit_generator.state = copy.deepcopy(snap["assignment_rng"])
        _restore_process(snap["process"])

    def draw(self) -> tuple[list, int, np.ndarray]:
        pools, _packed, illegal = build_pools(self.env, (("G1_RATE", 5),))
        if int(illegal) != 0:
            raise RuntimeError("candidate pool contains an illegal assignment")
        pool = pools[("G1_RATE", 5)]
        if not pool:
            raise RuntimeError("candidate pool is empty")
        index = int(self.assignment_rng.integers(0, len(pool)))
        return pool, index, np.asarray(pool[index], dtype=np.int64).copy()

    def install(self, assignment: np.ndarray) -> None:
        chosen = np.asarray(assignment, dtype=np.int64).copy()
        if not is_legal(chosen, int(self.env.config.num_uavs), int(self.env.config.num_buoys)):
            raise RuntimeError("branch assignment is illegal")
        self.option = make_option(self.env, chosen)
        self.pending = int(HOLD)

    def _urgency(self, bucket: dict, time_step: int) -> float:
        probe = TagBucket(
            int(bucket["admission_step"]),
            "attribution_probe",
            int(bucket["priority_tag"]),
            float(bucket["bits_remaining"]),
            int(bucket["insert_index"]),
        )
        return float(self.ledger.urgency(probe, int(time_step)))

    def physical(self) -> dict:
        if self.pending <= 0 or self.option is None:
            raise RuntimeError("physical step missing an installed hold")
        if not np.array_equal(self.option.association, self.option.targets):
            raise RuntimeError("INTERFACE")
        association = np.asarray(self.option.association, dtype=np.int64).copy()
        obs = build_actor_observation(self.env, self.option, self.last_movement)
        if obs.get("leaks_future_arrivals") or any(key in obs for key in FORBIDDEN_KEYS):
            raise RuntimeError("future arrival information entered the observation")
        with torch.no_grad():
            mu, _log_std = self.actor(
                torch.as_tensor(obs["self"], dtype=torch.float32),
                torch.as_tensor(obs["task"], dtype=torch.float32),
                torch.as_tensor(obs["communication"], dtype=torch.float32),
                torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
            )
            pre = mu.detach().cpu().numpy().astype(np.float64)
        delta = float(ALPHA_RESIDUAL) * np.tanh(pre)
        desired = clip_unit_step(fly_toward(self.env, association) + delta)
        projected = self.projector.project(self.env, desired, self.option)
        failure = projected.constraint_status != "ok" or not projected.finite
        executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
        executed = clip_unit_step(executed)
        time_step = int(self.env.t)
        buckets = copy_buckets(self.ledger)
        if self.hold_known_ids is None:
            known_ids = set(_inventory(self.ledger, time_step)["known_ids"])
        else:
            known_ids = set(self.hold_known_ids)
        served_before = np.asarray(self.env.cumulative_served_bits, dtype=np.float64).copy()
        followed = follow_step(self.env, self.ledger, executed, association, self.scenario_id)
        served_vec = np.asarray(self.env.cumulative_served_bits, dtype=np.float64) - served_before
        attributed = attribute_served(buckets, served_vec, time_step, known_ids, self._urgency)
        served_gap = abs(attributed["known_tagged"] + attributed["future_tagged"] - float(followed["served_tagged_bits"]))
        ordinary_gap = abs(attributed["known_ordinary"] + attributed["future_ordinary"] - float(followed["served_ordinary_bits"]))
        urgency_gap = abs(attributed["urgency_served"] - float(followed["urgency_served_bits"]))
        if max(served_gap, ordinary_gap, urgency_gap) > float(BUCKET_TOL):
            raise RuntimeError("FIFO attribution does not match the ledger drain")
        error = float(np.max(np.abs(self.env.conservation_error_bits())))
        if error > float(self.env.tol_bits):
            raise RuntimeError("ACCOUNTING")
        comm = communication_features(self.env)
        age_mean, age_max = queue_age(self.ledger, int(self.env.t))
        self.pending -= 1
        self.option.remaining_hold = int(self.pending)
        self.last_movement = executed.copy()
        generated = np.asarray(self.env.cumulative_generated_arrival_bits, dtype=np.float64)
        return {
            "t": time_step,
            "action": executed.copy(),
            "positions": np.asarray(self.env.uav_positions, dtype=np.float64).copy(),
            "payload": float(self.env.payload_bits),
            "q_bits": np.asarray(self.env.q_bits, dtype=np.float64).copy(),
            "generated": generated.copy(),
            "tags": tuple(followed["tags"]),
            "ledger": self.ledger.capture(),
            "assignment_rng": copy.deepcopy(self.assignment_rng.bit_generator.state),
            "env_rng": copy.deepcopy(self.env.rng.bit_generator.state),
            "served_tagged": float(followed["served_tagged_bits"]),
            "served_ordinary": float(followed["served_ordinary_bits"]),
            "dropped_tagged": float(followed["dropped_tagged_bits"]),
            "dropped_ordinary": float(followed["dropped_ordinary_bits"]),
            "attr": attributed,
            "attr_gap": float(max(served_gap, ordinary_gap, urgency_gap)),
            "sinr": float(np.mean(self.env.last_sinr)),
            "interference": float(np.mean(comm["total_interference"])),
            "path_step": float(np.sum(np.linalg.norm(executed, axis=-1) * self.env.config.step_length_m)),
            "safety": int(minimum_pairwise_distance(self.env.uav_positions) + 1.0e-9 < float(D_SAFE_M)),
            "constraint": int(failure),
            "min_distance_m": float(minimum_pairwise_distance(self.env.uav_positions)),
            "age_mean": float(age_mean),
            "age_max": float(age_max),
            "association": assignment_key(association),
        }


def _hold_metrics(rows: list[dict], start_payload: float) -> dict:
    attr = rows[0]["attr"]
    known_tagged = float(attr["known_tagged"])
    known_ordinary = float(attr["known_ordinary"])
    future_tagged = float(attr["future_tagged"])
    future_ordinary = float(attr["future_ordinary"])
    urgency = float(attr["urgency_served"])
    dropped_tagged = float(rows[0]["dropped_tagged"])
    dropped_ordinary = float(rows[0]["dropped_ordinary"])
    gap = float(rows[0]["attr_gap"])
    for row in rows[1:]:
        known_tagged += float(row["attr"]["known_tagged"])
        known_ordinary += float(row["attr"]["known_ordinary"])
        future_tagged += float(row["attr"]["future_tagged"])
        future_ordinary += float(row["attr"]["future_ordinary"])
        urgency += float(row["attr"]["urgency_served"])
        dropped_tagged += float(row["dropped_tagged"])
        dropped_ordinary += float(row["dropped_ordinary"])
        gap = max(gap, float(row["attr_gap"]))
    last = rows[-1]
    return {
        "payload_bits": float(last["payload"]) - float(start_payload),
        "known_tagged_served_bits": known_tagged,
        "known_ordinary_served_bits": known_ordinary,
        "future_tagged_served_bits": future_tagged,
        "future_ordinary_served_bits": future_ordinary,
        "all_tagged_served_bits": known_tagged + future_tagged,
        "age_weighted_service": urgency,
        "dropped_tagged_bits": dropped_tagged,
        "dropped_ordinary_bits": dropped_ordinary,
        "final_queue_bits": float(np.sum(last["q_bits"])),
        "final_age_mean": float(last["age_mean"]),
        "final_age_max": float(last["age_max"]),
        "mean_sinr": float(np.mean([row["sinr"] for row in rows])),
        "mean_interference": float(np.mean([row["interference"] for row in rows])),
        "safety_count": int(sum(int(row["safety"]) for row in rows)),
        "constraint_count": int(sum(int(row["constraint"]) for row in rows)),
        "min_distance_m": float(min(row["min_distance_m"] for row in rows)),
        "path_length_m": float(sum(row["path_step"] for row in rows)),
        "attribution_gap_bits": gap,
    }


def _candidate_row(scenario: int, decision_time: int, slot: int, pool_index: int, assignment, baseline: dict, metrics: dict, *, is_baseline: bool, interpretable: bool) -> dict:
    levels = service_levels(
        float(metrics["known_tagged_served_bits"]) - float(baseline["known_tagged_served_bits"]),
        float(metrics["payload_bits"]) - float(baseline["payload_bits"]),
        float(metrics["safety_count"]) - float(baseline["safety_count"]),
        float(metrics["constraint_count"]) - float(baseline["constraint_count"]),
        float(metrics["mean_sinr"]) - float(baseline["mean_sinr"]),
        float(metrics["mean_interference"]) - float(baseline["mean_interference"]),
    )
    all_tagged_delta = float(metrics["all_tagged_served_bits"]) - float(baseline["all_tagged_served_bits"])
    payload_delta = float(metrics["payload_bits"]) - float(baseline["payload_bits"])
    row = {
        "scenario_id": int(scenario),
        "decision_time": int(decision_time),
        "candidate_slot": int(slot),
        "pool_index": int(pool_index),
        "is_baseline": bool(is_baseline),
        "assignment": list(assignment_key(assignment)),
        "hindsight_diagnostic": True,
        "deployable_policy": False,
        "interpretable": bool(interpretable),
        "all_tagged_descriptive_flag": bool(all_tagged_delta > 1.0 and payload_delta >= -1.0),
        **metrics,
        "known_tagged_delta_bits": float(metrics["known_tagged_served_bits"]) - float(baseline["known_tagged_served_bits"]),
        "payload_delta_bits": payload_delta,
        "all_tagged_delta_bits": all_tagged_delta,
        "age_weighted_delta": float(metrics["age_weighted_service"]) - float(baseline["age_weighted_service"]),
        "safety_delta": float(metrics["safety_count"]) - float(baseline["safety_count"]),
        "constraint_delta": float(metrics["constraint_count"]) - float(baseline["constraint_count"]),
        "mean_sinr_delta": float(metrics["mean_sinr"]) - float(baseline["mean_sinr"]),
        "mean_interference_delta": float(metrics["mean_interference"]) - float(baseline["mean_interference"]),
        "s0": bool(levels["s0"] and interpretable and not is_baseline),
        "s1": bool(levels["s1"] and interpretable and not is_baseline),
        "s2": bool(levels["s2"] and interpretable and not is_baseline),
    }
    return row


def _run_reference(session: FrozenAssignmentSession) -> tuple[list[dict], list[dict]]:
    session.reset_episode()
    rows = []
    snapshots = []
    allowed = set(DECISION_TIMES) | {int(EXCLUDED_TAIL_TIME)}
    while int(session.env.t) < int(HORIZON):
        if session.pending <= 0:
            if int(session.env.t) not in allowed:
                raise RuntimeError(f"unexpected assignment boundary at t={int(session.env.t)}")
            if int(HOLD) != int(HOLD_STEPS):
                raise RuntimeError("hold contract drifted while stepping")
            snap = session.capture()
            snap["step_index"] = len(rows)
            pool, index, chosen = session.draw()
            session.install(chosen)
            if int(snap["t"]) in DECISION_TIMES:
                snap["baseline_index"] = int(index)
                snap["baseline_assignment"] = chosen.copy()
                snap["selection"] = select_registered(pool, index, CANDIDATE_LIMIT)
                snapshots.append(snap)
        session.budget.add("reference")
        rows.append(session.physical())
    if [int(item["t"]) for item in snapshots] != list(DECISION_TIMES):
        raise RuntimeError("reference trajectory did not snapshot t=0, 15, and 30")
    if len(rows) != int(HORIZON):
        raise RuntimeError("reference trajectory length is not the registered horizon")
    return rows, snapshots


def _roll(session: FrozenAssignmentSession, snap: dict, assignment: np.ndarray, kind: str) -> list[dict]:
    session.restore(snap)
    pool, index, drawn = session.draw()
    if int(index) != int(snap["baseline_index"]) or not np.array_equal(drawn, snap["baseline_assignment"]):
        raise RuntimeError("restored draw does not reproduce the reference sample")
    regenerated = select_registered(pool, index, CANDIDATE_LIMIT)
    if regenerated["pool_sha256"] != snap["selection"]["pool_sha256"]:
        raise RuntimeError("restored pool does not match the pre-decision pool")
    session.install(assignment)
    session.hold_known_ids = {int(item) for item in snap["known"]["known_ids"]}
    rows = []
    try:
        for _ in range(int(HOLD_STEPS)):
            if int(session.env.t) >= int(HORIZON):
                raise RuntimeError("hold crossed the episode horizon")
            session.budget.add(kind)
            rows.append(session.physical())
    finally:
        session.hold_known_ids = None
    return rows


def _summarize_episode(scenario: int, rows: list[dict]) -> dict:
    return {
        "scenario_id": int(scenario),
        "steps": len(rows),
        "payload_bits": float(rows[-1]["payload"]),
        "served_tagged_bits": float(sum(row["served_tagged"] for row in rows)),
        "served_ordinary_bits": float(sum(row["served_ordinary"] for row in rows)),
        "safety_count": int(sum(row["safety"] for row in rows)),
        "constraint_count": int(sum(row["constraint"] for row in rows)),
        "final_t": int(rows[-1]["t"]) + 1,
        "decision_times": list(DECISION_TIMES),
    }


def _public_snapshot(scenario: int, snap: dict, status: dict) -> dict:
    selection = snap["selection"]
    known = snap["known"]
    return {
        "scenario_id": int(scenario),
        "decision_time": int(snap["t"]),
        "pool_size": int(selection["pool_size"]),
        "unique_count": int(selection["unique_count"]),
        "duplicate_count": int(selection["duplicate_count"]),
        "pool_sha256": selection["pool_sha256"],
        "baseline_pool_index": int(selection["baseline_index"]),
        "baseline_assignment": list(selection["baseline_key"]),
        "selected": [
            {
                "slot": int(slot),
                "pool_index": int(selection["first_index"][key]),
                "assignment": list(key),
                "is_baseline": key == selection["baseline_key"],
            }
            for slot, key in enumerate(selection["selected"])
        ],
        "dedup_first_index": list(selection["dedup_first_index"]),
        "truncated": bool(selection["truncated"]),
        "known_tagged_bits": float(known["known_tagged_bits"]),
        "known_ordinary_bits": float(known["known_ordinary_bits"]),
        "known_bucket_count": len(known["known_ids"]),
        "fifo_blocked_bits": float(known["fifo_blocked_bits"]),
        "heads": known["heads"],
        "queue_bits": float(np.sum(snap["env"]["q_bits"])),
        "hindsight_diagnostic": True,
        **status,
    }


def diagnose_scenarios(actor: ResidualActor, budget: StepBudget) -> dict:
    baselines = []
    candidates = []
    manifests = []
    problems = []
    for scenario in SCENARIOS:
        session = FrozenAssignmentSession(actor, int(scenario), budget)
        rows, snapshots = _run_reference(session)
        baselines.append(_summarize_episode(int(scenario), rows))
        for snap in snapshots:
            start = int(snap["step_index"])
            reference = rows[start : start + int(HOLD_STEPS)]
            status = {
                "interpretation": "pending",
                "reference_match": False,
                "repeat_match": False,
                "exogenous_match": False,
                "posthoc_best": None,
                "pareto_slots": [],
            }
            try:
                baseline_rows = _roll(session, snap, snap["baseline_assignment"], "candidate")
                match = _window_match(reference, baseline_rows)
                status["reference_match"] = bool(match["ok"])
                status["match_gaps"] = {key: match[key] for key in match if key != "ok"}
                repeat_rows = _roll(session, snap, snap["baseline_assignment"], "repeat")
                repeat = _window_match(baseline_rows, repeat_rows)
                status["repeat_match"] = bool(repeat["ok"])
                status["repeat_gaps"] = {key: repeat[key] for key in repeat if key != "ok"}
                if not match["ok"] or not repeat["ok"]:
                    status["interpretation"] = "stopped_reference_mismatch"
                    problems.append(f"scenario {scenario} t={int(snap['t'])} baseline replay mismatch")
                    baseline_metrics = _hold_metrics(baseline_rows, float(snap["env"]["payload_bits"]))
                    baseline_key = snap["selection"]["baseline_key"]
                    baseline_slot = next(slot for slot, key in enumerate(snap["selection"]["selected"]) if key == baseline_key)
                    candidates.append(
                        _candidate_row(
                            int(scenario), int(snap["t"]), int(baseline_slot), int(snap["selection"]["first_index"][baseline_key]),
                            snap["baseline_assignment"], baseline_metrics, baseline_metrics, is_baseline=True, interpretable=False,
                        )
                    )
                    manifests.append(_public_snapshot(int(scenario), snap, status))
                    continue
                start_payload = float(snap["env"]["payload_bits"])
                baseline_metrics = _hold_metrics(baseline_rows, start_payload)
                baseline_key = snap["selection"]["baseline_key"]
                baseline_slot = next(slot for slot, key in enumerate(snap["selection"]["selected"]) if key == baseline_key)
                snapshot_rows = [
                    _candidate_row(
                        int(scenario), int(snap["t"]), int(baseline_slot),
                        int(snap["selection"]["first_index"][baseline_key]),
                        baseline_key, baseline_metrics, baseline_metrics, is_baseline=True, interpretable=True,
                    )
                ]
                exogenous_ok = True
                for slot, key in enumerate(snap["selection"]["selected"]):
                    if key == snap["selection"]["baseline_key"]:
                        continue
                    branch_rows = _roll(session, snap, np.asarray(key, dtype=np.int64), "candidate")
                    exogenous = _exogenous_match(baseline_rows, branch_rows)
                    exogenous_ok = exogenous_ok and bool(exogenous["ok"])
                    metrics = _hold_metrics(branch_rows, start_payload)
                    snapshot_rows.append(
                        _candidate_row(
                            int(scenario), int(snap["t"]), int(slot), int(snap["selection"]["first_index"][key]),
                            key, baseline_metrics, metrics, is_baseline=False, interpretable=bool(exogenous["ok"]),
                        )
                    )
                status["exogenous_match"] = bool(exogenous_ok)
                if not exogenous_ok:
                    status["interpretation"] = "stopped_exogenous_mismatch"
                    problems.append(f"scenario {scenario} t={int(snap['t'])} exogenous arrivals diverged")
                    for row in snapshot_rows:
                        row["interpretable"] = False
                        row["s0"] = False
                        row["s1"] = False
                        row["s2"] = False
                snapshot_rows.sort(key=lambda item: int(item["candidate_slot"]))
                if exogenous_ok:
                    status["interpretation"] = "interpreted"
                    status["posthoc_best"] = posthoc_best(snapshot_rows)
                    status["pareto_slots"] = pareto_slots(snapshot_rows)
                candidates.extend(snapshot_rows)
            except RuntimeError as exc:
                status["interpretation"] = "stopped_integrity_error"
                status["error"] = str(exc)
                problems.append(f"scenario {scenario} t={int(snap['t'])}: {exc}")
            manifests.append(_public_snapshot(int(scenario), snap, status))
    return {"baselines": baselines, "candidates": candidates, "snapshots": manifests, "problems": problems}


def _t0_problems(snapshots: list[dict], candidates: list[dict]) -> list[str]:
    problems = []
    for snap in snapshots:
        if int(snap["decision_time"]) == 0 and float(snap["known_tagged_bits"]) > float(BIT_TOL):
            problems.append(f"scenario {snap['scenario_id']} t=0 already has known tagged bits")
    for row in candidates:
        if int(row["decision_time"]) == 0 and float(row["known_tagged_served_bits"]) > float(BIT_TOL):
            problems.append(f"scenario {row['scenario_id']} t=0 served known tagged bits")
    return problems


def diagnose() -> dict:
    random.seed(PROCESS_SEED)
    np.random.seed(PROCESS_SEED)
    torch.manual_seed(PROCESS_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    identity = interface_identity()
    problems = contract_problems() + watched_problems() + history_problems()
    budget = StepBudget()
    body = {
        "identity": identity,
        "problems": problems,
        "baselines": [],
        "candidates": [],
        "snapshots": [],
        "budget": budget.as_dict(),
        "opportunity_published": False,
        "t0_negative_control": False,
        "formal_actor_sha256": file_sha256(FORMAL_ACTOR),
        "process_seed": int(PROCESS_SEED),
        "process_seed_role": PROCESS_SEED_ROLE,
        "deterministic_algorithms": True,
        "torch_threads": int(torch.get_num_threads()),
        "opportunity_evidence": None,
    }
    if problems:
        return body
    actor = load_actor()
    produced = diagnose_scenarios(actor, budget)
    body.update(produced)
    body["budget"] = budget.as_dict()
    t0 = _t0_problems(body["snapshots"], body["candidates"])
    body["problems"] = list(body["problems"]) + t0
    body["t0_negative_control"] = not t0
    interpreted = [item for item in body["snapshots"] if item.get("interpretation") == "interpreted"]
    publish = not body["problems"] and len(interpreted) == len(SCENARIOS) * len(DECISION_TIMES)
    body["opportunity_published"] = bool(publish)
    if publish:
        body["opportunity_evidence"] = opportunity_evidence(
            body["candidates"],
            expected_scenarios=len(SCENARIOS),
            expected_snapshots=len(SCENARIOS) * len(DECISION_TIMES),
        )
    else:
        body["opportunity_evidence"] = None
    return body
