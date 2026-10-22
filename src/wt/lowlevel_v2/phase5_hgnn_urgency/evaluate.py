"""Closed-loop metrics and high/low rank agreement. Reward is not rewritten."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase9.features import conflict_rate
from wt.htw_phase943.set_context import empty_assignment, make_snap, all_candidate_physics_set
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman

from .features import observable_exclusive_assignment
from .planner import Phase5Planner, first_step_entropy, make_phase5_option
from .protocol import HOLD_STEPS, assert_phase5_seed
from .worker import Phase5Worker


def _restore(env: L3BEnv, state: dict) -> None:
    env.t = int(state["t"])
    env.uav_positions = np.asarray(state["uav_positions"], dtype=np.float64).copy()
    env.queue = np.asarray(state["queue"], dtype=np.float64).copy()
    env.progress = np.asarray(state["progress"], dtype=np.float64).copy()
    env.served_k = np.asarray(state["served_k"], dtype=np.int64).copy()
    env.assignment = np.asarray(state["assignment"], dtype=np.int64).copy()
    env.payload_bits = float(state["payload_bits"])
    env.dropped = float(state["dropped"])
    env.last_sinr = np.asarray(state["last_sinr"], dtype=np.float64).copy()
    env.last_rate = np.asarray(state["last_rate"], dtype=np.float64).copy()


def _snapshot(env: L3BEnv) -> dict:
    return {
        "t": int(env.t),
        "uav_positions": env.uav_positions.copy(),
        "queue": env.queue.copy(),
        "progress": env.progress.copy(),
        "served_k": env.served_k.copy(),
        "assignment": env.assignment.copy(),
        "payload_bits": float(env.payload_bits),
        "dropped": float(env.dropped),
        "last_sinr": env.last_sinr.copy(),
        "last_rate": env.last_rate.copy(),
    }


def _conditional_logits(planner: Phase5Planner, env: L3BEnv, info: dict, last_arrival, last_positions, partial: np.ndarray) -> np.ndarray:
    from .features import graph_batch

    feature_urgency = None
    if planner.high_urgency:
        from .urgency import shared_task_urgency

        feature_urgency = shared_task_urgency(env.queue)
    batch = graph_batch(info, env, last_arrival, last_positions, urgency=feature_urgency)
    tensors = {}
    for key in ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu"):
        dtype = torch.bool if key.endswith("mask") else torch.float32
        tensors[key] = torch.as_tensor(batch[key], dtype=dtype)
    with torch.no_grad():
        encoded = planner.model.encode(
            tensors["uav"], tensors["buoy"], tensors["uav_mask"], tensors["buoy_mask"], tensors["global"], tensors["edge"], tensors["edge_uu"]
        )
        snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
        physics = torch.as_tensor(all_candidate_physics_set(snap, partial)[None, ...], dtype=torch.float32)
        logits = planner.model.decoder(encoded, partial, physics, np.asarray(batch["buoy_mask"][0], dtype=bool))
    return logits.detach().cpu().numpy()


def _hold_return(planner, actor, critic, *, use_urgency: bool, seed: int, state: dict, association: np.ndarray, urgency: np.ndarray, last_arrival, last_movement) -> float:
    worker = Phase5Worker(planner, actor, critic, use_urgency=use_urgency, seeds=(int(seed),))
    worker.reset(int(seed))
    _restore(worker.env, state)
    worker.last_arrival = np.asarray(last_arrival, dtype=np.float32).copy()
    worker.last_positions = np.asarray(state["uav_positions"], dtype=np.float32).copy()
    worker.last_movement = np.asarray(last_movement, dtype=np.float64).copy()
    worker.pending = int(HOLD_STEPS)
    info = worker.env.info()
    worker.option = make_phase5_option(worker.env, info, association, HOLD_STEPS, urgency, attach_urgency=use_urgency)
    total = 0.0
    for _ in range(int(HOLD_STEPS)):
        info, rec = worker.step(info, deterministic=True)
        total += float(rec["reward"])
        if rec["done"]:
            break
    return float(total)


def alignment_at_state(planner, actor, critic, *, use_urgency: bool, seed: int, state: dict, info: dict, assignment: np.ndarray, urgency: np.ndarray, last_arrival, last_movement) -> list[dict]:
    rows = []
    env = L3BEnv()
    env.reset(seed=int(seed))
    _restore(env, state)
    for focal in range(int(env.config.num_uavs)):
        taken = {int(assignment[uav]) for uav in range(int(env.config.num_uavs)) if uav != focal and int(assignment[uav]) >= 0}
        legal = [buoy for buoy in range(int(env.config.num_buoys)) if buoy not in taken and float(env.queue[buoy]) > 1.0e-9]
        if len(legal) < 3:
            continue
        partial = empty_assignment(int(env.config.num_uavs))
        for uav in range(int(env.config.num_uavs)):
            if uav != focal:
                partial[uav] = int(assignment[uav])
        logits = _conditional_logits(planner, env, info, last_arrival, last_movement, partial)
        scores = []
        returns = []
        for buoy in legal:
            trial = np.asarray(assignment, dtype=np.int64).copy()
            trial[focal] = int(buoy)
            scores.append(float(logits[focal, int(buoy)]))
            returns.append(
                _hold_return(
                    planner,
                    actor,
                    critic,
                    use_urgency=use_urgency,
                    seed=seed,
                    state=state,
                    association=trial,
                    urgency=urgency,
                    last_arrival=last_arrival,
                    last_movement=last_movement,
                )
            )
        scores_a = np.asarray(scores, dtype=np.float64)
        returns_a = np.asarray(returns, dtype=np.float64)
        q_choice = int(legal[int(np.argmax(scores_a))])
        u_choice = int(legal[int(np.argmax(returns_a))])
        span = float(np.max(returns_a) - np.min(returns_a))
        chosen = float(returns_a[int(np.argmax(scores_a))])
        regret = 0.0 if span <= 1.0e-8 else float((np.max(returns_a) - chosen) / span)
        rows.append(
            {
                "seed": int(seed),
                "focal": int(focal),
                "n_candidates": len(legal),
                "spearman": spearman(scores_a, returns_a),
                "kendall": kendall_tau(scores_a, returns_a),
                "top1_agreement": float(q_choice == u_choice),
                "normalized_regret": regret,
            }
        )
    return rows


def rollout_metrics(planner: Phase5Planner, actor, critic, *, use_urgency: bool, seed: int) -> dict:
    assert_phase5_seed(int(seed))
    worker = Phase5Worker(planner, actor, critic, use_urgency=use_urgency, seeds=(int(seed),))
    info = worker.reset(int(seed))
    path = 0.0
    reward_sum = 0.0
    entropies = []
    matches = []
    conflicts = []
    urgency_gaps = []
    completion = None
    align_rows: list[dict] = []
    while True:
        decision = worker.pending <= 0
        pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
        if decision and int(worker.env.t) == 15:
            print(f"alignment seed={seed} t=15", flush=True)
            state = _snapshot(worker.env)
            last_arrival = np.asarray(worker.last_arrival, dtype=np.float32).copy()
            last_movement = np.asarray(worker.last_movement, dtype=np.float64).copy()
            last_positions = np.asarray(worker.last_positions, dtype=np.float32).copy()
            assignment, urgency = planner.decide(worker.env, info, last_arrival, last_positions)
            align_rows.extend(
                alignment_at_state(
                    planner,
                    actor,
                    critic,
                    use_urgency=use_urgency,
                    seed=int(seed),
                    state=state,
                    info=info,
                    assignment=assignment,
                    urgency=urgency,
                    last_arrival=last_arrival,
                    last_movement=last_movement,
                )
            )
        label = observable_exclusive_assignment(worker.env) if decision else None
        if decision:
            entropies.append(first_step_entropy(planner, worker.env, info, worker.last_arrival, worker.last_positions))
        info, rec = worker.step(info, deterministic=True)
        path += float(np.sum(np.linalg.norm(worker.env.uav_positions - pos0, axis=-1)))
        reward_sum += float(rec["reward"])
        if decision and worker.option is not None and label is not None:
            chosen = np.asarray(worker.option.association, dtype=np.int64)
            matches.append(float(np.mean(chosen == label)))
            conflicts.append(conflict_rate(chosen, int(worker.env.config.num_buoys)))
            if worker.option.urgency is not None:
                taken = [int(b) for b in chosen if int(b) >= 0]
                rest = [b for b in range(int(worker.env.config.num_buoys)) if b not in set(taken)]
                if taken and rest:
                    urgency_gaps.append(float(np.mean(worker.option.urgency[taken]) - np.mean(worker.option.urgency[rest])))
        if completion is None and float(np.sum(worker.env.queue)) <= 1.0e-9:
            completion = int(worker.env.t)
        if info.get("terminated", False) or rec["preexisting_safety_violation"]:
            break
    return {
        "payload": float(worker.env.payload_bits),
        "residual_queue": float(np.sum(worker.env.queue)),
        "completion_time": int(worker.env.config.episode_horizon if completion is None else completion),
        "path_length_m": path,
        "energy": None,
        "trajectory_reward": reward_sum,
        "assignment_match": float(np.mean(matches)) if matches else float("nan"),
        "assignment_conflict": float(np.mean(conflicts)) if conflicts else float("nan"),
        "assignment_entropy": float(np.nanmean(np.asarray(entropies, dtype=np.float64))) if entropies else float("nan"),
        "urgency_gap": float(np.mean(urgency_gaps)) if urgency_gaps else float("nan"),
        "alignment": align_rows,
    }


def mean_alignment(rows: list[dict]) -> dict:
    if not rows:
        return {"spearman": float("nan"), "kendall": float("nan"), "top1_agreement": float("nan"), "normalized_regret": float("nan"), "n": 0}

    def avg(key: str) -> float:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")

    return {
        "spearman": avg("spearman"),
        "kendall": avg("kendall"),
        "top1_agreement": avg("top1_agreement"),
        "normalized_regret": avg("normalized_regret"),
        "n": len(rows),
    }


def judge(baseline_payload: float, full_payload: float, baseline_align: dict, full_align: dict) -> str:
    payload_up = float(full_payload) > float(baseline_payload)
    base_s = float(baseline_align["spearman"])
    full_s = float(full_align["spearman"])
    full_top = float(full_align["top1_agreement"])
    align_up = np.isfinite(full_s) and np.isfinite(base_s) and full_s > base_s + 0.1
    still_mismatch = (not np.isfinite(full_s)) or full_s < 0.3 or (not np.isfinite(full_top)) or full_top < 0.5
    if payload_up and align_up and not still_mismatch:
        return "A"
    if (payload_up or align_up) and still_mismatch:
        return "B"
    return "INCONCLUSIVE_BOUNDED_BUDGET"


def high_level_changes_when_urgency_zeroed(planner: Phase5Planner, seed: int) -> bool:
    if not planner.high_urgency:
        return False
    from wt.htw_phase943.evaluation import decode_assignment

    from .features import graph_batch

    env = L3BEnv()
    info = env.reset(seed=int(seed))
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_positions = np.asarray(env.uav_positions, dtype=np.float32).copy()
    real, _ = planner.decide(env, info, last_arrival, last_positions)
    zeros = np.zeros(env.config.num_buoys, dtype=np.float64)
    batch = graph_batch(info, env, last_arrival, last_positions, urgency=zeros)
    with torch.no_grad():
        wiped = decode_assignment(planner.model, batch, env)
    return not np.array_equal(np.asarray(real), np.asarray(wiped))


def urgency_sensitivity(actor, obs: dict, urgency: np.ndarray) -> float:
    from .actor import UrgencyResidualActor

    if not isinstance(actor, UrgencyResidualActor):
        return float("nan")
    self_f = torch.as_tensor(obs["self"], dtype=torch.float32)
    task_f = torch.as_tensor(obs["task"], dtype=torch.float32)
    comm_f = torch.as_tensor(obs["communication"], dtype=torch.float32)
    neigh = torch.as_tensor(obs["neighbor_set"], dtype=torch.float32)
    near = torch.as_tensor(obs["d_nearest"], dtype=torch.float32)
    low = torch.zeros_like(torch.as_tensor(urgency, dtype=torch.float32))
    high = torch.ones_like(low)
    with torch.no_grad():
        mu_low, _ = actor(self_f, task_f, comm_f, neigh, near, low)
        mu_high, _ = actor(self_f, task_f, comm_f, neigh, near, high)
    return float(torch.mean(torch.abs(mu_high - mu_low)).cpu())


__all__ = [
    "alignment_at_state",
    "high_level_changes_when_urgency_zeroed",
    "judge",
    "mean_alignment",
    "rollout_metrics",
    "urgency_sensitivity",
]
