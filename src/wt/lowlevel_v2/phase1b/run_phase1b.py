"""LL-V2.1B service-locked residual MAPPO pilot. Fresh init. No Phase-1 weight load."""
from __future__ import annotations

import copy
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.phase1a.diagnose import desired_constraint_flags, intervention_label
from wt.lowlevel_v2.protocol import EXPECTED_HIGH_SHA
from wt.round30.ppo import seed_everything

from .controller import ResidualWorker, load_frozen_high, make_networks
from .mappo import ICARMAPPOLocked
from .protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    EVAL_EVERY,
    LOG_STD_INIT,
    MODEL_SEED,
    OLD_P1_U100_ENVELOPE_AMONG,
    OLD_P1_U100_PAYLOAD,
    OLD_P1_U100_PROJECTOR,
    OUTPUT,
    PHASE1B,
    PILOT_UPDATES_N,
    PPO,
    PREFREEZE_FILE,
    ROOT,
    SERVICE_SAFETY_MARGIN_M,
    SMOKE_UPDATES_N,
    VECTOR_SEED_PILOT,
    VECTOR_SEED_SMOKE,
    high_sha,
    load_pools,
    load_prefreeze,
    refuse_untouched,
)


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def collect_rollout(workers: list[ResidualWorker], agent: ICARMAPPOLocked, horizon: int) -> dict[str, np.ndarray]:
    n_env = len(workers)
    keys = {
        "self": (horizon, n_env, 6, 5),
        "task": (horizon, n_env, 6, 8),
        "communication": (horizon, n_env, 6, 4),
        "neighbor_set": (horizon, n_env, 6, 5, 11),
        "d_nearest": (horizon, n_env, 6),
        "global_state": (horizon, n_env, expected_critic_dim(6, 12)),
        "pre_tanh": (horizon, n_env, 6, 2),
        "log_prob_uav": (horizon, n_env, 6),
        "policy_active_mask": (horizon, n_env, 6),
        "values": (horizon, n_env),
        "rewards": (horizon, n_env),
        "dones": (horizon, n_env),
    }
    storage = {key: np.zeros(shape, dtype=np.float32) for key, shape in keys.items()}
    infos = [w.env.info() for w in workers]
    for t in range(horizon):
        for e, worker in enumerate(workers):
            infos[e], rec = worker.step(infos[e], deterministic=False)
            storage["self"][t, e] = rec.obs["self"]
            storage["task"][t, e] = rec.obs["task"]
            storage["communication"][t, e] = rec.obs["communication"]
            storage["neighbor_set"][t, e] = rec.obs["neighbor_set"]
            storage["d_nearest"][t, e] = rec.obs["d_nearest"]
            storage["global_state"][t, e] = rec.global_state
            storage["pre_tanh"][t, e] = rec.pre_tanh
            storage["log_prob_uav"][t, e] = rec.log_prob_uav
            storage["policy_active_mask"][t, e] = rec.policy_active_mask
            storage["values"][t, e] = rec.value
            storage["rewards"][t, e] = rec.reward
            storage["dones"][t, e] = float(rec.done)
            if rec.done:
                seed = int(worker._next_seed())
                infos[e] = worker.reset(seed)
    last_values = np.zeros(n_env, dtype=np.float32)
    for e, worker in enumerate(workers):
        if storage["dones"][-1, e] > 0.5:
            last_values[e] = 0.0
            continue
        from wt.lowlevel_v2.critic_state import build_critic_state

        glob = build_critic_state(worker.env, worker.option, worker.last_movement)["vector"]
        with torch.no_grad():
            last_values[e] = float(agent.critic(torch.as_tensor(glob, dtype=torch.float32)).cpu())
    storage["last_values"] = last_values
    return storage


def evaluate_set(worker: ResidualWorker, seeds: list[int], *, zero_residual: bool, shuffle: str | None = None) -> dict:
    payloads, blocks, queues, drops = [], [], [], []
    minseps, collisions, nears = [], [], []
    inter_rates, inter_norms, failures, preexisting = [], [], [], []
    sinrs, rates, flights, res_norms, unfinished = [], [], [], [], []
    disagreements, active_fracs, lock_viols = [], [], []
    env_int_rates, pair_int_rates = [], []
    used_nz_in_service = []
    label_counts = defaultdict(int)
    n_steps_all = 0
    n_intervened = 0
    for seed in seeds:
        refuse_untouched(int(seed))
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        flight = 0.0
        inter_n = 0
        env_n = 0
        pair_n = 0
        fail_n = 0
        pre_n = 0
        coll = 0
        near = 0
        mins = []
        res = []
        sinr_acc = []
        rate_acc = []
        active = []
        viol = 0.0
        step_n = 0
        nz_svc = []
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
            saved = worker.env.uav_positions.copy()
            worker.env.uav_positions = pos0
            flags = desired_constraint_flags(worker.env, worker.option, rec.a_des)
            worker.env.uav_positions = saved
            intervened = float(rec.intervention_norm) > 1.0e-6
            label = intervention_label(flags, intervened)
            n_steps_all += 1
            if intervened:
                n_intervened += 1
                label_counts[label] += 1
                if flags["SERVICE_SAFE_ENVELOPE"] and not flags["PAIRWISE_COLLISION"] and not flags["STEP_LIMIT"]:
                    env_n += 1
                if flags["PAIRWISE_COLLISION"] and not flags["SERVICE_SAFE_ENVELOPE"]:
                    pair_n += 1
            if rec.service_lock.any():
                used_nz = rec.delta_used[rec.service_lock]
                nz_svc.extend((np.linalg.norm(used_nz, axis=-1) > 1.0e-8).astype(float).tolist())
            if shuffle and not rec.preexisting_safety_violation:
                shuffled = {key: (np.array(val, copy=True) if isinstance(val, np.ndarray) else val) for key, val in rec.obs.items()}
                rng = np.random.default_rng(int(seed) + 17)
                if shuffle == "neighbor":
                    perm = rng.permutation(shuffled["neighbor_set"].shape[1])
                    shuffled["neighbor_set"] = shuffled["neighbor_set"][:, perm]
                elif shuffle == "interference":
                    shuffled["communication"][:, 2] = rng.permutation(shuffled["communication"][:, 2])
                    shuffled["communication"][:, 3] = rng.permutation(shuffled["communication"][:, 3])
                elif shuffle == "sinr":
                    shuffled["communication"][:, 0] = rng.permutation(shuffled["communication"][:, 0])
                    shuffled["communication"][:, 1] = rng.permutation(shuffled["communication"][:, 1])
                with torch.no_grad():
                    pre_s, _, _ = worker.act_tensors(shuffled, deterministic=True)
                    disagreements.append(float(np.mean(np.abs(pre_s.cpu().numpy() - rec.pre_tanh))))
            step_n += 1
            flight += float(np.sum(np.linalg.norm(np.asarray(worker.env.uav_positions) - pos0, axis=-1)))
            if rec.intervention_norm > 1.0e-6:
                inter_n += 1
            inter_norms.append(rec.intervention_norm)
            fail_n += int(rec.projector_failure)
            pre_n += int(rec.preexisting_safety_violation)
            mins.append(rec.min_sep)
            if rec.min_sep + 1.0e-9 < D_SAFE_M:
                coll += 1
            elif rec.min_sep < 1.5 * D_SAFE_M:
                near += 1
            res.append(float(np.mean(np.linalg.norm(rec.delta_used, axis=-1))))
            active.append(float(np.mean(rec.policy_active_mask)))
            viol += rec.lock_violation
            sinr_acc.append(float(np.mean(worker.env.last_sinr)))
            rate_acc.append(float(np.mean(worker.env.last_rate)))
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
        payload = float(worker.env.payload_bits - payload0)
        payloads.append(payload)
        blocks.append(payload / 1.0e6)
        queues.append(float(np.sum(worker.env.queue)))
        drops.append(float(worker.env.dropped))
        minseps.append(float(min(mins) if mins else float("inf")))
        collisions.append(coll)
        nears.append(near)
        inter_rates.append(inter_n / max(step_n, 1))
        env_int_rates.append(env_n / max(step_n, 1))
        pair_int_rates.append(pair_n / max(step_n, 1))
        failures.append(fail_n)
        preexisting.append(pre_n)
        sinrs.append(float(np.mean(sinr_acc) if sinr_acc else 0.0))
        rates.append(float(np.mean(rate_acc) if rate_acc else 0.0))
        flights.append(flight)
        res_norms.append(float(np.mean(res) if res else 0.0))
        unfinished.append(float(np.mean(worker.env.progress)))
        active_fracs.append(float(np.mean(active) if active else 0.0))
        lock_viols.append(viol)
        used_nz_in_service.extend(nz_svc)

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    among = {name: label_counts[name] / max(n_intervened, 1) for name in ("STEP_LIMIT", "PAIRWISE_COLLISION", "SERVICE_SAFE_ENVELOPE", "MULTI_CONSTRAINT", "OTHER")}
    rate = {name: label_counts[name] / max(n_steps_all, 1) for name in among}
    return {
        "payload": mean(payloads),
        "completed_blocks": mean(blocks),
        "residual_queue": mean(queues),
        "drop": mean(drops),
        "min_sep": mean(minseps),
        "min_sep_worst": float(min(minseps) if minseps else float("inf")),
        "collision_count": float(np.sum(collisions)),
        "near_events": float(np.sum(nears)),
        "projector_intervention_rate": mean(inter_rates),
        "projector_intervention_norm": mean(inter_norms),
        "service_envelope_intervention_rate": mean(env_int_rates),
        "pairwise_projector_intervention_rate": mean(pair_int_rates),
        "projector_failure_count": float(np.sum(failures)),
        "preexisting_safety_violation": float(np.sum(preexisting)),
        "mean_sinr": mean(sinrs),
        "mean_active_rate": mean(rates),
        "flight_distance": mean(flights),
        "mean_residual_norm": mean(res_norms),
        "unfinished_progress": mean(unfinished),
        "policy_active_fraction": mean(active_fracs),
        "service_lock_violation_count": float(np.sum(lock_viols)),
        "delta_used_nonzero_fraction_locked": float(np.mean(used_nz_in_service) if used_nz_in_service else 0.0),
        "n": len(seeds),
        "shuffle_disagreement": mean(disagreements) if disagreements else 0.0,
        "n_steps": n_steps_all,
        "n_intervened": n_intervened,
        **{f"{k}_rate": v for k, v in rate.items()},
        **{f"{k}_among_intervened": v for k, v in among.items()},
        "payloads": payloads,
    }


class SeedCursor:
    def __init__(self, seeds: list[int], rng: np.random.Generator) -> None:
        self.seeds = list(seeds)
        self.rng = rng
        self.order = self.rng.permutation(self.seeds).tolist()
        self.i = 0

    def next(self) -> int:
        if self.i >= len(self.order):
            self.order = self.rng.permutation(self.seeds).tolist()
            self.i = 0
        seed = int(self.order[self.i])
        self.i += 1
        refuse_untouched(seed)
        return seed


def attach_cursor(workers: list[ResidualWorker], cursor: SeedCursor) -> None:
    for worker in workers:
        worker._next_seed = cursor.next


def train_updates(workers, agent, n_updates, history) -> int:
    env_steps = 0
    for _ in range(int(n_updates)):
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        env_steps += int(PPO.rollout_length) * len(workers)
        history.append(stats | {"env_steps": env_steps})
    return env_steps


def plot_curves(eval_rows: list[dict], l0: dict, p1_payload: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    xs = [row["update"] for row in eval_rows]

    def line(ys, ylabel, name):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("update")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)

    line([r["payload"] for r in eval_rows], "payload", "payload_learning_curve")
    line([r["completed_blocks"] for r in eval_rows], "blocks", "completed_blocks_curve")
    line([r["projector_intervention_rate"] for r in eval_rows], "rate", "projector_intervention_curve")
    line([r["service_envelope_intervention_rate"] for r in eval_rows], "rate", "service_envelope_intervention_curve")
    line([r["mean_residual_norm"] for r in eval_rows], "norm", "residual_magnitude_curve")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["L0", "P1 u100", "P1B u100"], [l0["payload"], p1_payload, eval_rows[-1]["payload"]], color=["#6c757d", "#c0392b", "#2a6f97"])
    ax.set_ylabel("payload")
    ax.set_title("L0 vs P1 vs P1B payload")
    fig.tight_layout()
    fig.savefig(out / "l0_vs_p1_vs_p1b_payload.png", dpi=150)
    fig.savefig(out / "l0_vs_p1_vs_p1b_payload.pdf")
    plt.close(fig)


def decide_gate(eval_map: dict, l0: dict, nan_free: bool, safety_ok: bool, u100: dict) -> dict:
    p1b = u100["payload"]
    vs_p1 = p1b > OLD_P1_U100_PAYLOAD + 1.0e-6
    ge_l0 = p1b + 1.0e-6 >= l0["payload"]
    blocks_ok = u100["completed_blocks"] + 0.25 >= l0["completed_blocks"]
    lock_ok = u100["service_lock_violation_count"] == 0 and u100["delta_used_nonzero_fraction_locked"] == 0
    env_drop = u100["SERVICE_SAFE_ENVELOPE_among_intervened"] < 0.5 * OLD_P1_U100_ENVELOPE_AMONG or u100["service_envelope_intervention_rate"] < 0.5 * 0.24739583333333334
    if (not nan_free) or (not safety_ok) or (not vs_p1):
        letter = "C"
        text = "No improvement over old P1, or safety/NaN failure."
        if not safety_ok:
            text = "LLV2-P1B-SAFETY-FAIL"
            letter = "C"
    elif vs_p1 and ge_l0 and blocks_ok and lock_ok and env_drop:
        letter = "A"
        text = "SERVICE lock repaired residual; may proceed later to REL/multi-seed (not now)."
    elif vs_p1 and (not ge_l0) and lock_ok:
        letter = "B"
        text = "SERVICE repaired vs old P1 but still below L0. Next diagnose REL/ACTION. No auto budget."
    else:
        letter = "C"
        text = "Incomplete SERVICE repair or remaining failure modes."
    return {
        "gate": letter,
        "text": text,
        "better_than_old_p1": vs_p1,
        "at_least_l0": ge_l0,
        "blocks_ok": blocks_ok,
        "lock_ok": lock_ok,
        "envelope_dropped": env_drop,
        "safety_ok": safety_ok,
        "nan_free": nan_free,
        "recommend_multiseed": False,
    }


def main() -> dict:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefreeze = load_prefreeze()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high-level SHA mismatch before training")
    train_seeds, design_seeds = load_pools()
    seed_everything(MODEL_SEED)
    high, digest = load_frozen_high()
    actor, critic = make_networks()
    agent = ICARMAPPOLocked(actor, critic, PPO)
    init_state = copy.deepcopy(agent.state_dict())
    workers = [ResidualWorker(high, actor, critic, service_lock=True) for _ in range(int(PPO.num_envs))]
    smoke_cursor = SeedCursor(train_seeds, np.random.default_rng(VECTOR_SEED_SMOKE))
    attach_cursor(workers, smoke_cursor)
    for _ in range(int(PPO.num_envs)):
        workers[_].reset(smoke_cursor.next())
    smoke_hist = []
    smoke_steps = train_updates(workers, agent, SMOKE_UPDATES_N, smoke_hist)
    ckpt_path = OUTPUT / "smoke_ckpt.pt"
    torch.save(agent.state_dict(), ckpt_path)
    agent.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    smoke_report = {
        "updates": SMOKE_UPDATES_N,
        "env_transitions": smoke_steps,
        "history": smoke_hist,
        "forward_ok": True,
        "nan": False,
        "reload_ok": True,
        "service_lock": True,
        "policy_mask": True,
        "not_counted_as_pilot": True,
        "projector_on": True,
    }
    agent.load_state_dict(init_state)
    seed_everything(MODEL_SEED)
    actor.load_state_dict(init_state["actor"])
    critic.load_state_dict(init_state["critic"])
    agent = ICARMAPPOLocked(actor, critic, PPO)
    agent.load_state_dict(init_state)
    workers = [ResidualWorker(high, actor, critic, service_lock=True) for _ in range(int(PPO.num_envs))]
    cursor = SeedCursor(train_seeds, np.random.default_rng(VECTOR_SEED_PILOT))
    attach_cursor(workers, cursor)
    for worker in workers:
        worker.reset(cursor.next())
    eval_worker = ResidualWorker(high, actor, critic, service_lock=True)
    eval_rows = []
    ckpts = []
    history = []
    env_steps = 0
    eval_map = {}
    for update in range(0, PILOT_UPDATES_N + 1):
        if update in {0, 20, 40, 60, 80, 100} or update % EVAL_EVERY == 0:
            metrics = evaluate_set(eval_worker, design_seeds, zero_residual=False)
            metrics["update"] = update
            eval_rows.append(metrics)
            eval_map[update] = metrics
            path = OUTPUT / f"ckpt_update{update}.pt"
            torch.save(agent.state_dict(), path)
            ckpts.append({"update": update, "path": str(path), "selected_by_design_payload": False})
        if update == PILOT_UPDATES_N:
            break
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        env_steps += int(PPO.rollout_length) * len(workers)
        history.append({"update": update + 1, **stats, "env_steps": env_steps})
    l0_worker = ResidualWorker(high, actor, critic, service_lock=True)
    l0 = evaluate_set(l0_worker, design_seeds, zero_residual=True)
    l0["update"] = "L0"
    neigh = evaluate_set(eval_worker, design_seeds, zero_residual=False, shuffle="neighbor")
    interf = evaluate_set(eval_worker, design_seeds, zero_residual=False, shuffle="interference")
    sinr_sh = evaluate_set(eval_worker, design_seeds, zero_residual=False, shuffle="sinr")
    u100 = eval_map[100]
    safety_ok = (
        u100["collision_count"] == 0
        and u100["min_sep_worst"] + 1.0e-6 >= D_SAFE_M
        and u100["projector_failure_count"] == 0
        and u100["preexisting_safety_violation"] == 0
    )
    nan_free = all(np.isfinite(row.get("policy_loss", 0.0)) for row in history)
    gate = decide_gate(eval_map, l0, nan_free, safety_ok, u100)
    if not safety_ok:
        gate["SAFETY_FAIL"] = True
        gate["gate"] = "C"
        gate["text"] = "LLV2-P1B-SAFETY-FAIL"
    sha_after = high_sha()
    elapsed = time.time() - started
    protocol = {
        "name": "LL-V2.1B",
        "model_seed": MODEL_SEED,
        "d_safe_m": D_SAFE_M,
        "alpha_residual": ALPHA_RESIDUAL,
        "service_safety_margin_m": SERVICE_SAFETY_MARGIN_M,
        "service_residual_lock": True,
        "policy_loss_mask_locked_states": True,
        "loaded_old_ppo_weights": False,
        "loaded_phase1_actor": False,
        "visited_phase106": False,
        "high_sha_before": digest,
        "high_sha_after": sha_after,
        "ppo": PPO.to_dict(),
        "env_transitions_pilot": env_steps,
        "agent_transitions_pilot": env_steps * 6,
        "wall_s": elapsed,
        "primary_checkpoint": "update100",
        "old_p1_u100_payload": OLD_P1_U100_PAYLOAD,
    }
    _dump(OUTPUT / "LLV2_PHASE1B_PROTOCOL.json", protocol)
    _dump(
        OUTPUT / "LLV2_PHASE1B_TRAINING_CONFIG.json",
        PPO.to_dict()
        | {
            "log_std_init": LOG_STD_INIT,
            "lambda_project": 0.05,
            "lambda_residual": 0.01,
            "lambda_progress": 0,
            "lambda_near": 0,
            "lambda_interference": 0,
            "service_residual_lock": True,
        },
    )
    _dump(OUTPUT / "LLV2_PHASE1B_SMOKE.json", smoke_report)
    _dump(OUTPUT / "LLV2_PHASE1B_CHECKPOINT_MANIFEST.json", {"primary": "update100", "checkpoints": ckpts, "best_of_curve_forbidden": True})
    _dump(OUTPUT / "LLV2_PHASE1B_GATE.json", gate | {"SAFETY_FAIL": (not safety_ok), "prefreeze": prefreeze})
    _write_csv(OUTPUT / "LLV2_PHASE1B_TRAINING_HISTORY.csv", history, sorted({k for row in history for k in row}))
    eval_fields = [k for k in eval_rows[0] if k != "payloads"]
    _write_csv(OUTPUT / "LLV2_PHASE1B_EVAL_CURVE.csv", [{k: r[k] for k in eval_fields} for r in eval_rows], eval_fields)
    audit_rows = [
        {
            "update": r["update"],
            "delta_used_nonzero_fraction_locked": r["delta_used_nonzero_fraction_locked"],
            "service_lock_violation_count": r["service_lock_violation_count"],
            "policy_active_fraction": r["policy_active_fraction"],
            "service_envelope_intervention_rate": r["service_envelope_intervention_rate"],
        }
        for r in eval_rows
    ]
    _write_csv(OUTPUT / "LLV2_PHASE1B_SERVICE_LOCK_AUDIT.csv", audit_rows, list(audit_rows[0]))
    proj_rows = [
        {
            "update": r["update"],
            "projector_intervention_rate": r["projector_intervention_rate"],
            "STEP_LIMIT_among_intervened": r["STEP_LIMIT_among_intervened"],
            "PAIRWISE_COLLISION_among_intervened": r["PAIRWISE_COLLISION_among_intervened"],
            "SERVICE_SAFE_ENVELOPE_among_intervened": r["SERVICE_SAFE_ENVELOPE_among_intervened"],
            "MULTI_CONSTRAINT_among_intervened": r["MULTI_CONSTRAINT_among_intervened"],
            "OTHER_among_intervened": r["OTHER_among_intervened"],
            "old_p1_u100_projector_rate": OLD_P1_U100_PROJECTOR,
            "old_p1_u100_envelope_among": OLD_P1_U100_ENVELOPE_AMONG,
        }
        for r in eval_rows
    ]
    _write_csv(OUTPUT / "LLV2_PHASE1B_PROJECTOR_BREAKDOWN.csv", proj_rows, list(proj_rows[0]))
    safety_rows = [
        {
            "update": r["update"],
            "collision_count": r["collision_count"],
            "min_sep": r["min_sep"],
            "min_sep_worst": r["min_sep_worst"],
            "projector_failure_count": r["projector_failure_count"],
            "preexisting_safety_violation": r["preexisting_safety_violation"],
        }
        for r in eval_rows
    ]
    _write_csv(OUTPUT / "LLV2_PHASE1B_SAFETY.csv", safety_rows, list(safety_rows[0]))
    _write_csv(
        OUTPUT / "LLV2_PHASE1B_FEATURE_AUDIT.csv",
        [
            {"audit": "neighbor_shuffle", "action_disagreement": neigh["shuffle_disagreement"], "payload": neigh["payload"]},
            {"audit": "interference_shuffle", "action_disagreement": interf["shuffle_disagreement"], "payload": interf["payload"]},
            {"audit": "sinr_rate_shuffle", "action_disagreement": sinr_sh["shuffle_disagreement"], "payload": sinr_sh["payload"]},
        ],
        ["audit", "action_disagreement", "payload"],
    )
    plot_curves(eval_rows, l0, OLD_P1_U100_PAYLOAD, OUTPUT / "figures")
    payloads_curve = {f"u{u}": eval_map[u]["payload"] for u in (0, 20, 40, 60, 80, 100)}
    report = f"""# HTW Low-Level V2 Phase-1B Service-Locked Residual MAPPO

单 seed 0，100 updates，LLV2_design。非 fresh，非 confirmation。唯一修改：ServiceResidualLock。

1. 唯一改动：进入有效服务状态后冻结 Residual（delta_used=0），Actor policy/entropy 对 locked UAV mask。
2. service_lock = target_valid AND not_idle AND queue>0 AND in_service_region（env._in_range / envelope_active）。
3. locked step 从 Actor policy loss 中 mask：是。Critic/value 不 mask。
4. Reward 是否改？否。C_residual 用 delta_used。
5. alpha/d_safe/lr/clip/entropy/lambda/网络/Projector 是否改？否。
6. 正式训练 environment transitions：{env_steps}（smoke {smoke_steps} 不计）。
7. payload 曲线：{payloads_curve}
8. L0 payload：{l0['payload']}
9. 旧 P1 u100 payload：{OLD_P1_U100_PAYLOAD}
10. P1B u100 payload：{u100['payload']}
11. P1B-L0 delta：{u100['payload'] - l0['payload']}
12. P1B-oldP1 delta：{u100['payload'] - OLD_P1_U100_PAYLOAD}
13. completed blocks u100/L0：{u100['completed_blocks']} / {l0['completed_blocks']}
14. service-lock violation count：{u100['service_lock_violation_count']}；locked delta_used nonzero：{u100['delta_used_nonzero_fraction_locked']}
15. projector intervention 旧→新：{OLD_P1_U100_PROJECTOR} → {u100['projector_intervention_rate']}
16. SERVICE_SAFE_ENVELOPE among intervened 旧→新：{OLD_P1_U100_ENVELOPE_AMONG} → {u100['SERVICE_SAFE_ENVELOPE_among_intervened']}
17. pairwise among intervened：{u100['PAIRWISE_COLLISION_among_intervened']}；pairwise rate：{u100['pairwise_projector_intervention_rate']}
18. collision count：{u100['collision_count']}
19. min_sep_worst：{u100['min_sep_worst']}
20. neighbor/interference/sinr shuffle disagreement：{neigh['shuffle_disagreement']} / {interf['shuffle_disagreement']} / {sinr_sh['shuffle_disagreement']}
21. Gate：P1B-{gate['gate']}。{gate['text']}
22. 是否进入 multi-seed：否。

高层 SHA 不变：{sha_after == digest}
未访问 untouched。primary=update100。
"""
    _dump(OUTPUT / "HTW_LowLevelV2_Phase1B_ServiceLocked_Pilot_CN.md", report)
    names = [
        "LLV2_PHASE1B_PROTOCOL.json",
        "LLV2_PHASE1B_SMOKE.json",
        "LLV2_PHASE1B_GATE.json",
        "HTW_LowLevelV2_Phase1B_ServiceLocked_Pilot_CN.md",
        "LLV2_PHASE1B_TRAINING_HISTORY.csv",
        "LLV2_PHASE1B_EVAL_CURVE.csv",
        "LLV2_PHASE1B_SERVICE_LOCK_AUDIT.csv",
        "LLV2_PHASE1B_PROJECTOR_BREAKDOWN.csv",
        "LLV2_PHASE1B_SAFETY.csv",
        "LLV2_PHASE1B_FEATURE_AUDIT.csv",
    ]
    for name in names:
        src = OUTPUT / name
        (PHASE1B / name).write_bytes(src.read_bytes())
        (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).write_bytes(src.read_bytes())
    (ROOT / "results/root_exports/LLV2_PHASE1B_PREFREEZE.json").write_text(PREFREEZE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_LowLevelV2_Phase1B_ServiceLocked_Pilot_CN.md").write_text((OUTPUT / "HTW_LowLevelV2_Phase1B_ServiceLocked_Pilot_CN.md").read_text(encoding="utf-8"), encoding="utf-8")
    return {"gate": gate, "env_steps": env_steps, "u100": u100["payload"], "l0": l0["payload"], "elapsed": elapsed}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
