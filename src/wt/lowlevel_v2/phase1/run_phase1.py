"""LL-V2.1 single-seed residual MAPPO pilot. No Phase-10.6 access. No FINAL-HIER edits."""
from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.protocol import EXPECTED_HIGH_SHA
from wt.round30.ppo import seed_everything

from .audit import audit_initial_geometry
from .controller import ResidualWorker, load_frozen_high, make_networks
from .mappo import ICARMAPPO
from .protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    EVAL_EVERY,
    FINAL_CKPT,
    LOG_STD_INIT,
    MODEL_SEED,
    OUTPUT,
    PHASE1,
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def measure_init_rms(worker: ResidualWorker) -> dict:
    worker.reset(220000)
    info = worker.env.info() if worker.env.t == 0 else worker.env.info()
    worker._refresh_option(info)
    obs = worker.step(info, deterministic=True, zero_residual=True)[0]
    _ = obs
    # rebuild obs at current state
    from wt.lowlevel_v2.observation import build_actor_observation

    worker.reset(220000)
    info = worker.env.info()
    worker._refresh_option(info)
    obs = build_actor_observation(worker.env, worker.option, worker.last_movement)
    worker.actor.eval()
    rms = []
    with torch.no_grad():
        for _ in range(64):
            pre, _ = worker.act_tensors(obs, deterministic=False)
            scaled = float(ALPHA_RESIDUAL) * torch.tanh(pre)
            rms.append(float(torch.sqrt(torch.mean(scaled.square()))))
    return {
        "log_std_init": LOG_STD_INIT,
        "measured_residual_rms": float(np.mean(rms)),
        "limit": 0.05,
        "ok": float(np.mean(rms)) <= 0.05 + 1.0e-9,
    }


def collect_rollout(workers: list[ResidualWorker], agent: ICARMAPPO, horizon: int) -> dict[str, np.ndarray]:
    n_env = len(workers)
    keys = {
        "self": (horizon, n_env, 6, 5),
        "task": (horizon, n_env, 6, 8),
        "communication": (horizon, n_env, 6, 4),
        "neighbor_set": (horizon, n_env, 6, 5, 11),
        "d_nearest": (horizon, n_env, 6),
        "global_state": (horizon, n_env, expected_critic_dim(6, 12)),
        "pre_tanh": (horizon, n_env, 6, 2),
        "log_prob": (horizon, n_env),
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
            storage["log_prob"][t, e] = rec.log_prob
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
    payloads = []
    blocks = []
    queues = []
    drops = []
    minseps = []
    collisions = []
    nears = []
    inter_rates = []
    inter_norms = []
    failures = []
    preexisting = []
    sinrs = []
    rates = []
    flights = []
    res_norms = []
    smooths = []
    unfinished = []
    disagreements = []
    for seed in seeds:
        refuse_untouched(int(seed))
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        flight = 0.0
        inter_n = 0
        fail_n = 0
        pre_n = 0
        coll = 0
        near = 0
        mins = []
        res = []
        prev_exec = None
        smooth = []
        sinr_acc = []
        rate_acc = []
        step_n = 0
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
            if shuffle and not rec.preexisting_safety_violation:
                obs = rec.obs
                shuffled = {key: np.array(val, copy=True) if isinstance(val, np.ndarray) else val for key, val in obs.items()}
                if shuffle == "neighbor":
                    rng = np.random.default_rng(int(seed) + 17)
                    perm = rng.permutation(shuffled["neighbor_set"].shape[1])
                    shuffled["neighbor_set"] = shuffled["neighbor_set"][:, perm]
                elif shuffle == "interference":
                    rng = np.random.default_rng(int(seed) + 31)
                    shuffled["communication"] = shuffled["communication"].copy()
                    shuffled["communication"][:, 2] = rng.permutation(shuffled["communication"][:, 2])
                    shuffled["communication"][:, 3] = rng.permutation(shuffled["communication"][:, 3])
                with torch.no_grad():
                    pre_s, _ = worker.act_tensors(shuffled, deterministic=True)
                    pre_o = torch.as_tensor(rec.pre_tanh)
                    disagreements.append(float(torch.mean(torch.abs(pre_s - pre_o))))
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
            res.append(float(np.mean(np.linalg.norm(rec.delta_tanh, axis=-1))))
            if prev_exec is not None:
                smooth.append(float(np.mean(np.linalg.norm(rec.a_exec - prev_exec, axis=-1))))
            prev_exec = rec.a_exec
            sinr_acc.append(float(np.mean(worker.env.last_sinr)))
            rate_acc.append(float(np.mean(worker.env.last_rate)))
            if rec.done:
                break
            if worker.env.t >= worker.env.config.episode_horizon:
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
        failures.append(fail_n)
        preexisting.append(pre_n)
        sinrs.append(float(np.mean(sinr_acc) if sinr_acc else 0.0))
        rates.append(float(np.mean(rate_acc) if rate_acc else 0.0))
        flights.append(flight)
        res_norms.append(float(np.mean(res) if res else 0.0))
        smooths.append(float(np.mean(smooth) if smooth else 0.0))
        unfinished.append(float(np.mean(worker.env.progress)))
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
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
        "projector_failure_count": float(np.sum(failures)),
        "preexisting_safety_violation": float(np.sum(preexisting)),
        "mean_sinr": mean(sinrs),
        "mean_active_rate": mean(rates),
        "flight_distance": mean(flights),
        "mean_residual_norm": mean(res_norms),
        "smoothness": mean(smooths),
        "unfinished_progress": mean(unfinished),
        "n": len(seeds),
        "shuffle_disagreement": mean(disagreements) if disagreements else 0.0,
        "payloads": payloads,
    }


def compare_update0_to_l0(worker: ResidualWorker, seed: int) -> dict:
    refuse_untouched(seed)
    info_a = worker.reset(seed)
    recs_a = []
    while True:
        info_a, rec = worker.step(info_a, deterministic=True, zero_residual=True)
        recs_a.append(rec)
        if rec.done or worker.env.t >= 48:
            break
    payload_l0 = float(worker.env.payload_bits)
    assigns_l0 = [r.a_base.copy() for r in recs_a]
    exec_l0 = [r.a_exec.copy() for r in recs_a]
    info_b = worker.reset(seed)
    recs_b = []
    while True:
        info_b, rec = worker.step(info_b, deterministic=True, zero_residual=False)
        recs_b.append(rec)
        if rec.done or worker.env.t >= 48:
            break
    payload_u0 = float(worker.env.payload_bits)
    ok = True
    reasons = []
    if len(recs_a) != len(recs_b):
        ok = False
        reasons.append("length")
    for a, b in zip(recs_a, recs_b):
        if not np.allclose(a.a_base, b.a_base, atol=1.0e-7):
            ok = False
            reasons.append("base")
            break
        if not np.allclose(a.a_exec, b.a_exec, atol=1.0e-6):
            ok = False
            reasons.append("exec")
            break
        if float(np.max(np.abs(b.delta_tanh))) > 1.0e-5:
            ok = False
            reasons.append("delta")
            break
    if abs(payload_l0 - payload_u0) > 1.0e-6:
        ok = False
        reasons.append("payload")
    return {"ok": ok, "reasons": reasons, "payload_l0": payload_l0, "payload_u0": payload_u0, "n": len(recs_a)}


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


def train_updates(workers, agent, n_updates, history, cursor_rng_note) -> int:
    _ = cursor_rng_note
    env_steps = 0
    for _ in range(int(n_updates)):
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        stats = agent.update(storage)
        env_steps += int(PPO.rollout_length) * len(workers)
        history.append(stats | {"env_steps": env_steps})
        if not all(np.isfinite(v) for k, v in stats.items() if isinstance(v, float)):
            raise RuntimeError("non-finite PPO stats")
    return env_steps


def plot_curves(eval_rows: list[dict], l0: dict, out: Path) -> None:
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
    line([r["min_sep"] for r in eval_rows], "m", "min_separation_curve")
    line([r["mean_residual_norm"] for r in eval_rows], "norm", "residual_magnitude_curve")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["L0", "ICAR-MAPPO u100"], [l0["payload"], eval_rows[-1]["payload"]], color=["#6c757d", "#2a6f97"])
    ax.set_ylabel("payload")
    ax.set_title("L0 vs ICAR-MAPPO payload")
    fig.tight_layout()
    fig.savefig(out / "l0_vs_icar_payload.png", dpi=150)
    fig.savefig(out / "l0_vs_icar_payload.pdf")
    plt.close(fig)


def decide_gate(eval_map: dict[int, dict], l0: dict, nan_free: bool, safety_ok: bool) -> dict:
    u0 = eval_map[0]
    u100 = eval_map[100]
    learned = u100["payload"] > u0["payload"] + 1.0e-6
    ge_l0 = u100["payload"] + 1.0e-6 >= l0["payload"]
    blocks_ok = u100["completed_blocks"] + 0.25 >= min(l0["completed_blocks"], u0["completed_blocks"])
    shield_high = u100["projector_intervention_rate"] >= 0.99
    if (not nan_free) or (not safety_ok):
        letter = "C"
        text = "Safety failed or non-finite training."
    elif learned and ge_l0 and blocks_ok and not shield_high:
        letter = "A"
        text = "Residual MAPPO worth multi-seed validation."
    elif learned and (not ge_l0) and safety_ok:
        letter = "B"
        text = "Learned vs update0 but still below L0 at 100 updates. Do not auto-extend budget."
    else:
        letter = "C"
        text = "No useful learning trend, shield takeover, or optimizer issue."
    return {
        "gate": letter,
        "text": text,
        "learned_vs_update0": learned,
        "at_least_l0": ge_l0,
        "blocks_ok": blocks_ok,
        "shield_dependence_high": shield_high,
        "safety_ok": safety_ok,
        "nan_free": nan_free,
        "recommend_multiseed": letter == "A",
    }


def main() -> dict:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefreeze = load_prefreeze()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high-level SHA mismatch before training")
    audit = audit_initial_geometry()
    _dump(OUTPUT / "LLV2_PHASE1_GEOMETRY_AUDIT.json", {k: v for k, v in audit.items() if k != "rows"})
    if not audit["ok"]:
        raise RuntimeError("infeasible_initial_geometry")
    train_seeds, design_seeds = load_pools()
    seed_everything(MODEL_SEED)
    high, digest = load_frozen_high()
    actor, critic = make_networks()
    agent = ICARMAPPO(actor, critic, PPO)
    init_state = copy.deepcopy(agent.state_dict())
    workers = [ResidualWorker(high, actor, critic) for _ in range(int(PPO.num_envs))]
    smoke_cursor = SeedCursor(train_seeds, np.random.default_rng(VECTOR_SEED_SMOKE))
    attach_cursor(workers, smoke_cursor)
    for worker, _ in zip(workers, range(int(PPO.num_envs))):
        worker.reset(smoke_cursor.next())
    rms = measure_init_rms(workers[0])
    if not rms["ok"]:
        raise RuntimeError(f"initial residual RMS {rms['measured_residual_rms']} exceeds 5%")
    eq = compare_update0_to_l0(workers[0], int(design_seeds[0]))
    if not eq["ok"]:
        raise RuntimeError(f"update0 vs L0 mismatch: {eq}")
    smoke_hist = []
    smoke_steps = train_updates(workers, agent, SMOKE_UPDATES_N, smoke_hist, "smoke")
    smoke_report = {
        "updates": SMOKE_UPDATES_N,
        "env_transitions": smoke_steps,
        "history": smoke_hist,
        "forward_ok": True,
        "nan": False,
        "reload_after_smoke": True,
        "not_counted_as_pilot": True,
        "init_rms": rms,
        "update0_vs_l0": eq,
    }
    agent.load_state_dict(init_state)
    if high_sha() != digest:
        raise RuntimeError("high-level SHA changed during smoke")
    seed_everything(MODEL_SEED)
    actor.load_state_dict(init_state["actor"])
    critic.load_state_dict(init_state["critic"])
    agent = ICARMAPPO(actor, critic, PPO)
    agent.load_state_dict(init_state)
    workers = [ResidualWorker(high, actor, critic) for _ in range(int(PPO.num_envs))]
    cursor = SeedCursor(train_seeds, np.random.default_rng(VECTOR_SEED_PILOT))
    attach_cursor(workers, cursor)
    for worker in workers:
        worker.reset(cursor.next())
    eval_worker = ResidualWorker(high, actor, critic)
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
    l0 = evaluate_set(eval_worker, design_seeds, zero_residual=True)
    l0["update"] = "L0"
    final = evaluate_set(eval_worker, design_seeds, zero_residual=False)
    neigh = evaluate_set(eval_worker, design_seeds, zero_residual=False, shuffle="neighbor")
    interf = evaluate_set(eval_worker, design_seeds, zero_residual=False, shuffle="interference")
    u100 = eval_map[100]
    safety_ok = (
        u100["collision_count"] == 0
        and u100["min_sep_worst"] + 1.0e-6 >= D_SAFE_M
        and u100["projector_failure_count"] == 0
        and u100["preexisting_safety_violation"] == 0
    )
    nan_free = all(np.isfinite(row.get("policy_loss", 0.0)) for row in history)
    gate = decide_gate(eval_map, l0, nan_free, safety_ok)
    sha_after = high_sha()
    elapsed = time.time() - started
    protocol = {
        "name": "LL-V2.1",
        "model_seed": MODEL_SEED,
        "d_safe_m": D_SAFE_M,
        "alpha_residual": ALPHA_RESIDUAL,
        "service_safety_margin_m": SERVICE_SAFETY_MARGIN_M,
        "loaded_old_ppo_weights": False,
        "visited_phase106": False,
        "high_sha_before": digest,
        "high_sha_after": sha_after,
        "ppo": PPO.to_dict(),
        "mappo_difference": "separate ResidualActor + CentralizedCritic 188-d; joint tanh log-prob over 6x2; log_prob of pre-projection residual only",
        "env_transitions_pilot": env_steps,
        "agent_transitions_pilot": env_steps * 6,
        "wall_s": elapsed,
        "logprob_uses_projected_action": False,
    }
    _dump(OUTPUT / "LLV2_PHASE1_PROTOCOL.json", protocol)
    _dump(OUTPUT / "LLV2_PHASE1_TRAINING_CONFIG.json", PPO.to_dict() | {"log_std_init": LOG_STD_INIT, "lambda_project": 0.05, "lambda_residual": 0.01, "lambda_progress": 0, "lambda_near": 0, "lambda_interference": 0})
    _dump(OUTPUT / "LLV2_PHASE1_SMOKE_REPORT.json", smoke_report)
    _dump(OUTPUT / "LLV2_PHASE1_CHECKPOINT_MANIFEST.json", {"primary": "update100", "checkpoints": ckpts, "best_of_curve_forbidden": True})
    _dump(OUTPUT / "LLV2_PHASE1_PILOT_GATE.json", gate | {"SAFETY_FAIL": (not safety_ok)})
    _write_csv(OUTPUT / "LLV2_PHASE1_TRAINING_HISTORY.csv", history, sorted({k for row in history for k in row}))
    eval_fields = [k for k in eval_rows[0] if k != "payloads"]
    _write_csv(OUTPUT / "LLV2_PHASE1_EVAL_CURVE.csv", [{k: r[k] for k in eval_fields} for r in eval_rows], eval_fields)
    design_row = {k: u100[k] for k in eval_fields if k != "update"}
    design_row["method"] = "ICAR-MAPPO-u100"
    l0_row = {k: l0[k] for k in eval_fields if k in l0}
    l0_row["method"] = "L0"
    _write_csv(OUTPUT / "LLV2_PHASE1_DESIGN_METRICS.csv", [l0_row, design_row], ["method"] + [k for k in eval_fields if k != "update"])
    safety_rows = [{"update": r["update"], "collision_count": r["collision_count"], "min_sep": r["min_sep"], "min_sep_worst": r["min_sep_worst"], "projector_intervention_rate": r["projector_intervention_rate"], "projector_failure_count": r["projector_failure_count"], "preexisting_safety_violation": r["preexisting_safety_violation"]} for r in eval_rows]
    _write_csv(OUTPUT / "LLV2_PHASE1_SAFETY_METRICS.csv", safety_rows, list(safety_rows[0]))
    _write_csv(
        OUTPUT / "LLV2_PHASE1_COMM_FEATURE_AUDIT.csv",
        [
            {"audit": "neighbor_shuffle", "action_disagreement": neigh["shuffle_disagreement"], "payload": neigh["payload"], "design_only": True},
            {"audit": "interference_shuffle", "action_disagreement": interf["shuffle_disagreement"], "payload": interf["payload"], "design_only": True},
        ],
        ["audit", "action_disagreement", "payload", "design_only"],
    )
    plot_curves(eval_rows, l0, OUTPUT / "figures")
    payloads_curve = {f"u{u}": eval_map[u]["payload"] for u in (0, 20, 40, 60, 80, 100)}
    report = f"""# HTW Low-Level V2 Phase-1 Pilot Report

单 seed 0，100 updates，LLV2_design 验证。非 confirmation，非 final untouched。未改 FINAL-HIER。未访问 106000–106063。

1. d_safe 冻结：{D_SAFE_M} m（仿真中心距，非真实机身直径）
2. alpha_residual：{ALPHA_RESIDUAL}
3. ServiceSafeEnvelope margin：{SERVICE_SAFETY_MARGIN_M} m
4. MAPPO：gamma={PPO.gamma}, gae_lambda={PPO.gae_lambda}, clip={PPO.clip_eps}, lr={PPO.learning_rate}, value_coef={PPO.value_coef}, entropy={PPO.entropy_coef}, max_grad={PPO.max_grad_norm}, rollout={PPO.rollout_length}, num_envs={PPO.num_envs}, minibatches={PPO.num_minibatches}, update_epochs={PPO.update_epochs}, 来自 Phase-10.1 PPOConfig。log_std_init={LOG_STD_INIT}。
5. 是否从旧 PPO 加载权重？**否。**
6. Actor：self, task, communication(SINR/rate/interference/INR), neighbor embedding, d_nearest（hold 与剩余时间在 task 内）。
7. Critic：188 维 global state。
8. Reward：r = delta_payload_bits/1e6 - 0.05*C_project - 0.01*C_residual；lambda_progress=lambda_near=lambda_interference=0。
9. environment transitions：{env_steps}
10. agent transitions：{env_steps * 6}
11. 训练耗时：{elapsed:.1f}s（含 design eval）
12. update0 payload：{eval_map[0]['payload']}
13. update 20/40/60/80/100 payload：{payloads_curve}
14. L0 payload：{l0['payload']}
15. update100 vs L0 delta：{u100['payload'] - l0['payload']}
16. completed blocks u0/u100/L0：{eval_map[0]['completed_blocks']} / {u100['completed_blocks']} / {l0['completed_blocks']}
17. collision count (u100)：{u100['collision_count']}
18. minimum separation worst (u100)：{u100['min_sep_worst']}
19. projector intervention rate u0/u100：{eval_map[0]['projector_intervention_rate']} / {u100['projector_intervention_rate']}
20. projector failure (u100)：{u100['projector_failure_count']}
21. neighbor shuffle disagreement：{neigh['shuffle_disagreement']}；interference shuffle：{interf['shuffle_disagreement']}
22. 高层 SHA 不变：{sha_after == digest} `{sha_after}`
23. 是否访问旧 final untouched？**否。**
24. Gate：LLV2-PILOT-{gate['gate']}。{gate['text']} SAFETY_FAIL={not safety_ok} shield_dependence_high={gate['shield_dependence_high']}
25. 是否建议 LL-V2.2 多 seed：{'是' if gate['recommend_multiseed'] else '否（本阶段不自动执行）'}

init residual RMS={rms['measured_residual_rms']} (limit 0.05)
smoke env transitions={smoke_steps}（不计入正式 pilot）
primary checkpoint=update100，禁止按 design 曲线选点。
"""
    _dump(OUTPUT / "HTW_LowLevelV2_Phase1_Pilot_Report_CN.md", report)
    _dump(PHASE1 / "LLV2_PHASE1_PREFREEZE.json", json.loads(PREFREEZE_FILE.read_text(encoding="utf-8")))
    for name in (
        "LLV2_PHASE1_PROTOCOL.json",
        "LLV2_PHASE1_TRAINING_CONFIG.json",
        "LLV2_PHASE1_SMOKE_REPORT.json",
        "LLV2_PHASE1_CHECKPOINT_MANIFEST.json",
        "LLV2_PHASE1_PILOT_GATE.json",
        "HTW_LowLevelV2_Phase1_Pilot_Report_CN.md",
    ):
        src = OUTPUT / name
        (PHASE1 / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    for name in (
        "LLV2_PHASE1_TRAINING_HISTORY.csv",
        "LLV2_PHASE1_EVAL_CURVE.csv",
        "LLV2_PHASE1_DESIGN_METRICS.csv",
        "LLV2_PHASE1_SAFETY_METRICS.csv",
        "LLV2_PHASE1_COMM_FEATURE_AUDIT.csv",
    ):
        (PHASE1 / name).write_bytes((OUTPUT / name).read_bytes())
        (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).write_bytes((OUTPUT / name).read_bytes())
    (ROOT / "results/root_exports/LLV2_PHASE1_PREFREEZE.json").write_text(PREFREEZE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_LowLevelV2_Phase1_Pilot_Report_CN.md").write_text((OUTPUT / "HTW_LowLevelV2_Phase1_Pilot_Report_CN.md").read_text(encoding="utf-8"), encoding="utf-8")
    _ = final
    return {"gate": gate, "elapsed": elapsed, "env_steps": env_steps, "u100": u100["payload"], "l0": l0["payload"]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
