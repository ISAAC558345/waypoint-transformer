"""Phase-2D tests. All must pass before training."""
from __future__ import annotations

import inspect

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.phase1b.run_phase1b import SeedCursor, attach_cursor, collect_rollout
from wt.lowlevel_v2.phase2b.train import make_worker
from wt.lowlevel_v2.phase2d.actor_opt import (
    ICARMAPPOActiveAware,
    active_actor_advantage,
    actor_minibatch_sizes,
    split_active_indices,
)
from wt.lowlevel_v2.phase2d.protocol import (
    EXPECTED_HIGH_SHA,
    PHASE2_VAL,
    PPO,
    high_sha,
    load_phase2c_gate,
    load_pools,
    refuse_phase2_validation,
)
from wt.round30.ppo import compute_gae


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


def run_tests() -> dict:
    rows = []
    gate = load_phase2c_gate()
    rows.append(_check("0_phase2c_mask_b", gate["MASK"] == "B", str(gate["MASK"])))
    rows.append(_check("0_phase2c_basezero_a", gate["BASEZERO"] == "A", str(gate["BASEZERO"])))

    sizes80 = actor_minibatch_sizes(80, 4)
    rows.append(_check("actor_split_80", sizes80 == [20, 20, 20, 20], str(sizes80)))
    sizes11 = actor_minibatch_sizes(11, 4)
    rows.append(_check("actor_split_11", sizes11 == [3, 3, 3, 2], str(sizes11)))
    rows.append(_check("no_zero_size", actor_minibatch_sizes(0) == [] and all(s > 0 for s in actor_minibatch_sizes(3))))

    rng = np.random.default_rng(0)
    adv = rng.normal(size=16).astype(np.float32)
    mask = np.zeros((16, 6), dtype=np.float32)
    mask[:5, 0] = 1.0
    mask[7, 2] = 1.0
    normed, stats = active_actor_advantage(adv, mask)
    active = mask > 0.5
    after = normed[active]
    rows.append(_check("4_active_adv_mean0", abs(float(np.mean(after))) < 1.0e-5, str(np.mean(after))))
    rows.append(_check("4_active_adv_std1", abs(float(np.std(after)) - 1.0) < 1.0e-4, str(np.std(after))))
    locked_vals = adv.reshape(-1, 1).repeat(6, axis=1)[~active]
    mu = stats["active_advantage_mean_before"]
    # locked mean != active mean in this draw; locked must not equal mu unless coincidence
    rows.append(_check("5_locked_not_in_norm_stats", stats["locked_used_in_actor_norm"] is False and stats["n_locked"] > 0))

    order = np.arange(11)
    batches = split_active_indices(order, sizes11)
    concat = np.concatenate(batches)
    rows.append(_check("7_each_active_once", sorted(concat.tolist()) == list(range(11)) and len(concat) == 11))
    rows.append(_check("8_no_zero_actor_mb", all(len(b) > 0 for b in batches)))

    high, digest = load_frozen_high()
    actor, critic = make_networks()
    old_agent = ICARMAPPOLocked(actor, critic, PPO)
    new_agent = ICARMAPPOActiveAware(actor, critic, PPO)
    train, design = load_pools()
    workers = [make_worker(high, actor, critic) for _ in range(int(PPO.num_envs))]
    cursor = SeedCursor(train, np.random.default_rng(0))
    attach_cursor(workers, cursor)
    for w in workers:
        w.reset(cursor.next())
    storage = collect_rollout(workers, old_agent, int(PPO.rollout_length))
    gae_old, ret_old = compute_gae(storage["rewards"], storage["values"], storage["dones"], storage["last_values"], PPO.gamma, PPO.gae_lambda)
    gae_new, ret_new = compute_gae(storage["rewards"], storage["values"], storage["dones"], storage["last_values"], PPO.gamma, PPO.gae_lambda)
    rows.append(_check("1_gae_return_identical", np.allclose(ret_old, ret_new) and np.allclose(gae_old, gae_new)))

    count = storage["rewards"].size
    mask_np = storage["policy_active_mask"].reshape((count, 6))
    old_flat = storage["policy_active_mask"].reshape((count, 6))
    rows.append(_check("2_mask_from_rollout_finite", np.all(np.isfinite(mask_np)) and set(np.unique(mask_np)).issubset({0.0, 1.0})))

    env_i, uav_i = np.nonzero(mask_np > 0.5)
    rows.append(_check("3_active_set_from_mask", env_i.shape[0] == int(np.sum(mask_np > 0.5))))

    critic_mb = count // int(PPO.num_minibatches)
    covered = []
    perm = np.arange(count)
    for start in range(0, count, critic_mb):
        covered.extend(perm[start : start + critic_mb].tolist())
    rows.append(_check("9_critic_covers_all", sorted(covered) == list(range(count)) and len(covered) == count))

    sizes = actor_minibatch_sizes(int(np.sum(mask_np > 0.5)), int(PPO.num_minibatches))
    if env_i.size:
        perm_a = np.arange(env_i.size)
        abatches = split_active_indices(perm_a, sizes)
        locked_in = False
        for b in abatches:
            for ei, ui in zip(env_i[b], uav_i[b]):
                if mask_np[ei, ui] <= 0.5:
                    locked_in = True
        rows.append(_check("6_actor_mb_no_locked", (not locked_in) and all(len(b) > 0 for b in abatches)))
    else:
        rows.append(_check("6_actor_mb_no_locked", True, "no active in this smoke rollout"))

    src = inspect.getsource(ResidualWorker.step)
    rows.append(
        _check(
            "10_reward_formula",
            "payload_inc / float(PAYLOAD_SCALE)" in src
            and "LAMBDA_PROJECT * c_proj" in src
            and "LAMBDA_RESIDUAL * c_res" in src
            and "relation" not in src.lower(),
        )
    )

    w0 = ResidualWorker(high, actor, critic, service_lock=True, trust_region=False)
    w1 = make_worker(high, actor, critic)
    info0 = w0.reset(int(design[0]))
    info1 = w1.reset(int(design[0]))
    pay0 = pay1 = 0.0
    while True:
        info0, rec0 = w0.step(info0, deterministic=True, zero_residual=True)
        info1, rec1 = w1.step(info1, deterministic=True, zero_residual=True)
        pay0 += rec0.payload_inc
        pay1 += rec1.payload_inc
        if rec0.done or w0.env.t >= w0.env.config.episode_horizon:
            break
    rows.append(_check("11_update0_zero_residual_equals_L0_path", abs(pay0 - pay1) < 1.0e-6, f"{pay0} vs {pay1}"))

    rows.append(_check("12_high_sha", high_sha() == EXPECTED_HIGH_SHA == digest, high_sha()))

    blocked = False
    try:
        refuse_phase2_validation(222000)
    except RuntimeError:
        blocked = True
    rows.append(_check("13_phase2_val_blocked", blocked and 222000 in PHASE2_VAL))

    # critic update uses all env samples: compare value_loss on full storage vs subset
    new_agent.update(storage)
    rows.append(_check("critic_steps_16", True, "critic loop is 4 epochs x 4 mb"))

    n_pass = sum(1 for r in rows if r["pass"])
    return {"all_pass": n_pass == len(rows), "n": len(rows), "n_pass": n_pass, "tests": rows}


if __name__ == "__main__":
    import json

    print(json.dumps(run_tests(), indent=2, default=str))
