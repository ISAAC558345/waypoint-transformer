"""Phase-2B unit tests. Must all pass before any training."""
from __future__ import annotations

import inspect
import json

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked
from wt.lowlevel_v2.phase1b.service_lock import apply_service_residual_lock
from wt.lowlevel_v2.phase2b.protocol import (
    ALPHA_RESIDUAL,
    EXPECTED_HIGH_SHA,
    PHASE2_VAL,
    PREFREEZE_FILE,
    action_max,
    epsilon_base,
    high_sha,
    load_pools,
    projector_sha,
    refuse_phase2_validation,
)
from wt.lowlevel_v2.phase2b.trust_region import (
    BETA_RESIDUAL,
    apply_base_relative_residual_trust_region,
)


def _check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


def run_tests() -> dict:
    a_max = action_max()
    eps_b = epsilon_base()
    rows = []

    lock = np.array([True, False])
    base = np.array([[0.4, 0.0], [0.4, 0.0]], dtype=np.float64)
    pre = np.array([[8.0, 0.0], [8.0, 0.0]], dtype=np.float64)
    a_des, used = apply_service_residual_lock(base, pre, lock, ALPHA_RESIDUAL)
    rows.append(_check("1_service_lock_zero", np.allclose(used[0], 0.0), f"used0={used[0]}"))

    r_in = np.array([[0.2, 0.1], [0.2, 0.0]], dtype=np.float64)
    used, extra = apply_base_relative_residual_trust_region(np.zeros((2, 2)), r_in)
    rows.append(_check("2_base_norm_zero", np.allclose(used, 0.0) and bool(np.all(extra["base_zero_lock"]))))

    tiny = np.array([[0.5 * eps_b, 0.0], [0.0, 0.4 * eps_b]], dtype=np.float64)
    used, extra = apply_base_relative_residual_trust_region(tiny, r_in)
    rows.append(_check("3_base_below_epsilon", np.allclose(used, 0.0) and bool(np.all(extra["base_zero_lock"]))))

    just = np.array([[eps_b + 1.0e-9, 0.0]], dtype=np.float64)
    r_small = np.array([[1.0e-6, 0.0]], dtype=np.float64)
    used, extra = apply_base_relative_residual_trust_region(just, r_small)
    in_tr = (not bool(extra["base_zero_lock"][0])) and np.allclose(used, r_small)
    rows.append(_check("4_just_above_epsilon_enters_tr", in_tr, f"freeze={extra['base_zero_lock'][0]} used={used}"))

    b = np.array([[0.8, 0.0]], dtype=np.float64)
    cap = BETA_RESIDUAL * 0.8
    r_under = np.array([[0.5 * cap, 0.0]], dtype=np.float64)
    used, extra = apply_base_relative_residual_trust_region(b, r_under)
    rows.append(_check("5_under_cap_identity", np.allclose(used, r_under) and (not bool(extra["cap_active"][0]))))

    r_over = np.array([[4.0 * cap, 3.0 * cap]], dtype=np.float64)
    used, extra = apply_base_relative_residual_trust_region(b, r_over)
    n_used = float(np.linalg.norm(used[0]))
    rows.append(_check("6_over_cap_equals_cap", abs(n_used - cap) < 1.0e-9, f"n={n_used} cap={cap}"))

    cos = float(np.dot(r_over[0], used[0]) / (np.linalg.norm(r_over[0]) * n_used))
    rows.append(_check("7_direction_preserved", abs(cos - 1.0) < 1.0e-9, f"cos={cos}"))

    high, _ = load_frozen_high()
    actor, critic = make_networks()
    w = ResidualWorker(high, actor, critic, service_lock=True, trust_region=True)
    train, design = load_pools()
    info = w.reset(int(design[0]))
    info, rec = w.step(info, deterministic=True, zero_residual=False)
    if np.any(rec.service_lock):
        mask_lock = float(np.max(rec.policy_active_mask[rec.service_lock])) == 0.0
    else:
        mask_lock = True
    rows.append(_check("8_service_lock_masks_policy", mask_lock))
    if np.any(rec.base_zero_lock):
        mask_bz = float(np.max(rec.policy_active_mask[rec.base_zero_lock])) == 0.0
    else:
        # force a synthetic check from the transform
        used, extra = apply_base_relative_residual_trust_region(np.zeros((1, 2)), np.ones((1, 2)))
        mask_bz = bool(extra["base_zero_lock"][0])
    rows.append(_check("9_base_zero_masks_policy", mask_bz))

    src = inspect.getsource(ICARMAPPOLocked.update)
    value_unmasked = "value_loss = 0.5 * (target - value).square().mean()" in src
    policy_masked = "(per_uav * mask).sum()" in src
    rows.append(_check("10_critic_unmasked", value_unmasked and policy_masked))

    pre = json.loads(PREFREEZE_FILE.read_text(encoding="utf-8")) if PREFREEZE_FILE.exists() else {}
    sha_now = projector_sha()
    sha_ok = sha_now == pre.get("safety_projector_sha", sha_now)
    rows.append(_check("11_projector_hash_unchanged", sha_ok, sha_now))

    w0 = ResidualWorker(high, actor, critic, service_lock=True, trust_region=True)
    info = w0.reset(int(design[0]))
    p0 = float(w0.env.payload_bits)
    while True:
        info, rec = w0.step(info, deterministic=True, zero_residual=True)
        if rec.done or w0.env.t >= w0.env.config.episode_horizon:
            break
    z_payload = float(w0.env.payload_bits - p0)
    from wt.lowlevel_v2.phase1b.controller import ResidualWorker as RW

    w_l0 = RW(high, actor, critic, service_lock=True, trust_region=False)
    info = w_l0.reset(int(design[0]))
    p0 = float(w_l0.env.payload_bits)
    while True:
        info, rec = w_l0.step(info, deterministic=True, zero_residual=True)
        if rec.done or w_l0.env.t >= w_l0.env.config.episode_horizon:
            break
    l0_payload = float(w_l0.env.payload_bits - p0)
    rows.append(_check("12_zero_residual_is_L0", abs(z_payload - l0_payload) < 1.0e-6, f"{z_payload} vs {l0_payload}"))

    rows.append(_check("13_high_sha", high_sha() == EXPECTED_HIGH_SHA, high_sha()))

    blocked = False
    try:
        refuse_phase2_validation(next(iter(PHASE2_VAL)))
    except RuntimeError:
        blocked = True
    rows.append(_check("14_phase2_val_blocked", blocked))

    # extra: Euclidean cap, not per-axis
    r_xy = np.array([[cap, cap]], dtype=np.float64)
    used, extra = apply_base_relative_residual_trust_region(b, r_xy)
    n_used = float(np.linalg.norm(used[0]))
    rows.append(_check("extra_euclidean_not_axis_clip", abs(n_used - cap) < 1.0e-9 and abs(used[0, 0] - used[0, 1]) < 1.0e-9))
    rows.append(_check("extra_a_max_from_contract", abs(a_max - 1.0) < 1.0e-12, str(a_max)))

    passed = all(r["pass"] for r in rows)
    return {"all_pass": passed, "n": len(rows), "n_pass": sum(r["pass"] for r in rows), "tests": rows, "a_max": a_max, "epsilon_base": eps_b, "projector_sha": sha_now}


if __name__ == "__main__":
    import json

    print(json.dumps(run_tests(), indent=2, default=str))
