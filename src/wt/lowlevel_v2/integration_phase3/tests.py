"""Phase-3 pretests. No optimizer.step. No low-level training."""
from __future__ import annotations

import numpy as np
import torch

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.utility_targets import soft_targets
from wt.lowlevel_v2.integration_phase2b.abs_loss import utility_losses_abs
from wt.round30.ppo import seed_everything

from .dataset import load_exec_split
from .exec_loss import executor_targets, utility_losses_exec


def q_construction_tests() -> dict:
    pos = np.array([2.0, 6.0, 4.0], dtype=np.float64)
    q_pos = executor_targets(pos)
    mixed = np.array([-1.0, 0.0, 3.0], dtype=np.float64)
    q_mix = executor_targets(mixed)
    tau_q = soft_targets(pos, 1.0)
    return {
        "pass": bool(
            abs(float(q_pos.sum()) - 1.0) < 1.0e-8
            and abs(float(q_mix.sum()) - 1.0) < 1.0e-8
            and np.all(q_mix > 0)
            and float(np.max(np.abs(q_pos - pos / pos.sum()))) < 1.0e-12
            and float(np.max(np.abs(q_pos - tau_q))) > 1.0e-4
        ),
        "q_pos": q_pos.tolist(),
        "q_mix": q_mix.tolist(),
        "differs_from_soft_targets": float(np.max(np.abs(q_pos - tau_q))),
    }


def loss_identity_tests() -> dict:
    seed_everything(0)
    data = load_exec_split("hl_exec_align_val")
    model = OrderFreeAssignmentModel("deepsets")
    encoded = _encode_one(model, data, 0, "cpu")
    snap = make_snap(data["uav_positions"][0], data["buoy_positions"][0])
    with torch.no_grad():
        exe = utility_losses_exec(model, encoded, snap, data["candidates"][0], data["utility"][0], data["assignment"][0], data["buoy_mask"][0])
        abso = utility_losses_abs(model, encoded, snap, data["candidates"][0], data["utility"][0], data["assignment"][0], data["buoy_mask"][0])
    q_exec = exe["q"].cpu().numpy()
    q_abs = abso["q"].cpu().numpy()
    cond_exec = float((-(exe["q"] * torch.log_softmax(exe["logp"], dim=0)).sum()).cpu())
    return {
        "pass": bool(
            abs(float(exe["rank"].cpu()) - float(abso["rank"].cpu())) < 1.0e-6
            and abs(float(exe["ref"].cpu()) - float(abso["ref"].cpu())) < 1.0e-6
            and float(np.max(np.abs(q_exec - q_abs))) > 1.0e-4
            and abs(float(exe["utility_exec"].cpu()) - cond_exec) > 1.0e-4
        ),
        "rank_delta": abs(float(exe["rank"].cpu()) - float(abso["rank"].cpu())),
        "ref_delta": abs(float(exe["ref"].cpu()) - float(abso["ref"].cpu())),
        "q_vs_tau_max": float(np.max(np.abs(q_exec - q_abs))),
        "exec_vs_cond": abs(float(exe["utility_exec"].cpu()) - cond_exec),
    }


def gradient_sanity() -> dict:
    seed_everything(0)
    data = load_exec_split("hl_exec_align_val")
    model = OrderFreeAssignmentModel("deepsets")
    model.train()
    encoded = _encode_one(model, data, 0, "cpu")
    snap = make_snap(data["uav_positions"][0], data["buoy_positions"][0])
    packed = utility_losses_exec(model, encoded, snap, data["candidates"][0], data["utility"][0], data["assignment"][0], data["buoy_mask"][0])
    packed["utility_exec"].backward()
    grads = [float(p.grad.abs().sum().cpu()) for p in model.parameters() if p.grad is not None]
    return {"pass": bool(sum(grads) > 0.0), "grad_abs_sum": float(sum(grads)), "optimizer_step": False}


def run_pretests() -> dict:
    q = q_construction_tests()
    ident = loss_identity_tests()
    grad = gradient_sanity()
    if not q["pass"]:
        raise RuntimeError("executor q construction failed")
    if not ident["pass"]:
        raise RuntimeError("L_exec identity vs ranking/reference or cond/tau separation failed")
    if not grad["pass"]:
        raise RuntimeError("L_exec gradient is zero")
    return {"q": q, "identity": ident, "grad": grad}


__all__ = ["gradient_sanity", "loss_identity_tests", "q_construction_tests", "run_pretests"]
