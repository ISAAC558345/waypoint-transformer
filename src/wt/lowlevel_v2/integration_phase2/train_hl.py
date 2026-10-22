"""Train DeepSets from scratch on frozen U_new. Checkpoint = best val candidate regret."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.utility_loss import candidate_logprobs, utility_losses
from wt.round30.ppo import seed_everything

from .protocol import BATCH, EPOCHS, LR, OUTPUT, VAL_EVERY


def _entropy(logp: np.ndarray) -> float:
    p = np.exp(logp - np.max(logp))
    p = p / max(float(p.sum()), 1.0e-12)
    return float(-np.sum(p * np.log(np.clip(p, 1.0e-12, 1.0))))


@torch.no_grad()
def split_metrics(model: OrderFreeAssignmentModel, data: dict[str, Any], *, mechanism: bool = True) -> dict[str, float]:
    from wt.htw_phase945.metrics import pool_metrics
    from wt.htw_phase9.features import conflict_rate

    model.eval()
    rows = []
    idle = []
    conflict = []
    entropy = []
    nlls = []
    for index in range(int(data["assignment"].shape[0])):
        encoded = _encode_one(model, data, index, "cpu")
        snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
        logp = candidate_logprobs(model, encoded, snap, data["candidates"][index], data["buoy_mask"][index])
        lp = logp.detach().cpu().numpy()
        row = pool_metrics(data["utility"][index], lp)
        u = np.asarray(data["utility"][index], dtype=np.float64)
        n = u.size
        top = int(np.argmax(lp))
        order_u = np.argsort(-u)
        top25 = set(order_u[: max(1, int(np.ceil(0.25 * n)))].tolist())
        row["hit_top25"] = float(top in top25)
        rows.append(row)
        entropy.append(_entropy(lp))
        if mechanism:
            from wt.htw_phase943.set_likelihood_dp import set_nll

            nlls.append(float(set_nll(model, encoded, snap, np.asarray(data["assignment"][index], dtype=np.int64), data["buoy_mask"][index]).cpu()))
        chosen = np.asarray(data["candidates"][index][top], dtype=np.int64)
        idle.append(float(np.mean(chosen[:6] < 0)))
        conflict.append(float(conflict_rate(chosen, 12)))
    keys = rows[0].keys()
    out = {key: float(np.nanmean([row[key] for row in rows])) for key in keys}
    out["entropy"] = float(np.mean(entropy))
    out["set_nll"] = float(np.mean(nlls)) if nlls else float("nan")
    out["idle_fraction"] = float(np.mean(idle))
    out["conflict_rate"] = float(np.mean(conflict))
    return out


def load_align_split(name: str) -> dict[str, Any]:
    import json

    packed = np.load(OUTPUT / f"{name}.npz")
    data = {key: packed[key] for key in packed.files}
    man = json.loads((OUTPUT / f"{name}_manifest.json").read_text(encoding="utf-8"))
    data["meta"] = man["meta"]
    data["hashes"] = man["hashes"]
    data["split"] = man["split"]
    data["seeds"] = man["seeds"]
    data["pool_size"] = man["pool_size"]
    return data


def train_seed_process(init_seed: int) -> dict[str, Any]:
    import os

    os.environ["OMP_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    train = load_align_split("hl_rl_align_train")
    val = load_align_split("hl_rl_align_val")
    info = train_deepsets(train, val, int(init_seed))
    info.pop("model", None)
    return info


def train_deepsets(train: dict[str, Any], val: dict[str, Any], init_seed: int) -> dict[str, Any]:
    torch.set_num_threads(2)
    seed_everything(int(init_seed))
    model = OrderFreeAssignmentModel("deepsets").to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    count = int(train["assignment"].shape[0])
    best = {"regret": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        losses = []
        gnorms = []
        model.train()
        for start in range(0, count, BATCH):
            index = order[start : start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            running = []
            scale = float(len(index))
            for item in index:
                encoded = _encode_one(model, train, int(item), "cpu")
                snap = make_snap(train["uav_positions"][int(item)], train["buoy_positions"][int(item)])
                packed = utility_losses(
                    model,
                    encoded,
                    snap,
                    train["candidates"][int(item)],
                    train["utility"][int(item)],
                    train["assignment"][int(item)],
                    train["buoy_mask"][int(item)],
                )
                total = packed["total"]
                (total / scale).backward()
                running.append(float(total.detach().cpu()))
                del packed, total, encoded
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            optimizer.step()
            losses.append(float(np.mean(running)))
        row: dict[str, Any] = {
            "encoder": "deepsets",
            "model_seed": int(init_seed),
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "grad_norm": float(np.mean(gnorms)),
        }
        if (epoch + 1) % int(VAL_EVERY) == 0 or epoch + 1 == EPOCHS:
            val_row = split_metrics(model, val, mechanism=False)
            row.update({f"val_{k}": v for k, v in val_row.items()})
            regret = float(val_row["top1_regret"])
            if regret < best["regret"] - 1.0e-9:
                best = {"regret": regret, "epoch": epoch + 1, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            print(
                f"deepsets seed={init_seed} epoch {epoch + 1}/{EPOCHS} loss={row['train_loss']:.4f} val_regret={regret:.1f} spearman={val_row['spearman']:.3f}",
                flush=True,
            )
        else:
            print(f"deepsets seed={init_seed} epoch {epoch + 1}/{EPOCHS} loss={row['train_loss']:.4f}", flush=True)
        history.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT / f"deepsets_rlalign_seed{int(init_seed)}_last.pt")
    if best["state"] is None:
        best = {"regret": float("nan"), "epoch": EPOCHS, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    path = OUTPUT / f"deepsets_rlalign_seed{int(init_seed)}.pt"
    torch.save(model.state_dict(), path)
    return {
        "best_epoch": best["epoch"],
        "best_val_regret": best["regret"],
        "history": history,
        "checkpoint": str(path),
        "model": model,
    }


__all__ = ["load_align_split", "split_metrics", "train_deepsets", "train_seed_process"]
