"""Train utility-aligned objective. Same Adam/lr/epochs/batch as Phase-9.4.3."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import BATCH, EPOCHS, LR
from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.round30.ppo import seed_everything

from .metrics import pool_metrics
from .protocol import OUTPUT, VAL_EVERY
from .utility_loss import candidate_logprobs, utility_losses

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


@torch.no_grad()
def split_metrics(model: OrderFreeAssignmentModel, data: dict[str, Any], max_items: int | None = None) -> dict[str, float]:
    model.eval()
    n = int(data["assignment"].shape[0])
    take = n if max_items is None else min(n, int(max_items))
    rows = []
    for index in range(take):
        encoded = _encode_one(model, data, index, "cpu")
        snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
        logp = candidate_logprobs(model, encoded, snap, data["candidates"][index], data["buoy_mask"][index])
        rows.append(pool_metrics(data["utility"][index], logp.detach().cpu().numpy()))
    keys = rows[0].keys()
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


def train_utility_model(
    model: OrderFreeAssignmentModel,
    train: dict[str, Any],
    val: dict[str, Any],
    name: str,
    init_seed: int,
) -> tuple[OrderFreeAssignmentModel, dict[str, Any]]:
    if "utility" not in train or "candidates" not in train:
        raise RuntimeError("frozen utility dataset missing; refuse to regenerate")
    seed_everything(int(init_seed))
    model = model.to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    count = int(train["assignment"].shape[0])
    best = {"regret": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        losses = []
        model.train()
        for start in range(0, count, BATCH):
            index = order[start : start + BATCH]
            batch_loss = []
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
                batch_loss.append(packed["total"])
            loss = torch.stack(batch_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row: dict[str, Any] = {"encoder": name, "model_seed": init_seed, "epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if (epoch + 1) % int(VAL_EVERY) == 0 or epoch + 1 == EPOCHS:
            val_row = split_metrics(model, val)
            row.update({f"val_{k}": v for k, v in val_row.items()})
            regret = float(val_row["top1_regret"])
            if regret < best["regret"] - 1.0e-9:
                best = {"regret": regret, "epoch": epoch + 1, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            torch.save(model.state_dict(), OUTPUT / f"{name}_seed{init_seed}_last.pt")
            print(f"{name} seed={init_seed} epoch {epoch + 1}/{EPOCHS} loss={row['train_loss']:.4f} val_regret={regret:.1f}", flush=True)
        else:
            print(f"{name} seed={init_seed} epoch {epoch + 1}/{EPOCHS} loss={row['train_loss']:.4f}", flush=True)
        history.append(row)
    if best["state"] is None:
        best = {"regret": float("nan"), "epoch": EPOCHS, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    path = OUTPUT / f"{name}_seed{init_seed}.pt"
    torch.save(model.state_dict(), path)
    return model, {"best_epoch": best["epoch"], "best_val_regret": best["regret"], "history": history, "checkpoint": str(path)}
