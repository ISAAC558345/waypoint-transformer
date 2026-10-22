"""Train DeepSets from scratch with executor-normalized utility. Checkpoint unchanged."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.lowlevel_v2.integration_phase2.train_hl import split_metrics
from wt.round30.ppo import seed_everything

from .dataset import load_exec_split
from .exec_loss import utility_losses_exec
from .protocol import BATCH, EPOCHS, LR, OUTPUT, TRAIN_NAME, VAL_EVERY, VAL_NAME


def train_exec(train: dict[str, Any], val: dict[str, Any], init_seed: int) -> dict[str, Any]:
    torch.set_num_threads(2)
    seed_everything(int(init_seed))
    model = OrderFreeAssignmentModel("deepsets").to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    count = int(train["assignment"].shape[0])
    best = {"regret": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(EPOCHS):
        order = np.random.permutation(count)
        bucket = {k: [] for k in ("total", "exec", "rank", "ref", "grad")}
        model.train()
        for start in range(0, count, BATCH):
            index = order[start : start + BATCH]
            optimizer.zero_grad(set_to_none=True)
            scale = float(len(index))
            for item in index:
                encoded = _encode_one(model, train, int(item), "cpu")
                snap = make_snap(train["uav_positions"][int(item)], train["buoy_positions"][int(item)])
                packed = utility_losses_exec(
                    model,
                    encoded,
                    snap,
                    train["candidates"][int(item)],
                    train["utility"][int(item)],
                    train["assignment"][int(item)],
                    train["buoy_mask"][int(item)],
                )
                (packed["total"] / scale).backward()
                bucket["total"].append(float(packed["total"].detach().cpu()))
                bucket["exec"].append(float(packed["utility_exec"].detach().cpu()))
                bucket["rank"].append(float(packed["rank"].detach().cpu()))
                bucket["ref"].append(float(packed["ref"].detach().cpu()))
                del packed, encoded
            bucket["grad"].append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            optimizer.step()
        row: dict[str, Any] = {
            "encoder": "deepsets",
            "model_seed": int(init_seed),
            "epoch": epoch + 1,
            "L_total": float(np.mean(bucket["total"])),
            "L_exec": float(np.mean(bucket["exec"])),
            "L_rank": float(np.mean(bucket["rank"])),
            "L_ref": float(np.mean(bucket["ref"])),
            "grad_norm": float(np.mean(bucket["grad"])),
        }
        if (epoch + 1) % int(VAL_EVERY) == 0 or epoch + 1 == EPOCHS:
            val_row = split_metrics(model, val, mechanism=False)
            row.update({f"val_{k}": v for k, v in val_row.items()})
            regret = float(val_row["top1_regret"])
            if regret < best["regret"] - 1.0e-9:
                best = {"regret": regret, "epoch": epoch + 1, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            print(
                f"exec seed={init_seed} epoch {epoch + 1}/{EPOCHS} L_exec={row['L_exec']:.3f} val_regret={regret:.1f} spearman={val_row['spearman']:.3f}",
                flush=True,
            )
        else:
            print(f"exec seed={init_seed} epoch {epoch + 1}/{EPOCHS} L_exec={row['L_exec']:.3f} L_rank={row['L_rank']:.3f}", flush=True)
        history.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT / f"deepsets_exec_seed{int(init_seed)}_last.pt")
    if best["state"] is None:
        best = {"regret": float("nan"), "epoch": EPOCHS, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    path = OUTPUT / f"deepsets_exec_seed{int(init_seed)}.pt"
    torch.save(model.state_dict(), path)
    return {
        "best_epoch": best["epoch"],
        "best_val_regret": best["regret"],
        "history": history,
        "checkpoint": str(path),
        "from_old_checkpoint": False,
    }


def train_seed_process(init_seed: int) -> dict[str, Any]:
    import os

    os.environ["OMP_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    train = load_exec_split(TRAIN_NAME)
    val = load_exec_split(VAL_NAME)
    info = train_exec(train, val, int(init_seed))
    info.pop("model", None)
    return info


__all__ = ["train_exec", "train_seed_process"]
