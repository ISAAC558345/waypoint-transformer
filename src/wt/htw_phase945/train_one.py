"""Train one encoder seed from the frozen utility NPZ. Never regenerates H=15 utilities."""
from __future__ import annotations

import os

# Pin BLAS/OpenMP before importing torch. Two concurrent PyTorch CPU jobs on
# Windows caused ACCESS_VIOLATION (0xC0000005 / null-pointer read dialogs).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase944.tables import write_csv

from .protocol import OUTPUT
from .run_phase945 import load_split
from .training import train_utility_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", choices=("deepsets", "hgnn"))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    train_path = OUTPUT / "phase945_train.npz"
    val_path = OUTPUT / "phase945_val.npz"
    if not train_path.exists() or not val_path.exists():
        raise RuntimeError("frozen utility dataset missing; refuse to regenerate")
    ckpt = OUTPUT / f"{args.encoder}_seed{args.seed}.pt"
    if ckpt.exists():
        print(json.dumps({"skip": True, "encoder": args.encoder, "seed": args.seed}), flush=True)
        return
    train = load_split("phase945_train")
    val = load_split("phase945_val")
    if "utility" not in train or "candidates" not in train:
        raise RuntimeError("utility/candidates missing on frozen split")
    print(f"train {args.encoder} seed={args.seed}", flush=True)
    _model, info = train_utility_model(OrderFreeAssignmentModel(args.encoder), train, val, args.encoder, int(args.seed))
    write_csv(OUTPUT / f"phase945_train_history_{args.encoder}_{args.seed}.csv", info["history"])
    print(json.dumps({"skip": False, "encoder": args.encoder, "seed": args.seed, "best_epoch": info["best_epoch"], "best_val_regret": info["best_val_regret"]}), flush=True)


if __name__ == "__main__":
    main()
