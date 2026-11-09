"""Show the frozen training entries.

This script does not start a new training run and does not overwrite checkpoints.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the frozen high-level and low-level training entries.")
    parser.parse_args()
    print("High-level scheme C checkpoint: checkpoints/formal/scheme_c_epoch1.pt")
    print("Low-level MAPPO-V2 checkpoint: checkpoints/formal/mappo_v2_seed0.pt")
    print("Existing low-level module: python -m wt.control.mappo.run_phase6g")
    print("Existing high-level module: python -m wt.algorithms.ranking_loss.train")
    print("Both commands belong to completed experiments. This entry does not launch them.")


if __name__ == "__main__":
    main()