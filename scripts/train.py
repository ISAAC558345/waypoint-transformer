"""Show the frozen training entries.

This script does not start a new training run and does not overwrite checkpoints.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the frozen high-level and low-level training entries.")
    parser.parse_args()
    print("High-level scheme C checkpoint: results/phase6if_ranking_loss/runs/c/epoch_001.pt")
    print("Low-level MAPPO-V2 checkpoint: src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt")
    print("Existing low-level module: python -m wt.benchmark_v2.phase6g_lowlevel_mappo.run_phase6g")
    print("Existing high-level module: python -m wt.benchmark_v2.phase6if_ranking_loss.train")
    print("Both commands belong to completed experiments. This entry does not launch them.")


if __name__ == "__main__":
    main()
