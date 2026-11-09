"""Preview the fixed L8 contract. Training starts only with --run."""
from __future__ import annotations

import argparse
import json


def config() -> dict:
    return {
        "mode": "config_only",
        "comparison": "C_LISTNET versus R_EXPECTED_REGRET",
        "new_training": "R only, 120 updates from epoch 6, empty Adam",
        "reused_without_training": ["C_LISTNET L3 B10", "E6", "L3 A10"],
        "loss": "softmax temperature 1 expected normalized payload regret",
        "train_scenarios": list(range(322016, 322022)),
        "train_times": [15, 30],
        "comparison_scenarios": list(range(322024, 322032)),
        "comparison_times": [15, 30],
        "candidates_per_state": 128,
        "sampling": "stored zero-arrival J15 top 32 plus rank-stratified 96, seed 605",
        "label": "score_label zero_arrivals false horizon 15",
        "budget": {
            "training_updates": 120,
            "backward_passes": 120,
            "replay_environment_steps": 240,
            "candidate_rollouts": 2048,
            "candidate_environment_steps": 30720,
            "environment_steps": 30960,
            "scoring_forwards_planned": 196,
            "scoring_forwards_cap": 200,
        },
        "command": "py -3 -m wt.research.payload_objective --run",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(config(), ensure_ascii=False, indent=2), flush=True)
        return
    from wt.research.payload_objective.execute import run

    run()


if __name__ == "__main__":
    main()