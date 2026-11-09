"""Preview the fixed L9 contract. Training starts only with --run."""
from __future__ import annotations

import argparse
import json


def config() -> dict:
    return {
        "mode": "config_only",
        "comparison": "R28_U120 versus reused R12_U120",
        "new_training": "R28 only, 120 updates from epoch 6, empty Adam, 28 states",
        "visit_rule": "states sorted by scenario and time; update i uses state i mod 28",
        "visits": {"first_8_states": 5, "other_20_states": 4, "r12_each_state": 10},
        "reused_without_training": ["R12_U120 L8 R10", "E6", "L3A10", "C_LISTNET"],
        "loss": "L8 expected_regret_loss, softmax temperature 1",
        "train_states": "L2 322016-322021 plus L8 322024-322031, t15 and t30",
        "comparison_scenarios": list(range(322032, 322040)),
        "comparison_times": [15, 30],
        "budget": {
            "training_updates": 120,
            "backward_passes": 120,
            "old_state_environment_steps": 0,
            "replay_environment_steps": 240,
            "candidate_rollouts": 2048,
            "candidate_environment_steps": 30720,
            "environment_steps": 30960,
            "scoring_forwards_planned": 200,
            "scoring_forwards_cap": 204,
        },
        "command": "py -3 -m wt.research.arrival_coverage --run",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(config(), ensure_ascii=False, indent=2), flush=True)
        return
    from wt.research.arrival_coverage.execute import run

    run()


if __name__ == "__main__":
    main()