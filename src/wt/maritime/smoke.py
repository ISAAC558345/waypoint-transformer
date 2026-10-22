"""Run a deterministic, dependency-light stage 1A environment rollout.

Execute from the repository root with::

    PYTHONPATH=src python -m wt.maritime.smoke

The module neither calls a training entry point nor writes any artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .env import MaritimeEnv


SMOKE_SEED = 2023
ACTION_COMPONENT_LIMIT = 1.5


def run_smoke(seed: int = SMOKE_SEED) -> dict[str, Any]:
    """Run one complete episode and return its eight summary statistics."""
    environment = MaritimeEnv()
    observation, info = environment.reset(seed=seed)
    generator = np.random.default_rng(seed)

    if not np.all(np.isfinite(observation)):
        raise RuntimeError("reset returned a non-finite observation")

    episode_steps = 0
    cumulative_reward = 0.0
    cumulative_communication_bits = 0.0
    total_movement_distance_m = 0.0
    boundary_violation_count = 0
    action_clipping_count = 0

    while True:
        action = generator.uniform(
            -ACTION_COMPONENT_LIMIT,
            ACTION_COMPONENT_LIMIT,
            size=environment.action_dim,
        )
        observation, reward, terminated, truncated, info = environment.step(action)

        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                f"step {episode_steps + 1} returned a non-finite observation"
            )
        if not np.isfinite(reward):
            raise RuntimeError(f"step {episode_steps + 1} returned a non-finite reward")

        episode_steps += 1
        cumulative_reward += reward
        cumulative_communication_bits += (
            float(info["total_rate_bps"]) * environment.config.delta_t
        )
        total_movement_distance_m += float(
            np.linalg.norm(info["applied_displacement_m"])
        )
        boundary_violation_count += int(info["boundary_violation"])
        action_clipping_count += int(info["action_component_clipped"])

        if terminated or truncated:
            break

    return {
        "episode_steps": episode_steps,
        "cumulative_reward": float(cumulative_reward),
        "cumulative_communication_bits": float(cumulative_communication_bits),
        "total_movement_distance_m": float(total_movement_distance_m),
        "final_position_m": np.asarray(info["uav_position_m"]).copy(),
        "remaining_energy": float(info["remaining_energy"]),
        "boundary_violation_count": boundary_violation_count,
        "action_clipping_count": action_clipping_count,
    }


def main() -> None:
    """Run the smoke rollout and print exactly eight human-readable fields."""
    summary = run_smoke()
    print(f"episode_steps: {summary['episode_steps']}")
    print(f"cumulative_reward: {summary['cumulative_reward']:.6f}")
    print(
        "cumulative_communication_bits: "
        f"{summary['cumulative_communication_bits']:.6f}"
    )
    print(
        f"total_movement_distance_m: {summary['total_movement_distance_m']:.6f}"
    )
    final_position = np.asarray(summary["final_position_m"])
    print(
        "final_position_m: "
        f"[{final_position[0]:.6f}, {final_position[1]:.6f}]"
    )
    print(f"remaining_energy: {summary['remaining_energy']:.6f}")
    print(
        f"boundary_violation_count: {summary['boundary_violation_count']}"
    )
    print(f"action_clipping_count: {summary['action_clipping_count']}")


if __name__ == "__main__":
    main()
