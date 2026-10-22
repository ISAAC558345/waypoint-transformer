"""Read-only K-step sample availability and terminal-boundary audit.

No samples are materialized or used for learning.  A valid pair starts at
``observation[t]`` and targets ``observation[t+K]``, represented by
``next_observations[t+K-1]``.  Every intervening transition must be real and no
done flag may occur before the target transition.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dataset_io import (
    SPLIT_NAMES,
    deterministic_episode_split,
    load_dataset,
    validate_dataset,
)
from .trajectory import EpisodeTrajectory


DEFAULT_K_VALUES = (1, 5, 10, 20, 30)
AUDIT_COLUMNS = (
    "split",
    "policy_id",
    "num_buoys",
    "k",
    "episode_count",
    "transition_count",
    "candidate_anchor_count",
    "valid_sample_count",
    "invalid_short_horizon_count",
    "invalid_terminal_crossing_count",
    "invalid_padding_count",
    "short_episode_count",
    "terminal_boundary_violation_count",
)


@dataclass(frozen=True)
class KStepPair:
    """Index metadata for one safe within-episode ``(t, t+K)`` pair."""

    episode_id: str
    start_timestep: int
    target_timestep: int
    k: int


def enumerate_k_step_pairs(
    episode: EpisodeTrajectory, k: int
) -> List[KStepPair]:
    """Enumerate valid pairs without crossing padding, done, or episode end."""
    validated_k = _positive_k(k)
    pairs: List[KStepPair] = []
    for start in range(episode.episode_length):
        outcome = _classify_anchor(episode, start, validated_k)
        if outcome == "valid":
            pairs.append(
                KStepPair(
                    episode_id=episode.episode_id,
                    start_timestep=start,
                    target_timestep=start + validated_k,
                    k=validated_k,
                )
            )
    return pairs


def get_k_step_state_pair(
    episode: EpisodeTrajectory, pair: KStepPair
) -> Tuple[np.ndarray, np.ndarray]:
    """Return copied ``(s_t, s_t_plus_K)`` arrays for a validated pair."""
    if pair.episode_id != episode.episode_id:
        raise ValueError("pair episode_id does not match the trajectory")
    if pair.k < 1 or pair.target_timestep != pair.start_timestep + pair.k:
        raise ValueError("inconsistent KStepPair indices")
    if (
        pair.start_timestep < 0
        or pair.start_timestep >= episode.episode_length
        or _classify_anchor(episode, pair.start_timestep, pair.k) != "valid"
    ):
        raise ValueError("KStepPair is not valid for this episode")
    source = episode.observations[pair.start_timestep].copy()
    future = episode.next_observations[pair.target_timestep - 1].copy()
    return source, future


def audit_k_steps(
    episodes: Sequence[EpisodeTrajectory],
    split_assignments: Optional[Mapping[str, str]] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> List[Dict[str, Any]]:
    """Return per-split, per-policy, per-scale K-step availability rows."""
    if split_assignments is None:
        assignments = deterministic_episode_split(episodes)
    else:
        assignments = {
            str(episode_id): str(split)
            for episode_id, split in split_assignments.items()
        }
    validate_dataset(episodes, assignments)
    validated_k_values = _validate_k_values(k_values)

    grouped: Dict[Tuple[str, str, int], List[EpisodeTrajectory]] = {}
    for episode in episodes:
        key = (
            assignments[episode.episode_id],
            episode.policy_id,
            episode.num_buoys,
        )
        grouped.setdefault(key, []).append(episode)

    rows: List[Dict[str, Any]] = []
    split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    for (split_name, policy_id, num_buoys), selected in sorted(
        grouped.items(),
        key=lambda item: (
            split_order[item[0][0]],
            item[0][1],
            item[0][2],
        ),
    ):
        for k in validated_k_values:
            row = _audit_group(
                selected, split_name, policy_id, num_buoys, k
            )
            if row["terminal_boundary_violation_count"] != 0:
                raise RuntimeError(
                    "K-step audit detected a terminal/padding boundary violation"
                )
            rows.append(row)
    return rows


def audit_k_step_availability(
    episodes: Sequence[EpisodeTrajectory],
    split_assignments: Optional[Mapping[str, str]] = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> List[Dict[str, Any]]:
    """Compatibility alias for :func:`audit_k_steps`."""
    return audit_k_steps(episodes, split_assignments, k_values)


def write_k_step_audit_csv(
    rows: Sequence[Mapping[str, Any]], output_path: Any
) -> Path:
    """Write deterministic audit rows to the required CSV artifact.

    ``(rows, output_path)`` is canonical.  ``(output_path, rows)`` is accepted
    for compatibility with early round-02 integration code.
    """
    if isinstance(rows, (str, Path)):
        rows, output_path = output_path, rows
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing K-step audit: {path}"
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(AUDIT_COLUMNS))
        writer.writeheader()
        for row in rows:
            missing = [name for name in AUDIT_COLUMNS if name not in row]
            if missing:
                raise ValueError(f"audit row is missing columns: {missing}")
            writer.writerow({name: row[name] for name in AUDIT_COLUMNS})
    return path


def run_k_step_audit(
    dataset_dir: Any,
    output_csv: Any,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> List[Dict[str, Any]]:
    """Load a saved dataset, run the read-only audit, and write its CSV."""
    root = Path(dataset_dir)
    episodes = load_dataset(root)
    import json

    with (root / "dataset_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assignments = manifest["split_assignments"]
    rows = audit_k_steps(episodes, assignments, k_values)
    write_k_step_audit_csv(rows, output_csv)
    return rows


def _audit_group(
    episodes: Sequence[EpisodeTrajectory],
    split_name: str,
    policy_id: str,
    num_buoys: int,
    k: int,
) -> Dict[str, Any]:
    counts = {
        "valid": 0,
        "short_horizon": 0,
        "terminal_crossing": 0,
        "padding": 0,
    }
    terminal_boundary_violation_count = 0
    for episode in episodes:
        for start in range(episode.episode_length):
            outcome = _classify_anchor(episode, start, k)
            counts[outcome] += 1
        for pair in enumerate_k_step_pairs(episode, k):
            if _pair_crosses_boundary(episode, pair):
                terminal_boundary_violation_count += 1
    candidate_count = int(
        sum(episode.episode_length for episode in episodes)
    )
    classified_count = int(sum(counts.values()))
    if classified_count != candidate_count:
        raise RuntimeError("K-step anchor classification is not exhaustive")
    return {
        "split": split_name,
        "policy_id": policy_id,
        "num_buoys": int(num_buoys),
        "k": int(k),
        "episode_count": len(episodes),
        "transition_count": candidate_count,
        "candidate_anchor_count": candidate_count,
        "valid_sample_count": counts["valid"],
        "invalid_short_horizon_count": counts["short_horizon"],
        "invalid_terminal_crossing_count": counts["terminal_crossing"],
        "invalid_padding_count": counts["padding"],
        "short_episode_count": int(
            sum(episode.episode_length < k for episode in episodes)
        ),
        "terminal_boundary_violation_count": (
            terminal_boundary_violation_count
        ),
    }


def _classify_anchor(
    episode: EpisodeTrajectory, start: int, k: int
) -> str:
    target_transition = start + k - 1
    if target_transition >= episode.episode_length:
        return "short_horizon"
    window = episode.valid_transition_mask[start : target_transition + 1]
    if window.shape != (k,) or not np.all(window):
        return "padding"
    # A done flag on target_transition leads to the legitimate terminal state
    # s_(t+K). A done flag earlier would cross the boundary and is forbidden.
    if target_transition > start and (
        np.any(episode.terminated[start:target_transition])
        or np.any(episode.truncated[start:target_transition])
    ):
        return "terminal_crossing"
    return "valid"


def _pair_crosses_boundary(
    episode: EpisodeTrajectory, pair: KStepPair
) -> bool:
    start = pair.start_timestep
    target_transition = pair.target_timestep - 1
    if target_transition >= episode.episode_length:
        return True
    if not np.all(
        episode.valid_transition_mask[start : target_transition + 1]
    ):
        return True
    return bool(
        np.any(episode.terminated[start:target_transition])
        or np.any(episode.truncated[start:target_transition])
    )


def _validate_k_values(k_values: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(k_values, (str, bytes)) or not k_values:
        raise ValueError("k_values must be a non-empty sequence")
    values = tuple(_positive_k(value) for value in k_values)
    if len(values) != len(set(values)):
        raise ValueError("k_values must not contain duplicates")
    return values


def _positive_k(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("K must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError("K must be a positive integer")
    return result


def _parse_k_values(value: str) -> Tuple[int, ...]:
    return _validate_k_values(
        tuple(int(part.strip()) for part in value.split(",") if part.strip())
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point for auditing an already-saved dataset."""
    parser = argparse.ArgumentParser(
        description="Audit safe within-episode K-step sample availability."
    )
    parser.add_argument("dataset_dir")
    parser.add_argument("output_csv")
    parser.add_argument(
        "--k-values",
        default=",".join(str(value) for value in DEFAULT_K_VALUES),
        help="comma-separated positive K values",
    )
    args = parser.parse_args(argv)
    rows = run_k_step_audit(
        args.dataset_dir, args.output_csv, _parse_k_values(args.k_values)
    )
    print(f"wrote {len(rows)} K-step audit rows to {args.output_csv}")


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_COLUMNS",
    "DEFAULT_K_VALUES",
    "KStepPair",
    "audit_k_step_availability",
    "audit_k_steps",
    "enumerate_k_step_pairs",
    "get_k_step_state_pair",
    "run_k_step_audit",
    "write_k_step_audit_csv",
]
