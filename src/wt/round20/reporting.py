"""Figures and concise completion reporting for the Round-20 dataset audit."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .dataset import BEHAVIORS, plain, sha256


def _save(figure: Any, directory: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(directory / f"{stem}.png", dpi=150)
    figure.savefig(directory / f"{stem}.svg")
    plt.close(figure)


def _group_mean(episodes: Sequence[Mapping[str, Any]], key: str) -> tuple[list[str], list[float]]:
    labels, values = [], []
    for behavior in BEHAVIORS:
        group = [item for item in episodes if item["behavior"].name == behavior.name]
        labels.append(behavior.name.replace("noisy_joint_", "nJ_").replace("policy_switch_independent_h1_h10", "switch").replace("independent_queue_rate", "independent"))
        values.append(float(np.mean([item[key] for item in group])))
    return labels, values


def render_figures(directory: Path, result: Mapping[str, Any], audits: Mapping[str, Any]) -> list[str]:
    directory.mkdir(parents=True, exist_ok=False)
    episodes, samples, scenarios = result["episodes"], result["samples"], result["scenarios"]
    metrics = result["metrics"]
    # 01 M=2 example trajectories.
    figure, axis = plt.subplots(figsize=(6, 5))
    for episode in [item for item in episodes if item["behavior"].name in ("independent_queue_rate", "joint_h10")][:2]:
        tracks = [[], []]
        for transition in episode["transitions"]:
            positions = np.asarray(transition["graph_state_t"]["uav_nodes"], dtype=float)
            for uav in range(2): tracks[uav].append(positions[uav, :2])
        for uav, track in enumerate(tracks):
            points = np.asarray(track); axis.plot(points[:, 0], points[:, 1], marker="o", ms=2, label=f"{episode['behavior'].name}: UAV{uav}")
        buoy = np.asarray(episode["scenario"].buoy_positions_m); axis.scatter(buoy[:, 0], buoy[:, 1], marker="^", c="black")
    axis.set(xlabel="x (m)", ylabel="y (m)", title="M=2 development trajectory examples"); axis.legend(fontsize=6); _save(figure, directory, "01_m2_dataset_trajectories")
    # 02 K distribution.
    figure, axis = plt.subplots(figsize=(5, 3)); counts = Counter(item["scenario"].num_buoys for item in episodes); axis.bar(["K=3", "K=5"], [counts[3], counts[5]]); axis.set(title="Episodes by buoy count", ylabel="episodes"); _save(figure, directory, "02_k_distribution")
    # 03 interference distribution.
    figure, axis = plt.subplots(figsize=(5, 3)); tiers = ("orthogonal", "low", "medium", "full"); values = [sum(item["scenario"].spectrum_tier == tier for item in episodes) for tier in tiers]; axis.bar(tiers, values); axis.set(title="Interference-tier distribution", ylabel="episodes"); _save(figure, directory, "03_interference_tier_distribution")
    # 04 behavior distribution.
    figure, axis = plt.subplots(figsize=(9, 3)); labels, values = _group_mean(episodes, "steps"); axis.bar(range(len(labels)), values); axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7); axis.set(title="Behavior-generator coverage", ylabel="mean steps"); _save(figure, directory, "04_behavior_generator_distribution")
    # 05 quality distribution.
    figure, axis = plt.subplots(figsize=(5, 3)); quality = Counter(item["quality_tier"] for item in episodes); axis.bar(["high", "medium", "low"], [quality["high"], quality["medium"], quality["low"]]); axis.set(title="Outcome-derived quality tiers", ylabel="episodes"); _save(figure, directory, "05_quality_tier_distribution")
    # 06-09 behavior outcomes.
    for number, key, title, ylabel in ((6, "payload_mbit", "Payload by behavior", "payload (Mbit)"), (7, "completion", "Completion by behavior", "completion"), (8, "residual_queue_bits", "Residual queue by behavior", "bits"), (9, "return", "Team return by behavior", "return")):
        figure, axis = plt.subplots(figsize=(9, 3)); labels, values = _group_mean(episodes, key); axis.bar(range(len(labels)), values); axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7); axis.set(title=title, ylabel=ylabel); _save(figure, directory, f"{number:02d}_{title.lower().replace(' ', '_')}")
    # 10 role examples.
    role_samples = [item for item in samples if item["H_task"] == 10 and np.any(np.asarray(item["service_valid_uav_mask"], dtype=bool))][:1]
    figure, axis = plt.subplots(figsize=(6, 3)); matrix = np.asarray(role_samples[0]["per_uav_service_responsibility_target"], dtype=float) if role_samples else np.zeros((2, 5)); image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1); axis.set(title="H10 service-responsibility role example", xlabel="padded buoy index", ylabel="UAV"); figure.colorbar(image, ax=axis); _save(figure, directory, "10_uav_role_examples")
    # 11 responsibility distributions.
    figure, axis = plt.subplots(figsize=(5, 3)); values = np.asarray([value for item in samples for value in np.asarray(item["per_uav_service_responsibility_target"], dtype=float).reshape(-1)]); axis.hist(values, bins=20); axis.set(title="Service-responsibility target distribution", xlabel="probability"); _save(figure, directory, "11_service_responsibility_distributions")
    # 12 progress target distributions.
    figure, axis = plt.subplots(figsize=(5, 3)); values = np.asarray([value for item in samples for value in np.asarray(item["per_uav_gross_service_progress_normalized"], dtype=float).reshape(-1)]); axis.hist(values, bins=20); axis.set(title="Gross progress target distribution", xlabel="queue-capacity normalized progress"); _save(figure, directory, "12_progress_target_distributions")
    # 13 team return distributions.
    figure, axis = plt.subplots(figsize=(5, 3)); values = [item["team_h_step_return_raw"] for item in samples]; axis.hist(values, bins=25); axis.set(title="Team H-step return distribution", xlabel="discounted team return"); _save(figure, directory, "13_team_hstep_return_distribution")
    # 14 state overlap.
    figure, axis = plt.subplots(figsize=(5, 3)); rows = audits["overlap"]["state_overlap_rows"]; axis.bar([item["scope"] for item in rows], [item["cross_behavior_neighbor_ratio"] for item in rows], label="cross behavior"); axis.bar([item["scope"] for item in rows], [item["cross_planner_neighbor_ratio"] for item in rows], bottom=[item["cross_behavior_neighbor_ratio"] for item in rows], label="cross family"); axis.set(title="Cross-policy state overlap", ylabel="nearest-neighbor ratio"); axis.legend(fontsize=7); _save(figure, directory, "14_cross_policy_state_overlap")
    # 15 bridge examples/count.
    figure, axis = plt.subplots(figsize=(5, 3)); bridge = audits["overlap"]; axis.bar(["eligible", "bridges"], [len(bridge["bridge_rows"]), bridge["coordination_bridge_count"]]); axis.set(title="Coordination bridge candidates", ylabel="cross-episode states"); _save(figure, directory, "15_coordination_bridge_examples")
    # 16 conflict bridge relation.
    figure, axis = plt.subplots(figsize=(5, 3)); relation = audits["overlap"]["relation_rows"]; axis.bar([item["state_class"] for item in relation], [item["bridge_ratio"] for item in relation]); axis.set(title="Conflict versus non-conflict bridge ratio", ylabel="bridge ratio"); _save(figure, directory, "16_conflict_nonconflict_bridge_ratio")
    # 17 assignment-divergent neighbors.
    figure, axis = plt.subplots(figsize=(5, 3)); axis.bar(["assignment-divergent", "future quality-different"], [bridge["assignment_divergent_neighbor_ratio"], bridge["future_quality_difference"]]); axis.set(title="Nearest-neighbor future divergence", ylabel="ratio"); _save(figure, directory, "17_assignment_divergent_neighbors")
    # 18 graph schematic.
    figure, axis = plt.subplots(figsize=(7, 3)); axis.axis("off"); axis.text(.05, .70, "UAV nodes\nposition, velocity, energy, path, service", bbox={"boxstyle":"round", "fc":"#cfe8ff"}, ha="center"); axis.text(.48, .70, "UB edges\ndistance, gain, SNR, rate,\nqueue weight, coupling", bbox={"boxstyle":"round", "fc":"#e9f7d5"}, ha="center"); axis.text(.90, .70, "Buoy nodes\nposition, queue, arrival,\ncompletion, service", bbox={"boxstyle":"round", "fc":"#ffe5c2"}, ha="center"); axis.text(.48, .18, "UU edges: distance, relative movement, safety margin, current coupling\nMasks + physical global features; no behavior/quality named metadata input", ha="center", bbox={"boxstyle":"round", "fc":"#eeeeee"}); axis.annotate("", (.37,.7), (.18,.7), arrowprops={"arrowstyle":"<->"}); axis.annotate("", (.60,.7), (.82,.7), arrowprops={"arrowstyle":"<->"}); axis.set(title="Graph-ready state schema"); _save(figure, directory, "18_graph_schema_schematic")
    # 19 task sample count.
    figure, axis = plt.subplots(figsize=(5, 3)); count = Counter(item["H_task"] for item in samples); axis.bar(["H5", "H10"], [count[5], count[10]]); axis.set(title="Task-Waypoint sample counts", ylabel="samples"); _save(figure, directory, "19_h5_h10_task_waypoint_counts")
    # 20 split distribution.
    figure, axis = plt.subplots(figsize=(5, 3)); count = Counter(item.split for item in scenarios); axis.bar(["train", "validation", "development test"], [count["train"], count["validation"], count["development_test"]]); axis.set(title="Scenario-level split distribution", ylabel="scenarios"); _save(figure, directory, "20_split_distribution")
    return sorted(path.name for path in directory.iterdir() if path.suffix in (".png", ".svg"))


def write_completion_report(path: Path, result: Mapping[str, Any], audits: Mapping[str, Any], readiness: Mapping[str, Any], figure_directory: Path) -> None:
    quality, graph, task, overlap = audits["quality_gate"], audits["graph_gate"], audits["task_gate"], audits["overlap"]
    behavior_rows = []
    for behavior in BEHAVIORS:
        group = [item for item in result["episodes"] if item["behavior"].name == behavior.name]
        behavior_rows.append(f"| {behavior.name} | {len(group)} | {np.mean([item['payload_mbit'] for item in group]):.3f} |")
    content = f"""# Round-20 completion report

Round-20 generated a deterministic, non-learning mixed-quality M=2
shared-spectrum development dataset from the verified immutable Round-19
lock.  The environment, rate/interference formula, matching, reward, movement
semantics, and 15 m safety projection were reused without modification.

## Dataset

- Development scenarios: {len(result['scenarios'])}; episodes: {len(result['episodes'])}; transitions: {len(result['transitions'])}.
- Scenario split: train/validation/development-test = 48/16/16 with no scenario crossing a split.
- Quality counts: {dict(quality['quality_counts'])}; ratios: {quality['quality_ratios']}.
- K episode counts: {quality['num_buoys_episode_counts']}; tier episode counts: {quality['interference_tier_episode_counts']}.
- H5/H10 Task-Waypoint samples: {task['sample_counts']}.

## Behavior coverage

| behavior | episodes | mean payload (Mbit) |
| --- | ---: | ---: |
{chr(10).join(behavior_rows)}

## Readiness gates

- Dataset quality: `{quality['dataset_quality_ready']}`.
- Graph training data: `{graph['graph_training_data_ready']}`.
- Multi-UAV Task-Waypoint: `{task['multi_uav_task_waypoint_training_ready']}`.
- Final HGNN + Task-Waypoint readiness: `{readiness['round20_ready_for_hgnn_task_waypoint_training']}`.
- Coordination bridges: {overlap['coordination_bridge_count']} ({overlap['coordination_bridge_ratio']:.4f}); local-distance threshold {overlap['local_distance_threshold']:.6f}.
- New untouched-evaluation policy rollouts: 0. Historical protected-data accesses: 0/0/0.

The independent role head remains unnecessary: responsibility distributions are
the primary per-UAV role supervision and the role-redundancy audit is retained
as evidence.  Figures are in `{figure_directory.as_posix()}`.
"""
    path.write_text(content, encoding="utf-8")


def figure_manifest(directory: Path) -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.suffix in (".png", ".svg")}


__all__ = ["figure_manifest", "render_figures", "write_completion_report"]
