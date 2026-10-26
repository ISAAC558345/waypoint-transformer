"""Check the committed E2 artifacts before any reward-form arithmetic."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess

from .protocol import (
    E2_COMMIT,
    E2_DIR,
    EXPECTED_EPISODE_ROWS,
    EXPECTED_GATE,
    EXPECTED_SCENARIOS,
    EXPECTED_SCHEMA,
    EXPECTED_STEP_ROWS,
    EXPECTED_STEPS_PER_EPISODE,
    ROOT,
    SOURCE_NAMES,
)

STEP_FIELDS = (
    "scenario_id",
    "time_step",
    "served_bits",
    "payload_term",
    "urgency_served_bits",
    "urgency_term",
    "projection_penalty",
    "residual_penalty",
    "penalty",
    "base_reward",
    "payload_bits",
)
EPISODE_FIELDS = ("scenario_id", "steps", "served_bits", "urgency_served_bits", "payload_bits", "sum_penalty")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(relative: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.longpaths=true", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def _git_dirty(relative: str) -> bool:
    result = subprocess.run(
        ["git", "-c", "core.longpaths=true", "status", "--porcelain", "--", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _read_csv(path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_sources() -> dict:
    problems: list[str] = []
    hashes: dict[str, str] = {}
    commits: dict[str, str] = {}
    present = {}
    for name in SOURCE_NAMES:
        path = E2_DIR / name
        relative = f"results/phase6i_priority_tag_reward_scale_e2/{name}"
        if not path.is_file():
            problems.append(f"missing {relative}")
            continue
        present[name] = path
        hashes[name] = _sha256(path)
        commits[name] = _git_commit(relative)
        if commits[name] != E2_COMMIT:
            problems.append(f"{relative} last commit is {commits[name] or 'missing'}")
        if _git_dirty(relative):
            problems.append(f"{relative} differs from the committed E2 artifact")
    steps: list[dict[str, str]] = []
    episodes: list[dict[str, str]] = []
    summary = {}
    schema = {}
    if "step_rows.csv" in present:
        steps = _read_csv(present["step_rows.csv"])
    if "episode_rows.csv" in present:
        episodes = _read_csv(present["episode_rows.csv"])
    if "summary.json" in present:
        summary = json.loads(present["summary.json"].read_text(encoding="utf-8"))
    if "schema.json" in present:
        schema = json.loads(present["schema.json"].read_text(encoding="utf-8"))
    if len(steps) != EXPECTED_STEP_ROWS:
        problems.append(f"step row count is {len(steps)}")
    if len(episodes) != EXPECTED_EPISODE_ROWS:
        problems.append(f"episode row count is {len(episodes)}")
    if schema.get("schema_version") != EXPECTED_SCHEMA:
        problems.append("schema version does not match the E2 record")
    if summary.get("gate") != EXPECTED_GATE:
        problems.append("E2 summary gate is not the recorded inconclusive gate")
    if tuple(int(item) for item in summary.get("scenarios", [])) != EXPECTED_SCENARIOS:
        problems.append("E2 scenario list is not the eight train scenarios")
    missing_step = [field for field in STEP_FIELDS if steps and field not in steps[0]]
    missing_episode = [field for field in EPISODE_FIELDS if episodes and field not in episodes[0]]
    if missing_step or missing_episode:
        problems.append(f"missing fields step={missing_step} episode={missing_episode}")
    if not problems:
        problems.extend(_reconcile(steps, episodes))
    return {
        "source_ok": not problems,
        "problems": problems,
        "sha256": hashes,
        "git_commit": commits,
        "expected_commit": E2_COMMIT,
        "steps": steps,
        "episodes": episodes,
        "summary": summary,
        "schema": schema,
    }


def _reconcile(steps: list[dict[str, str]], episodes: list[dict[str, str]]) -> list[str]:
    problems = []
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in steps:
        scenario = int(row["scenario_id"])
        grouped.setdefault(scenario, []).append(row)
        payload = float(row["payload_term"])
        served = float(row["served_bits"]) / 1.0e6
        urgency = float(row["urgency_served_bits"]) / 1.0e6
        penalty = float(row["projection_penalty"]) + float(row["residual_penalty"])
        base = payload - penalty
        if abs(payload - served) > 1.0e-8:
            problems.append(f"scenario {scenario} step payload term does not match served bits")
            break
        if abs(float(row["urgency_term"]) - urgency) > 1.0e-8:
            problems.append(f"scenario {scenario} urgency term does not match urgency served bits")
            break
        if abs(float(row["penalty"]) - penalty) > 1.0e-8 or abs(float(row["base_reward"]) - base) > 1.0e-8:
            problems.append(f"scenario {scenario} base reward does not match payload minus penalties")
            break
    if problems:
        return problems
    if tuple(sorted(grouped)) != EXPECTED_SCENARIOS:
        return ["step scenarios do not match the E2 scenario list"]
    for scenario in EXPECTED_SCENARIOS:
        rows = grouped[scenario]
        if len(rows) != EXPECTED_STEPS_PER_EPISODE:
            return [f"scenario {scenario} does not have 48 steps"]
        if [int(row["time_step"]) for row in rows] != list(range(EXPECTED_STEPS_PER_EPISODE)):
            return [f"scenario {scenario} time steps are not 0 through 47"]
    by_episode = {int(row["scenario_id"]): row for row in episodes}
    for scenario in EXPECTED_SCENARIOS:
        episode = by_episode[scenario]
        rows = grouped[scenario]
        served = sum(float(row["served_bits"]) for row in rows)
        urgency = sum(float(row["urgency_served_bits"]) for row in rows)
        penalty = sum(float(row["penalty"]) for row in rows)
        if abs(served - float(episode["served_bits"])) > 1.0:
            return [f"scenario {scenario} step served bits do not match the episode row"]
        if abs(urgency - float(episode["urgency_served_bits"])) > 1.0:
            return [f"scenario {scenario} step urgency bits do not match the episode row"]
        if abs(penalty - float(episode["sum_penalty"])) > 1.0e-8:
            return [f"scenario {scenario} step penalties do not match the episode row"]
        if abs(float(rows[-1]["payload_bits"]) - float(episode["payload_bits"])) > 1.0:
            return [f"scenario {scenario} cumulative payload does not match the episode row"]
    return []
