"""Recompute the frozen E10 table. This module does not rewrite those artifacts."""
from __future__ import annotations

import csv
import json
import subprocess

import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.criteria import choose_gate, score_l3
from wt.benchmark_v2.phase6i_priority_tag_direct_service_reward.protocol import (
    GATE_FAIL,
    PRIOR_CHECKPOINT_SHA256,
)
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import LEGACY_ACTOR
from wt.benchmark_v2.phase6i_priority_tag_reward.smoke import file_sha256, watched_hashes

from .protocol import E10_RESULT_COMMIT, E10_ROWS, E10_SOURCE_COMMIT, E10_SUMMARY, L3_CHECKPOINT_SHA256, ROOT


def _checkpoint_path(variant: str, seed: int):
    if variant == "L0":
        return ROOT / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo" / "checkpoints" / "phase6g_seed0.pt"
    if variant == "L3":
        return ROOT / "results" / "phase6i_priority_tag_direct_service_reward" / "checkpoints" / f"l3_seed{int(seed)}.pt"
    return ROOT / "results" / "phase6i_priority_tag_reward" / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"


def load_e10_rows() -> list[dict]:
    with E10_ROWS.open(encoding="utf-8", newline="") as handle:
        parsed = []
        for raw in csv.DictReader(handle):
            item = {"variant": raw["variant"]}
            for key, value in raw.items():
                if key == "variant" or value is None or value == "":
                    continue
                if value in ("true", "false"):
                    item[key] = value == "true"
                    continue
                item[key] = float(value)
            item["model_seed"] = int(item["model_seed"])
            item["scenario_id"] = int(item["scenario_id"])
            parsed.append(item)
    return parsed


def row_inventory(rows: list[dict]) -> dict:
    keys = [(row["variant"], int(row["model_seed"]), int(row["scenario_id"])) for row in rows]
    expected_jobs = [("L0", 0)] + [(variant, seed) for variant in ("L1", "L2", "L3") for seed in (0, 1, 2)]
    pool = [int(scenario) for scenario in DEV_POOL]
    problems = []
    if len(keys) != len(set(keys)):
        problems.append("variant/model_seed/scenario_id combinations are not unique")
    if len(rows) != 320:
        problems.append(f"row count is {len(rows)}")
    for variant, seed in expected_jobs:
        found = sorted(row["scenario_id"] for row in rows if row["variant"] == variant and int(row["model_seed"]) == int(seed))
        if found != pool:
            problems.append(f"{variant} seed {seed} is not aligned to the development pool")
    return {"rows": len(rows), "unique_keys": len(set(keys)), "problems": problems, "development_pool": [pool[0], pool[-1]]}


def recompute_gate(rows: list[dict]) -> dict:
    score = score_l3(rows)
    gate = choose_gate(smoke_ok=True, train_ok=True, replay_ok=True, hashes_ok=True, score=score)
    saved = json.loads(E10_SUMMARY.read_text(encoding="utf-8"))
    gaps = []
    for fresh, old in zip(score["seeds"], saved["score"]["seeds"]):
        for metric in ("tagged_versus_l0", "payload_versus_l0", "tagged_versus_l2", "payload_versus_l2"):
            for bound in ("mean", "ci_low", "ci_high"):
                gap = abs(float(fresh[metric][bound]) - float(old[metric][bound]))
                if gap > 1.0e-6:
                    gaps.append({"seed": fresh["model_seed"], "metric": metric, "bound": bound, "gap": gap})
    safety = []
    index = {(row["variant"], int(row["model_seed"]), int(row["scenario_id"])): row for row in rows}
    expected = ((1, 321123), (2, 321104))
    for seed, scenario in expected:
        current = index[("L3", seed, scenario)]["safety_violations"]
        baseline = index[("L0", 0, scenario)]["safety_violations"]
        safety.append(
            {
                "model_seed": seed,
                "scenario_id": scenario,
                "l3_safety_violations": current,
                "l0_safety_violations": baseline,
                "matches_reported_event": bool(current == 1.0 and baseline == 0.0),
            }
        )
    return {
        "gate": gate,
        "matches_saved_fail_gate": bool(gate == GATE_FAIL == saved["scientific_gate"]),
        "passing_seeds": int(score["passing_seeds"]),
        "interval_gaps": gaps,
        "safety_events": safety,
        "seeds": score["seeds"],
    }


def checkpoint_inventory() -> dict:
    problems = []
    found = {}
    for (variant, seed), expected in PRIOR_CHECKPOINT_SHA256.items():
        digest = file_sha256(_checkpoint_path(variant, int(seed)))
        found[f"{variant}_seed{int(seed)}"] = digest
        if digest != expected:
            problems.append(f"{variant} seed {seed} hash changed")
    for seed, expected in L3_CHECKPOINT_SHA256.items():
        digest = file_sha256(_checkpoint_path("L3", int(seed)))
        found[f"L3_seed{int(seed)}"] = digest
        if digest != expected:
            problems.append(f"L3 seed {seed} hash changed")
    legacy = file_sha256(LEGACY_ACTOR)
    found["legacy"] = legacy
    if legacy != "44d44cf96f666aa7810bcd9b6546e2956d35fbf280e5f312c983444b25dbce14":
        problems.append("legacy actor hash changed")
    return {"hashes": found, "problems": problems, "watched": watched_hashes()}


def source_commit_audit() -> dict:
    listing = subprocess.run(
        ["git", "-c", "core.longpaths=true", "show", "--name-only", "--format=", E10_SOURCE_COMMIT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    paths = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    keys = []
    recorded = []
    for seed in L3_CHECKPOINT_SHA256:
        blob = torch.load(_checkpoint_path("L3", int(seed)), map_location="cpu", weights_only=False)
        keys = sorted(str(key) for key in blob.keys())
        recorded.append(
            {
                "model_seed": int(seed),
                "source_commit": blob.get("source_commit"),
                "has_implementation_hash": any("code" in str(key) or "worktree" in str(key) for key in blob.keys()),
            }
        )
    return {
        "recorded_source_commit": E10_SOURCE_COMMIT,
        "git_exit_code": int(listing.returncode),
        "source_commit_paths": paths,
        "source_commit_is_instruction_only": paths == ["docs/research/HTW_UrgencyAware_Direct_Tag_Reward_Experiment_Instructions_CN.md"],
        "checkpoint_keys": keys,
        "checkpoints": recorded,
        "implementation_bytes_at_training_time": "missing",
        "result_commit_after_training": E10_RESULT_COMMIT,
        "note": "The checkpoint records the instruction commit and no implementation digest. Current source is not treated as proof of the bytes that ran.",
    }
