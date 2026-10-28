"""E9 runner. One preflight episode, then the saved 224-row table if the budget allows."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL
from wt.benchmark_v2.phase6i_priority_tag_eval_context.context import file_sha256
from wt.benchmark_v2.phase6i_priority_tag_eval_context.protocol import HASHED_RELATIVE
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR, OUTPUT as REWARD_OUTPUT, TRAIN_SEEDS, assert_dev_scenario
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.analyze import summarize_attribution
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.decide import next_design
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.reconcile import load_saved_rows, reconcile_saved, saved_index
from wt.benchmark_v2.phase6i_priority_tag_reward_attribution.replay import attach_origin_splits, load_actor, public_row

from .replay import replay_one

from .compare import choose_gate, episode_status
from .context import CONTEXT_LABEL, apply_historical_context, context_is_historical
from .protocol import (
    CHECKPOINT_SHA256,
    DECOMPOSITION_TOLERANCE_BITS,
    E8_COMMIT,
    E8_CONTEXT,
    EPISODE_STEPS,
    FULL_EPISODE_COUNT,
    GATE_BLOCKED,
    GLOBAL_SEED_ROLE,
    HISTORICAL_INHERITED_GLOBAL_SEED,
    INSTRUCTION_COMMIT,
    OUTPUT,
    PATH_TOLERANCE_M,
    PREFLIGHT_MODEL_SEED,
    PREFLIGHT_SCENARIO,
    PREFLIGHT_VARIANT,
    REPORT,
    ROOT,
    SCHEMA_VERSION,
    SOURCE_RESULT_COMMIT,
    TIME_LIMIT_SECONDS,
)
from .report import render_report


STEP_FIELDS = (
    "variant",
    "model_seed",
    "scenario_id",
    "env_reset_seed",
    "assignment_rng_seed",
    "process_rng_seed",
    "process_rng_role",
    "step",
    "age_service_bits",
    "tag_increment_service_bits",
    "combined_urgency_served_bits",
    "served_tagged_bits",
    "served_ordinary_bits",
    "tagged_served_below_full_boost_bits",
    "tagged_served_fully_saturated_bits",
    "ledger_urgency_bits",
    "decomposition_gap_bits",
    "served_bits",
    "dropped_bits",
    "payload_bits",
    "q_bits",
    "path_step_m",
    "ledger_gap_bits",
)


def _cell(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return value


def _write_rows(path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        public = [public_row(row) for row in rows]
        names: list[str] = []
        for row in public:
            for key in row:
                if key not in names and not isinstance(row[key], (dict, list)):
                    names.append(str(key))
        fields = tuple(names)
        payload = public
    else:
        payload = rows
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in payload:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _checkpoint_path(variant: str, seed: int):
    if variant == "L0":
        return FORMAL_ACTOR
    return REWARD_OUTPUT / "checkpoints" / f"{variant.lower()}_seed{int(seed)}.pt"


def _jobs() -> list[tuple[str, int]]:
    return [("L0", 0)] + [(variant, int(seed)) for variant in ("L1", "L2") for seed in TRAIN_SEEDS]


def _code_hashes() -> dict[str, str]:
    return {relative: file_sha256(ROOT / relative) for relative in HASHED_RELATIVE}


def _checkpoint_hashes() -> dict[str, str]:
    found = {}
    for variant, seed in _jobs():
        path = _checkpoint_path(variant, seed)
        found[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
    return found


def _problems_before(saved_rows: list[dict], reconciliation: dict, code_hashes: dict, checkpoints: dict) -> list[str]:
    problems = []
    if not reconciliation["matches"]:
        problems.append("saved csv and summary do not reconcile")
    if len(saved_rows) != int(FULL_EPISODE_COUNT):
        problems.append(f"saved row count {len(saved_rows)} != {FULL_EPISODE_COUNT}")
    if len(DEV_POOL) * len(_jobs()) != int(FULL_EPISODE_COUNT):
        problems.append("development pool and job count do not make 224 episodes")
    e8 = json.loads(E8_CONTEXT.read_text(encoding="utf-8"))
    for relative, expected in e8["code_hashes"].items():
        if code_hashes.get(relative) != expected:
            problems.append(f"code hash drifted: {relative}")
    for (variant, seed), expected in CHECKPOINT_SHA256.items():
        path = _checkpoint_path(variant, seed)
        actual = checkpoints[str(path.relative_to(ROOT)).replace("\\", "/")]
        if actual != expected:
            problems.append(f"checkpoint hash drifted: {variant} seed {seed}")
    for scenario in DEV_POOL:
        assert_dev_scenario(int(scenario))
    for row in saved_rows:
        scenario = int(row["scenario_id"])
        assert_dev_scenario(scenario)
        if scenario not in set(int(item) for item in DEV_POOL):
            problems.append(f"saved scenario {scenario} is outside the development pool")
    if int(PREFLIGHT_SCENARIO) not in set(int(item) for item in DEV_POOL):
        problems.append("preflight scenario is outside the development pool")
    return problems


def _apply_checked() -> dict:
    applied = apply_historical_context()
    if not context_is_historical(applied):
        raise RuntimeError("process context is not the documented historical context")
    return applied


def _run_episode(actor, variant: str, seed: int, scenario: int, saved: dict) -> tuple[dict, list[dict], dict]:
    if int(actor_steps_expected()) != int(EPISODE_STEPS):
        raise RuntimeError("episode length drifted")
    row, steps = replay_one(actor, variant, int(seed), int(scenario))
    if len(steps) != int(EPISODE_STEPS):
        raise RuntimeError(f"{variant} seed {seed} scenario {scenario} returned {len(steps)} steps")
    status = episode_status(row, saved)
    row["matches_saved"] = bool(status["matches_saved"])
    row["max_bit_gap"] = float(status["max_bit_gap"])
    row["max_saved_gap_bits"] = float(status["max_bit_gap"])
    row["path_gap_m"] = float(status["path_gap_m"])
    row["identity_gap_bits"] = float(status["identity_gap_bits"])
    row["ledger_urgency_gap_bits"] = float(status["ledger_urgency_gap_bits"])
    return row, steps, status


def actor_steps_expected() -> int:
    return int(EPISODE_STEPS)


def _head() -> str:
    return subprocess.check_output(
        ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _manifest(applied: dict, code_hashes: dict, checkpoints: dict) -> dict:
    return {
        "context_label": CONTEXT_LABEL,
        "historical_note": (
            "Seed 2 is the inherited process state from the last train_variant call before evaluate_all. "
            "It is reapplied before every episode and is not a canonical evaluation seed."
        ),
        "process_rng_seed": int(HISTORICAL_INHERITED_GLOBAL_SEED),
        "process_rng_role": GLOBAL_SEED_ROLE,
        "canonical_explicit_seed": None,
        "torch_num_threads": int(applied["torch_num_threads"]),
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "actor_eval_mode": True,
        "deterministic_actions": True,
        "env_reset_seed": "scenario_id",
        "assignment_rng_seed": "7000000+scenario_id",
        "reapplied_before_every_episode": True,
        "preflight": {"variant": PREFLIGHT_VARIANT, "model_seed": PREFLIGHT_MODEL_SEED, "scenario_id": PREFLIGHT_SCENARIO},
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "git_commit": _head(),
        "instruction_commit": INSTRUCTION_COMMIT,
        "e8_commit": E8_COMMIT,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "code_hashes": code_hashes,
        "checkpoint_sha256": checkpoints,
        "applied": applied,
    }


def _summary_shell(reconciliation: dict, applied: dict) -> dict:
    return {
        "gate": GATE_BLOCKED,
        "instruction_commit": INSTRUCTION_COMMIT,
        "e8_commit": E8_COMMIT,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "context_label": CONTEXT_LABEL,
        "process_rng_role": GLOBAL_SEED_ROLE,
        "process_rng_seed": int(HISTORICAL_INHERITED_GLOBAL_SEED),
        "canonical_explicit_seed": None,
        "reconciliation": reconciliation,
        "preflight": {},
        "replay": {"completed_episodes": 0, "elapsed_seconds": 0.0, "stopped_early": True},
        "formal_hash_unchanged": False,
        "applied_context": applied,
        "attribution": None,
        "locked_rule_letter": None,
        "failure": None,
        "problems": [],
        "retrained": False,
        "merged": False,
        "tolerances": {
            "accounting_bits": 1.0,
            "path_m": float(PATH_TOLERANCE_M),
            "decomposition_bits": float(DECOMPOSITION_TOLERANCE_BITS),
        },
    }


def _publish(summary: dict, rows: list[dict], steps: list[dict], manifest: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_rows(OUTPUT / "scenario_attribution.csv", rows)
    _write_rows(OUTPUT / "step_attribution.csv", steps, STEP_FIELDS)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "runtime_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema = {
        "schema_version": SCHEMA_VERSION,
        "synthetic_tag": True,
        "gate": summary.get("gate"),
        "instruction_commit": INSTRUCTION_COMMIT,
        "e8_commit": E8_COMMIT,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "process_rng_role": GLOBAL_SEED_ROLE,
        "canonical_explicit_seed": None,
        "accounting_tolerance_bits": 1.0,
        "path_tolerance_m": float(PATH_TOLERANCE_M),
        "decomposition_tolerance_bits": float(DECOMPOSITION_TOLERANCE_BITS),
        "retrained": False,
        "merged": False,
        "held_out_splits_read": False,
        "thresholds_changed_after_results": False,
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")


def _replay_fields(rows: list[dict], elapsed: float, stopped: bool) -> dict:
    if not rows:
        return {
            "completed_episodes": 0,
            "elapsed_seconds": float(elapsed),
            "stopped_early": stopped,
            "max_bit_gap": None,
            "max_path_gap_m": None,
            "max_identity_gap_bits": None,
            "max_ledger_urgency_gap_bits": None,
        }
    return {
        "completed_episodes": len(rows),
        "elapsed_seconds": float(elapsed),
        "stopped_early": bool(stopped),
        "max_bit_gap": float(max(float(row["max_bit_gap"]) for row in rows)),
        "max_path_gap_m": float(max(float(row["path_gap_m"]) for row in rows)),
        "max_identity_gap_bits": float(max(float(row["identity_gap_bits"]) for row in rows)),
        "max_ledger_urgency_gap_bits": float(max(float(row["ledger_urgency_gap_bits"]) for row in rows)),
    }


def main() -> None:
    saved_rows = load_saved_rows()
    index = saved_index(saved_rows)
    reconciliation = reconcile_saved()
    code_hashes = _code_hashes()
    checkpoints = _checkpoint_hashes()
    applied = _apply_checked()
    manifest = _manifest(applied, code_hashes, checkpoints)
    summary = _summary_shell(reconciliation, applied)
    problems = _problems_before(saved_rows, reconciliation, code_hashes, checkpoints)
    summary["problems"] = problems
    if problems:
        summary["formal_hash_unchanged"] = code_hashes.get(
            "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"
        ) == CHECKPOINT_SHA256[("L0", 0)]
        _publish(summary, [], [], manifest)
        print(summary["gate"])
        return
    actors = {}
    key = (PREFLIGHT_VARIANT, int(PREFLIGHT_MODEL_SEED))
    actors[key] = load_actor(*key)
    saved = index[(PREFLIGHT_VARIANT, int(PREFLIGHT_MODEL_SEED), int(PREFLIGHT_SCENARIO))]
    started = time.perf_counter()
    row, steps, status = _run_episode(actors[key], PREFLIGHT_VARIANT, PREFLIGHT_MODEL_SEED, PREFLIGHT_SCENARIO, saved)
    elapsed = float(time.perf_counter() - started)
    estimate = elapsed * float(FULL_EPISODE_COUNT)
    summary["preflight"] = {
        "scenario_id": int(PREFLIGHT_SCENARIO),
        "seconds": elapsed,
        "estimated_seconds": estimate,
        "matches_saved": bool(status["matches_saved"]),
        "max_bit_gap": float(status["max_bit_gap"]),
        "path_gap_m": float(status["path_gap_m"]),
        "identity_gap_bits": float(status["identity_gap_bits"]),
        "ledger_urgency_gap_bits": float(status["ledger_urgency_gap_bits"]),
    }
    print(
        f"preflight gap={status['max_bit_gap']:.6g} path={status['path_gap_m']:.6g} seconds={elapsed:.3f} estimate={estimate:.1f}",
        flush=True,
    )
    if not status["matches_saved"] or estimate > float(TIME_LIMIT_SECONDS):
        if estimate > float(TIME_LIMIT_SECONDS):
            summary["problems"].append("full replay estimate exceeds 10 minutes")
        if not status["matches_saved"]:
            summary["problems"].append("preflight episode missed the saved row")
            summary["failure"] = {
                "variant": PREFLIGHT_VARIANT,
                "model_seed": int(PREFLIGHT_MODEL_SEED),
                "scenario_id": int(PREFLIGHT_SCENARIO),
                "bit_gaps": status["bit_gaps"],
                "path_gap_m": float(status["path_gap_m"]),
                "identity_gap_bits": float(status["identity_gap_bits"]),
            }
        summary["replay"] = _replay_fields([row], elapsed, True)
        summary["locked_rule_letter"] = "C"
        after = _code_hashes()
        summary["formal_hash_unchanged"] = after == code_hashes
        _publish(summary, [row], steps, manifest)
        print(summary["gate"])
        return
    rows = []
    all_steps = []
    batch_started = time.perf_counter()
    stopped = False
    failure = None
    for variant, seed in _jobs():
        actor_key = (variant, int(seed))
        if actor_key not in actors:
            actors[actor_key] = load_actor(variant, int(seed))
        actor = actors[actor_key]
        if bool(actor.training):
            raise RuntimeError(f"{variant} seed {seed} left evaluation mode")
        for scenario in DEV_POOL:
            saved_row = index[(variant, int(seed), int(scenario))]
            episode, episode_steps, episode_status_row = _run_episode(actor, variant, int(seed), int(scenario), saved_row)
            rows.append(episode)
            all_steps.extend(episode_steps)
            print(
                f"{variant} seed {int(seed)} scenario {int(scenario)} bit={episode['max_bit_gap']:.6g} "
                f"path={episode['path_gap_m']:.6g} age={episode['age_service_bits']:.6g} "
                f"tag_boost={episode['tag_increment_service_bits']:.6g}",
                flush=True,
            )
            if not episode_status_row["matches_saved"]:
                stopped = True
                failure = {
                    "variant": variant,
                    "model_seed": int(seed),
                    "scenario_id": int(scenario),
                    "bit_gaps": episode_status_row["bit_gaps"],
                    "path_gap_m": float(episode_status_row["path_gap_m"]),
                    "identity_gap_bits": float(episode_status_row["identity_gap_bits"]),
                }
                break
        if stopped:
            break
    batch_elapsed = float(time.perf_counter() - batch_started)
    after = _code_hashes()
    summary["formal_hash_unchanged"] = after == code_hashes and _checkpoint_hashes() == checkpoints
    summary["replay"] = _replay_fields(rows, batch_elapsed, stopped or len(rows) != int(FULL_EPISODE_COUNT))
    summary["failure"] = failure
    reproduced = bool(len(rows) == int(FULL_EPISODE_COUNT) and all(row["matches_saved"] for row in rows))
    decomposition_ok = bool(
        reproduced
        and float(summary["replay"]["max_identity_gap_bits"]) <= float(DECOMPOSITION_TOLERANCE_BITS)
        and float(summary["replay"]["max_ledger_urgency_gap_bits"]) <= float(DECOMPOSITION_TOLERANCE_BITS)
    )
    attribution = None
    l2_groups = None
    if reproduced and decomposition_ok and summary["formal_hash_unchanged"]:
        attach_origin_splits(rows)
        attribution = summarize_attribution(rows)
        l2_groups = [group for group in attribution["groups"] if group["variant"] == "L2"]
        summary["l2_tagged_positive_seeds"] = int(
            sum(float(group["intervals"]["served_tagged_bits"]["mean"]) > 0.0 for group in l2_groups)
        )
    summary["attribution"] = attribution
    summary["gate"] = choose_gate(
        integrity_ok=bool(summary["formal_hash_unchanged"] and not problems),
        budget_ok=True,
        reproduced=reproduced,
        decomposition_ok=decomposition_ok,
        l2_groups=l2_groups,
    )
    if summary["gate"] == "PRIORITY_TAG_ATTRIBUTION_BLOCKED":
        summary["locked_rule_letter"] = "C"
    elif l2_groups is not None:
        summary["locked_rule_letter"] = next_design(True, True, bool(reconciliation["matches"]), l2_groups)
    _publish(summary, rows, all_steps, manifest)
    print(summary["gate"])


if __name__ == "__main__":
    main()
