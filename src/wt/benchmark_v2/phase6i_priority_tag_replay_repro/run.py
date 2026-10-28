"""One-scenario reproducibility diagnostic. It does not train or merge."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import LOG_STD_INIT
from wt.benchmark_v2.phase6i_priority_tag_reward.actor import load_formal_actor
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import FORMAL_ACTOR, assert_dev_scenario
from wt.round30.ppo import seed_everything

from .compare import (
    bits_within_tolerance,
    cause_letter,
    choose_gate,
    first_difference,
    path_matches,
    saved_gaps,
    step_diffs,
)
from .protocol import (
    ACCOUNTING_TOLERANCE_BITS,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    INHERITED_PROCESS_SEED,
    INHERITED_TORCH_THREADS,
    INSTRUCTION_COMMIT,
    MODEL_SEED,
    OUTPUT,
    PATH_EQUAL_METERS,
    REPORT,
    ROOT,
    SCENARIO_ID,
    SCHEMA_VERSION,
    SOURCE_DIAGNOSTIC_COMMIT,
    SOURCE_EXPERIMENT_COMMIT,
    SOURCE_ROWS,
    WATCHED_RELATIVE,
)
from .report import render_report
from .trace import trace_episode


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _blob_pair(relative: str) -> dict:
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", "diff", "--quiet", SOURCE_EXPERIMENT_COMMIT, "--", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "sha256": _sha256(ROOT / relative),
        "matches_source": completed.returncode == 0,
    }


def _load_saved() -> dict:
    with SOURCE_ROWS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "L0" and int(float(row["model_seed"])) == int(MODEL_SEED) and int(float(row["scenario_id"])) == int(SCENARIO_ID):
                return {key: float(value) if key != "variant" else value for key, value in row.items()}
    raise RuntimeError("saved L0 row is missing")


def _runtime() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads_before": int(torch.get_num_threads()),
        "deterministic_algorithms_before": bool(torch.are_deterministic_algorithms_enabled()),
        "requirements_numpy": "1.21.6",
        "requirements_torch": "1.13.0",
        "head": _git(["rev-parse", "HEAD"]),
        "worktree": _git(["status", "--porcelain"]),
    }


def _csv_rows(comparison: str, left: list[dict], right: list[dict], first_index: int | None) -> list[dict]:
    rows = []
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        diffs = step_diffs(lhs, rhs)
        rows.append(
            {
                "comparison": comparison,
                "step_index": index,
                "is_first_mismatch": bool(first_index is not None and index == int(first_index)),
                "obs_equal": lhs["obs_sha256"] == rhs["obs_sha256"],
                "association_equal": bool(np.array_equal(lhs["association"], rhs["association"])),
                "payload_left": float(lhs["payload"]),
                "payload_right": float(rhs["payload"]),
                **{f"{key}_abs": float(value) for key, value in diffs.items()},
            }
        )
    return rows


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _cell(value):
    if isinstance(value, float):
        return format(value, ".16g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _publish(summary: dict, rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "step_diff.csv", rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": int(SCENARIO_ID),
        "model_seed": int(MODEL_SEED),
        "accounting_tolerance_bits": ACCOUNTING_TOLERANCE_BITS,
        "path_equal_meters": PATH_EQUAL_METERS,
        "tolerance_changed_after_results": False,
        "retrained": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "merged": False,
        "gate": summary.get("gate"),
        "cause_letter": summary.get("cause_letter"),
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")


def main() -> None:
    runtime_before = _runtime()
    blobs = {relative: _blob_pair(relative) for relative in WATCHED_RELATIVE}
    formal_sha = _sha256(FORMAL_ACTOR)
    hashes_ok = bool(
        formal_sha == FORMAL_ACTOR_SHA256
        and all(item["matches_source"] for item in blobs.values())
    )
    split_ok = True
    split_error = ""
    try:
        assert_dev_scenario(int(SCENARIO_ID))
    except RuntimeError as exc:
        split_ok = False
        split_error = str(exc)
    saved = _load_saved()
    summary = {
        "instruction_commit": INSTRUCTION_COMMIT,
        "source_experiment_commit": SOURCE_EXPERIMENT_COMMIT,
        "source_diagnostic_commit": SOURCE_DIAGNOSTIC_COMMIT,
        "runtime_before": runtime_before,
        "formal_actor_sha256": formal_sha,
        "blobs_match_source_experiment": hashes_ok,
        "blob_pairs": blobs,
        "split_ok": split_ok,
        "split_error": split_error,
        "scenario_id": int(SCENARIO_ID),
        "accounting_tolerance_bits": ACCOUNTING_TOLERANCE_BITS,
        "tolerance_changed_after_results": False,
        "retrained": False,
        "merged": False,
        "step_logs_in_saved_evaluation": False,
        "inherited_process": {
            "torch_threads": int(INHERITED_TORCH_THREADS),
            "process_seed": int(INHERITED_PROCESS_SEED),
            "source": "phase6i train_variant calls torch.set_num_threads(1) and seed_everything(model_seed); evaluate_all does not reset either flag",
        },
    }
    if not split_ok or formal_sha != FORMAL_ACTOR_SHA256:
        summary["gate"] = GATE_BLOCKED
        summary["cause_letter"] = "F"
        summary["problems"] = [split_error or "formal checkpoint hash differs"]
        _publish(summary, [])
        print(summary["gate"], summary["cause_letter"])
        return
    actor = load_formal_actor(FORMAL_ACTOR, LOG_STD_INIT)
    actor.eval()
    fresh = trace_episode(actor, instrumented=False)
    instrumented = trace_episode(actor, instrumented=True)
    torch.set_num_threads(int(INHERITED_TORCH_THREADS))
    seed_everything(int(INHERITED_PROCESS_SEED))
    try:
        inherited = trace_episode(actor, instrumented=False)
    except RuntimeError as exc:
        summary["gate"] = "REPLAY_DIVERGENCE_INCONCLUSIVE"
        summary["cause_letter"] = "F"
        summary["problems"] = [f"inherited process flags did not complete the episode: {exc}"]
        summary["formal_actor_sha256_after"] = _sha256(FORMAL_ACTOR)
        _publish(summary, [])
        print(summary["gate"], summary["cause_letter"])
        return
    fresh_vs_instrument = first_difference(fresh["steps"], instrumented["steps"])
    fresh_vs_inherited = first_difference(fresh["steps"], inherited["steps"])
    fresh_gaps = saved_gaps(fresh["episode"], saved)
    inherited_gaps = saved_gaps(inherited["episode"], saved)
    instrument_gaps = saved_gaps(instrumented["episode"], saved)
    reset_equal = bool(
        fresh["reset"]["q_sha256"] == inherited["reset"]["q_sha256"]
        and fresh["reset"]["position_sha256"] == inherited["reset"]["position_sha256"]
        and float(fresh["reset"]["payload"]) == float(inherited["reset"]["payload"])
    )
    inherited_matches = bool(bits_within_tolerance(inherited_gaps) and path_matches(inherited_gaps))
    fresh_matches = bool(bits_within_tolerance(fresh_gaps) and path_matches(fresh_gaps))
    instrumented_same = fresh_vs_instrument is None
    first_field = None if fresh_vs_inherited is None else fresh_vs_inherited.get("field")
    summary.update(
        {
            "runtime_after_inherited_flags": {
                "torch_threads": int(torch.get_num_threads()),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            },
            "reset_equal": reset_equal,
            "fresh_reset": {key: fresh["reset"][key] for key in ("t", "payload", "q_sha256", "position_sha256")},
            "inherited_reset": {key: inherited["reset"][key] for key in ("t", "payload", "q_sha256", "position_sha256")},
            "fresh_vs_instrumented": fresh_vs_instrument,
            "fresh_vs_inherited": fresh_vs_inherited,
            "fresh_gaps": fresh_gaps,
            "instrumented_gaps": instrument_gaps,
            "inherited_gaps": inherited_gaps,
            "fresh_matches_saved": fresh_matches,
            "inherited_matches_saved": inherited_matches,
            "instrumented_same_as_fresh": instrumented_same,
            "formal_actor_sha256_after": _sha256(FORMAL_ACTOR),
        }
    )
    summary["cause_letter"] = cause_letter(
        hashes_ok=bool(hashes_ok and summary["formal_actor_sha256_after"] == FORMAL_ACTOR_SHA256),
        split_ok=split_ok,
        reset_equal=reset_equal,
        inherited_matches=inherited_matches,
        first_field=first_field,
    )
    summary["gate"] = choose_gate(
        hashes_ok=bool(hashes_ok and summary["formal_actor_sha256_after"] == FORMAL_ACTOR_SHA256),
        split_ok=split_ok,
        instrumented_same=instrumented_same,
        inherited_matches=inherited_matches,
        first_step_found=fresh_vs_inherited is not None,
        fresh_matches=fresh_matches,
    )
    rows = _csv_rows("fresh_vs_instrumented", fresh["steps"], instrumented["steps"], None if fresh_vs_instrument is None else fresh_vs_instrument.get("step_index"))
    rows.extend(
        _csv_rows(
            "fresh_vs_inherited",
            fresh["steps"],
            inherited["steps"],
            None if fresh_vs_inherited is None else fresh_vs_inherited.get("step_index"),
        )
    )
    _publish(summary, rows)
    print(summary["gate"], summary["cause_letter"], None if fresh_vs_inherited is None else fresh_vs_inherited.get("step_index"))


if __name__ == "__main__":
    main()
