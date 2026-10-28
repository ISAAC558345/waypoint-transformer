"""Run the attribution diagnostic. This module does not train or merge."""
from __future__ import annotations

import csv
import hashlib
import json

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL
from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import TRAIN_SEEDS, WATCHED_PATHS

from .analyze import summarize_attribution
from .decide import next_design
from .protocol import (
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    GATE_COMPLETE,
    GATE_TIME_LIMIT,
    INSTRUCTION_COMMIT,
    OUTPUT,
    REPORT,
    ROOT,
    SCHEMA_VERSION,
    SOURCE_OUTPUT,
    SOURCE_REPORT,
    SOURCE_RESULT_COMMIT,
    SOURCE_ROWS,
    SOURCE_SUMMARY,
    TIME_LIMIT_SECONDS,
)
from .reconcile import load_saved_rows, reconcile_saved, saved_index
from .replay import load_actor, public_row, replay_saved, uninstrumented_gaps
from .report import render_report


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes() -> dict:
    paths = [ROOT / item for item in WATCHED_PATHS]
    paths.extend(
        [
            SOURCE_ROWS,
            SOURCE_SUMMARY,
            SOURCE_REPORT,
            ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag" / "ledger.py",
            ROOT / "src" / "wt" / "benchmark_v2" / "phase6i_priority_tag" / "follow.py",
        ]
    )
    paths.extend(sorted((SOURCE_OUTPUT / "checkpoints").glob("*.pt")))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path) for path in paths}


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public = [public_row(row) for row in rows]
    if not public:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in public:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in public:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _cell(value):
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _schema(summary: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "synthetic_tag": True,
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "instruction_commit": INSTRUCTION_COMMIT,
        "formal_actor_sha256": summary.get("formal_actor_sha256"),
        "next_design": summary.get("next_design"),
        "thresholds_changed_after_results": False,
        "retrained": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "merged": False,
    }


def _publish(summary: dict, rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "scenario_attribution.csv", rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "schema.json").write_text(json.dumps(_schema(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")


def _base(reconciliation: dict, before: dict, probe: dict, replay: dict) -> dict:
    return {
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "instruction_commit": INSTRUCTION_COMMIT,
        "formal_actor_sha256": before.get(
            "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"
        ),
        "reconciliation": reconciliation,
        "probe": probe,
        "replay": replay,
        "watched_sha256_before": before,
        "attribution": None,
        "next_design": None,
        "merged": False,
        "retrained": False,
        "problems": list(reconciliation["problems"]),
    }


def main() -> None:
    before = _hashes()
    reconciliation = reconcile_saved()
    index = saved_index(load_saved_rows())
    full_count = len(DEV_POOL) * (1 + 2 * len(TRAIN_SEEDS))
    probe_rows, probe_meta = replay_saved(index, scenarios=[int(DEV_POOL[0])], jobs=[("L0", 0)])
    estimate = float(probe_meta["first_scenario_seconds"]) * float(full_count)
    probe = {
        "scenario_id": int(DEV_POOL[0]),
        "first_scenario_seconds": probe_meta["first_scenario_seconds"],
        "estimated_seconds": estimate,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "full_episode_count": full_count,
        "matches_saved": bool(probe_rows[0]["matches_saved"]),
        "decomposition_ok": bool(probe_rows[0]["decomposition_ok"]),
        "max_saved_gap_bits": float(probe_rows[0]["max_saved_gap_bits"]),
    }
    replay = {
        "completed_episodes": int(probe_meta["completed_episodes"]),
        "elapsed_seconds": float(probe_meta["elapsed_seconds"]),
        "stopped_early": bool(probe_meta["stopped_early"]),
        "max_saved_gap_bits": float(probe_rows[0]["max_saved_gap_bits"]),
        "max_decomposition_gap_bits": float(probe_rows[0]["decomposition_gap_bits"]),
    }
    summary = _base(reconciliation, before, probe, replay)
    if before.get("src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt") != FORMAL_ACTOR_SHA256:
        summary["problems"].append("formal checkpoint hash changed before replay")
        summary["gate"] = GATE_BLOCKED
        summary["next_design"] = "C"
        _publish(summary, probe_rows)
        print(summary["gate"], summary["next_design"])
        return
    if not probe["matches_saved"] or not probe["decomposition_ok"]:
        summary["gate"] = GATE_BLOCKED
        summary["next_design"] = "C"
        summary["problems"].append("first replay does not reproduce the saved evaluation")
        _publish(summary, probe_rows)
        print(summary["gate"], summary["next_design"])
        return
    if not reconciliation["matches"]:
        summary["problems"].append("saved csv, summary, or report do not reconcile")
    if estimate > float(TIME_LIMIT_SECONDS):
        summary["gate"] = GATE_TIME_LIMIT
        summary["problems"].append("full replay estimate exceeds 10 minutes")
        _publish(summary, probe_rows)
        print(summary["gate"], estimate)
        return
    rows, meta = replay_saved(index)
    after = _hashes()
    summary["replay"] = {
        "completed_episodes": int(meta["completed_episodes"]),
        "elapsed_seconds": float(meta["elapsed_seconds"]),
        "stopped_early": bool(meta["stopped_early"]),
        "max_saved_gap_bits": float(max(float(row["max_saved_gap_bits"]) for row in rows)),
        "max_decomposition_gap_bits": float(max(float(row["decomposition_gap_bits"]) for row in rows)),
    }
    summary["watched_sha256_after"] = after
    if after != before:
        changed = sorted(key for key in before if before[key] != after.get(key))
        summary["problems"].append("watched files changed during replay: " + ",".join(changed))
        summary["gate"] = GATE_BLOCKED
        summary["next_design"] = "C"
        _publish(summary, rows)
        print(summary["gate"], summary["next_design"])
        return
    if meta["stopped_early"] or len(rows) != full_count:
        failed = rows[-1]
        saved = index[(str(failed["variant"]), int(failed["model_seed"]), int(failed["scenario_id"]))]
        summary["failure"] = {
            "variant": str(failed["variant"]),
            "model_seed": int(failed["model_seed"]),
            "scenario_id": int(failed["scenario_id"]),
            "field_gaps": failed.get("_field_gaps", {}),
            "decomposition_gap_bits": float(failed["decomposition_gap_bits"]),
            "uninstrumented_field_gaps": uninstrumented_gaps(
                load_actor(str(failed["variant"]), int(failed["model_seed"])),
                str(failed["variant"]),
                int(failed["model_seed"]),
                int(failed["scenario_id"]),
                saved,
            ),
        }
        summary["problems"].append(
            f"{failed['variant']} seed {int(failed['model_seed'])} scenario {int(failed['scenario_id'])} "
            "exceeds the 1-bit reproduction tolerance"
        )
        summary["gate"] = GATE_BLOCKED
        summary["next_design"] = "C"
        _publish(summary, rows)
        print(summary["gate"], summary["next_design"])
        return
    attribution = summarize_attribution(rows)
    summary["attribution"] = attribution
    summary["replay"]["max_origin_split_gap_bits"] = attribution["max_origin_split_gap_bits"]
    l2_rows = [group for group in attribution["groups"] if group["variant"] == "L2"]
    origin_ok = float(attribution["max_origin_split_gap_bits"]) <= 1.0
    if not origin_ok:
        summary["problems"].append("buoy-level tagged split does not add up")
    summary["next_design"] = next_design(
        True,
        bool(attribution["max_decomposition_gap_bits"] <= 1.0 and origin_ok),
        bool(reconciliation["matches"]),
        l2_rows,
    )
    summary["gate"] = GATE_COMPLETE if summary["next_design"] != "C" and not summary["problems"] else GATE_BLOCKED
    _publish(summary, rows)
    print(summary["gate"], summary["next_design"], summary["replay"]["completed_episodes"])


if __name__ == "__main__":
    main()
