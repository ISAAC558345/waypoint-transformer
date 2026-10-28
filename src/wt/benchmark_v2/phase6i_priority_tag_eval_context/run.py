"""Spawn two fresh processes from one manifest. This module does not train."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys

import numpy as np
import torch

from wt.benchmark_v2.phase6i_priority_tag_reward.protocol import assert_dev_scenario

from .compare import choose_gate, ledger_ok, matches_saved, traces_consistent
from .context import build_manifest, code_hashes, file_sha256, manifest_is_explicit
from .protocol import (
    ACCOUNTING_TOLERANCE_BITS,
    FORMAL_ACTOR_SHA256,
    GATE_BLOCKED,
    INSTRUCTION_COMMIT,
    OUTPUT,
    PATH_TOLERANCE_M,
    REPORT,
    ROOT,
    SCENARIO_ID,
    SCHEMA_VERSION,
    SOURCE_E7_COMMIT,
    SOURCE_ROWS,
)
from .report import render_report


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _versions() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _load_saved() -> dict:
    with SOURCE_ROWS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "L0" and int(float(row["model_seed"])) == 0 and int(float(row["scenario_id"])) == int(SCENARIO_ID):
                parsed = {key: float(value) if key != "variant" else value for key, value in row.items()}
                parsed["model_seed"] = int(parsed["model_seed"])
                parsed["scenario_id"] = int(parsed["scenario_id"])
                return parsed
    raise RuntimeError("saved L0 row is missing")


def _spawn(manifest_path, output_path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "-m", "wt.benchmark_v2.phase6i_priority_tag_eval_context.child", "--manifest", str(manifest_path), "--output", str(output_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
        "accounting_tolerance_bits": ACCOUNTING_TOLERANCE_BITS,
        "path_tolerance_m": PATH_TOLERANCE_M,
        "tolerance_changed_after_results": False,
        "historical_global_seed_role": "historical_inherited_process_seed",
        "canonical_explicit_seed": None,
        "retrained": False,
        "formal_system_modified": False,
        "held_out_splits_read": False,
        "merged": False,
        "gate": summary.get("gate"),
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(summary), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split_ok = True
    split_error = ""
    try:
        assert_dev_scenario(int(SCENARIO_ID))
    except RuntimeError as exc:
        split_ok = False
        split_error = str(exc)
    formal_before = file_sha256(ROOT / "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt")
    manifest = build_manifest(versions=_versions(), commit=_git(["rev-parse", "HEAD"]), code_hashes=code_hashes())
    manifest_path = OUTPUT / "runtime_context.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_sha = file_sha256(manifest_path)
    summary = {
        "instruction_commit": INSTRUCTION_COMMIT,
        "source_e7_commit": SOURCE_E7_COMMIT,
        "head": manifest["git_commit"],
        "formal_actor_sha256": formal_before,
        "manifest_sha256": manifest_sha,
        "manifest_explicit": manifest_is_explicit(manifest),
        "split_ok": split_ok,
        "split_error": split_error,
        "accounting_tolerance_bits": ACCOUNTING_TOLERANCE_BITS,
        "path_tolerance_m": PATH_TOLERANCE_M,
        "tolerance_changed_after_results": False,
        "retrained": False,
        "merged": False,
        "held_out_splits_read": False,
    }
    if not split_ok or formal_before != FORMAL_ACTOR_SHA256 or not manifest_is_explicit(manifest):
        summary["gate"] = GATE_BLOCKED
        summary["problems"] = [split_error or "formal hash or manifest label failed before the two processes"]
        _publish(summary, [])
        print(summary["gate"])
        return
    processes = []
    traces = []
    for name in ("run_a", "run_b"):
        output = OUTPUT / f"{name}.json"
        completed = _spawn(manifest_path, output)
        processes.append({"name": name, "returncode": int(completed.returncode), "stderr": completed.stderr[-2000:], "stdout": completed.stdout[-500:]})
        if completed.returncode != 0 or not output.exists():
            summary["processes"] = processes
            summary["gate"] = GATE_BLOCKED
            summary["problems"] = [f"{name} exited {completed.returncode}"]
            _publish(summary, [])
            print(summary["gate"], name)
            return
        traces.append(json.loads(output.read_text(encoding="utf-8")))
    saved = _load_saved()
    compared = traces_consistent(traces[0]["steps"], traces[1]["steps"])
    match_a = matches_saved(traces[0], saved)
    match_b = matches_saved(traces[1], saved)
    ledger_a = ledger_ok(traces[0])
    ledger_b = ledger_ok(traces[1])
    formal_after = file_sha256(ROOT / "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt")
    hashes_ok = bool(formal_before == FORMAL_ACTOR_SHA256 and formal_after == FORMAL_ACTOR_SHA256 and traces[0]["applied"]["torch_initial_seed"] == traces[1]["applied"]["torch_initial_seed"])
    summary.update(
        {
            "processes": processes,
            "applied": [traces[0]["applied"], traces[1]["applied"]],
            "reset": [traces[0]["reset"], traces[1]["reset"]],
            "reset_equal": traces[0]["reset"] == traces[1]["reset"],
            "process_comparison": {key: value for key, value in compared.items() if key != "rows"},
            "saved_match": {"run_a": match_a, "run_b": match_b},
            "ledger_ok": {"run_a": ledger_a, "run_b": ledger_b},
            "formal_actor_sha256_after": formal_after,
            "actor_training": [traces[0]["actor_training"], traces[1]["actor_training"]],
        }
    )
    summary["gate"] = choose_gate(
        hashes_ok=hashes_ok,
        split_ok=split_ok,
        manifest_ok=bool(manifest_is_explicit(manifest) and traces[0]["actor_training"] is False and traces[1]["actor_training"] is False),
        processes_consistent=bool(compared["consistent"] and traces[0]["reset"] == traces[1]["reset"]),
        both_match_saved=bool(match_a["matches"] and match_b["matches"]),
        ledger_ok_both=bool(ledger_a and ledger_b),
    )
    _publish(summary, compared.get("rows") or [])
    print(summary["gate"])


if __name__ == "__main__":
    main()
