"""Registered R1 entry. Four formal arms and the evidence index, then stop."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE
from wt.legacy_formal_repro_r1.evidence import build_archive
from wt.legacy_formal_repro_r1.failures import build_failure_index
from wt.legacy_formal_repro_r1.protocol import (
    CONFIG_DIR,
    EPISODE_DIR,
    HISTORICAL_REQUIREMENTS,
    HORIZON,
    INSTRUCTION_COMMIT,
    OUTPUT,
    ROOT,
    SCENARIOS,
    STEP_BUDGET,
)
from wt.legacy_formal_repro_r1.report_cn import render_report, render_runtime
from wt.legacy_formal_repro_r1.runner import dump_json, model_identity, pair_rows, public_row, run_registered


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _pins() -> dict[str, str]:
    found = {}
    if not HISTORICAL_REQUIREMENTS.is_file():
        return found
    for line in HISTORICAL_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        name = line.split("==", 1)[0].strip().lower()
        if name in {"numpy", "torch"}:
            found[name] = line.strip()
    return found


def _runtime() -> dict:
    import numpy
    import torch

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "historical_requirements_sha256": hashlib.sha256(HISTORICAL_REQUIREMENTS.read_bytes()).hexdigest(),
        "historical_pins": _pins(),
    }


def _write_steps(rows: list[dict]) -> None:
    path = OUTPUT / "steps.csv"
    header = "scenario_id,policy,time,served_bits,cumulative_payload_bits\n"
    body = []
    for row in rows:
        for step in row["steps"]:
            body.append(
                f"{row['scenario_id']},{row['policy']},{step['time']},{step['served_bits']:.10f},{step['cumulative_payload_bits']:.10f}"
            )
    path.write_text(header + "\n".join(body) + "\n", encoding="utf-8")


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    EPISODE_DIR.mkdir(parents=True, exist_ok=True)
    official_before = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
    archive = build_archive()
    failures = build_failure_index()
    dump_json(OUTPUT / "evidence_index.json", archive)
    dump_json(OUTPUT / "failure_index.json", failures)
    executed = run_registered(CONFIG_DIR)
    official_after = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
    rows = [public_row(row) for row in executed["rows"]]
    for row in rows:
        dump_json(EPISODE_DIR / f"{row['scenario_id']}_{row['policy']}.json", row)
    _write_steps(rows)
    runtime = _runtime()
    manifest = {
        "implementation_commit": _git_head(),
        "instruction_commit": INSTRUCTION_COMMIT,
        "scenarios": list(SCENARIOS),
        "horizon": HORIZON,
        "step_budget": STEP_BUDGET,
        "environment_steps": int(executed["environment_steps"]),
        "stopped_early": bool(executed["stopped_early"]),
        "training_updates": 0,
        "downloads": 0,
        "official_policy_path": str(POLICY_FILE),
        "official_policy_sha256_before": official_before,
        "official_policy_sha256_after": official_after,
        "official_policy_unchanged": official_before == official_after,
        "models": model_identity(),
        "runtime": runtime,
        "pairs": pair_rows(rows),
        "arms": rows,
        "archive_summary": {
            "recovered_count": sum(1 for row in archive["recovered"] if row["status"] == "RECOVERED"),
            "missing_count": len(archive["missing"]),
            "missing": archive["missing"],
            "current_scheme_c_epoch_matches_archived_blob": archive["current_scheme_c_epoch_matches_archived_blob"],
        },
        "failure_summary": {
            "source_commit": failures["source_commit"],
            "source_path": failures["source_path"],
            "signature_count": failures["signature_count"],
            "counts": failures["counts"],
            "current_formal_path_count": failures["current_formal_path_count"],
            "unittest_footer": failures["unittest_footer"],
        },
        "checks_note": "必要检查在本运行之后单独执行，结果写入 necessary_checks.json。",
    }
    dump_json(OUTPUT / "run_manifest.json", manifest)
    runtime_path = ROOT / "docs" / "research" / "Legacy_Formal_Repro_R1_Runtime_CN.md"
    report_path = ROOT / "docs" / "research" / "Legacy_Formal_Repro_R1_Report_CN.md"
    runtime_path.write_text(render_runtime(runtime), encoding="utf-8")
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = main()
    print(json.dumps({"environment_steps": result["environment_steps"], "stopped_early": result["stopped_early"], "pairs": result["pairs"]}, ensure_ascii=False))
