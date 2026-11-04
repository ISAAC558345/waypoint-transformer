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


def _saved_rows() -> list[dict]:
    rows = []
    for scenario_id in SCENARIOS:
        for policy in ("scheme_c", "g1_rate"):
            path = EPISODE_DIR / f"{scenario_id}_{policy}.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _official_unchanged() -> tuple[str, bool]:
    digest = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", "diff", "--", "results/phase6iaa_formal_planner/active_policy.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return digest, completed.returncode == 0 and completed.stdout.strip() == ""


def _write_manifest(rows: list[dict], *, stopped_early: bool, checks_note: str, output_note: str) -> dict:
    failures = build_failure_index()
    dump_json(OUTPUT / "failure_index.json", failures)
    archive_path = OUTPUT / "evidence_index.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    current = ROOT / "results/phase6if_ranking_loss/runs/c/epoch_001.pt"
    current_hash = hashlib.sha256(current.read_bytes()).hexdigest() if current.is_file() else None
    archived_hash = next(
        (row["sha256"] for row in archive["checkpoints_hashed"] if row["path"].endswith("runs/c/epoch_001.pt")),
        None,
    )
    archive["current_scheme_c_epoch_matches_archived_blob"] = bool(current_hash is not None and current_hash == archived_hash)
    dump_json(archive_path, archive)
    official_hash, official_clean = _official_unchanged()
    runtime = _runtime()
    manifest = {
        "runner_commit": "ed2c87c83dcdaccee1420ee3e14a8dfa19982e92",
        "instruction_commit": INSTRUCTION_COMMIT,
        "scenarios": list(SCENARIOS),
        "horizon": HORIZON,
        "step_budget": STEP_BUDGET,
        "environment_steps": int(sum(int(row["environment_steps"]) for row in rows)),
        "stopped_early": bool(stopped_early),
        "training_updates": 0,
        "scheme_c_calls": int(sum(int(row["scheme_c_calls"]) for row in rows)),
        "low_level_completed_steps": int(sum(int(row["environment_steps"]) for row in rows)),
        "downloads": 0,
        "official_policy_path": str(POLICY_FILE),
        "official_policy_sha256": official_hash,
        "official_policy_unchanged": official_clean,
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
        "checks_note": checks_note,
        "output_note": output_note,
    }
    dump_json(OUTPUT / "run_manifest.json", manifest)
    (ROOT / "docs" / "research" / "Legacy_Formal_Repro_R1_Runtime_CN.md").write_text(render_runtime(runtime), encoding="utf-8")
    (ROOT / "docs" / "research" / "Legacy_Formal_Repro_R1_Report_CN.md").write_text(render_report(manifest), encoding="utf-8")
    return manifest


def main() -> dict:
    if any(EPISODE_DIR.glob("*.json")):
        raise RuntimeError("R1 episode outputs already exist; refusing another environment sample")
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
    if official_before != official_after:
        raise RuntimeError("the official active policy file changed during the registered run")
    return _write_manifest(
        rows,
        stopped_early=bool(executed["stopped_early"]),
        checks_note="必要检查在本运行之后单独执行，结果写入 necessary_checks.json。",
        output_note="Episode outputs were written directly under results/legacy_formal_repro_r1.",
    )


def finalize_saved() -> dict:
    """Write the manifest from episode files that already exist. This does not step the environment."""
    rows = _saved_rows()
    if sum(int(row["environment_steps"]) for row in rows) > STEP_BUDGET:
        raise RuntimeError("saved environment steps exceed the registered budget")
    return _write_manifest(
        rows,
        stopped_early=len(rows) != 4 or any(int(row["environment_steps"]) != HORIZON for row in rows),
        checks_note="必要检查在本清单之后单独执行，结果写入 necessary_checks.json。",
        output_note="The first manifest write used the source directory as the repository root. Episode bytes were moved to results/legacy_formal_repro_r1 without running another environment step.",
    )


def cli(argv: list[str] | None = None) -> int:
    """Default entry writes nothing. Rerun and finalize stay outside this command."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("read-only boundary: this command does not rewrite the saved R1 manifest or load a model.")
        print("inspect: py -3 -m wt.legacy_formal_repro_r1a.inspect --artifact-root results/legacy_formal_repro_r1")
        print("export: py -3 -m wt.legacy_formal_repro_r1a.export --artifact-root results/legacy_formal_repro_r1 --output results/legacy_formal_repro_r1a")
        print("future rerun is outside this package and is not executed: py -3 -m wt.legacy_formal_repro_r1.run --rerun")
        return 0
    if "--rerun" in args or "--finalize" in args:
        print("refusing to rerun or rewrite the saved R1 manifest from this entry")
        return 2
    print("unrecognized arguments; no files were written")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
