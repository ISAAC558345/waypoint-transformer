"""Restore original summary bytes into a separate archive. Missing files stay missing."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from wt.evaluation.scheme_c_runner.protocol import (
    ARCHIVE_DIR,
    EVIDENCE_COMMIT,
    REMOVAL_COMMIT,
    ROOT,
    SUMMARY_SOURCES,
    checkpoint_sources,
    episode_sources,
    training_sources,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_bytes(commit: str, relative: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", "cat-file", "blob", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_hash(commit: str, relative: str) -> str | None:
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", "rev-parse", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _on_disk(relative: str) -> Path | None:
    suffix = Path(relative)
    exact = [ROOT / relative, ROOT / "backup" / relative]
    for candidate in exact:
        if candidate.is_file():
            return candidate
    backup = ROOT / "backup"
    if not backup.is_dir():
        return None
    for candidate in backup.rglob(suffix.name):
        if candidate.is_file() and candidate.parts[-len(suffix.parts) :] == suffix.parts:
            return candidate
    return None


def _definition(relative: str, payload: dict | None) -> str:
    if relative.endswith("phase6iab_planner_stability/summary.json"):
        return (
            "AB summary fields are the saved development-set pair statistics. "
            "validation_is_development_set marks this file as a development summary. "
            "scheme_mean_payload and g1_mean_payload average payload_bits over the saved pairs. "
            "bit_tie is the file's own tie threshold."
        )
    if relative.endswith("phase6iac_independent_holdout/summary.json"):
        return (
            "AC summary fields are the saved holdout pair statistics. "
            "This archive copies that file and does not rerun the holdout. "
            "scheme_mean_payload and g1_mean_payload average payload_bits over the saved pairs."
        )
    if relative.endswith("phase6if_ranking_loss/summary.json"):
        return "No summary.json is defined because the original path has no file."
    if payload is None:
        return "Original bytes were not found, so no statistic is defined from this path."
    keys = ", ".join(list(payload)[:12])
    return f"Original JSON keys begin with: {keys}."


def restore_file(relative: str, *, copy_bytes: bool, origin: str = "auto") -> dict:
    disk = None if origin == "git" else _on_disk(relative)
    source = "working_tree_or_backup" if disk is not None else "git_blob"
    data = disk.read_bytes() if disk is not None else _git_bytes(EVIDENCE_COMMIT, relative)
    record = {
        "path": relative,
        "status": "RECOVERED" if data is not None else "MISSING",
        "source": None if data is None else source,
        "source_commit": None if data is None or source != "git_blob" else EVIDENCE_COMMIT,
        "source_path": None if disk is None else str(disk),
        "sha256": None if data is None else _sha256(data),
        "bytes": 0 if data is None else len(data),
        "archive_path": None,
        "statistic_definition": _definition(relative, None),
    }
    if data is None or not copy_bytes:
        if data is not None and not copy_bytes:
            record["status"] = "HASHED_NOT_COPIED"
            record["git_blob"] = _git_hash(EVIDENCE_COMMIT, relative)
            record["statistic_definition"] = "Checkpoint bytes are identified by hash and are not copied into this archive."
        return record
    destination = ARCHIVE_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    record["archive_path"] = str(destination.relative_to(ROOT)).replace("\\", "/")
    parsed = None
    if relative.endswith(".json"):
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            parsed = None
    record["statistic_definition"] = _definition(relative, parsed if isinstance(parsed, dict) else None)
    if isinstance(parsed, dict):
        record["stored_fields"] = {
            key: parsed[key]
            for key in (
                "gate",
                "pairs",
                "validation_is_development_set",
                "deployment_verified",
                "trained",
                "bit_tie",
            )
            if key in parsed
        }
    return record


def build_archive() -> dict:
    copied = [restore_file(path, copy_bytes=True) for path in (*SUMMARY_SOURCES, *episode_sources(), *training_sources())]
    hashed = [restore_file(path, copy_bytes=False, origin="git") for path in checkpoint_sources()]
    current_checkpoint = ROOT / "checkpoints/formal/scheme_c_epoch1.pt"
    current_hash = _sha256(current_checkpoint.read_bytes()) if current_checkpoint.is_file() else None
    archived_hash = next(
        (row["sha256"] for row in hashed if row["path"].endswith("runs/c/epoch_001.pt")),
        None,
    )
    created = subprocess.run(
        ["git", "-c", "core.longpaths=true", "show", "-s", "--format=%cI", EVIDENCE_COMMIT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "evidence_commit": EVIDENCE_COMMIT,
        "evidence_commit_time": created.stdout.strip() if created.returncode == 0 else None,
        "removal_commit": REMOVAL_COMMIT,
        "removal_note": "The removal commit dropped these generated files from the working tree. Recovered bytes come from the preceding published tree.",
        "new_run_directory": "results/reproducibility/formal_scheme_c",
        "recovered": copied,
        "checkpoints_hashed": hashed,
        "current_scheme_c_epoch_matches_archived_blob": bool(current_hash is not None and current_hash == archived_hash),
        "missing": [row["path"] for row in copied + hashed if row["status"] == "MISSING"],
    }