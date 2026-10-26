"""Read committed review inputs. This module does not recompute rewards."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess

from .protocol import (
    E1_SCHEMA,
    E1_SCHEMA_FILE,
    E2_COMMIT,
    E2_FILES,
    E2_GATE,
    E3_COMMIT,
    E3_FILES,
    E3_GATE,
    FORMAL_FILES,
    ROOT,
)


def _sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _numeric_assignments(relative: str) -> dict[str, float]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            found[target.id] = float(node.value.value)
    return found


def _record(relative: str, expected_commit: str | None, problems: list[str]) -> dict:
    path = ROOT / relative
    if not path.is_file():
        problems.append(f"missing {relative}")
        return {"path": relative, "present": False}
    commit = _git_commit(relative)
    dirty = _git_dirty(relative)
    if dirty:
        problems.append(f"dirty {relative}")
    if expected_commit is not None and commit != expected_commit:
        problems.append(f"commit mismatch {relative}: {commit}")
    return {
        "path": relative,
        "present": True,
        "sha256": _sha256(relative),
        "git_commit": commit,
        "clean": not dirty,
    }


def load_review_inputs() -> dict:
    """Collect hashes and already stored summaries. No environment and no reward arithmetic."""
    problems: list[str] = []
    e2 = [_record(path, E2_COMMIT, problems) for path in E2_FILES]
    e3 = [_record(path, E3_COMMIT, problems) for path in E3_FILES]
    e1 = _record(E1_SCHEMA_FILE, None, problems)
    formal = [_record(path, None, problems) for path in FORMAL_FILES]
    e2_summary = _load_json(E2_FILES[2]) if (ROOT / E2_FILES[2]).is_file() else {}
    e3_summary = _load_json(E3_FILES[1]) if (ROOT / E3_FILES[1]).is_file() else {}
    e2_schema = _load_json(E2_FILES[3]) if (ROOT / E2_FILES[3]).is_file() else {}
    e3_schema = _load_json(E3_FILES[2]) if (ROOT / E3_FILES[2]).is_file() else {}
    e1_schema = _load_json(E1_SCHEMA_FILE) if (ROOT / E1_SCHEMA_FILE).is_file() else {}
    if e2_summary.get("gate") != E2_GATE:
        problems.append("E2 gate is not the committed inconclusive gate")
    if e3_summary.get("gate") != E3_GATE:
        problems.append("E3 gate is not the committed inconclusive gate")
    if e1_schema.get("schema_version") != E1_SCHEMA:
        problems.append("E1 schema version is not synthetic_priority_bucket_v1")
    recorded = e3_summary.get("source_sha256", {})
    for item in e2[:4]:
        name = str(item["path"]).rsplit("/", 1)[-1]
        if item.get("sha256") and recorded.get(name) != item.get("sha256"):
            problems.append(f"E3 recorded hash differs for {name}")
    watched = e2_summary.get("watched_sha256", {})
    for item in formal:
        previous = watched.get(item["path"])
        if previous is not None and item.get("sha256") != previous:
            problems.append(f"formal file hash changed since E2: {item['path']}")
    assignments = _numeric_assignments("src/wt/lowlevel_v2/phase1/protocol.py") if (ROOT / FORMAL_FILES[0]).is_file() else {}
    worker = (ROOT / FORMAL_FILES[2]).read_text(encoding="utf-8") if (ROOT / FORMAL_FILES[2]).is_file() else ""
    phase6g = (ROOT / FORMAL_FILES[1]).read_text(encoding="utf-8") if (ROOT / FORMAL_FILES[1]).is_file() else ""
    formula_ok = (
        "r_comm = float(served_bits) / float(block_bits)" in worker
        and "reward = r_comm - float(LAMBDA_PROJECT) * c_proj - float(LAMBDA_RESIDUAL) * c_res" in worker
        and "LAMBDA_PROJECT" in phase6g
        and "LAMBDA_PROJECT =" not in phase6g
    )
    if not formula_ok:
        problems.append("formal low-level reward formula could not be verified")
    return {
        "problems": problems,
        "e2_files": e2,
        "e3_files": e3,
        "e1_schema_file": e1,
        "formal_files": formal,
        "e2_summary": e2_summary,
        "e3_summary": e3_summary,
        "e2_schema": e2_schema,
        "e3_schema": e3_schema,
        "e1_schema": e1_schema,
        "reward_constants": assignments,
        "formal_formula_ok": formula_ok,
    }
