"""Inventory files for a cleanup plan. Does not delete or edit experiment state."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "outputs"
SKIP_DIRS = {".git", ".venv", "node_modules"}
LINEAGE_TOKENS = (
    "registry",
    "manifest",
    "holdout",
    "integrity",
    "criterion",
    "phase4",
    "gate",
    "b2",
    "extratree",
    "shadow",
    "hindsight",
    "deployment",
    "failure_attribution",
    "scenario",
    "checkpoint",
    "label",
)
KEEP_EXT = {".py", ".pt", ".pth", ".ckpt", ".pkl", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
JUNK_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage(rel: str) -> bool:
    text = rel.lower()
    return any(token in text for token in LINEAGE_TOKENS)


_PATH_RE = re.compile(r"[A-Za-z0-9_.\\/-]{8,}\.[A-Za-z0-9]{1,5}")


def _mention_sets() -> tuple[set[str], set[str], set[str], set[str]]:
    code_paths, report_paths, code_names, report_names = set(), set(), set(), set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and name != "__pycache__"]
        for name in filenames:
            path = Path(dirpath) / name
            ext = path.suffix.lower()
            if ext not in {".py", ".md"} or path.stat().st_size > 1_500_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").replace("\\", "/")
            except OSError:
                continue
            paths = {item.lower().lstrip("./") for item in _PATH_RE.findall(text)}
            names = {Path(item).name for item in paths}
            if ext == ".py":
                code_paths.update(paths)
                code_names.update(names)
            else:
                report_paths.update(paths)
                report_names.update(names)
    return code_paths, report_paths, code_names, report_names


def _referenced(rel: str, paths: set[str], names: set[str], basename_counts: dict[str, int]) -> bool:
    lowered = rel.lower()
    if lowered in paths:
        return True
    parts = lowered.split("/")
    for index in range(1, len(parts)):
        suffix = "/".join(parts[index:])
        if len(suffix) > 12 and suffix in paths:
            return True
    base = Path(lowered).name
    return basename_counts[base] == 1 and len(base) > 12 and base in names


def _classify(rel: str, path: Path, code_ref: bool, report_ref: bool, lineage: bool, duplicate: bool) -> tuple[str, str, str, str]:
    name = path.name.lower()
    ext = path.suffix.lower()
    if name in JUNK_NAMES or "__pycache__" in rel.split("/") or ext == ".pyc":
        return "DELETE_CANDIDATE", "delete", "low", "temporary cache or OS junk; not an experiment record"
    if lineage or ext in KEEP_EXT or name.startswith("htw_") or "agents.md" in name:
        if duplicate and ext == ".md" and "/outputs/" in f"/{rel.lower()}":
            return "ARCHIVE", "archive", "medium", "byte-identical copy of an audit report already stored at another path"
        return "KEEP", "keep", "high", "code, model, config, or experimental lineage"
    if ext in {".csv", ".json", ".md"}:
        if duplicate and "/outputs/" in f"/{rel.lower()}":
            return "ARCHIVE", "archive", "medium", "byte-identical duplicate of a report or table"
        return "KEEP", "keep", "high", "result, report, or structured record"
    if name.endswith("_cache.json") or "ull_cache" in name:
        return "ARCHIVE", "archive", "medium", "regenerable label cache; the label CSV remains the experimental record"
    return "KEEP", "keep", "high", "unclassified file kept because cleanup must not drop unique records"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    code_paths, report_paths, code_names, report_names = _mention_sets()
    files = []
    empty_dirs = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        if not dirnames and not filenames:
            empty_dirs.append(_rel(Path(dirpath)))
        for name in filenames:
            files.append(Path(dirpath) / name)
    counts = Counter(path.name.lower() for path in files)
    hashes = {}
    hash_groups = defaultdict(list)
    for path in files:
        if path.suffix.lower() not in {".md", ".json", ".csv"}:
            continue
        if counts[path.name.lower()] < 2 or path.stat().st_size > 25_000_000:
            continue
        try:
            digest = _hash(path)
        except OSError:
            continue
        hashes[_rel(path)] = digest
        hash_groups[digest].append(_rel(path))
    duplicates = {item for group in hash_groups.values() if len(group) > 1 for item in group}
    rows = []
    for path in files:
        rel = _rel(path)
        if rel.startswith("src/wt/lowlevel_v2/project_cleanup_audit/outputs/"):
            continue
        stat = path.stat()
        lineage = _lineage(rel)
        code_ref = _referenced(rel, code_paths, code_names, counts)
        report_ref = _referenced(rel, report_paths, report_names, counts)
        category, action, risk, reason = _classify(rel, path, code_ref, report_ref, lineage, rel in duplicates)
        if category == "DELETE_CANDIDATE" and (lineage or code_ref or report_ref):
            category, action, risk, reason = "KEEP", "keep", "high", "protected because it is referenced or part of lineage"
        rows.append(
            {
                "path": rel,
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "file_type": path.suffix.lower() or "none",
                "referenced_by_code": int(code_ref),
                "referenced_by_report": int(report_ref),
                "experimental_lineage": int(lineage),
                "category": category,
                "action": action,
                "risk": risk,
                "reason": reason,
            }
        )
    rows.sort(key=lambda row: row["path"])
    inventory = OUT / "cleanup_file_inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_files": len(rows),
        "n_empty_dirs": len(empty_dirs),
        "empty_dirs": empty_dirs[:50],
        "by_category": dict(Counter(row["category"] for row in rows)),
        "by_action": dict(Counter(row["action"] for row in rows)),
        "delete_bytes": sum(row["size_bytes"] for row in rows if row["category"] == "DELETE_CANDIDATE"),
        "archive_bytes": sum(row["size_bytes"] for row in rows if row["category"] == "ARCHIVE"),
        "total_bytes": sum(row["size_bytes"] for row in rows),
    }
    (OUT / "cleanup_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
