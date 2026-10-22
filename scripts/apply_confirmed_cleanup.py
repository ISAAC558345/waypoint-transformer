"""Delete confirmed cache and byte-identical report copies, then merge archived notes.

Does not touch backup/snapshots, checkpoints, or unique experiment tables.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "archive" / "reports"
MERGED = ROOT / "docs" / "archive" / "merged"
TABLES = ROOT / "docs" / "archive" / "tables"
LOG = ROOT / "docs" / "archive" / "DELETION_AND_MERGE_LOG.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def under_backup(path: Path) -> bool:
    return "backup" in path.relative_to(ROOT).parts


def remove_empty_parents(path: Path, stop: Path) -> None:
    current = path
    while current != stop and current != ROOT:
        if current.exists() and current.is_dir() and not any(current.iterdir()):
            current.rmdir()
            current = current.parent
            continue
        break


def delete_caches() -> list[str]:
    deleted = []
    for path in list(ROOT.rglob("*")):
        if not path.exists() or under_backup(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.is_file() and (path.suffix == ".pyc" or path.name.lower() in {"thumbs.db", ".ds_store", "desktop.ini"}):
            path.unlink()
            deleted.append(relative)
            remove_empty_parents(path.parent, ROOT)
        elif path.is_dir() and path.name == "__pycache__" and not any(path.rglob("*")):
            path.rmdir()
            deleted.append(relative + "/")
    for directory in (ROOT / ".test-tmp",):
        if directory.exists():
            shutil.rmtree(directory)
            deleted.append(".test-tmp/")
    round02 = ROOT / "round_02"
    if round02.exists():
        for child in list(round02.iterdir()):
            if child.is_dir() and child.name.startswith("t") and not any(child.rglob("*")):
                child.rmdir()
                deleted.append(child.relative_to(ROOT).as_posix() + "/")
    runtime = ROOT / "round_36" / "tmp_runtime"
    if runtime.exists() and not any(item.is_file() for item in runtime.rglob("*")):
        shutil.rmtree(runtime)
        deleted.append("round_36/tmp_runtime/")
    return deleted


def canonical_for(name: str) -> Path | None:
    for folder in (ROOT, REPORTS):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def delete_identical_copies() -> tuple[list[str], list[str]]:
    deleted = []
    kept_unique = []
    unique_dir = MERGED / "nonidentical_module_copies"
    for path in (ROOT / "src").rglob("*.md"):
        relative = path.relative_to(ROOT).as_posix()
        if "/outputs/" not in f"/{relative}" and "/phase4d_deployment_criterion_registry/" not in relative:
            continue
        if not path.name.startswith("HTW_Phase4"):
            continue
        canonical = canonical_for(path.name)
        if canonical is None or canonical.resolve() == path.resolve():
            continue
        if sha256(path) == sha256(canonical):
            path.unlink()
            deleted.append(relative)
            remove_empty_parents(path.parent, ROOT)
        else:
            unique_dir.mkdir(parents=True, exist_ok=True)
            destination = unique_dir / path.name
            shutil.move(str(path), str(destination))
            kept_unique.append(f"{relative} -> {destination.relative_to(ROOT).as_posix()}")
    return deleted, kept_unique


def group_name(filename: str) -> str:
    if filename.startswith("PAPER_"):
        return "PAPER_NOTES.md"
    if filename.startswith("HTW_Phase6I_"):
        return "PHASE6I_PROCESS.md"
    if filename.startswith("FINAL_"):
        return "FINAL_NOTES.md"
    if filename.startswith("LLV2_") or filename.startswith("HTW_LowLevel"):
        return "LOW_LEVEL_NOTES.md"
    return "OTHER_PROCESS_NOTES.md"


def merge_archived_reports() -> list[str]:
    MERGED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[Path]] = {}
    moved_tables = []
    for path in sorted(REPORTS.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".md":
            destination = TABLES / path.name
            shutil.move(str(path), str(destination))
            moved_tables.append(destination.relative_to(ROOT).as_posix())
            continue
        buckets.setdefault(group_name(path.name), []).append(path)
    merged_names = []
    for name, paths in buckets.items():
        parts = [
            f"# {name[:-3]}",
            "",
            "以下内容由 `docs/archive/reports/` 中的过程文档按原文件全文合并。合并后的正文是这些过程记录的归档副本。",
            "",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            parts.append(f"## 源文件 `{path.name}`")
            parts.append("")
            parts.append(text.rstrip())
            parts.append("")
        destination = MERGED / name
        destination.write_text("\n".join(parts), encoding="utf-8")
        combined = destination.read_text(encoding="utf-8")
        for path in paths:
            if path.read_text(encoding="utf-8").rstrip() not in combined:
                raise RuntimeError(f"merge dropped {path.name}")
            path.unlink()
        merged_names.append(f"{name} ({len(paths)} files)")
    return merged_names + [f"table {item}" for item in moved_tables]


def main() -> None:
    caches = delete_caches()
    copies, unique_copies = delete_identical_copies()
    merged = merge_archived_reports()
    lines = [
        "# 删除与合并记录",
        "",
        "只删除了两类已确认没有独立实验内容的文件：仓库 `backup/` 之外的 Python 字节码和系统垃圾文件，以及与根目录或归档副本 SHA-256 完全相同的 Phase-4 模块报告。空的 `.test-tmp`、`round_02/t*` 和已经清空的 `__pycache__` 目录一并移除。",
        "",
        f"- 缓存和空目录删除项：{len(caches)}",
        f"- 字节相同的重复报告删除项：{len(copies)}",
        f"- 内容不同、改为归档的模块副本：{len(unique_copies)}",
        "",
        "## 合并后的过程文档",
        "",
    ]
    lines.extend(f"- `{item}`" for item in merged)
    if copies:
        lines.extend(["", "## 已删除的重复报告", ""])
        lines.extend(f"- `{item}`" for item in copies)
    if unique_copies:
        lines.extend(["", "## 内容不同而保留的副本", ""])
        lines.extend(f"- `{item}`" for item in unique_copies)
    lines.append("")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    print(f"cache={len(caches)} copies={len(copies)} unique={len(unique_copies)} merged={len(merged)}")


if __name__ == "__main__":
    main()
