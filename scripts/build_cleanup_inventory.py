"""Build the cleanup inventory and archive unreferenced root reports.

This script does not delete phase directories, checkpoints, or result files.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".venv", "node_modules"}
ARCHIVE = ROOT / "docs" / "archive" / "reports"
KEEP_ROOT_NAMES = {
    "README.md",
    "AGENTS.md",
    "requirements.txt",
    "environment.yml",
    "CURRENT_ROUND.txt",
    "ROUND_VERSIONING.md",
    "round_manifest.json",
    "LICENSE",
    "pyproject.toml",
}
HASH_TARGETS = [
    "phase6if_ranking_loss/runs/c/epoch_001.pt",
    "phase6id_joint_utility_training/epoch_006.pt",
    "phase5l_joint_encoder/joint_utility_encoder.pt",
    "phase5k_joint_decoder/matching_decoder.pt",
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt",
    "phase6iaa_formal_planner/active_policy.json",
    "phase6iab_planner_stability/summary.json",
    "phase6iac_independent_holdout/summary.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def referenced_reports() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "src").rglob("*.py"):
        if any(part in SKIP for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        start = 0
        while True:
            index = text.find("HTW_", start)
            if index < 0:
                break
            end = text.find(".md", index)
            if 0 < end - index < 180:
                name = text[index : end + 3]
                if all(char not in name for char in " /\\*"):
                    names.add(name)
            start = index + 4
    return names


def category(relative: str) -> str:
    path = relative.replace("\\", "/")
    name = Path(path).name
    if path.startswith("backup/"):
        return "snapshot_backup"
    if path.startswith("round_") or "/round" in path:
        return "historical_round"
    if path.startswith("src/") and path.endswith(".py"):
        return "source"
    if path.startswith("tests/"):
        return "test"
    if path.endswith(".pt") or "checkpoint" in path:
        return "checkpoint"
    if name.startswith("HTW_") and name.endswith(".md"):
        return "phase_report"
    if name.startswith("PAPER_") or name.startswith("FINAL_"):
        return "paper_or_final_note"
    if path.endswith(".md"):
        return "document"
    if path.endswith((".csv", ".json", ".jsonl", ".npz")):
        return "experiment_result"
    if path.endswith((".png", ".pdf", ".svg")):
        return "figure"
    if path.endswith((".pyc",)) or "/__pycache__/" in path:
        return "cache"
    return "other"


def action_for(relative: str, referenced: set[str]) -> str:
    path = relative.replace("\\", "/")
    name = Path(path).name
    kind = category(path)
    if kind == "snapshot_backup":
        return "retain"
    if kind in {"checkpoint", "experiment_result", "source", "test", "historical_round"}:
        return "retain"
    if name in KEEP_ROOT_NAMES:
        return "retain"
    if kind == "phase_report" and "/" not in path:
        if name in referenced:
            return "retain"
        return "move"
    if kind in {"paper_or_final_note", "document"} and "/" not in path and name not in referenced:
        return "move"
    return "retain"


def main() -> None:
    referenced = referenced_reports()
    archive = ARCHIVE
    archive.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[str, str]] = []
    for path in list(ROOT.iterdir()):
        if not path.is_file():
            continue
        relative = path.name
        if action_for(relative, referenced) != "move":
            continue
        destination = archive / path.name
        if destination.exists():
            destination = archive / f"root_{path.name}"
        shutil.move(str(path), str(destination))
        moved.append((relative, destination.relative_to(ROOT).as_posix()))
    rows = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in SKIP]
        for filename in filenames:
            if filename.endswith(".pyc"):
                continue
            path = Path(dirpath) / filename
            relative = path.relative_to(ROOT).as_posix()
            kind = category(relative)
            suggested = "moved" if any(relative == destination for _, destination in moved) else action_for(relative, referenced)
            rows.append((relative, kind, suggested))
    out_dir = ROOT / "docs" / "archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "REPOSITORY_CLEANUP_INVENTORY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "category", "code_referenced_report", "contains_unique_research_record", "action"])
        for relative, kind, suggested in rows:
            name = Path(relative).name
            referenced_flag = "yes" if name in referenced else "no"
            unique = "yes" if kind in {"checkpoint", "experiment_result", "phase_report", "source", "historical_round", "paper_or_final_note"} else "not_assessed_as_unique_model"
            writer.writerow([relative, kind, referenced_flag, unique, suggested])
    hashes = []
    for relative in HASH_TARGETS:
        path = ROOT / relative
        hashes.append((relative, sha256(path) if path.is_file() else "MISSING"))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    version = subprocess.check_output(["py", "-3", "--version"], cwd=ROOT, text=True).strip()
    counts: dict[str, int] = {}
    for _, kind, _ in rows:
        counts[kind] = counts.get(kind, 0) + 1
    lines = [
        "# 仓库清理盘点",
        "",
        "本清单在移动未被代码引用的根目录过程报告之前生成规则，并在同一次运行中完成这些移动。没有删除 phase 目录、checkpoint、实验结果或 Markdown 正文。没有执行 `git reset` 或 `git clean`。",
        "",
        f"- 分支：`{branch}`",
        f"- HEAD：`{revision}`",
        "- 工作区在盘点前已有大量未提交修改。这些修改原样保留。",
        f"- Python：`{version}`",
        f"- 纳入清单的文件数：{len(rows)}。`__pycache__` 与 `.pyc` 没有逐个列入。",
        f"- 本次从仓库根目录移入 `docs/archive/reports/` 的报告数：{len(moved)}。",
        "- 删除清单：空。阶段报告都含有该阶段的 Gate、参数或失败分析；被代码引用的报告留在根目录，其余过程报告归档而不是删除。",
        "",
        "逐文件登记在 `docs/archive/REPOSITORY_CLEANUP_INVENTORY.csv`。列为相对路径、类别、报告名是否出现在 `src` 的 Python 源码中、是否属于模型或实验记录、以及保留或已移动。",
        "",
        "## 类别计数",
        "",
        "| 类别 | 文件数 |",
        "| --- | --- |",
    ]
    for kind, count in sorted(counts.items(), key=lambda item: -item[1]):
        lines.append(f"| {kind} | {count} |")
    lines.extend(["", "## 关键文件 SHA-256", "", "| 路径 | SHA-256 |", "| --- | --- |"])
    for relative, digest in hashes:
        lines.append(f"| `{relative}` | `{digest}` |")
    lines.extend(["", "## 已归档的根目录报告", ""])
    for source, destination in moved:
        lines.append(f"- `{source}` → `{destination}`")
    lines.extend(
        [
            "",
            "## 留在根目录的原因",
            "",
            "审计脚本通过 `ROOT / \"HTW_....md\"` 读取或写回一部分报告。这些文件名出现在 `src` 中，移动会让旧入口找不到报告，因此保留在根目录。`README.md`、`AGENTS.md`、`requirements.txt`、`ROUND_VERSIONING.md` 和 `round_manifest.json` 也保留。",
            "",
            "`backup/` 与 `round_XX/` 含有历史快照和轮次产物，按原位保留，不整目录删除。",
            "",
        ]
    )
    (out_dir / "REPOSITORY_CLEANUP_INVENTORY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"rows={len(rows)} moved={len(moved)}")


if __name__ == "__main__":
    main()
