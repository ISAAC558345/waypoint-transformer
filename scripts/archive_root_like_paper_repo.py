"""Move root process reports and phase outputs into a paper-repo layout.

Path literals that join the repository root to a moved name are updated.
Round locks, backup snapshots, and src/wt/round* stay untouched.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "archive" / "phase_reports"
EXPORTS = ROOT / "results" / "root_exports"
KEEP_FILES = {
    "README.md",
    "LICENSE",
    "AGENTS.md",
    "requirements.txt",
    "environment.yml",
    "CURRENT_ROUND.txt",
    "ROUND_VERSIONING.md",
    "round_manifest.json",
}
KEEP_DIRS = {
    ".git",
    ".vscode",
    "backup",
    "configs",
    "docs",
    "experiments",
    "final_artifacts",
    "logs",
    "paper_assets",
    "report",
    "results",
    "scripts",
    "src",
    "tests",
}
PUBLISH_FILES = {
    "src/wt/lowlevel_v2/integration_phase1/run_compat.py",
    "src/wt/lowlevel_v2/integration_phase2/run_phase2.py",
    "src/wt/lowlevel_v2/integration_phase2a/run_audit.py",
    "src/wt/lowlevel_v2/integration_phase2b/run_phase2b.py",
    "src/wt/lowlevel_v2/integration_phase2c/run_phase2c.py",
    "src/wt/lowlevel_v2/integration_phase2d/run_phase2d.py",
    "src/wt/lowlevel_v2/integration_phase3/run_phase3.py",
    "src/wt/lowlevel_v2/phase1/run_phase1.py",
    "src/wt/lowlevel_v2/phase1b/run_phase1b.py",
    "src/wt/lowlevel_v2/phase2/run_phase2.py",
    "src/wt/lowlevel_v2/phase3/run_phase3.py",
    "src/wt/lowlevel_v2/phase4a/run_phase4a.py",
    "src/wt/lowlevel_v2/phase4b/redraw_figures.py",
    "src/wt/lowlevel_v2/phase4b/run_phase4b.py",
}
JOIN = 'ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name'


def skip_rewrite(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    if not parts:
        return True
    if parts[0] in {"backup", "results"} or parts[0].startswith("round_"):
        return True
    if parts[:3] == ("docs", "archive", "phase_reports") or parts[:3] == ("docs", "archive", "merged"):
        return True
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "wt" and parts[2].startswith("round"):
        return True
    if parts[0] == "tests" and path.name.startswith("test_round"):
        return True
    return False


def move_root_entries() -> tuple[dict[str, str], list[str]]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    moved_dirs: list[str] = []
    for path in list(ROOT.iterdir()):
        if path.name in KEEP_FILES or path.name in KEEP_DIRS or path.name.startswith("."):
            continue
        if path.name.startswith("round_"):
            continue
        if path.is_dir() and path.name.startswith("phase"):
            destination = ROOT / "results" / path.name
            if destination.exists():
                raise RuntimeError(f"destination exists: {destination}")
            shutil.move(str(path), str(destination))
            mapping[path.name] = f"results/{path.name}"
            moved_dirs.append(path.name)
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() == ".md":
            destination = REPORTS / path.name
            relative = destination.relative_to(ROOT).as_posix()
        else:
            destination = EXPORTS / path.name
            relative = destination.relative_to(ROOT).as_posix()
        if destination.exists():
            raise RuntimeError(f"destination exists: {destination}")
        shutil.move(str(path), str(destination))
        mapping[path.name] = relative
    return mapping, moved_dirs


def rewrite_literals(mapping: dict[str, str]) -> int:
    pattern = re.compile(r"""(?P<prefix>\b(?:ROOT|root)\s*/\s*)(?P<q>["'])(?P<name>[^"']+)(?P=q)""")
    changed = 0
    suffixes = {".py", ".md", ".json", ".txt"}
    roots = [ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "docs", ROOT / "configs"]
    for base in roots:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes or skip_rewrite(path):
                continue
            text = path.read_text(encoding="utf-8")

            def replace(match: re.Match[str]) -> str:
                name = match.group("name")
                if name in mapping:
                    return f"{match.group('prefix')}{match.group('q')}{mapping[name]}{match.group('q')}"
                head, sep, tail = name.partition("/")
                if sep and head in mapping:
                    return f"{match.group('prefix')}{match.group('q')}{mapping[head]}/{tail}{match.group('q')}"
                return match.group(0)

            updated = pattern.sub(replace, text)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                changed += 1
    return changed


def rewrite_publishers() -> None:
    for relative in PUBLISH_FILES:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        updated = text.replace("ROOT / name", JOIN)
        if updated == text:
            raise RuntimeError(f"publish path not found in {relative}")
        path.write_text(updated, encoding="utf-8")
    audit = ROOT / "src/wt/benchmark_v2/phase6iac_independent_holdout/audit.py"
    text = audit.read_text(encoding="utf-8")
    old = "if not (root / name).is_file():"
    new = 'if not (root / "docs/archive/phase_reports" / name).is_file():'
    if old not in text:
        raise RuntimeError("holdout report check not found")
    audit.write_text(text.replace(old, new, 1), encoding="utf-8")


def rewrite_docs(moved_dirs: list[str]) -> None:
    targets = [ROOT / "docs" / "SYSTEM_OVERVIEW_CN.md", * (ROOT / "configs").rglob("*.json")]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        updated = text
        for name in sorted(moved_dirs, key=len, reverse=True):
            updated = re.sub(
                rf"(?<![\w./-]){re.escape(name)}/",
                f"results/{name}/",
                updated,
            )
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def main() -> None:
    mapping, moved_dirs = move_root_entries()
    rewritten = rewrite_literals(mapping)
    rewrite_publishers()
    rewrite_docs(moved_dirs)
    (REPORTS / "README.md").write_text(
        "# 历史过程报告\n\n这些 Markdown 是各阶段审计和实验记录，已从仓库根目录移出。当前系统结论以 `docs/SYSTEM_OVERVIEW_CN.md` 为准。\n",
        encoding="utf-8",
    )
    print(f"mapped={len(mapping)} dirs={len(moved_dirs)} rewritten_files={rewritten}")


if __name__ == "__main__":
    main()
