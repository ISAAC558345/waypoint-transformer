"""Compare unittest signatures against the design-baseline worktree."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASELINE = Path(r"E:\srdp\wt_m0_baseline")
BASELINE_LOG = Path(r"E:\srdp\m0_baseline_unittest.log")
OUT = ROOT / "results" / "ocean_observation_m0"
SUMMARY = re.compile(r"^Ran (\d+) tests in .*")
FAILED = re.compile(r"^FAILED \(failures=(\d+), errors=(\d+), skipped=(\d+)\)$")
OK_SKIP = re.compile(r"^OK \(skipped=(\d+)\)$")


def signatures(text: str) -> list:
    found = []
    for line in text.splitlines():
        if line.startswith("FAIL:") or line.startswith("ERROR:"):
            found.append(line.strip())
    return sorted(set(found))


def summary(text: str) -> dict:
    tests_run = failures = errors = skipped = None
    for line in text.splitlines():
        ran = SUMMARY.match(line.strip())
        if ran:
            tests_run = int(ran.group(1))
        failed = FAILED.match(line.strip())
        if failed:
            failures = int(failed.group(1))
            errors = int(failed.group(2))
            skipped = int(failed.group(3))
        ok_skip = OK_SKIP.match(line.strip())
        if ok_skip:
            failures = 0
            errors = 0
            skipped = int(ok_skip.group(1))
    if tests_run is None:
        raise RuntimeError("unittest summary missing")
    return {
        "tests_run": tests_run,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def run_suite(root: Path, log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for key in ("GIT_INDEX_FILE", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE"):
        env.pop(key, None)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )


def main() -> None:
    package = ROOT / "src" / "wt" / "ocean_observation_m0"
    test_file = ROOT / "tests" / "test_ocean_observation_m0.py"
    overlay_package = BASELINE / "src" / "wt" / "ocean_observation_m0"
    overlay_test = BASELINE / "tests" / "test_ocean_observation_m0.py"
    overlay_log = OUT / "overlay_tests.log"
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASELINE_LOG, OUT / "baseline_tests.log")
    try:
        shutil.copytree(package, overlay_package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(test_file, overlay_test)
        run_suite(BASELINE, overlay_log)
    finally:
        shutil.rmtree(overlay_package, ignore_errors=True)
        overlay_test.unlink(missing_ok=True)
    baseline_text = (OUT / "baseline_tests.log").read_text(encoding="utf-8", errors="replace")
    overlay_text = overlay_log.read_text(encoding="utf-8", errors="replace")
    base_sigs = signatures(baseline_text)
    over_sigs = signatures(overlay_text)
    new_sigs = sorted(set(over_sigs) - set(base_sigs))
    lost_sigs = sorted(set(base_sigs) - set(over_sigs))
    related = "not_run"
    related_log = OUT / "related_tests.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with related_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.test_ocean_observation_m0"],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    related = "OK" if proc.returncode == 0 else f"exit_{proc.returncode}"
    payload = {
        "baseline": summary(baseline_text),
        "overlay": summary(overlay_text),
        "baseline_signature_count": len(base_sigs),
        "overlay_signature_count": len(over_sigs),
        "new_failure_count": len(new_sigs),
        "new_failure_signatures": new_sigs,
        "lost_baseline_signature_count": len(lost_sigs),
        "lost_baseline_signatures": lost_sigs,
        "related_tests": related,
        "unittest_processes": 3,
        "suite_is_green": False,
        "historical_failures_retained": True,
    }
    (OUT / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("baseline", "overlay", "new_failure_count", "lost_baseline_signature_count", "related_tests")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
