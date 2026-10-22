"""Project-local Windows temporary-directory and spawn-safe process utilities."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TMP_RUNTIME = ROOT / "round_36" / "tmp_runtime"
TMP_TESTS = ROOT / "round_36" / "tmp_tests"


def configure_project_tempdir(kind: str = "runtime") -> Path:
    """Set only this process and inherited child process temporary locations."""
    target = TMP_RUNTIME if kind == "runtime" else TMP_TESTS
    target.mkdir(parents=True, exist_ok=True)
    value = str(target)
    for key in ("TEMP", "TMP", "TMPDIR"):
        os.environ[key] = value
    tempfile.tempdir = value
    return target


def temp_worker(value: int) -> tuple[int, bool, str]:
    """Top-level pickle-safe worker for spawn and tempfile smoke tests."""
    path = Path(tempfile.mkdtemp(prefix="round36_worker_"))
    probe = path / "probe.txt"
    probe.write_text(str(value), encoding="utf-8")
    content = probe.read_text(encoding="utf-8")
    shutil.rmtree(path)
    return int(value), content == str(value), tempfile.gettempdir()


def process_smoke(workers: int) -> list[tuple[int, bool, str]]:
    """Spawn independent workers with deterministic dispatch and ordering."""
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
        return list(pool.map(temp_worker, range(int(workers))))


__all__ = ["TMP_RUNTIME", "TMP_TESTS", "configure_project_tempdir", "process_smoke", "temp_worker"]
