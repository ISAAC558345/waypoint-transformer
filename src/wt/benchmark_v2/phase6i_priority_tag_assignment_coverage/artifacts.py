"""Shared artifact writers. Boolean cells stay 1 or 0 and come before the integer check."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from .protocol import ROOT


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (list, dict)):
        return json.dumps(plain(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: cell(row.get(key)) for key in fields})


def git_bytes(commit: str, relative: str) -> bytes:
    proc = subprocess.run(
        ["git", "-c", "core.longpaths=true", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot read {relative} at {commit}")
    return proc.stdout


def text_matches(path: Path, blob: bytes) -> bool:
    current = path.read_bytes()
    if current == blob:
        return True
    if path.suffix in {".py", ".md", ".json", ".csv", ".txt"}:
        return current.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
    return False
