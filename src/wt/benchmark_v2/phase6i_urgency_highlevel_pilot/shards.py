"""Read frozen Phase-6I-B shards from git blobs. This module does not write the dataset."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _paths

from .protocol import DATASET, FROZEN_SHARD_COMMIT, ROOT, frozen_relpath


class ShardError(RuntimeError):
    """Raised when a frozen shard cannot be aligned with its manifest."""


def _git_blob(relpath: str) -> bytes:
    spec = f"{FROZEN_SHARD_COMMIT}:{relpath}"
    completed = subprocess.run(
        ["git", "-c", "core.longpaths=true", "cat-file", "blob", spec],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ShardError(f"frozen blob is unavailable: {spec}: {message}")
    return bytes(completed.stdout)


def _refuse_dataset_write(path: Path) -> None:
    resolved = Path(path).resolve()
    dataset = DATASET.resolve()
    if resolved == dataset or dataset in resolved.parents:
        raise ShardError(f"stage D0 refused to write inside the frozen dataset: {resolved}")


def read_frozen_bytes(scenario_id: int, time_step: int, kind: str) -> bytes:
    """Return the historical shard bytes. An on-disk copy must match and is never replaced."""
    relpath = frozen_relpath(scenario_id, time_step, kind)
    blob = _git_blob(relpath)
    disk = _paths(int(scenario_id), int(time_step))[kind]
    if disk.is_file():
        local = disk.read_bytes()
        if hashlib.sha256(local).digest() != hashlib.sha256(blob).digest():
            raise ShardError(f"on-disk {kind} does not match frozen blob {relpath}")
    return blob


def load_manifest(scenario_id: int, time_step: int) -> dict:
    disk = _paths(int(scenario_id), int(time_step))["manifest"]
    blob = read_frozen_bytes(scenario_id, time_step, "manifest")
    if not disk.is_file():
        raise ShardError(f"working tree manifest is missing: {disk}")
    if hashlib.sha256(disk.read_bytes()).digest() != hashlib.sha256(blob).digest():
        raise ShardError(f"working tree manifest does not match the frozen blob for {scenario_id} t={time_step}")
    payload = json.loads(blob.decode("utf-8"))
    if int(payload["scenario_id"]) != int(scenario_id) or int(payload["time_step"]) != int(time_step):
        raise ShardError("manifest identity does not match the requested state")
    if str(payload.get("split")) != "train":
        raise ShardError("manifest split is not train")
    return payload


def load_npz(scenario_id: int, time_step: int, kind: str) -> tuple[dict[str, np.ndarray], str]:
    if kind not in {"state", "candidates"}:
        raise ShardError(f"npz loader refuses {kind}")
    raw = read_frozen_bytes(scenario_id, time_step, kind)
    digest = hashlib.sha256(raw).hexdigest()
    with np.load(io.BytesIO(raw)) as shard:
        arrays = {key: np.asarray(shard[key]) for key in shard.files}
    return arrays, digest


def dataset_fingerprint(scenarios: tuple[int, ...]) -> list[dict]:
    """Hash the working-tree dataset files for the pilot scenarios."""
    rows = []
    for scenario in scenarios:
        folder = DATASET / "train" / f"scenario_{int(scenario)}"
        if not folder.is_dir():
            rows.append({"scenario_id": int(scenario), "missing_directory": True})
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return rows
