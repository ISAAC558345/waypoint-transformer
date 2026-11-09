"""Locate the project root from this package, not from the working directory."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Walk upward until the project marker is found. This does not require .git."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        marker = candidate / "src" / "wt" / "common" / "paths.py"
        if marker.is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("project root was not found from wt.common.paths")