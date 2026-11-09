"""Read-only resource locations and legacy logical paths."""
from __future__ import annotations

import json
from pathlib import Path

from wt.common.paths import repo_root


def _table() -> dict:
    return json.loads((repo_root() / "configs" / "resources.json").read_text(encoding="utf-8"))


def checkpoint(name: str) -> Path:
    relative = _table()["checkpoints"][name]
    return repo_root() / relative


def policy_file() -> Path:
    return repo_root() / _table()["policy"]


def resolve_logical(logical: str) -> Path:
    """Map a historical logical path to its current file. Longest prefix wins."""
    normalized = logical.replace("\\", "/").lstrip("./")
    table = _table()["logical_paths"]
    if normalized in table:
        return repo_root() / table[normalized]
    match = ""
    for prefix in table:
        if (normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/")) and len(prefix) > len(match):
            match = prefix
    if not match:
        return repo_root() / normalized
    suffix = normalized[len(match):].lstrip("/")
    base = repo_root() / table[match]
    return base / suffix if suffix else base
