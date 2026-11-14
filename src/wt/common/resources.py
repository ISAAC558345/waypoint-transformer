"""Locate current resources and resolve historical file aliases without moving bytes."""
from __future__ import annotations

import json
from pathlib import Path

from wt.common.paths import repo_root


def _table() -> dict:
    return json.loads((repo_root() / "configs" / "resources.json").read_text(encoding="utf-8"))


def checkpoint(name: str) -> Path:
    return repo_root() / _table()["checkpoints"][name]


def policy_file() -> Path:
    return repo_root() / _table()["policy"]


def resolve_logical(logical: str) -> Path:
    """Resolve exact file moves and composed directory renames to a final path."""
    normalized = str(logical).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if Path(normalized).is_absolute():
        return Path(normalized)
    config = _table()
    aliases = {}
    index = config.get("legacy_path_index")
    if index:
        aliases.update(json.loads((repo_root() / index).read_text(encoding="utf-8"))["paths"])
    aliases.update(config.get("logical_paths", {}))
    seen = set()
    for _ in range(len(aliases) + 1):
        if normalized in seen:
            raise ValueError(f"Cyclic resource alias: {logical}")
        seen.add(normalized)
        match = normalized if normalized in aliases else ""
        if not match:
            for prefix in aliases:
                if normalized.startswith(prefix.rstrip("/") + "/") and len(prefix) > len(match):
                    match = prefix
        if not match:
            return repo_root() / normalized
        target = aliases[match] + normalized[len(match):]
        if target == normalized:
            return repo_root() / target
        normalized = target
    raise ValueError(f"Resource alias chain is too long: {logical}")
