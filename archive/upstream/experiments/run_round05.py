"""Repository-local entry point for the Round-5 experiment runner."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = repo_root()
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from wt.legacy.causal_offline_policy.round5_runner import main
from wt.common.paths import repo_root


if __name__ == "__main__":
    raise SystemExit(main())