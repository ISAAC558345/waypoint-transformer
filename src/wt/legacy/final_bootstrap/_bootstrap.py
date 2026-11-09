"""Ensure `src/` is on sys.path so `import wt` works for `python -m src.wt.legacy.final_bootstrap.*`."""
from __future__ import annotations

import sys
from pathlib import Path
from wt.common.paths import repo_root

_ROOT = repo_root()
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))