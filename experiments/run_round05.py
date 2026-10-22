"""Repository-local entry point for the Round-5 experiment runner."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from wt.round5.round5_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
