"""Command-line entrypoint for the audit-only Round-13 formal run."""
from __future__ import annotations
import argparse
from pathlib import Path
from .audit import run_round13

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.formal:
        parser.error("Round-13 exposes only --formal; it never trains a model")
    print(run_round13(args.root, output=args.output))

if __name__ == "__main__":
    main()
