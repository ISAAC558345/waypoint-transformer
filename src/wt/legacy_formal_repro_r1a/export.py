"""Write a portable view beside the saved R1 artifacts, never over them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wt.legacy_formal_repro_r1a.read import assert_export_target, build_view


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a derived R1 view to a new directory.")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    view = build_view(args.artifact_root)
    dest = assert_export_target(args.artifact_root, args.output)
    dest.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_environment_steps": view["source_environment_steps"],
        "source_steps_are_a_new_run": False,
        "reader_environment_steps": 0,
        "arms": view["arms"],
        "pairs": view["pairs"],
        "missing_original_summaries": view["missing_original_summaries"],
    }
    _write(dest / "derived_manifest.json", view)
    _write(dest / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
