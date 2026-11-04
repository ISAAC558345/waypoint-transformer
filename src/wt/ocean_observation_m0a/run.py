"""Write the frozen geometry, then the one analytical comparison."""
from __future__ import annotations

import json
from pathlib import Path

from wt.ocean_observation_m0a.compare import comparison_table
from wt.ocean_observation_m0a.geometry import geometry_manifest
from wt.ocean_observation_m0a.schedule import build_schedule

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results" / "ocean_observation_m0a"


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_geometry() -> dict:
    payload = geometry_manifest()
    _dump(OUTPUT / "geometry_manifest.json", payload)
    return payload


def write_comparison() -> dict:
    geometry_path = OUTPUT / "geometry_manifest.json"
    if not geometry_path.is_file():
        raise RuntimeError("the geometry manifest is missing")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    schedule = build_schedule(geometry["geometries"])
    table = comparison_table(schedule)
    _dump(OUTPUT / "schedule.json", schedule)
    _dump(OUTPUT / "comparison.json", table)
    return table


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        table = write_comparison()
        print(table["conclusion"], len(table["rows"]))
    else:
        payload = write_geometry()
        for row in payload["geometries"]:
            print(row["geometry"], row["fixed"]["xy_m"], row["fixed"]["covers_all_buoys"], [(item["uav"], item["receive_xy_m"], item["covers"]) for item in row["moving"]["visits"]])
