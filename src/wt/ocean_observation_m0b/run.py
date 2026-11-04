"""Write the derived M0B view into its own directory."""
from __future__ import annotations

import json
from pathlib import Path

from wt.ocean_observation_m0b.derive import ROOT, build_view

OUTPUT = ROOT / "results" / "ocean_observation_m0b"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    view = build_view()
    targets = (
        OUTPUT / "derived_rows.json",
        OUTPUT / "difference_table.json",
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write(
        OUTPUT / "derived_rows.json",
        {
            "source_result_commit": view["source_result_commit"],
            "environment_steps": 0,
            "independent_experiment": False,
            "equivalence_label": view["equivalence_label"],
            "derived_method_row_count": view["derived_method_row_count"],
            "revised_c_rows": view["revised_c_rows"],
            "b2_rows": view["b2_rows"],
            "timing_corrections": view["timing_corrections"],
            "identity": view["identity"],
            "energy_limit_J": view["energy_limit_J"],
        },
    )
    _write(
        OUTPUT / "difference_table.json",
        {
            "rule": "B2 and C both use the shared half-duplex order",
            "rows": view["difference_table"],
            "proposed_cross_uav_overlaps": view["proposed_cross_uav_overlaps"],
            "scheduled_overlaps": view["scheduled_overlaps"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
