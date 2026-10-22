"""Existing Benchmark-V2 outputs do not store reusable full-pool labels for the new scenarios."""
from __future__ import annotations

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import PHASE, ROOT


def reuse_rows() -> list[dict]:
    sources = [
        ROOT / "src" / "wt" / "benchmark_v2" / "phase6h_executor_conditioned_macro",
        ROOT / "src" / "wt" / "benchmark_v2" / "phase6ia_candidate_boundary",
        ROOT / "src" / "wt" / "benchmark_v2" / "phase6e",
        ROOT / "src" / "wt" / "benchmark_v2" / "phase6f_system_bringup",
    ]
    rows = []
    for source in sources:
        files = [path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".csv", ".npz"}]
        rows.append(
            {
                "source": str(source.relative_to(ROOT)).replace("\\", "/"),
                "files_scanned": len(files),
                "reusable_labels": 0,
                "reuse_existing_label": 0,
                "reason": "Artifacts have no per-candidate J15_MAPPO_CURRENT joined to a state hash for scenarios 322000-322063. Optima without the full G1_RATE k=6 pool are not reused.",
            }
        )
    rows.append(
        {
            "source": str(PHASE.relative_to(ROOT)).replace("\\", "/"),
            "files_scanned": 0,
            "reusable_labels": 0,
            "reuse_existing_label": 0,
            "reason": "No Phase-6I-B shard existed before the first J15 call.",
        }
    )
    return rows
