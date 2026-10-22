"""Read the saved full-episode comparisons. Does not rerun episodes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = (
    ("validation 322048-322063", ROOT / "results" / "phase6iab_planner_stability" / "summary.json"),
    ("reserved test 322064-322079", ROOT / "results" / "phase6iac_independent_holdout" / "summary.json"),
)


def main() -> None:
    for label, path in SUMMARIES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        paired = payload["paired"]
        print(
            f"{label}: mean={paired['mean']:.3f} bit "
            f"ci=[{paired['ci_low']:.3f}, {paired['ci_high']:.3f}] "
            f"positive={payload['positive_scenarios']} "
            f"tie={payload['tie_scenarios']} "
            f"negative={payload['negative_scenarios']}"
        )


if __name__ == "__main__":
    main()
